import time

import numpy as np
import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from .utils import preprocess, reshape_patches

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

# Map TensorRT dtypes to torch dtypes
_trt_to_torch = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32:   torch.int32,
    trt.int8:    torch.int8,
}


class DINOv2_TRT:
    """TensorRT-accelerated DINOv2 inference.

    Args:
        model_weights: Path to serialized TensorRT engine (.trt file).
        img_size: int for aspect-ratio preserving resize (short side), or
            (h, w) tuple for exact resize. Default 256.
        timing: Print per-call timing breakdown. Default False.
    """

    def __init__(self, model_weights, img_size=256, timing=False):
        self.timing = timing
        if isinstance(img_size, int):
            self.imgsz = img_size
            self.fixed_size = None
        else:
            self.imgsz = tuple(img_size)
            self.fixed_size = self.imgsz  # (h, w)

        with open(model_weights, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = self.engine.get_tensor_name(0)
        self._current_input_shape = None
        self.input_host = None
        self.input_dev = None
        self.output_tensors = {}
        self.output_names = []

        # Discover output tensor names
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                self.output_names.append(name)

        # Pre-allocate buffers for fixed-size mode
        if self.fixed_size is not None:
            h, w = self.fixed_size
            self._setup_buffers((1, 3, h, w))

    def _setup_buffers(self, input_shape):
        """(Re)allocate I/O buffers for the given input shape."""
        if input_shape == self._current_input_shape:
            return
        self._current_input_shape = input_shape

        self.context.set_input_shape(self.input_name, input_shape)

        # Input buffer
        dtype = self.engine.get_tensor_dtype(self.input_name)
        np_dtype = trt.nptype(dtype)
        self.input_host = cuda.pagelocked_empty(int(np.prod(input_shape)), np_dtype)
        self.input_dev = cuda.mem_alloc(self.input_host.nbytes)
        self.context.set_tensor_address(self.input_name, int(self.input_dev))

        # Output buffers
        self.output_tensors = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = _trt_to_torch[dtype]
            tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
            self.output_tensors[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

    def _infer(self, inp):
        self._setup_buffers(inp.shape)
        np.copyto(self.input_host, inp.flatten())
        cuda.memcpy_htod_async(self.input_dev, self.input_host, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        return [self.output_tensors[n] for n in self.output_names]

    def warmup(self, iters=3):
        """Run dummy inferences to warm up the GPU."""
        if self.fixed_size is not None:
            h, w = self.fixed_size
        else:
            h, w = self.imgsz, self.imgsz
        dummy = np.zeros((1, 3, h, w), dtype=np.float32)
        for _ in range(iters):
            self._infer(dummy)

    def embed(self, bgr_img, reshape=False):
        """Run DINOv2 feature extraction on a BGR image.

        Args:
            bgr_img: Input image in BGR format (numpy array, H x W x 3).
            reshape: If True, reshape output to spatial grid (1, h, w, D)
                based on the preprocessed image. Default False.

        Returns:
            If reshape=False: torch.Tensor on GPU of shape (1, num_patches+1, 768).
            If reshape=True:  torch.Tensor on GPU of shape (1, h, w, 768).
        """
        t0 = time.time()
        inp = preprocess(bgr_img, self.imgsz)
        t1 = time.time()
        raw = self._infer(inp)
        t2 = time.time()
        out = raw[0]  # (1, num_patches+1, 768)
        if reshape:
            out = reshape_patches(out, inp.shape[2:])
        t3 = time.time()
        if self.timing:
            print(f"[DINOv2_TRT] preprocess: {t1-t0:.4f}s, inference: {t2-t1:.4f}s, postprocess: {t3-t2:.4f}s")
        return out
