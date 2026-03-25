import time

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch

from .utils import preprocess, postprocess

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

# Map TensorRT dtypes to torch dtypes
_trt_to_torch = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
}


class FastSAM_TRT:
    """TensorRT-accelerated FastSAM inference.

    Args:
        model_weights: Path to serialized TensorRT engine (.trt file).
        img_size: Input image size (square). Default 256.
        retina_masks: Use native (retina) mask processing. Default True.
        conf: Confidence threshold for NMS. Default 0.25.
        iou: IoU threshold for NMS. Default 0.7.
        agnostic_nms: Class-agnostic NMS. Default False.
    """

    def __init__(
        self,
        model_weights,
        img_size=256,
        retina_masks=True,
        conf=0.25,
        iou=0.7,
        agnostic_nms=False,
        timing=False
    ):
        self.imgsz = img_size
        self.retina_masks = retina_masks
        self.conf = conf
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.timing = timing

        with open(model_weights, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        input_name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_name, (1, 3, img_size, img_size))

        # Allocate buffers: page-locked host for input, torch CUDA tensors for outputs
        all_output_names = []
        self.input_host = None
        self.input_dev = None
        self.input_name = input_name
        self.output_tensors = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                # Input: page-locked host buffer + pycuda device buffer for htod copy
                np_dtype = trt.nptype(dtype)
                self.input_host = cuda.pagelocked_empty(int(np.prod(shape)), np_dtype)
                self.input_dev = cuda.mem_alloc(self.input_host.nbytes)
                self.context.set_tensor_address(name, int(self.input_dev))
            else:
                # Output: torch CUDA tensor — TRT writes directly into it
                torch_dtype = _trt_to_torch[dtype]
                tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
                self.output_tensors[name] = tensor
                self.context.set_tensor_address(name, tensor.data_ptr())
                all_output_names.append(name)

        # Only read the two outputs postprocess needs (detections and proto)
        self.output_names = [all_output_names[0], all_output_names[5]]

    def _infer(self, inp):
        np.copyto(self.input_host, inp.flatten())
        cuda.memcpy_htod_async(self.input_dev, self.input_host, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        return [self.output_tensors[n] for n in self.output_names]

    def warmup(self, iters=3):
        """Run dummy inferences to warm up the GPU."""
        dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        for _ in range(iters):
            self._infer(dummy)

    def segment(self, bgr_img):
        """Run segmentation on a BGR image.

        Args:
            bgr_img: Input image in BGR format (numpy array).

        Returns:
            Mask tensor on GPU of shape (N, H, W) where N is the number of detected objects.
        """
        t0 = time.time()
        inp = preprocess(bgr_img, self.imgsz)
        t1 = time.time()
        preds = self._infer(inp)
        t2 = time.time()
        result = postprocess(preds, inp, bgr_img, self.retina_masks, self.conf, self.iou, self.agnostic_nms)
        t3 = time.time()
        if self.timing: 
            print(f"[FastSAM_TRT] preprocess: {t1-t0:.4f}s, inference: {t2-t1:.4f}s, postprocess: {t3-t2:.4f}s")
        return result[0]
