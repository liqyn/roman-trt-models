import numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

# Map TensorRT dtypes to torch dtypes
_trt_to_torch = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32:   torch.int32,
    trt.int8:    torch.int8,
}


class TRTEngine:
    """Base class for TensorRT-accelerated inference.

    Handles engine loading, I/O buffer allocation, inference execution,
    and warmup. Supports both fixed and dynamic input shapes.
    Uses only torch.cuda for GPU memory and stream management (no pycuda).

    Args:
        model_weights: Path to serialized TensorRT engine (.trt file).
        input_shape: Fixed input shape tuple (e.g. (1, 3, 256, 256)).
            If None, buffers are allocated dynamically on first inference.
        timing: Print per-call timing breakdown. Default False.
    """

    def __init__(self, model_weights, input_shape=None, timing=False):
        self.timing = timing
        with open(model_weights, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.input_name = self.engine.get_tensor_name(0)
        self._current_input_shape = None
        self.input_host = None
        self.input_dev = None
        self.output_tensors = {}

        # Discover all output tensor names
        self._all_output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                self._all_output_names.append(name)

        # By default, _infer returns all outputs; subclasses can override
        self.output_names = list(self._all_output_names)

        # Pre-allocate buffers for fixed-size mode
        if input_shape is not None:
            self._setup_buffers(input_shape)

    def _setup_buffers(self, input_shape):
        """(Re)allocate I/O buffers for the given input shape."""
        if input_shape == self._current_input_shape:
            return
        self._current_input_shape = input_shape

        self.context.set_input_shape(self.input_name, input_shape)

        # Input buffers: pinned host tensor + CUDA device tensor
        dtype = self.engine.get_tensor_dtype(self.input_name)
        torch_dtype = _trt_to_torch[dtype]
        self.input_host = torch.empty(input_shape, dtype=torch_dtype).pin_memory()
        self.input_dev = torch.empty(input_shape, dtype=torch_dtype, device='cuda')
        self.context.set_tensor_address(self.input_name, self.input_dev.data_ptr())

        # Output buffers (allocate ALL outputs for TRT, read only self.output_names)
        self.output_tensors = {}
        for name in self._all_output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = _trt_to_torch[dtype]
            tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
            self.output_tensors[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

    def _infer(self, inp):
        self._setup_buffers(inp.shape)
        self.input_host.copy_(torch.from_numpy(inp))
        with torch.cuda.stream(self.stream):
            self.input_dev.copy_(self.input_host, non_blocking=True)
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return [self.output_tensors[n] for n in self.output_names]

    def warmup(self, input_shape, iters=3):
        """Run dummy inferences to warm up the GPU."""
        dummy = np.zeros(input_shape, dtype=np.float32)
        for _ in range(iters):
            self._infer(dummy)
