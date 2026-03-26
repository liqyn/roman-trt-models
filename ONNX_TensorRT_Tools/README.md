# ONNX TensorRT Tools

Shared utilities for TensorRT-accelerated inference, used by the FastSAM, YOLOv8, and DINOv2 TRT packages.

## Components

### TRTEngine

Base class for TensorRT inference. Handles engine loading, I/O buffer allocation, async inference, and warmup. Supports both fixed and dynamic input shapes.

```python
from onnx_trt_tools import TRTEngine

class MyModel(TRTEngine):
    def __init__(self, weights, img_size=640, timing=False):
        super().__init__(weights, input_shape=(1, 3, img_size, img_size), timing=timing)
        # Optionally filter which outputs _infer() returns:
        self.output_names = [self._all_output_names[0]]

    def warmup(self, iters=3):
        super().warmup((1, 3, 640, 640), iters)
```

For dynamic input shapes, omit `input_shape` and buffers will be (re)allocated on each call to `_infer()`:

```python
super().__init__(weights, input_shape=None)
```

### onnx2trt

Converts an ONNX model to a TensorRT engine with optimization profiles.

```python
from onnx_trt_tools import onnx2trt

# Fixed-size engine
onnx2trt("model.onnx", "model.trt", input_name="images",
         min_sz=(1, 640, 640), opt_sz=(1, 640, 640), max_sz=(1, 640, 640))

# Dynamic-size engine with FP16
onnx2trt("model.onnx", "model.trt", input_name="pixel_values",
         min_sz=(1, 256, 256), opt_sz=(1, 256, 512), max_sz=(1, 768, 768),
         fp16=True)
```

### letterbox_preprocess

YOLO-style letterbox preprocessing: aspect-ratio preserving resize with padding, BGR-to-RGB conversion, and normalization to `[0, 1]`.

```python
from onnx_trt_tools import letterbox_preprocess

inp = letterbox_preprocess(bgr_img, imgsz=640)  # (1, 3, 640, 640) float32
```
