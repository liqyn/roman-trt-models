# YOLOv8 COCO Detector — TensorRT

TensorRT-accelerated YOLOv8 object detection with 80 COCO classes. Exported to ONNX/TensorRT from [Ultralytics](https://docs.ultralytics.com/models/yolov8).

## Python API

### Convert weights

```python
from yolov8_trt import pt2onnx, onnx2trt

pt2onnx("yolov8n.pt", "yolov8n_640.onnx", img_size=640)
onnx2trt("yolov8n_640.onnx", "yolov8n_640.trt", img_size=640, fp16=True)
```

### Run inference

```python
import cv2
from yolov8_trt import YOLOv8_TRT

model = YOLOv8_TRT("yolov8n_640.trt", img_size=640, conf=0.25, iou=0.7, timing=True)
model.warmup()

img = cv2.imread("image.jpg")
dets = model.detect(img)  # (N, 6) tensor on GPU: [x1, y1, x2, y2, conf, class_id]
```

### Visualize results

```python
from yolov8_trt import draw_detections, COCO_NAMES

result = draw_detections(img, dets, class_names=COCO_NAMES)
cv2.imwrite("result.jpg", result)
```

## CLI Scripts

From the `scripts/` directory:

### Export ONNX

```bash
python3 pt2onnx.py --weights <pt_weights_path> --output <onnx_path> --img-size <size>
```

### Export TensorRT

```bash
python3 onnx2trt.py --onnx <onnx_path> --output <trt_path> --img-size <size> \
    --opt-batch <optimal_batch_size> --max-batch <max_batch_size>

# Add --fp16 for half-precision
python3 onnx2trt.py --onnx <onnx_path> --output <trt_path> --fp16
```

### Inference

```bash
# Default PyTorch model
python3 infer_default.py --weights <pt_weights_path> --img-size <size>

# ONNX model
python3 infer_onnx.py --weights <onnx_path> --img-size <size>

# TensorRT engine
python3 infer_trt.py --weights <trt_path> --img-size <size>
```
