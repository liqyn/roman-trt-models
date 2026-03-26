# YOLOv8 COCO Detector — TensorRT

TensorRT deployment of [YOLOv8](https://docs.ultralytics.com/models/yolov8) from Ultralytics. Detection model with COCO classes by default.

## Python API

### Convert weights

```python
from yolov8_trt import pt2onnx, onnx2trt

pt2onnx("yolov8n.pt", "yolov8n_640.onnx", img_size=640)
onnx2trt("yolov8n_640.onnx", "yolov8n_640.trt", img_size=640, fp16=True)
```

`img_size` is the square input resolution. Images are letterbox-preprocessed (aspect-ratio preserving resize + padding to `img_size × img_size`) during inference, so the engine always receives a fixed square input.

### Run inference

```python
import cv2
from yolov8_trt import YOLOv8_TRT

model = YOLOv8_TRT("yolov8n_640.trt", img_size=640, conf=0.25, iou=0.7, timing=True)
model.warmup()

img = cv2.imread("image.jpg")
dets = model.detect(img)  # (N, 6) tensor on GPU: [x1, y1, x2, y2, conf, class_id]
```

`img_size` must match the engine's input resolution. Input images of any size are automatically letterbox-preprocessed to fit.

### Visualize results

```python
from yolov8_trt import draw_detections, COCO_NAMES

result = draw_detections(img, dets, class_names=COCO_NAMES)
cv2.imwrite("result.jpg", result)
```

## CLI Scripts

From the `scripts/` directory:

```bash
# Export ONNX
python3 pt2onnx.py --weights <pt_weights_path> --output <onnx_path> --img-size <size>

# Export TensorRT
python3 onnx2trt.py --onnx <onnx_path> --output <trt_path> --img-size <size> \
    --opt-batch <optimal_batch_size> --max-batch <max_batch_size> [--fp16]

# Inference (Ultralytics, ONNX, TensorRT)
python3 infer_default.py --weights <pt_weights_path> --img-size <size>
python3 infer_onnx.py --weights <onnx_path> --img-size <size>
python3 infer_trt.py --weights <trt_path> --img-size <size>
```
