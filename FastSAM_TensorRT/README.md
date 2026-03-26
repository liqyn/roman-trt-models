# Fast Segment Anything — TensorRT

TensorRT deployment of [FastSAM](https://docs.ultralytics.com/models/fast-sam) from Ultralytics

**Refer from**
[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)

#### Rewritten by @liqyn from https://github.com/ChuRuaNh0/FastSam_Awsome_TensorRT

## Python API

### Convert weights

```python
from fastsam_trt import pt2onnx, onnx2trt

pt2onnx("FastSAM-x.pt", "FastSAM-x_256.onnx", img_size=256)
onnx2trt("FastSAM-x_256.onnx", "FastSAM-x_256.trt", img_size=256, fp16=True)
```

`img_size` is the square input resolution. Images are letterbox-preprocessed (aspect-ratio preserving resize + padding to `img_size × img_size`) during inference, so the engine always receives a fixed square input. Larger values increase accuracy but cost more compute.

### Run inference

```python
import cv2
from fastsam_trt import FastSAM_TRT

model = FastSAM_TRT("FastSAM-x_256.trt", img_size=256, conf=0.25, iou=0.7, timing=True)
model.warmup()

img = cv2.imread("image.jpg")
masks = model.segment(img)  # (N, H, W) tensor on GPU
```

`img_size` must match the engine's input resolution. Input images of any size are automatically letterbox-preprocessed to fit.

### Visualize masks

```python
from fastsam_trt import overlay

masks_np = masks.cpu().numpy()
viz = img.copy()
for mask in masks_np:
    color = tuple(np.random.randint(0, 255, 3).tolist())
    viz = overlay(viz, mask, color=color, alpha=0.5)
cv2.imwrite("result.jpg", viz)
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
