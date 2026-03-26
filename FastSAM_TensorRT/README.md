![](assets/logo.png)

# Fast Segment Anything — TensorRT

[[`📕Paper`](https://arxiv.org/pdf/2306.12156.pdf)] [[`🤗HuggingFace Demo`](https://huggingface.co/spaces/An-619/FastSAM)] [[`Colab demo`](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing)] [[`Replicate demo & API`](https://replicate.com/casia-iva-lab/fastsam)] [[`Model Zoo`](#model-checkpoints)] [[`BibTeX`](#citing-fastsam)]

The **Fast Segment Anything Model(FastSAM)** is a CNN Segment Anything Model trained by only 2% of the SA-1B dataset published by SAM authors. The FastSAM achieve a comparable performance with
the SAM method at **50× higher run-time speed**.

**🍇 Refer from**
[Original](https://github.com/CASIA-IVA-Lab/FastSAM)

#### Rewritten by @andyli27 from https://github.com/ChuRuaNh0/FastSam_Awsome_TensorRT

## Python API

### Convert weights

```python
from fastsam_trt import pt2onnx, onnx2trt

pt2onnx("FastSAM-x.pt", "FastSAM-x_256.onnx", img_size=256)
onnx2trt("FastSAM-x_256.onnx", "FastSAM-x_256.trt", img_size=256, fp16=True)
```

### Run inference

```python
import cv2
from fastsam_trt import FastSAM_TRT

model = FastSAM_TRT("FastSAM-x_256.trt", img_size=256, conf=0.25, iou=0.7, timing=True)
model.warmup()

img = cv2.imread("image.jpg")
masks = model.segment(img)  # (N, H, W) tensor on GPU
```

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