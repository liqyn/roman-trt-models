# DINOv2 — TensorRT

TensorRT deployment of [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) from HuggingFace Transformers.

**Model details**
- Patch size: 14 px
- Feature dimension: 768
- Output: `last_hidden_state` — shape `(1, num_patches+1, 768)`; index 0 is the CLS token, indices 1: are patch tokens

**Refer from** [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)


## Python API

### Convert weights

```python
from dinov2_trt import pt2onnx, onnx2trt

# HuggingFace -> ONNX (requires transformers)
pt2onnx("facebook/dinov2-base", "dinov2-base_256.onnx", img_size=256)

# Fixed-size engine
onnx2trt("dinov2-base_256.onnx", "dinov2-base_256.trt", img_size=256, fp16=True)

# Dynamic-size engine (for varying aspect ratios)
onnx2trt("dinov2-base_256.onnx", "dinov2-base_dynamic.trt",
         min_sz=(1, 256, 256), opt_sz=(1, 256, 512), max_sz=(1, 768, 768),
         fp16=True)
```

### Run inference

```python
import cv2
from dinov2_trt import DINOv2_TRT

# Fixed-size: all images resized to exact (h, w)
model = DINOv2_TRT("dinov2-base_256.trt", img_size=(256, 256), timing=True)

# Dynamic aspect-ratio: shortest side scaled to 256, aspect ratio preserved
model = DINOv2_TRT("dinov2-base_dynamic.trt", img_size=256, timing=True)

model.warmup()

img = cv2.imread("image.jpg")
features = model.embed(img)               # (1, num_patches+1, 768)
patches  = model.embed(img, reshape=True) # (1, h, w, 768) — CLS token dropped, reshaped to preprocessed image dimensions
```

### Visualize features

```python
from dinov2_trt import visualize_features

viz = visualize_features(features, img_shape=img.shape[:2])
cv2.imwrite("features.jpg", viz)
```

## CLI Scripts

From the `scripts/` directory:

### Export ONNX

```bash
python3 pt2onnx.py --model <hf_model_name_or_path> --output <onnx_path> --img-size <h> [<w>]
```

`--img-size` accepts a single int (square) or two ints `h w`. Default: `256`.

### Export TensorRT

```bash
# Fixed-size engine
python3 onnx2trt.py --onnx <onnx_path> --output <trt_path> --img-size <h> [<w>]

# Dynamic-size engine
python3 onnx2trt.py --onnx <onnx_path> --output <trt_path> \
    --min-sz 1,256,256 --opt-sz 1,256,512 --max-sz 1,768,768

# Add --fp16 for half-precision
python3 onnx2trt.py --onnx <onnx_path> --output <trt_path> --fp16
```

`--min-sz`, `--opt-sz`, `--max-sz` are comma-separated `batch,h,w` tuples (minimum, optimal, maximum), used to export models that support dynamic input sizes. When provided, `--img-size` is ignored.

### Inference

```bash
# Default HuggingFace model
python3 infer_default.py --model <hf_model_name_or_path> --img-size <h> [<w>]

# ONNX model
python3 infer_onnx.py --weights <onnx_path> --img-size <h> [<w>]

# TensorRT engine
python3 infer_trt.py --weights <trt_path> --img-size <h> [<w>]
```