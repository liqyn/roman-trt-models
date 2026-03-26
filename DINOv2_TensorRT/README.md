# DINOv2 — TensorRT

TensorRT deployment of [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) (and other DINO models) from HuggingFace Transformers.

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

`img_size` accepts a single int (square) or an `(h, w)` tuple.

`min_sz`, `opt_sz`, `max_sz` are `(batch, h, w)` tuples that define the TensorRT optimization profile for dynamic input sizes (minimum, optimal, maximum). When provided, `img_size` is ignored.

### Run inference

`img_size` controls preprocessing behavior:

- **`int`** — aspect-ratio preserving resize: the shortest side is scaled to `img_size`, and the aspect ratio is preserved. Use this with a dynamic-size engine.
- **`(h, w)` tuple** — exact resize to the given dimensions. Use this with an appropriate fixed-size engine.

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
patches  = model.embed(img, reshape=True) # (1, h, w, 768) — CLS token dropped
```

For dynamic engines, the preprocessed dimensions must fall within the `min_sz`–`max_sz` range defined during conversion.

### Visualize features

```python
from dinov2_trt import visualize_features

viz = visualize_features(features, img_shape=img.shape[:2])
cv2.imwrite("features.jpg", viz)
```

## CLI Scripts

From the `scripts/` directory. All scripts mirror the Python API above.

```bash
# Export ONNX
python3 pt2onnx.py --model <hf_model_name_or_path> --output <onnx_path> --img-size <h> [<w>]

# Export TensorRT (fixed-size)
python3 onnx2trt.py --onnx <onnx_path> --output <trt_path> --img-size <h> [<w>] [--fp16]

# Export TensorRT (dynamic-size)
python3 onnx2trt.py --onnx <onnx_path> --output <trt_path> \
    --min-sz 1,256,256 --opt-sz 1,256,512 --max-sz 1,768,768 [--fp16]

# Inference (HF transformers, ONNX, TensorRT)
python3 infer_default.py --model <hf_model_name_or_path> --img-size <h> [<w>]
python3 infer_onnx.py --weights <onnx_path> --img-size <h> [<w>]
python3 infer_trt.py --weights <trt_path> --img-size <h> [<w>]
```
