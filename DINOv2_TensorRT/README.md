# DINOv2 TensorRT

TensorRT deployment of [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) from HuggingFace Transformers.

**Model details**
- Patch size: 14 px
- Feature dimension: 768
- Output: `last_hidden_state` — shape `(1, num_patches+1, 768)`; index 0 is the CLS token, indices 1: are patch tokens

**Refer from** [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

## Install

```
pip install -e .
```

To run tests and scripts:
```
cd scripts
```

### Test Default

```
python3 infer_default.py --model <hf_model_name_or_path>
```

## Export ONNX
```
python3 pt2onnx.py --model <hf_model_name_or_path> --output <onnx_weights_path> --img-size <square_image_size>
```

### Test ONNX

```
python3 infer_onnx.py --weights <onnx_weights_path> --img-size <square_image_size>
```

## Export TensorRT
```
python3 onnx2trt.py --onnx <onnx_weights_path> --output <trt_weights_path> --img-size <square_image_size> --opt-batch <optimal_batch_size> --max-batch <max_batch_size>
```
Add the `--fp16` flag to enable half-precision.

### Test TensorRT
```
python3 infer_trt.py --weights <trt_weights_path> --img-size <square_image_size>
```