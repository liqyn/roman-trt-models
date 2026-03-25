![](assets/logo.png)

# Fast Segment Anything

[[`📕Paper`](https://arxiv.org/pdf/2306.12156.pdf)] [[`🤗HuggingFace Demo`](https://huggingface.co/spaces/An-619/FastSAM)] [[`Colab demo`](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing)] [[`Replicate demo & API`](https://replicate.com/casia-iva-lab/fastsam)] [[`Model Zoo`](#model-checkpoints)] [[`BibTeX`](#citing-fastsam)]

The **Fast Segment Anything Model(FastSAM)** is a CNN Segment Anything Model trained by only 2% of the SA-1B dataset published by SAM authors. The FastSAM achieve a comparable performance with
the SAM method at **50× higher run-time speed**.

**🍇 Refer from**
[Original](https://github.com/CASIA-IVA-Lab/FastSAM)

#### Rewritten by @andyli27 from https://github.com/ChuRuaNh0/FastSam_Awsome_TensorRT

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
python3 infer_default.py --weights <pt_weights_path>
```

## Export ONNX
```
python3 pt2onnx.py --weights <pt_weights_path> --output <onnx_weights_path> --img-size <square_image_size>
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