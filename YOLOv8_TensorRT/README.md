# YOLOv8 COCO Detector — TensorRT

TensorRT-accelerated YOLOv8 object detection with 80 COCO classes. Exported to ONNX/TensorRT from [Ultralytics](https://docs.ultralytics.com/models/yolov8).

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
