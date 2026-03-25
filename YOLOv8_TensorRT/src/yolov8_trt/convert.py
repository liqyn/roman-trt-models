import torch

from onnx_trt_tools import onnx2trt as _onnx2trt


def pt2onnx(weights, output, img_size=640):
    """Convert a YOLOv8 detection PyTorch model to ONNX format.

    The ONNX model is always exported in float32. Use onnx2trt(..., fp16=True)
    to enable FP16 precision at the TensorRT level.

    Args:
        weights: Path to YOLOv8 .pt weights file.
        output: Output path for the ONNX model.
        img_size: Input image size (square). Default 640.
    """
    import onnx
    from onnxsim import simplify
    from ultralytics import YOLO

    device = torch.device("cuda")
    model = YOLO(weights).model.eval().to(device)

    img = torch.zeros(1, 3, img_size, img_size, device=device)
    dynamic = {
        'images':  {0: 'batch', 2: 'height', 3: 'width'},
        'output0': {0: 'batch', 2: 'anchors'},
    }

    torch.onnx.export(
        model, img, output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output0'],
        dynamic_axes=dynamic,
    )

    # Optimize the exported ONNX model
    onnx_model = onnx.load(output)
    onnx_model, ok = simplify(onnx_model)
    if not ok:
        print("Warning: YOLOv8 onnxsim simplification failed")

    onnx.save(onnx_model, output)
    print(f'Saved YOLOv8 ONNX model to {output}')


def onnx2trt(onnx_path, output, img_size=640, opt_batch=1, max_batch=1, fp16=False):
    """Convert a YOLOv8 ONNX model to a TensorRT engine.

    Args:
        onnx_path: Path to the ONNX model file.
        output: Output path for the TensorRT engine.
        img_size: Input image size (square). Default 640.
        opt_batch: Optimal batch size for the engine. Default 1.
        max_batch: Maximum batch size for the engine. Default 1.
        fp16: Enable FP16 precision in the TensorRT engine. Default False.
    """
    s = img_size
    _onnx2trt(
        onnx_path, output, input_name="images",
        min_sz=(1, s, s), opt_sz=(opt_batch, s, s), max_sz=(max_batch, s, s),
        fp16=fp16,
    )
    print(f"Saved YOLOv8 TRT engine to {output}")
