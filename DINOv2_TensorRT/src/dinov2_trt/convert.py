import torch

from onnx_trt_tools import onnx2trt as _onnx2trt


def pt2onnx(model_name, output, img_size=256):
    """Export a DINOv2 model from HuggingFace to ONNX format.

    The ONNX model is always exported in float32. Use onnx2trt(..., fp16=True)
    to enable FP16 precision at the TensorRT level.

    Args:
        model_name: HuggingFace model name or local path (e.g. 'facebook/dinov2-base').
        output: Output path for the ONNX model.
        img_size: int (square) or (h, w) tuple used for the dummy input shape.
            The exported ONNX model supports dynamic spatial dims regardless.
            Default 256.
    """
    import onnx
    from onnxsim import simplify
    from transformers import AutoModel

    class _Wrapper(torch.nn.Module):
        """Expose a single positional tensor input for ONNX export."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).last_hidden_state

    device = torch.device("cuda")
    # avoid using SDPA which is not supported by ONNX opset 17
    base_model = AutoModel.from_pretrained(model_name, attn_implementation="eager").eval().to(device)
    wrapper = _Wrapper(base_model)

    if isinstance(img_size, int):
        h, w = img_size, img_size
    else:
        h, w = img_size
    dummy = torch.zeros(1, 3, h, w, device=device)
    dynamic = {
        'pixel_values':      {0: 'batch', 2: 'height', 3: 'width'},
        'last_hidden_state': {0: 'batch', 1: 'sequence'},
    }

    torch.onnx.export(
        wrapper, dummy, output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['last_hidden_state'],
        dynamic_axes=dynamic,
    )

    # Optimize the exported ONNX model
    onnx_model = onnx.load(output)
    onnx_model, ok = simplify(onnx_model)
    if not ok:
        print("Warning: DINOv2 onnxsim simplification failed")

    onnx.save(onnx_model, output)
    print(f'Saved DINOv2 ONNX model to {output}')


def onnx2trt(onnx_path, output, img_size=256, min_sz=None, opt_sz=None, max_sz=None, fp16=False):
    """Convert an ONNX DINOv2 model to a TensorRT engine.

    For a fixed-size engine, pass ``img_size`` (int or (h, w) tuple).
    For a dynamic-range engine, pass ``min_sz``, ``opt_sz``, and ``max_sz``
    as 3-tuples of ``(batch, h, w)``.  When the explicit size tuples are
    provided, ``img_size`` is ignored.

    Args:
        onnx_path: Path to the ONNX model file.
        output: Output path for the TensorRT engine.
        img_size: Convenience param for fixed-size engines. int (square) or
            (h, w) tuple. Ignored when min_sz/opt_sz/max_sz are given.
            Default 256.
        min_sz: Min (batch, h, w) for the optimization profile.
        opt_sz: Optimal (batch, h, w) for the optimization profile.
        max_sz: Max (batch, h, w) for the optimization profile.
        fp16: Enable FP16 precision in the TensorRT engine. Default False.
    """
    if min_sz is None or opt_sz is None or max_sz is None:
        if isinstance(img_size, int):
            h, w = img_size, img_size
        else:
            h, w = img_size
        min_sz = (1, h, w)
        opt_sz = (1, h, w)
        max_sz = (1, h, w)
    _onnx2trt(
        onnx_path, output, input_name="pixel_values",
        min_sz=min_sz, opt_sz=opt_sz, max_sz=max_sz,
        fp16=fp16,
    )
    print(f"Saved DINOv2 TRT engine to {output}")
