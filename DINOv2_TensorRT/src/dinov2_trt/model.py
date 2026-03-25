import time

import numpy as np

from onnx_trt_tools import TRTEngine

from .utils import preprocess, reshape_patches


class DINOv2_TRT(TRTEngine):
    """TensorRT-accelerated DINOv2 inference.

    Args:
        model_weights: Path to serialized TensorRT engine (.trt file).
        img_size: int for aspect-ratio preserving resize (short side), or
            (h, w) tuple for exact resize. Default 256.
        timing: Print per-call timing breakdown. Default False.
    """

    def __init__(self, model_weights, img_size=256, timing=False):
        if isinstance(img_size, int):
            self.imgsz = img_size
            self.fixed_size = None
            input_shape = None
        else:
            self.imgsz = tuple(img_size)
            self.fixed_size = self.imgsz  # (h, w)
            input_shape = (1, 3, self.imgsz[0], self.imgsz[1])

        super().__init__(model_weights, input_shape=input_shape, timing=timing)

    def warmup(self, iters=3):
        """Run dummy inferences to warm up the GPU."""
        if self.fixed_size is not None:
            h, w = self.fixed_size
        else:
            h, w = self.imgsz, self.imgsz
        super().warmup((1, 3, h, w), iters)

    def embed(self, bgr_img, reshape=False):
        """Run DINOv2 feature extraction on a BGR image.

        Args:
            bgr_img: Input image in BGR format (numpy array, H x W x 3).
            reshape: If True, reshape output to spatial grid (1, h, w, D)
                based on the preprocessed image. Default False.

        Returns:
            If reshape=False: torch.Tensor on GPU of shape (1, num_patches+1, 768).
            If reshape=True:  torch.Tensor on GPU of shape (1, h, w, 768).
        """
        t0 = time.time()
        inp = preprocess(bgr_img, self.imgsz)
        t1 = time.time()
        raw = self._infer(inp)
        t2 = time.time()
        out = raw[0]  # (1, num_patches+1, 768)
        if reshape:
            out = reshape_patches(out, inp.shape[2:])
        t3 = time.time()
        if self.timing:
            print(f"[DINOv2_TRT] preprocess: {t1-t0:.4f}s, inference: {t2-t1:.4f}s, postprocess: {t3-t2:.4f}s")
        return out
