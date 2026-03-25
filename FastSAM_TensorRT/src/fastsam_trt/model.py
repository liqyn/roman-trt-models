import time

import numpy as np

from onnx_trt_tools import TRTEngine

from .utils import preprocess, postprocess


class FastSAM_TRT(TRTEngine):
    """TensorRT-accelerated FastSAM inference.

    Args:
        model_weights: Path to serialized TensorRT engine (.trt file).
        img_size: Input image size (square). Default 256.
        retina_masks: Use native (retina) mask processing. Default True.
        conf: Confidence threshold for NMS. Default 0.25.
        iou: IoU threshold for NMS. Default 0.7.
        agnostic_nms: Class-agnostic NMS. Default False.
    """

    def __init__(
        self,
        model_weights,
        img_size=256,
        retina_masks=True,
        conf=0.25,
        iou=0.7,
        agnostic_nms=False,
        timing=False
    ):
        super().__init__(model_weights, input_shape=(1, 3, img_size, img_size), timing=timing)
        self.imgsz = img_size
        self.retina_masks = retina_masks
        self.conf = conf
        self.iou = iou
        self.agnostic_nms = agnostic_nms

        # Only read the two outputs postprocess needs (detections and proto)
        self.output_names = [self._all_output_names[0], self._all_output_names[5]]

    def warmup(self, iters=3):
        """Run dummy inferences to warm up the GPU."""
        super().warmup((1, 3, self.imgsz, self.imgsz), iters)

    def segment(self, bgr_img):
        """Run segmentation on a BGR image.

        Args:
            bgr_img: Input image in BGR format (numpy array).

        Returns:
            Mask tensor on GPU of shape (N, H, W) where N is the number of detected objects.
        """
        t0 = time.time()
        inp = preprocess(bgr_img, self.imgsz)
        t1 = time.time()
        preds = self._infer(inp)
        t2 = time.time()
        result = postprocess(preds, inp, bgr_img, self.retina_masks, self.conf, self.iou, self.agnostic_nms)
        t3 = time.time()
        if self.timing:
            print(f"[FastSAM_TRT] preprocess: {t1-t0:.4f}s, inference: {t2-t1:.4f}s, postprocess: {t3-t2:.4f}s")
        return result[0]
