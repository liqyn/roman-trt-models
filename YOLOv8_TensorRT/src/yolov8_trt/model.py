import time

import numpy as np

from onnx_trt_tools import TRTEngine

from .utils import preprocess, postprocess


class YOLOv8_TRT(TRTEngine):
    """TensorRT-accelerated YOLOv8 detection inference.

    Args:
        model_weights: Path to serialized TensorRT engine (.trt file).
        img_size: Input image size (square). Default 640.
        conf: Confidence threshold for NMS. Default 0.25.
        iou: IoU threshold for NMS. Default 0.7.
        agnostic_nms: Class-agnostic NMS. Default False.
        max_det: Maximum number of detections. Default 300.
    """

    def __init__(
        self,
        model_weights,
        img_size=640,
        conf=0.25,
        iou=0.7,
        agnostic_nms=False,
        max_det=100,
        timing=False
    ):
        super().__init__(model_weights, input_shape=(1, 3, img_size, img_size), timing=timing)
        self.imgsz = img_size
        self.conf = conf
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det

        # Detection model has a single output
        self.output_names = [self._all_output_names[0]]

    def warmup(self, iters=3):
        """Run dummy inferences to warm up the GPU."""
        super().warmup((1, 3, self.imgsz, self.imgsz), iters)

    def detect(self, bgr_img):
        """Run detection on a BGR image.

        Args:
            bgr_img: Input image in BGR format (numpy array).

        Returns:
            Tensor on GPU of shape (N, 6) with columns [x1, y1, x2, y2, conf, class_id].
        """
        t0 = time.time()
        inp = preprocess(bgr_img, self.imgsz)
        t1 = time.time()
        preds = self._infer(inp)
        t2 = time.time()
        result = postprocess(
            preds[0], inp, bgr_img,
            self.conf, self.iou, self.agnostic_nms, self.max_det,
        )
        t3 = time.time()
        if self.timing:
            print(f"{'[YOLOv8_TRT]':<14} preprocess: {t1-t0:.4f}s, inference: {t2-t1:.4f}s, postprocess: {t3-t2:.4f}s")
        return result[0]
