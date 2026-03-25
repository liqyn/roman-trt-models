import cv2
import numpy as np
import torch
from ultralytics.yolo.utils import ops

from onnx_trt_tools import letterbox_preprocess

# COCO class names
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def preprocess(img_origin, imgsz=640):
    return letterbox_preprocess(img_origin, imgsz)


def postprocess(preds, img, orig_img, conf=0.25, iou=0.7, agnostic_nms=False, max_det=100):
    """Postprocess YOLOv8 detection output.

    Args:
        preds: Raw model output tensor of shape (1, 4+nc, num_anchors).
        img: Preprocessed input (for shape info).
        orig_img: Original BGR image (for box scaling).
        conf: Confidence threshold.
        iou: IoU threshold for NMS.
        agnostic_nms: Class-agnostic NMS.
        max_det: Maximum number of detections.

    Returns:
        List of tensors, one per batch image, each of shape (N, 6)
        with columns [x1, y1, x2, y2, conf, class_id].
    """
    p = ops.non_max_suppression(preds, conf, iou, agnostic=agnostic_nms, max_det=max_det)
    for i, pred in enumerate(p):
        if len(pred):
            orig = orig_img[i] if isinstance(orig_img, list) else orig_img
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig.shape)
    return p


def draw_detections(image, detections, class_names=None):
    """Draw bounding boxes and labels on an image.

    Args:
        image: BGR image (numpy array).
        detections: Tensor of shape (N, 6) with [x1, y1, x2, y2, conf, class_id].
        class_names: List of class names. Defaults to COCO_NAMES.

    Returns:
        Image with drawn detections.
    """
    if class_names is None:
        class_names = COCO_NAMES
    result = np.copy(image)
    if not len(detections):
        return result
    dets = detections.cpu().numpy() if isinstance(detections, torch.Tensor) else detections
    for x1, y1, x2, y2, conf, cls_id in dets:
        cls_id = int(cls_id)
        color = _class_color(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls_id]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return result


def _class_color(cls_id):
    """Deterministic color per class id."""
    np.random.seed(cls_id)
    return tuple(int(c) for c in np.random.randint(0, 255, 3))
