import cv2
import numpy as np
import torch
from ultralytics.yolo.utils import ops


def preprocess(img_origin, imgsz=256):
    h, w = img_origin.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    dw, dh = (imgsz - nw) / 2, (imgsz - nh) / 2
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    resized = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    inp = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.transpose(np.array([inp], dtype=np.float32) / 255.0, (0, 3, 1, 2))


def postprocess(preds, img, orig_img, retina_masks=True, conf=0.25, iou=0.7, agnostic_nms=False):
    """Postprocess FastSAM output.

    Returns:
        List of mask tensors per batch image, each of shape (N, H, W) or None if no detections.
    """
    p = ops.non_max_suppression(preds[0], conf, iou, agnostic=agnostic_nms, max_det=100, nc=1)
    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
    for i, pred in enumerate(p):
        orig = orig_img[i] if isinstance(orig_img, list) else orig_img
        if not len(pred):
            results.append(None)
            continue
        if retina_masks:
            if not isinstance(orig_img, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig.shape)
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig.shape[:2])
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
            if not isinstance(orig_img, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig.shape)
        results.append(masks)
    return results


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
