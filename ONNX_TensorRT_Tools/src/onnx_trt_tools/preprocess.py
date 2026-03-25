import cv2
import numpy as np


def letterbox_preprocess(img_bgr, imgsz, pad_value=(114, 114, 114)):
    """Resize and letterbox-pad a BGR image for YOLO-style inference.

    Preserves aspect ratio by scaling so the longer side fits ``imgsz``,
    then pads the shorter side with ``pad_value``.

    Args:
        img_bgr: BGR image as a numpy array (H, W, 3), uint8.
        imgsz: Target square size (int).
        pad_value: Padding color in RGB. Default (114, 114, 114).

    Returns:
        numpy.ndarray: Float32 array of shape (1, 3, imgsz, imgsz), normalized to [0, 1].
    """
    h, w = img_bgr.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    dw, dh = (imgsz - nw) / 2, (imgsz - nh) / 2
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    resized = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (nw, nh))
    inp = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=pad_value)
    return np.transpose(np.array([inp], dtype=np.float32) / 255.0, (0, 3, 1, 2))
