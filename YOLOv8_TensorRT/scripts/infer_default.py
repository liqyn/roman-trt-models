import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO
from yolov8_trt import draw_detections

conf = 0.25
iou = 0.7

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--img-size', type=int, default=640, help='input image size')
    opt = parser.parse_args()

    model = YOLO(opt.weights)
    img = cv2.imread('images/cat.jpg')

    warmup_iters = 3
    warmup_img = np.zeros_like(img)
    for _ in range(warmup_iters):
        model(warmup_img, imgsz=opt.img_size, iou=iou, conf=conf,
              max_det=100, device='cuda', verbose=False)

    t1 = time.time()
    results = model(img, imgsz=opt.img_size, iou=iou, conf=conf,
                    max_det=100, device='cuda', verbose=False)
    print(f"[Inference time]: {time.time() - t1:.4f}s")

    boxes = results[0].boxes
    dets = np.column_stack([
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy()[:, None],
        boxes.cls.cpu().numpy()[:, None],
    ])
    print("[Output]:", dets.shape)

    result = draw_detections(img, dets)
    cv2.imwrite("images/cat_default.jpg", result)
