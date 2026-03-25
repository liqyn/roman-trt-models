import cv2
import numpy as np
import time
import argparse
from random import randint
from ultralytics import YOLO
from fastsam_trt import overlay

retina_masks = True
conf = 0.25
iou = 0.7

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--img-size', type=int, default=256, help='input image size')
    opt = parser.parse_args()

    model = YOLO(opt.weights)
    img = cv2.imread('images/cat.jpg')

    warmup_iters = 3
    warmup_img = np.zeros_like(img)
    for _ in range(warmup_iters):
        model(warmup_img, imgsz=opt.img_size, retina_masks=retina_masks, iou=iou, conf=conf,
              max_det=100, device='cuda', verbose=False)

    t1 = time.time()
    results = model(img, imgsz=opt.img_size, retina_masks=retina_masks, iou=iou, conf=conf,
                    max_det=100, device='cuda', verbose=False)
    print(f"[Inference time]: {time.time() - t1:.4f}s")

    masks = results[0].masks.data.cpu().numpy()
    print("[Output]:", masks.shape)

    image_with_masks = np.copy(img)
    for mask_i in masks:
        rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=1)
    cv2.imwrite("images/cat_default.jpg", image_with_masks)
