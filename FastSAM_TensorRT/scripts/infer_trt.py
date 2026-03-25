import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import torch
import cv2
import time
import argparse
from random import randint
from fastsam_trt import FastSAM_TRT, overlay

retina_masks = True
conf = 0.25
iou = 0.7
agnostic_nms = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--img-size', type=int, default=256, help='input image size')
    opt = parser.parse_args()

    model = FastSAM_TRT(model_weights=opt.weights, img_size=opt.img_size, retina_masks=retina_masks, 
                        conf=conf, iou=iou, agnostic_nms=agnostic_nms, timing=True)
    model.warmup()
    img = cv2.imread('images/cat.jpg')
    masks = model.segment(img).cpu().numpy()
    print("[Output]:", masks.shape)

    image_with_masks = np.copy(img)
    for mask_i in masks:
        rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=1)
    cv2.imwrite("images/cat_trt.jpg", image_with_masks)
