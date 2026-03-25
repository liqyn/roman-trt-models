import numpy as np
import cv2
import argparse
from yolov8_trt import YOLOv8_TRT, draw_detections

conf = 0.25
iou = 0.7
agnostic_nms = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--img-size', type=int, default=640, help='input image size')
    opt = parser.parse_args()

    model = YOLOv8_TRT(model_weights=opt.weights, img_size=opt.img_size,
                       conf=conf, iou=iou, agnostic_nms=agnostic_nms, timing=True)
    model.warmup()
    img = cv2.imread('images/cat.jpg')
    dets = model.detect(img)
    print("[Output]:", dets.shape)

    result = draw_detections(img, dets)
    cv2.imwrite("images/cat_trt.jpg", result)
