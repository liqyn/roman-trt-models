import onnxruntime
import cv2
import numpy as np
import torch
import time
import argparse
from yolov8_trt import preprocess, postprocess, draw_detections

conf = 0.25
iou = 0.7
agnostic_nms = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--img-size', type=int, default=640, help='input image size')
    opt = parser.parse_args()

    img = cv2.imread('images/cat.jpg')
    inp = preprocess(img, imgsz=opt.img_size)
    print('Input:', inp.shape)

    model = onnxruntime.InferenceSession(opt.weights, providers=['CUDAExecutionProvider'])
    out_names = [model.get_outputs()[0].name]
    ort_inputs = {model.get_inputs()[0].name: inp}

    warmup_iters = 3
    warmup_inputs = {k: np.zeros_like(v) for k, v in ort_inputs.items()}
    for _ in range(warmup_iters):
        model.run(out_names, warmup_inputs)

    t1 = time.time()
    raw = model.run(out_names, ort_inputs)
    print(f"[Inference time]: {time.time() - t1:.4f}s")

    preds = torch.from_numpy(raw[0])
    dets = postprocess(preds, inp, img, conf, iou, agnostic_nms)
    print("[Output]:", dets[0].shape)

    result = draw_detections(img, dets[0])
    cv2.imwrite("images/cat_onnx.jpg", result)
