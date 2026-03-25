import onnxruntime
import cv2
import time
import argparse
import numpy as np
from dinov2_trt import preprocess, visualize_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='ONNX model path')
    parser.add_argument('--img-size', type=int, nargs='+', default=[256],
                        help='input image size: single int (aspect-ratio preserving) '
                             'or two ints h w (exact resize)')
    opt = parser.parse_args()

    img_size = opt.img_size[0] if len(opt.img_size) == 1 else tuple(opt.img_size)
    img = cv2.imread("images/cat.jpg")
    inp = preprocess(img, imgsz=img_size)
    print('Input:', inp.shape)

    model = onnxruntime.InferenceSession(opt.weights, providers=['CUDAExecutionProvider'])
    ort_inputs = {model.get_inputs()[0].name: inp}

    warmup_iters = 3
    warmup_inputs = {k: np.zeros_like(v) for k, v in ort_inputs.items()}
    for _ in range(warmup_iters):
        model.run(None, warmup_inputs)

    t1 = time.time()
    raw = model.run(None, ort_inputs)
    print(f"[Inference time]: {time.time() - t1:.4f}s")

    last_hidden_state = raw[0]
    print("[Output]:", last_hidden_state.shape)  # (1, num_patches+1, 768)

    viz = visualize_features(last_hidden_state, img_shape=inp.shape[2:])
    viz = cv2.resize(viz, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("images/cat_onnx.jpg", viz)
