import cv2
import time
import argparse
import torch
from transformers import AutoModel
from dinov2_trt import visualize_features
from dinov2_trt.utils import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='facebook/dinov2-base',
                        help='HuggingFace model name or local path')
    parser.add_argument('--img-size', type=int, nargs='+', default=[256],
                        help='input image size: single int (aspect-ratio preserving) '
                             'or two ints h w (exact resize)')
    opt = parser.parse_args()

    img_size = opt.img_size[0] if len(opt.img_size) == 1 else tuple(opt.img_size)
    model = AutoModel.from_pretrained(opt.model).eval().to('cuda')

    img = cv2.imread("images/cat.jpg")
    inp = preprocess(img, imgsz=img_size)
    inputs = {'pixel_values': torch.from_numpy(inp).to('cuda')}

    warmup_iters = 3
    warmup_inputs = {k: torch.zeros_like(v) for k, v in inputs.items()}
    for _ in range(warmup_iters):
        with torch.no_grad():
            model(**warmup_inputs)

    t1 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"[Inference time]: {time.time() - t1:.4f}s")

    last_hidden_state = outputs.last_hidden_state
    print("[Output]:", last_hidden_state.shape)  # (1, num_patches+1, 768)

    viz = visualize_features(last_hidden_state, img_shape=inp.shape[2:])
    viz = cv2.resize(viz, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("images/cat_default.jpg", viz)
