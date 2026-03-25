import cv2
import argparse
from dinov2_trt import DINOv2_TRT, preprocess, visualize_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='TRT engine path')
    parser.add_argument('--img-size', type=int, nargs='+', default=[256],
                        help='input image size: single int (aspect-ratio preserving) '
                             'or two ints h w (exact resize)')
    opt = parser.parse_args()

    img_size = opt.img_size[0] if len(opt.img_size) == 1 else tuple(opt.img_size)
    model = DINOv2_TRT(model_weights=opt.weights, img_size=img_size, timing=True)
    model.warmup()
    img = cv2.imread("images/cat.jpg")
    last_hidden_state = model.embed(img)
    print("[Output]:", last_hidden_state.shape)  # (1, num_patches+1, 768)

    viz = visualize_features(last_hidden_state, img_shape=img.shape[:2])
    viz = cv2.resize(viz, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("images/cat_trt.jpg", viz)
