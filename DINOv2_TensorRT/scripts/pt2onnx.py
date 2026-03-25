import argparse
from dinov2_trt import pt2onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DINOv2 to ONNX")
    parser.add_argument('--model', type=str, default='facebook/dinov2-base',
                        help='HuggingFace model name or local path')
    parser.add_argument('--output', type=str, required=True, help='output ONNX model path')
    parser.add_argument('--img-size', type=int, nargs='+', default=[256],
                        help="input image size: single int (square) or two ints (h w)")
    opt = parser.parse_args()

    img_size = opt.img_size[0] if len(opt.img_size) == 1 else tuple(opt.img_size)
    pt2onnx(opt.model, opt.output, img_size=img_size)
