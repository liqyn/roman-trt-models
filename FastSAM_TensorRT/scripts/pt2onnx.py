import argparse
from fastsam_trt import pt2onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='FastSAM .pt weights path')
    parser.add_argument('--output', type=str, required=True, help='output ONNX model path')
    parser.add_argument('--img-size', type=int, default=256, help='input image size')
    opt = parser.parse_args()

    pt2onnx(opt.weights, opt.output, img_size=opt.img_size)
