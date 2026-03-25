import argparse
from yolov8_trt import pt2onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='YOLOv8 .pt weights path')
    parser.add_argument('--output', type=str, required=True, help='output ONNX model path')
    parser.add_argument('--img-size', type=int, default=640, help='input image size')
    opt = parser.parse_args()

    pt2onnx(opt.weights, opt.output, img_size=opt.img_size)
