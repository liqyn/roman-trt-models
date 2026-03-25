import argparse
from fastsam_trt import onnx2trt

def main():
    p = argparse.ArgumentParser(description="Convert FastSAM ONNX to TRT")
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--opt-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=1)
    p.add_argument("--fp16", action="store_true", help="enable fp16 precision")
    args = p.parse_args()

    onnx2trt(args.onnx, args.output, img_size=args.img_size, opt_batch=args.opt_batch,
             max_batch=args.max_batch, fp16=args.fp16)

if __name__ == "__main__":
    main()
