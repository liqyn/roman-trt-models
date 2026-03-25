import argparse
from dinov2_trt import onnx2trt

def parse_3tuple(s):
    """Parse a comma-separated 3-tuple: batch,h,w"""
    parts = [int(x) for x in s.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 3 comma-separated ints (batch,h,w), got {len(parts)}")
    return tuple(parts)

def main():
    p = argparse.ArgumentParser(description="Convert DINOv2 ONNX to TRT")
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--img-size", type=int, nargs='+', default=[256],
                   help="fixed image size: single int (square) or two ints (h w). "
                        "Ignored when --min-sz/--opt-sz/--max-sz are given.")
    p.add_argument("--min-sz", type=parse_3tuple, default=None,
                   help="min shape as batch,h,w (e.g. 1,256,256)")
    p.add_argument("--opt-sz", type=parse_3tuple, default=None,
                   help="optimal shape as batch,h,w")
    p.add_argument("--max-sz", type=parse_3tuple, default=None,
                   help="max shape as batch,h,w")
    p.add_argument('--fp16', action='store_true', help='enable FP16 precision')
    args = p.parse_args()

    img_size = args.img_size[0] if len(args.img_size) == 1 else tuple(args.img_size)
    onnx2trt(args.onnx, args.output, img_size=img_size,
             min_sz=args.min_sz, opt_sz=args.opt_sz, max_sz=args.max_sz,
             fp16=args.fp16)

if __name__ == "__main__":
    main()
