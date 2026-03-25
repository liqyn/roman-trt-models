import tensorrt as trt


def onnx2trt(onnx_path, output, input_name, min_sz, opt_sz, max_sz, fp16=False):
    """Convert an ONNX model to a TensorRT engine.

    Args:
        onnx_path: Path to the ONNX model file.
        output: Output path for the TensorRT engine.
        input_name: Name of the input tensor (e.g. 'images', 'pixel_values').
        min_sz: Min (batch, h, w) for the optimization profile.
        opt_sz: Optimal (batch, h, w) for the optimization profile.
        max_sz: Max (batch, h, w) for the optimization profile.
        fp16: Enable FP16 precision in the TensorRT engine. Default False.
    """
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name,
                      (min_sz[0], 3, min_sz[1], min_sz[2]),
                      (opt_sz[0], 3, opt_sz[1], opt_sz[2]),
                      (max_sz[0], 3, max_sz[1], max_sz[2]))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    with open(output, "wb") as f:
        f.write(engine)

    print(f"Saved TRT engine to {output}")
