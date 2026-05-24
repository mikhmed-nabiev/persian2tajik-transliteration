import shutil
from pathlib import Path

import tensorrt as trt

_LOGGER = trt.Logger(trt.Logger.WARNING)


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    profiles: list[dict[str, tuple]],
) -> None:
    builder = trt.Builder(_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, _LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    for profile_shapes in profiles:
        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in profile_shapes.items():
            profile.set_shape(name, lo, opt, hi)
        config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path.write_bytes(engine_bytes)
    print(f"Saved engine: {engine_path}")


def export_to_tensorrt(onnx_dir: str, output_dir: str) -> None:
    """Convert LSTM ONNX models to TensorRT FP16 engines.

    Requires TensorRT to be installed (`pip install -e ".[trt]"`).
    The source ONNX files must already exist (run `python commands.py export` first).
    """
    src = Path(onnx_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    _build_engine(
        src / "encoder.onnx",
        dst / "encoder.engine",
        profiles=[
            {
                "src": ((1, 1), (4, 32), (8, 256)),
            }
        ],
    )
    _build_engine(
        src / "decoder_step.onnx",
        dst / "decoder_step.engine",
        profiles=[
            {
                "dec_input": ((1,), (4,), (8,)),
                "hidden": ((2, 1, 256), (2, 4, 256), (2, 8, 256)),
                "cell": ((2, 1, 256), (2, 4, 256), (2, 8, 256)),
                "encoder_outputs": ((1, 1, 512), (4, 32, 512), (8, 256, 512)),
            }
        ],
    )
    shutil.copy(src / "vocab.json", dst / "vocab.json")
    print(f"TensorRT conversion complete. Engines saved to {dst}")
