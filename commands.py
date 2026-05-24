"""Single CLI entry point for all project operations.

Usage:
    python commands.py download
    python commands.py train
    python commands.py infer "به نام خداوند"
    python commands.py eval-ood
    python commands.py export --checkpoint_path models/best.ckpt --output_dir models/onnx
    python commands.py quantize --onnx_dir models/onnx --output_dir models/onnx_int8
    python commands.py validate-onnx --onnx_dir models/onnx_int8
    python commands.py convert-trt --onnx_dir models/onnx/lstm --output_dir models/trt/lstm
"""

import subprocess
import sys


def download(
    parstext_dir: str = "ParsText/data/aligned/csv",
    stibiumghost_dir: str = "data/raw/stibiumghost",
    processed_dir: str = "data/processed",
) -> None:
    """Download and preprocess all training data."""
    from persian_tajik_translit.data.download import download_data

    download_data(
        parstext_dir=parstext_dir,
        stibiumghost_dir=stibiumghost_dir,
        processed_dir=processed_dir,
    )


def train() -> None:
    """Launch model training (delegates to train.py with Hydra)."""
    subprocess.run([sys.executable, "train.py"] + sys.argv[2:], check=True)


def infer(
    text: str = "",
    direction: str = "fa2tg",
    onnx_dir: str = "models/onnx_int8",
    input_file: str = "",
    output_file: str = "",
    num_beams: int = 4,
) -> None:
    """Transliterate text using the ONNX model."""
    from infer import infer as _infer

    _infer(
        text=text,
        direction=direction,
        onnx_dir=onnx_dir,
        input_file=input_file,
        output_file=output_file,
        num_beams=num_beams,
    )


def eval_ood(
    checkpoint_path: str = "",
    onnx_dir: str = "models/onnx_int8",
    source: str = "both",
) -> None:
    """Run OOD evaluation on FLORES-200 and Tajik Wikipedia."""
    from eval_ood import eval_ood as _eval_ood

    _eval_ood(checkpoint_path=checkpoint_path, onnx_dir=onnx_dir, source=source)


def export(
    checkpoint_path: str,
    output_dir: str = "models/onnx",
    processed_dir: str = "data/processed",
    opset: int = 17,
) -> None:
    """Export any trained checkpoint to ONNX (byt5, mt5, char_transformer, lstm)."""
    from persian_tajik_translit.export.onnx_export import export_to_onnx

    export_to_onnx(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        processed_dir=processed_dir,
        opset=opset,
    )


def quantize(
    onnx_dir: str = "models/onnx",
    output_dir: str = "models/onnx_int8",
) -> None:
    """Apply int8 dynamic quantization to ONNX models."""
    from persian_tajik_translit.export.onnx_export import quantize_onnx

    quantize_onnx(onnx_dir=onnx_dir, output_dir=output_dir)


def convert_trt(
    onnx_dir: str = "models/onnx/lstm",
    output_dir: str = "models/trt/lstm",
) -> None:
    """Convert LSTM ONNX models to TensorRT FP16 engines (requires TRT: uv pip install -e '.[trt]')."""
    from persian_tajik_translit.export.trt_export import export_to_tensorrt

    export_to_tensorrt(onnx_dir=onnx_dir, output_dir=output_dir)


def validate_onnx(
    onnx_dir: str = "models/onnx_int8",
    test_parquet: str = "data/processed/test.parquet",
    n_samples: int = 200,
) -> None:
    """Validate ONNX parity against PyTorch outputs."""
    from persian_tajik_translit.export.onnx_export import validate_parity

    validate_parity(onnx_dir=onnx_dir, test_parquet=test_parquet, n_samples=n_samples)


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "download": download,
            "train": train,
            "infer": infer,
            "eval-ood": eval_ood,
            "export": export,
            "quantize": quantize,
            "validate-onnx": validate_onnx,
            "convert-trt": convert_trt,
        }
    )
