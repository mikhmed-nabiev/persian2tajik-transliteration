"""Inference entry point — loads ONNX model, auto-detects model type.

Usage:
    python infer.py "به نام خداوند جان و خرد"                          # byt5 fa→tg
    python infer.py "Ба номи Худованд" --direction tg2fa                 # byt5 tg→fa
    python infer.py "به نام" --onnx_dir models/onnx/lstm                 # LSTM
    python infer.py "به نام" --onnx_dir models/onnx/char_transformer     # char-transformer
    python infer.py --input_file sentences.txt                           # batch from file
"""

import json
import sys
from pathlib import Path

ONNX_DIR = "models/onnx_int8"
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


# ─────────────────────────────── detection ───────────────────────────────────


def _detect_model_type(onnx_dir: Path) -> str:
    """Infer model type from ONNX file layout."""
    if not (onnx_dir / "vocab.json").exists():
        return "byt5"
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    return "lstm" if "hidden" in out_names else "char_transformer"


# ─────────────────────────────── byt5 / mt5 ─────────────────────────────────


def _load_byt5(onnx_dir: Path):
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = ORTModelForSeq2SeqLM.from_pretrained(str(onnx_dir))
    return tokenizer, model


def _infer_byt5(tokenizer, model, texts, direction, num_beams, max_new_tokens):
    prefix = "to_tajik: " if direction == "fa2tg" else "to_farsi: "
    inputs = tokenizer(
        [prefix + t for t in texts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    out_ids = model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


# ──────────────────────────────── LSTM ───────────────────────────────────────


def _infer_lstm(onnx_dir: Path, texts: list[str], direction: str, max_length: int) -> list[str]:
    import numpy as np
    import onnxruntime as ort

    meta = json.loads((onnx_dir / "vocab.json").read_text())
    token_to_idx: dict = meta["token_to_idx"]
    idx_to_token = {int(v): k for k, v in token_to_idx.items()}
    prefix = meta["prefix_to_tajik"] if direction == "fa2tg" else meta["prefix_to_farsi"]
    max_length = int(meta.get("max_length", max_length))

    enc_sess = ort.InferenceSession(
        str(onnx_dir / "encoder.onnx"), providers=["CPUExecutionProvider"]
    )
    dec_sess = ort.InferenceSession(
        str(onnx_dir / "decoder_step.onnx"), providers=["CPUExecutionProvider"]
    )

    results = []
    for text in texts:
        full = prefix + text
        ids = [BOS_IDX] + [token_to_idx.get(ch, UNK_IDX) for ch in full] + [EOS_IDX]
        src = np.array([ids], dtype=np.int64)  # (1, src_len)

        encoder_outputs, hidden, cell = enc_sess.run(None, {"src": src})

        dec_input = np.array([BOS_IDX], dtype=np.int64)  # (batch=1,)
        output_ids = []
        for _ in range(max_length):
            logits, hidden, cell = dec_sess.run(
                None,
                {
                    "dec_input": dec_input,
                    "hidden": hidden,
                    "cell": cell,
                    "encoder_outputs": encoder_outputs,
                },
            )
            next_token = int(np.argmax(logits[0]))
            if next_token == EOS_IDX:
                break
            output_ids.append(next_token)
            dec_input = np.array([next_token], dtype=np.int64)

        results.append("".join(idx_to_token.get(i, "") for i in output_ids))
    return results


# ───────────────────────── char_transformer ──────────────────────────────────


def _infer_char_transformer(
    onnx_dir: Path, texts: list[str], direction: str, max_length: int
) -> list[str]:
    import numpy as np
    import onnxruntime as ort

    meta = json.loads((onnx_dir / "vocab.json").read_text())
    token_to_idx: dict = meta["token_to_idx"]
    idx_to_token = {int(v): k for k, v in token_to_idx.items()}
    prefix = meta["prefix_to_tajik"] if direction == "fa2tg" else meta["prefix_to_farsi"]
    max_length = int(meta.get("max_length", max_length))

    enc_sess = ort.InferenceSession(
        str(onnx_dir / "encoder.onnx"), providers=["CPUExecutionProvider"]
    )
    dec_sess = ort.InferenceSession(
        str(onnx_dir / "decoder_step.onnx"), providers=["CPUExecutionProvider"]
    )

    results = []
    for text in texts:
        full = prefix + text
        ids = [BOS_IDX] + [token_to_idx.get(ch, UNK_IDX) for ch in full] + [EOS_IDX]
        src_len = len(ids)
        # char_transformer encoder: src is (src_len, batch), mask is (batch, src_len)
        src = np.array(ids, dtype=np.int64).reshape(src_len, 1)
        src_mask = np.zeros((1, src_len), dtype=bool)

        (memory,) = enc_sess.run(None, {"src": src, "src_key_padding_mask": src_mask})
        # memory: (src_len, 1, d_model)

        ys = np.array([[BOS_IDX]], dtype=np.int64)  # (tgt_len=1, batch=1)
        output_ids = []
        for _ in range(max_length):
            tgt_len = ys.shape[0]
            tgt_mask = np.triu(np.full((tgt_len, tgt_len), float("-inf"), dtype=np.float32), k=1)
            (next_logits,) = dec_sess.run(None, {"ys": ys, "memory": memory, "tgt_mask": tgt_mask})
            next_token = int(np.argmax(next_logits[0]))
            if next_token == EOS_IDX:
                break
            output_ids.append(next_token)
            ys = np.vstack([ys, np.array([[next_token]], dtype=np.int64)])

        results.append("".join(idx_to_token.get(i, "") for i in output_ids))
    return results


# ─────────────────────────────── public API ──────────────────────────────────


def infer(
    text: str = "",
    direction: str = "fa2tg",
    onnx_dir: str = ONNX_DIR,
    input_file: str = "",
    output_file: str = "",
    num_beams: int = 4,
    max_new_tokens: int = 512,
) -> None:
    """Transliterate text using the ONNX model (byt5, mt5, lstm, or char_transformer).

    Args:
        text: Single input text to transliterate.
        direction: "fa2tg" (Farsi→Tajik) or "tg2fa" (Tajik→Farsi).
        onnx_dir: Directory containing ONNX model files.
        input_file: Text file with one sentence per line.
        output_file: Write results here instead of stdout.
        num_beams: Beam width (byt5/mt5 only).
        max_new_tokens: Max output length.
    """
    onnx_path = Path(onnx_dir)
    if not onnx_path.exists():
        print(
            f"ONNX model not found at '{onnx_dir}'. "
            "Run: python commands.py export --checkpoint_path <path> --output_dir models/onnx/<name>",
            file=sys.stderr,
        )
        sys.exit(1)

    if input_file:
        texts = Path(input_file).read_text(encoding="utf-8").splitlines()
        texts = [line.strip() for line in texts if line.strip()]
    elif text:
        texts = [text]
    else:
        texts = [line.strip() for line in sys.stdin if line.strip()]

    model_type = _detect_model_type(onnx_path)
    print(f"[{model_type}] {len(texts)} sentence(s) ...", file=sys.stderr)

    if model_type == "byt5":
        tokenizer, model = _load_byt5(onnx_path)
        results = _infer_byt5(tokenizer, model, texts, direction, num_beams, max_new_tokens)
    elif model_type == "lstm":
        results = _infer_lstm(onnx_path, texts, direction, max_new_tokens)
    else:
        results = _infer_char_transformer(onnx_path, texts, direction, max_new_tokens)

    output = "\n".join(results)
    if output_file:
        Path(output_file).write_text(output, encoding="utf-8")
        print(f"Wrote {len(results)} transliterations to {output_file}")
    else:
        print(output)


if __name__ == "__main__":
    import fire

    fire.Fire(infer)
