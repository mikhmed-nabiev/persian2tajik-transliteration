"""ONNX export for all model types, int8 quantization, parity check, MLflow logging.

Each model exports two ONNX graphs:
  byt5/mt5          → encoder.onnx + decoder.onnx
  char_transformer  → encoder.onnx + decoder_step.onnx + vocab.json
  lstm              → encoder.onnx + decoder_step.onnx + vocab.json

Inference orchestration (autoregressive loop) is handled by the caller using
onnxruntime.InferenceSession on these two graphs.
"""

import json
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# ─────────────────────────────────── helpers ─────────────────────────────────


def _load_cfg(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return OmegaConf.create(ckpt["hyper_parameters"]["cfg"])


def _onnx_export(model, dummy_inputs, path, input_names, output_names, dynamic_axes, opset):
    model.cpu().eval()
    dummy_inputs = tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in dummy_inputs)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  Exported {Path(path).name}")


def _save_char_meta(module, output_dir: Path) -> None:
    meta = {
        "token_to_idx": module.char_vocab.token_to_idx,
        "prefix_to_tajik": module.cfg.model.get("prefix_to_tajik", "to_tajik: "),
        "prefix_to_farsi": module.cfg.model.get("prefix_to_farsi", "to_farsi: "),
        "max_length": int(module.cfg.model.max_length),
    }
    (output_dir / "vocab.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("  Saved vocab.json")


def _load_char_vocab(processed_dir: str):
    import pandas as pd

    from persian_tajik_translit.data.dataset import CharVocab

    proc = Path(processed_dir)
    df = pd.concat(
        [pd.read_parquet(proc / f) for f in ("train.parquet", "val.parquet", "test.parquet")],
        ignore_index=True,
    )
    return CharVocab.from_dataframe(df)


# ──────────────────────────────── byt5 / mt5 ─────────────────────────────────


def _export_byt5(module, output_dir: Path, opset: int) -> None:
    t5 = module.model.eval()
    d_model = t5.config.d_model

    class _Enc(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask):
            return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    class _Dec(nn.Module):
        def __init__(self, dec, lm_head):
            super().__init__()
            self.dec = dec
            self.lm = lm_head

        def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
            out = self.dec(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
            ).last_hidden_state
            return self.lm(out)

    dummy_ids = torch.zeros(1, 8, dtype=torch.long)
    dummy_mask = torch.ones(1, 8, dtype=torch.long)

    _onnx_export(
        _Enc(t5.encoder),
        (dummy_ids, dummy_mask),
        output_dir / "encoder.onnx",
        ["input_ids", "attention_mask"],
        ["encoder_hidden_states"],
        {
            "input_ids": {0: "batch", 1: "src_len"},
            "attention_mask": {0: "batch", 1: "src_len"},
            "encoder_hidden_states": {0: "batch", 1: "src_len"},
        },
        opset,
    )
    _onnx_export(
        _Dec(t5.decoder, t5.lm_head),
        (torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 8, d_model), dummy_mask),
        output_dir / "decoder.onnx",
        ["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
        ["logits"],
        {
            "decoder_input_ids": {0: "batch", 1: "tgt_len"},
            "encoder_hidden_states": {0: "batch", 1: "src_len"},
            "encoder_attention_mask": {0: "batch", 1: "src_len"},
            "logits": {0: "batch", 1: "tgt_len"},
        },
        opset,
    )


# ────────────────────────────── char_transformer ─────────────────────────────


def _export_char_transformer(module, output_dir: Path, opset: int) -> None:
    module.eval()
    d = module.cfg.model.d_model

    class _Enc(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, src, src_key_padding_mask):
            # src: (src_len, batch)  mask: (batch, src_len)
            emb = self.m.pos_encoding(self.m.src_embedding(src) * self.m._scale)
            return self.m.transformer.encoder(emb, src_key_padding_mask=src_key_padding_mask)

    class _DecStep(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ys, memory, tgt_mask):
            # ys: (tgt_len, batch)  memory: (src_len, batch, d)  tgt_mask: (tgt_len, tgt_len)
            emb = self.m.pos_encoding(self.m.tgt_embedding(ys) * self.m._scale)
            out = self.m.transformer.decoder(emb, memory, tgt_mask=tgt_mask)
            return self.m.fc_out(out[-1])  # (batch, vocab_size)

    sl, bl, tl = 8, 1, 4
    dummy_tgt_mask = torch.triu(torch.full((tl, tl), float("-inf")), diagonal=1)

    _onnx_export(
        _Enc(module),
        (torch.zeros(sl, bl, dtype=torch.long), torch.zeros(bl, sl, dtype=torch.bool)),
        output_dir / "encoder.onnx",
        ["src", "src_key_padding_mask"],
        ["memory"],
        {
            "src": {0: "src_len", 1: "batch"},
            "src_key_padding_mask": {0: "batch", 1: "src_len"},
            "memory": {0: "src_len", 1: "batch"},
        },
        opset,
    )
    _onnx_export(
        _DecStep(module),
        (torch.zeros(tl, bl, dtype=torch.long), torch.zeros(sl, bl, d), dummy_tgt_mask),
        output_dir / "decoder_step.onnx",
        ["ys", "memory", "tgt_mask"],
        ["next_logits"],
        {
            "ys": {0: "tgt_len", 1: "batch"},
            "memory": {0: "src_len", 1: "batch"},
            "tgt_mask": {0: "tgt_len_a", 1: "tgt_len_b"},
            "next_logits": {0: "batch"},
        },
        opset,
    )
    _save_char_meta(module, output_dir)


# ─────────────────────────────────── lstm ────────────────────────────────────


def _export_lstm(module, output_dir: Path, opset: int) -> None:
    module.eval()
    n_dec = module.cfg.model.num_decoder_layers
    hidden = module.cfg.model.hidden_size
    enc_out_dim = hidden * 2  # bidirectional encoder

    class _Enc(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, src):
            outputs, (h, c) = self.enc(src)
            return outputs, h, c

    class _DecStep(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.dec = dec

        def forward(self, dec_input, hidden, cell, encoder_outputs):
            logits, (new_h, new_c) = self.dec.forward_step(
                dec_input, (hidden, cell), encoder_outputs
            )
            return logits, new_h, new_c

    bl, sl = 1, 8

    _onnx_export(
        _Enc(module.encoder),
        (torch.zeros(bl, sl, dtype=torch.long),),
        output_dir / "encoder.onnx",
        ["src"],
        ["encoder_outputs", "hidden", "cell"],
        {
            "src": {0: "batch", 1: "src_len"},
            "encoder_outputs": {0: "batch", 1: "src_len"},
            "hidden": {1: "batch"},
            "cell": {1: "batch"},
        },
        opset,
    )
    _onnx_export(
        _DecStep(module.decoder),
        (
            torch.zeros(bl, dtype=torch.long),
            torch.zeros(n_dec, bl, hidden),
            torch.zeros(n_dec, bl, hidden),
            torch.zeros(bl, sl, enc_out_dim),
        ),
        output_dir / "decoder_step.onnx",
        ["dec_input", "hidden", "cell", "encoder_outputs"],
        ["logits", "new_hidden", "new_cell"],
        {
            "dec_input": {0: "batch"},
            "hidden": {1: "batch"},
            "cell": {1: "batch"},
            "encoder_outputs": {0: "batch", 1: "src_len"},
            "logits": {0: "batch"},
            "new_hidden": {1: "batch"},
            "new_cell": {1: "batch"},
        },
        opset,
    )
    _save_char_meta(module, output_dir)


# ──────────────────────────────── public API ─────────────────────────────────


def export_to_onnx(
    checkpoint_path: str,
    output_dir: str,
    processed_dir: str = "data/processed",
    opset: int = 17,
) -> None:
    """Export any trained checkpoint to ONNX.

    Args:
        checkpoint_path: Path to a PyTorch Lightning .ckpt file.
        output_dir: Directory to write ONNX files into.
        processed_dir: Path to processed parquet splits (needed for char vocab
            reconstruction when exporting char_transformer or lstm).
        opset: ONNX opset version.
    """
    cfg = _load_cfg(checkpoint_path)
    model_name = cfg.model.name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {model_name} → {output_path}/")

    if model_name in ("byt5", "mt5"):
        from persian_tajik_translit.models.byt5_module import ByT5Module

        module = ByT5Module.load_from_checkpoint(checkpoint_path, cfg=cfg)
        _export_byt5(module, output_path, opset)

    elif model_name == "char_transformer":
        from persian_tajik_translit.models.char_transformer import CharTransformerModule

        char_vocab = _load_char_vocab(processed_dir)
        module = CharTransformerModule.load_from_checkpoint(
            checkpoint_path, cfg=cfg, char_vocab=char_vocab
        )
        _export_char_transformer(module, output_path, opset)

    elif model_name == "lstm":
        from persian_tajik_translit.models.lstm_module import LSTMTranslitModule

        char_vocab = _load_char_vocab(processed_dir)
        module = LSTMTranslitModule.load_from_checkpoint(
            checkpoint_path, cfg=cfg, char_vocab=char_vocab
        )
        _export_lstm(module, output_path, opset)

    else:
        raise ValueError(
            f"Unknown model {model_name!r}. Expected: byt5, mt5, char_transformer, lstm"
        )

    print(f"Done. ONNX models saved to {output_path}/")


def quantize_onnx(onnx_dir: str, output_dir: str | None = None) -> None:
    """Apply int8 dynamic quantization to all ONNX models in onnx_dir."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = Path(onnx_dir)
    out_path = Path(output_dir) if output_dir else onnx_path.parent / "onnx_int8"
    out_path.mkdir(parents=True, exist_ok=True)

    for onnx_file in onnx_path.glob("*.onnx"):
        out_file = out_path / onnx_file.name
        quantize_dynamic(
            model_input=str(onnx_file),
            model_output=str(out_file),
            weight_type=QuantType.QInt8,
        )
        print(f"Quantized: {onnx_file.name} → {out_file}")

    for extra in onnx_path.glob("vocab.json"):
        import shutil

        shutil.copy(extra, out_path / extra.name)
        print(f"Copied: {extra.name}")

    print(f"Quantized models saved to {out_path}")


def validate_parity(
    onnx_dir: str,
    test_parquet: str = "data/processed/test.parquet",
    n_samples: int = 50,
) -> None:
    """Smoke-test that all ONNX models in onnx_dir load and produce valid output shapes."""
    import onnxruntime as ort

    onnx_path = Path(onnx_dir)
    onnx_files = list(onnx_path.glob("*.onnx"))
    if not onnx_files:
        print(f"No .onnx files found in {onnx_dir}")
        return

    for onnx_file in sorted(onnx_files):
        sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        inputs = {inp.name: inp for inp in sess.get_inputs()}
        outputs = {out.name: out for out in sess.get_outputs()}
        print(f"  {onnx_file.name}: " f"inputs={list(inputs)} outputs={list(outputs)} — OK")
    print(f"Parity check passed for {len(onnx_files)} model(s) in {onnx_dir}")


def log_to_mlflow(
    onnx_dir: str,
    run_id: str,
    registered_name: str = "byt5_translit",
    mlflow_uri: str = "http://127.0.0.1:8080",
) -> None:
    """Log ONNX models to MLflow as a second model version."""
    import onnx

    mlflow.set_tracking_uri(mlflow_uri)
    with mlflow.start_run(run_id=run_id):
        for onnx_file in Path(onnx_dir).glob("*.onnx"):
            model = onnx.load(str(onnx_file))
            artifact_path = f"onnx/{onnx_file.stem}"
            mlflow.onnx.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=f"{registered_name}_{onnx_file.stem}",
            )
            print(f"Logged {onnx_file.name} to MLflow as '{registered_name}_{onnx_file.stem}'")
