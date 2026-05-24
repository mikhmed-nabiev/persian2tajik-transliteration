"""Triton Python backend for LSTM transliteration via ONNX Runtime."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import triton_python_backend_utils as pb_utils

PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        model_config = json.loads(args["model_config"])

        def _param(key: str, default: str) -> str:
            return model_config.get("parameters", {}).get(key, {}).get("string_value", default)

        onnx_dir = Path(_param("onnx_dir", "/workspace/models/onnx_int8/lstm"))
        self.direction = _param("direction", "fa2tg")
        self.max_length = int(_param("max_length", "512"))

        meta = json.loads((onnx_dir / "vocab.json").read_text(encoding="utf-8"))
        self.token_to_idx: dict = meta["token_to_idx"]
        self.idx_to_token = {int(v): k for k, v in self.token_to_idx.items()}
        self.prefix_to_tajik: str = meta["prefix_to_tajik"]
        self.prefix_to_farsi: str = meta["prefix_to_farsi"]
        self.vocab_max_length: int = int(meta.get("max_length", self.max_length))

        providers = ["CPUExecutionProvider"]
        self.enc_sess = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), providers=providers)
        self.dec_sess = ort.InferenceSession(
            str(onnx_dir / "decoder_step.onnx"), providers=providers
        )

    def _infer_one(self, text: str, direction: str) -> str:
        prefix = self.prefix_to_tajik if direction == "fa2tg" else self.prefix_to_farsi
        full = prefix + text
        ids = [BOS_IDX] + [self.token_to_idx.get(ch, UNK_IDX) for ch in full] + [EOS_IDX]
        src = np.array([ids], dtype=np.int64)

        encoder_outputs, hidden, cell = self.enc_sess.run(None, {"src": src})

        dec_input = np.array([BOS_IDX], dtype=np.int64)
        output_ids = []
        for _ in range(self.vocab_max_length):
            logits, hidden, cell = self.dec_sess.run(
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

        return "".join(self.idx_to_token.get(i, "") for i in output_ids)

    def execute(self, requests: list) -> list:
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            texts = [item.decode("utf-8") for item in input_tensor.as_numpy().flatten().tolist()]

            dir_tensor = pb_utils.get_input_tensor_by_name(request, "direction")
            if dir_tensor is not None:
                direction = dir_tensor.as_numpy().flatten()[0].decode("utf-8")
            else:
                direction = self.direction

            results = [self._infer_one(t, direction) for t in texts]

            output_array = np.array([r.encode("utf-8") for r in results], dtype=object)
            output_tensor = pb_utils.Tensor("transliteration", output_array)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self) -> None:
        del self.enc_sess
        del self.dec_sess
