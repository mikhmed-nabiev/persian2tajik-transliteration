# Persian ↔ Tajik Transliteration

<!-- Bidirectional Tajik (Cyrillic) ↔ Persian (Perso-Arabic) transliteration built on **ByT5-small**,
with a four-way architecture ablation, cycle-consistency loss, and Ezafe auxiliary head. -->

Wraps a full MLOps pipeline: PyTorch Lightning · Hydra · DVC · MLflow · ONNX · Triton.

<!-- Reproduces and extends Merchant, Kakolu Ramarao & Tang (2025, arXiv:2502.20047). -->

---

## Project structure

<!-- ```
.
├── configs/                        # Hydra config groups
│   ├── config.yaml                 # root config with defaults
│   ├── model/lstm.yaml
# │   ├── model/{byt5,mt5,char_transformer}.yaml
│   ├── data/parstext.yaml
│   ├── train/default.yaml
│   └── eval/default.yaml
├── data/                           # managed by DVC
│   ├── raw/                        # stibiumghost clone
│   └── processed/                  # parquet splits (train/val/test/flores_ood)
├── models/                         # checkpoints + ONNX (DVC)
├── paper/                          # LaTeX source + compiled PDF
├── persian_tajik_translit/         # Python package
│   ├── data/
│   │   ├── download.py             # download_data()
│   │   └── dataset.py              # TransliterationDataset, TransliterationDataModule
│   ├── models/
# │   │   ├── byt5_module.py          # ByT5 / mT5 LightningModule
# │   │   ├── char_transformer.py     # char-transformer LightningModule
│   │   └── lstm_module.py          # LSTM seq2seq LightningModule
│   ├── training/losses.py          # CycleLoss, EzafeHead
│   ├── eval/metrics.py             # chrF++, seq_acc, CER, lev_ratio, vowel_f1
│   └── export/
│       ├── onnx_export.py          # ONNX export, int8 quant, parity check
│       └── trt_export.py           # TensorRT FP16 engine build
# ├── triton_models/byt5_translit/    # Triton Python backend
│   ├── config.pbtxt
│   ├── 1/model.py
│   └── test_client.py
├── plots/                          # saved training plots
├── commands.py                     # single fire CLI entry point
├── train.py                        # Hydra training entry
├── infer.py                        # inference entry (public API)
├── eval_ood.py                     # FLORES-200 + Wikipedia OOD eval
├── pyproject.toml                  # uv project + ruff config
└── uv.lock
``` -->

```
.
├── configs/                        # Hydra config groups
│   ├── config.yaml                 # root config with defaults
│   ├── model/lstm.yaml
│   ├── data/parstext.yaml
│   ├── train/default.yaml
│   └── eval/default.yaml
├── data/                           # managed by DVC
│   ├── raw/                        # stibiumghost clone
│   └── processed/                  # parquet splits (train/val/test/flores_ood)
├── models/                         # checkpoints + ONNX (DVC)
├── paper/                          # LaTeX source + compiled PDF
├── persian_tajik_translit/         # Python package
│   ├── data/
│   │   ├── download.py             # download_data()
│   │   └── dataset.py              # TransliterationDataset, TransliterationDataModule
│   ├── models/
│   │   └── lstm_module.py          # LSTM seq2seq LightningModule
│   ├── training/losses.py          # CycleLoss, EzafeHead
│   ├── eval/metrics.py             # chrF++, seq_acc, CER, lev_ratio, vowel_f1
│   └── export/
│       ├── onnx_export.py          # ONNX export, int8 quant, parity check
│       └── trt_export.py           # TensorRT FP16 engine build
├── triton_models/lstm_translit/    # Triton Python backend
│   ├── config.pbtxt
│   ├── 1/model.py
│   └── test_client.py
├── plots/                          # saved training plots
├── commands.py                     # single fire CLI entry point
├── train.py                        # Hydra training entry
├── infer.py                        # inference entry (public API)
├── eval_ood.py                     # FLORES-200 + Wikipedia OOD eval
├── pyproject.toml                  # uv project + ruff config
└── uv.lock
```

---

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install pre-commit hooks
pre-commit install

# Verify hooks pass
pre-commit run -a
```

---

## Train

### 1. Download and preprocess data

```bash
python commands.py download
```

This clones the stibiumghost corpus, reads ParsText CSVs from `ParsText/data/aligned/csv/`,
downloads FLORES-200 (deduplicated against training data), and saves parquet splits to
`data/processed/`.

### 2. Start MLflow tracking server (separate terminal)

```bash
mlflow server --host 127.0.0.1 --port 8080
```

### 3. Train

<!-- ```bash
# # Primary model — ByT5-small with cycle loss + Ezafe head
# python train.py

# # mT5-small baseline (subword arm of the ablation)
# python train.py model=mt5

# # Merchant et al. 2025 character-transformer baseline
# python train.py model=char_transformer

# Classical LSTM baseline
python train.py model=lstm

# # Quick smoke test (200 steps)
# python train.py model=char_transformer train.max_steps=200

# Override any config value on the command line (Hydra syntax)
python train.py train.lr=1e-4 train.batch_size=8
``` -->

```bash
# Train LSTM
python train.py model=lstm

# Quick smoke test (200 training steps, validate every 50)
python train.py model=lstm train.max_steps=200 train.val_check_interval=50

# Without a running MLflow server (logs to ./mlruns locally)
python train.py model=lstm train.mlflow_uri=file:./mlruns

# Override any config value on the command line (Hydra syntax)
python train.py model=lstm train.lr=1e-4 train.batch_size=8
```

Checkpoints are saved to `models/` (best validation chrF++, top-2 + last).
All metrics are logged to MLflow at `http://127.0.0.1:8080`.

---

## Export

### ONNX export

<!-- All four model types are supported. The output files vary by architecture: -->

| Model  | Files produced                                    |
| ------ | ------------------------------------------------- |
| `lstm` | `encoder.onnx`, `decoder_step.onnx`, `vocab.json` |

<!-- ```bash
# # ByT5 / mT5
# python commands.py export \
#     --checkpoint_path models/byt5-best.ckpt \
#     --output_dir models/onnx/byt5

# LSTM (requires --processed_dir for vocab reconstruction)
python commands.py export \
    --checkpoint_path models/lstm-best.ckpt \
    --output_dir models/onnx/lstm \
    --processed_dir data/processed
``` -->

```bash
# LSTM (requires --processed_dir for vocab reconstruction)
python commands.py export \
    --checkpoint_path models/lstm-best.ckpt \
    --output_dir models/onnx/lstm \
    --processed_dir data/processed
```

### int8 quantization

```bash
python commands.py quantize \
    --onnx_dir models/onnx/lstm \
    --output_dir models/onnx_int8/lstm
```

Also copies `vocab.json` (needed for <!-- char_transformer / -->lstm inference).

### Parity validation

```bash
python commands.py validate-onnx --onnx_dir models/onnx_int8/lstm
```

Loads each `.onnx` file in the directory and prints its input/output names and shapes,
confirming the graphs are well-formed and loadable by ONNX Runtime.

### TensorRT conversion

Converts the LSTM ONNX models to TensorRT FP16 engines. Requires a CUDA GPU and the
`tensorrt-cu12` package (not installed by default):

```bash
# Build FP16 engines (source dir must contain encoder.onnx + decoder_step.onnx)
python commands.py convert-trt \
    --onnx_dir models/onnx/lstm \
    --output_dir models/trt/lstm
```

Outputs `encoder.engine`, `decoder_step.engine`, and `vocab.json` in `models/trt/lstm/`.
Engines are GPU- and driver-specific and are excluded from git and DVC.

### Complete conversion workflow

```bash
# 1. Export checkpoint → ONNX (fp32)
python commands.py export \
    --checkpoint_path models/lstm-best.ckpt \
    --output_dir models/onnx/lstm \
    --processed_dir data/processed

# 2. Quantize → int8
python commands.py quantize \
    --onnx_dir models/onnx/lstm \
    --output_dir models/onnx_int8/lstm

# 3. Validate graphs load correctly
python commands.py validate-onnx --onnx_dir models/onnx_int8/lstm

# 4. (Optional) Build TensorRT FP16 engines
python commands.py convert-trt \
    --onnx_dir models/onnx/lstm \
    --output_dir models/trt/lstm

# 5. Run inference
python infer.py "به نام خداوند" --onnx_dir models/onnx_int8/lstm
```

---

## Infer

```bash
# Single Farsi → Tajik (defaults to models/onnx_int8/)
python infer.py "به نام خداوند جان و خرد"

# Single Tajik → Farsi
python infer.py "Ба номи Худованд ҷону хирад" --direction tg2fa

# Use a specific model directory (auto-detected: lstm)
python infer.py "به نام" --onnx_dir models/onnx/lstm

# Batch from file (one sentence per line)
python infer.py --input_file sentences.txt --output_file results.txt

# Via commands.py
python commands.py infer "سلام دنیا" --onnx_dir models/onnx/lstm
```

`infer.py` auto-detects the model type from the ONNX files in the directory.
Run export first if the ONNX model is not present (see Export section below).

> **Note:** Persian/Tajik text pasted into a terminal may display reversed — this is a
> terminal RTL rendering limitation. The actual string passed to the program is correct.

---

<!-- ## OOD Evaluation

```bash
# Both FLORES-200 and Wikipedia
python eval_ood.py

# FLORES-200 only
python eval_ood.py --source flores

# Wikipedia only
python eval_ood.py --source wiki

# From a specific checkpoint
python eval_ood.py --checkpoint_path models/byt5-epoch=05-val_chrf_pp=82.50.ckpt
``` -->

<!-- Results are logged to MLflow under the `persian-tajik-translit-eval` experiment. -->

<!-- --- -->

## Inference server

### Triton

```bash
# Pull ONNX models from DVC first
dvc pull models

# Start Triton server (Docker required)
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "$(pwd)/triton_models:/models" \
    -v "$(pwd)/models:/workspace/models" \
    tritonserver-ort tritonserver --model-repository=/models

# Test the server
python triton_models/test_client.py
```

---

## Data versioning (DVC)

```bash
# Pull processed data from Google Drive
dvc pull data/processed.dvc

# Pull trained models
dvc pull models.dvc

# After adding new data or models, push to remote
dvc push
```

Two separate remotes are configured in `.dvc/config`:

- `data_remote` — Google Drive folder for datasets
- `models_remote` — Google Drive folder for model checkpoints and ONNX artifacts

---

## Metrics

All five metrics are logged per epoch to MLflow:

| Metric      | Description                                                                            |
| ----------- | -------------------------------------------------------------------------------------- |
| `chrf_pp`   | chrF++ (β=2, char-order=6, word-order=2) — primary, comparable to Merchant et al. 2025 |
| `seq_acc`   | Fraction of outputs character-identical to reference                                   |
| `cer`       | Character Error Rate (normalised Levenshtein)                                          |
| `lev_ratio` | Levenshtein ratio (1 − CER)                                                            |
| `vowel_f1`  | Precision/recall over vowel characters targeting the Ezafe failure mode                |

<!-- --- -->

<!-- ## Paper

The accompanying paper is in [`paper/`](paper/persian_tajik_transliteration.pdf). -->

<!-- It describes the four-way architecture ablation, cycle-consistency loss formulation,
Ezafe auxiliary head, and OOD evaluation protocol. -->
