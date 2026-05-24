"""Training entry point.

Usage:
    python train.py
    python train.py model=mt5 train.max_steps=5000
    python train.py model=char_transformer train.max_steps=200
"""

import subprocess

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from persian_tajik_translit.data.dataset import TransliterationDataModule


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _build_model(cfg: DictConfig, data_module: TransliterationDataModule):
    model_name = cfg.model.name

    if model_name in ("byt5", "mt5"):
        from persian_tajik_translit.models.byt5_module import ByT5Module

        return ByT5Module(cfg)

    if model_name == "char_transformer":
        from persian_tajik_translit.models.char_transformer import CharTransformerModule

        if data_module.char_vocab is None:
            raise RuntimeError(
                "char_vocab must be built before model creation; call setup() first."
            )
        return CharTransformerModule(cfg, data_module.char_vocab)

    if model_name == "lstm":
        from persian_tajik_translit.models.lstm_module import LSTMTranslitModule

        if data_module.char_vocab is None:
            raise RuntimeError(
                "char_vocab must be built before model creation; call setup() first."
            )
        return LSTMTranslitModule(cfg, data_module.char_vocab)

    raise ValueError(f"Unknown model name: {model_name!r}")


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.data.seed, workers=True)

    use_hf = cfg.model.name in ("byt5", "mt5")
    data_module = TransliterationDataModule(
        processed_dir=cfg.data.processed_dir,
        model_name=cfg.model.get("model_name", "google/byt5-small"),
        use_hf_tokenizer=use_hf,
        max_length=cfg.model.max_length,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        prefix_to_tajik=cfg.model.get("prefix_to_tajik", "to_tajik: "),
        prefix_to_farsi=cfg.model.get("prefix_to_farsi", "to_farsi: "),
    )
    data_module.setup()

    model = _build_model(cfg, data_module)

    git_sha = _get_git_sha()
    mlflow.set_tracking_uri(cfg.train.mlflow_uri)
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.train.mlflow_experiment,
        tracking_uri=cfg.train.mlflow_uri,
        tags={"git_sha": git_sha, "model": cfg.model.name},
    )
    mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        filename=f"{cfg.model.name}-{{epoch:02d}}-{{val/chrf_pp:.2f}}",
        monitor=cfg.train.best_metric,
        mode=cfg.train.best_metric_mode,
        save_top_k=cfg.train.save_top_k,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_steps=cfg.train.max_steps,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip,
        val_check_interval=cfg.train.val_check_interval,
        limit_val_batches=cfg.train.limit_val_batches,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=mlf_logger,
        callbacks=[checkpoint_cb],
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=data_module)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    train()
