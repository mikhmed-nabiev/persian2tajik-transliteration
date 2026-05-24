"""PyTorch Lightning module wrapping ByT5-small / mT5-small for transliteration."""

from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, T5ForConditionalGeneration

from persian_tajik_translit.eval.metrics import compute_all_metrics
from persian_tajik_translit.training.losses import CycleLoss, EzafeHead


class ByT5Module(pl.LightningModule):
    """Bidirectional Tajik↔Persian transliterator built on ByT5-small or mT5-small.

    Both directions share one model via task prefix:
        "to_tajik: " + Farsi input
        "to_farsi: " + Tajik input
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.model = T5ForConditionalGeneration.from_pretrained(cfg.model.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

        if cfg.model.use_ezafe_head:
            d_model = self.model.config.d_model
            self.ezafe_head = EzafeHead(d_model, cfg.model.ezafe_head_hidden)
        else:
            self.ezafe_head = None

        if cfg.model.use_cycle_loss:
            self.cycle_loss_fn = CycleLoss(tau=cfg.train.gumbel_tau)
        else:
            self.cycle_loss_fn = None

        self._val_preds: list[str] = []
        self._val_refs: list[str] = []

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> Any:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=self.ezafe_head is not None,
        )
        ce_loss = outputs.loss
        total_loss = ce_loss
        self.log("train/ce_loss", ce_loss, prog_bar=False, sync_dist=True)

        if self.cycle_loss_fn is not None and self.global_step >= self.cfg.train.cycle_start_step:
            forward_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=labels.clamp(min=0),
            ).logits
            cycle_loss = self.cycle_loss_fn(
                model=self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                forward_logits=forward_logits,
                reverse_prefix_ids=input_ids,
                reverse_prefix_mask=attention_mask,
            )
            total_loss = total_loss + self.cfg.train.lambda_cyc * cycle_loss
            self.log("train/cycle_loss", cycle_loss, prog_bar=False, sync_dist=True)

        if self.ezafe_head is not None and "ezafe_labels" in batch:
            encoder_hidden = outputs.encoder_last_hidden_state
            ezafe_loss = self.ezafe_head(
                encoder_hidden,
                batch["ezafe_labels"],
                attention_mask.float(),
            )
            total_loss = total_loss + self.cfg.train.lambda_ez * ezafe_loss
            self.log("train/ezafe_loss", ezafe_loss, prog_bar=False, sync_dist=True)

        self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        with torch.no_grad():
            val_outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            self.log("val/loss", val_outputs.loss, prog_bar=True, sync_dist=True)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.cfg.model.val_max_new_tokens,
                num_beams=self.cfg.model.val_beam_size,
            )
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        refs = [
            self.tokenizer.decode([tok for tok in seq if tok != -100], skip_special_tokens=True)
            for seq in labels
        ]
        self._val_preds.extend(preds)
        self._val_refs.extend(refs)

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        metrics = compute_all_metrics(self._val_preds, self._val_refs)
        for metric_name, value in metrics.items():
            self.log(f"val/{metric_name}", value, prog_bar=metric_name == "chrf_pp", sync_dist=True)
        self._val_preds.clear()
        self._val_refs.clear()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            betas=(self.cfg.train.beta1, self.cfg.train.beta2),
            weight_decay=self.cfg.train.weight_decay,
        )
        total_steps = self.cfg.train.max_steps
        warmup_steps = self.cfg.train.warmup_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            import math

            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def generate(self, text: str, num_beams: int = 4) -> str:
        """Translate a single input string. Prefix must already be included."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.cfg.model.max_length,
            truncation=True,
        ).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=self.cfg.model.max_length,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
