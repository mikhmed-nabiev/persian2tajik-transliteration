"""Character-level Transformer — reproduction of Merchant et al. (2025).

4+4 encoder/decoder layers, d_model=256, d_ff=1024, 4 heads, dropout=0.1.
"""

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from persian_tajik_translit.data.dataset import BOS_IDX, EOS_IDX, PAD_IDX, CharVocab
from persian_tajik_translit.eval.metrics import compute_all_metrics


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        return self.dropout(token_emb + self.pe[: token_emb.size(0)])


class CharTransformerModule(pl.LightningModule):
    """Char-level transformer seq2seq reproducing the Merchant et al. 2025 baseline."""

    def __init__(self, cfg: DictConfig, char_vocab: CharVocab) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["char_vocab"])
        self.cfg = cfg
        self.char_vocab = char_vocab

        vocab_size = len(char_vocab)
        d_model = cfg.model.d_model
        nhead = cfg.model.nhead
        d_ff = cfg.model.d_ff
        dropout = cfg.model.dropout
        n_enc = cfg.model.num_encoder_layers
        n_dec = cfg.model.num_decoder_layers

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=n_enc,
            num_decoder_layers=n_dec,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._scale = math.sqrt(d_model)

        self._val_preds: list[str] = []
        self._val_refs: list[str] = []

    def _encode_src(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_encoding(self.src_embedding(src) * self._scale)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        src = batch["input_ids"].T
        tgt_full = batch["labels"].T
        tgt_in = tgt_full[:-1]
        tgt_out = tgt_full[1:]

        src_key_padding_mask = src.T == PAD_IDX
        tgt_key_padding_mask = tgt_in.T == PAD_IDX
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(0), device=src.device)

        memory = self._encode_src(src, src_key_padding_mask)
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt_in) * self._scale)
        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.fc_out(out)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=PAD_IDX
        )
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        src = batch["input_ids"].T
        tgt_full = batch["labels"].T
        tgt_in = tgt_full[:-1]
        tgt_out = tgt_full[1:]
        src_key_padding_mask = src.T == PAD_IDX
        tgt_key_padding_mask = tgt_in.T == PAD_IDX
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(0), device=src.device)
        with torch.no_grad():
            memory = self._encode_src(src, src_key_padding_mask)
            tgt_emb = self.pos_encoding(self.tgt_embedding(tgt_in) * self._scale)
            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = self.fc_out(out)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=PAD_IDX
            )
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
            preds = self._greedy_decode(src, src_key_padding_mask, max_len=src.size(0) + 10)
        pred_texts = [self.char_vocab.decode(pred) for pred in preds]
        ref_texts = batch["target_text"]
        self._val_preds.extend(pred_texts)
        self._val_refs.extend(ref_texts)

    def _greedy_decode(
        self, src: torch.Tensor, src_key_padding_mask: torch.Tensor, max_len: int
    ) -> list[list[int]]:
        batch_size = src.size(1)
        memory = self._encode_src(src, src_key_padding_mask)
        ys = torch.full((1, batch_size), BOS_IDX, dtype=torch.long, device=src.device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        result = [[] for _ in range(batch_size)]
        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(0), device=src.device)
            tgt_emb = self.pos_encoding(self.tgt_embedding(ys) * self._scale)
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            next_logits = self.fc_out(out[-1])
            next_tokens = next_logits.argmax(dim=-1)
            for batch_idx in range(batch_size):
                if not done[batch_idx]:
                    token = next_tokens[batch_idx].item()
                    if token == EOS_IDX:
                        done[batch_idx] = True
                    else:
                        result[batch_idx].append(token)
            ys = torch.cat([ys, next_tokens.unsqueeze(0)], dim=0)
            if done.all():
                break
        return result

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        metrics = compute_all_metrics(self._val_preds, self._val_refs)
        for metric_name, value in metrics.items():
            self.log(f"val/{metric_name}", value, prog_bar=metric_name == "chrf_pp", sync_dist=True)
        self._val_preds.clear()
        self._val_refs.clear()

    def configure_optimizers(self) -> dict:
        import math

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
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
