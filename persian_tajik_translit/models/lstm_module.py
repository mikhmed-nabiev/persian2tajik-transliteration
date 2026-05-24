"""Character-level LSTM seq2seq with Bahdanau attention — classical baseline."""

import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from persian_tajik_translit.data.dataset import PAD_IDX, CharVocab
from persian_tajik_translit.eval.metrics import compute_all_metrics


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden: int, dec_hidden: int) -> None:
        super().__init__()
        self.attn = nn.Linear(enc_hidden + dec_hidden, dec_hidden)
        self.v = nn.Linear(dec_hidden, 1, bias=False)

    def forward(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_len = encoder_outputs.size(1)
        hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
        energy = torch.tanh(self.attn(torch.cat([hidden_expanded, encoder_outputs], dim=2)))
        attn_weights = F.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_decoder_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_decoder_layers = num_decoder_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.hidden_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = torch.tanh(self.hidden_proj(torch.cat([hidden[-2], hidden[-1]], dim=1)))
        cell = torch.tanh(self.cell_proj(torch.cat([cell[-2], cell[-1]], dim=1)))
        hidden = hidden.unsqueeze(0).expand(self.num_decoder_layers, -1, -1).contiguous()
        cell = cell.unsqueeze(0).expand(self.num_decoder_layers, -1, -1).contiguous()
        return outputs, (hidden, cell)


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        enc_hidden: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.attention = BahdanauAttention(enc_hidden, hidden_size)
        self.lstm = nn.LSTM(
            embed_dim + enc_hidden,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size + enc_hidden + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        tgt_token: torch.Tensor,
        hidden: tuple,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple]:
        embedded = self.dropout(self.embedding(tgt_token.unsqueeze(1)))
        context, _ = self.attention(hidden[0][-1], encoder_outputs)
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.lstm(lstm_input, hidden)
        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        )
        return prediction, hidden


class LSTMTranslitModule(pl.LightningModule):
    """LSTM seq2seq transliterator with Bahdanau attention."""

    def __init__(self, cfg: DictConfig, char_vocab: CharVocab) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["char_vocab"])
        self.cfg = cfg
        self.char_vocab = char_vocab

        vocab_size = len(char_vocab)
        hidden = cfg.model.hidden_size
        embed_dim = hidden // 2

        self.encoder = LSTMEncoder(
            vocab_size,
            embed_dim,
            hidden,
            cfg.model.num_encoder_layers,
            cfg.model.dropout,
            num_decoder_layers=cfg.model.num_decoder_layers,
        )
        self.decoder = LSTMDecoder(
            vocab_size,
            embed_dim,
            hidden,
            cfg.model.num_decoder_layers,
            cfg.model.dropout,
            enc_hidden=hidden * 2,
        )
        self._val_preds: list[str] = []
        self._val_refs: list[str] = []

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        src = batch["input_ids"]
        tgt = batch["labels"]
        encoder_outputs, hidden = self.encoder(src)

        tgt_len = tgt.size(1)
        vocab_size = len(self.char_vocab)
        batch_size = src.size(0)
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)

        dec_input = tgt[:, 0]
        for step_idx in range(tgt_len - 1):
            dec_output, hidden = self.decoder.forward_step(dec_input, hidden, encoder_outputs)
            outputs[:, step_idx] = dec_output
            teacher_force = random.random() < 0.5
            dec_input = tgt[:, step_idx + 1] if teacher_force else dec_output.argmax(dim=1)

        loss = F.cross_entropy(
            outputs.reshape(-1, vocab_size),
            tgt[:, 1:].reshape(-1),
            ignore_index=PAD_IDX,
        )
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        src = batch["input_ids"]
        tgt = batch["labels"]
        encoder_outputs, hidden = self.encoder(src)

        tgt_len = tgt.size(1)
        vocab_size = len(self.char_vocab)
        batch_size = src.size(0)
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)

        dec_input = tgt[:, 0]
        with torch.no_grad():
            for step_idx in range(tgt_len - 1):
                dec_output, hidden = self.decoder.forward_step(dec_input, hidden, encoder_outputs)
                outputs[:, step_idx] = dec_output
                dec_input = dec_output.argmax(dim=1)

        loss = F.cross_entropy(
            outputs.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1), ignore_index=PAD_IDX
        )
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        pred_ids = outputs.argmax(dim=2)
        pred_texts = [self.char_vocab.decode(row.tolist()) for row in pred_ids]
        ref_texts = batch["target_text"]
        self._val_preds.extend(pred_texts)
        self._val_refs.extend(ref_texts)

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
        return {"optimizer": optimizer}
