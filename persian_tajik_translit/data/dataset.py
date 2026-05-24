"""Dataset and DataModule for Tajik↔Persian transliteration."""

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


class CharVocab:
    """Character-level vocabulary for custom models (char-transformer, LSTM)."""

    def __init__(self, chars: list[str]) -> None:
        self.token_to_idx: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for char in sorted(set(chars)):
            if char not in self.token_to_idx:
                self.token_to_idx[char] = len(self.token_to_idx)
        self.idx_to_token: dict[int, str] = {v: k for k, v in self.token_to_idx.items()}

    def encode(self, text: str) -> list[int]:
        return [BOS_IDX] + [self.token_to_idx.get(ch, UNK_IDX) for ch in text] + [EOS_IDX]

    def decode(self, indices: list[int]) -> str:
        result = []
        for idx in indices:
            token = self.idx_to_token.get(idx, "")
            if token in ("<bos>", "<pad>"):
                continue
            if token == "<eos>":
                break
            result.append(token)
        return "".join(result)

    def __len__(self) -> int:
        return len(self.token_to_idx)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CharVocab":
        all_chars = list("".join(df["persian"].tolist() + df["tajik"].tolist()))
        return cls(all_chars)


class TransliterationDataset(Dataset):
    """Yields (source_text, target_text, direction) triples for both directions."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer=None,
        char_vocab: CharVocab | None = None,
        max_length: int = 512,
        prefix_to_tajik: str = "to_tajik: ",
        prefix_to_farsi: str = "to_farsi: ",
    ) -> None:
        self.tokenizer = tokenizer
        self.char_vocab = char_vocab
        self.max_length = max_length
        self.prefix_to_tajik = prefix_to_tajik
        self.prefix_to_farsi = prefix_to_farsi
        self.records: list[tuple[str, str]] = []
        for _, row in df.iterrows():
            self.records.append((prefix_to_tajik + row["persian"], row["tajik"]))
            self.records.append((prefix_to_farsi + row["tajik"], row["persian"]))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        source, target = self.records[idx]
        if self.tokenizer is not None:
            encoded_source = self.tokenizer(
                source,
                max_length=self.max_length,
                truncation=True,
                return_tensors=None,
            )
            encoded_target = self.tokenizer(
                target,
                max_length=self.max_length,
                truncation=True,
                return_tensors=None,
            )
            labels = encoded_target["input_ids"].copy()
            labels = [tok if tok != self.tokenizer.pad_token_id else -100 for tok in labels]
            return {
                "input_ids": encoded_source["input_ids"],
                "attention_mask": encoded_source["attention_mask"],
                "labels": labels,
                "source_text": source,
                "target_text": target,
            }
        else:
            source_ids = self.char_vocab.encode(source)[: self.max_length]
            target_ids = self.char_vocab.encode(target)[: self.max_length]
            return {
                "input_ids": source_ids,
                "labels": target_ids,
                "source_text": source,
                "target_text": target,
            }


def _hf_collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    max_src = max(len(item["input_ids"]) for item in batch)
    max_tgt = max(len(item["labels"]) for item in batch)
    input_ids, attention_masks, labels = [], [], []
    for item in batch:
        src_len = len(item["input_ids"])
        tgt_len = len(item["labels"])
        input_ids.append(item["input_ids"] + [pad_token_id] * (max_src - src_len))
        attention_masks.append(item["attention_mask"] + [0] * (max_src - src_len))
        labels.append(item["labels"] + [-100] * (max_tgt - tgt_len))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "source_text": [item["source_text"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
    }


def _char_collate_fn(batch: list[dict], pad_idx: int = PAD_IDX) -> dict:
    max_src = max(len(item["input_ids"]) for item in batch)
    max_tgt = max(len(item["labels"]) for item in batch)
    input_ids, labels = [], []
    for item in batch:
        src_pad = max_src - len(item["input_ids"])
        tgt_pad = max_tgt - len(item["labels"])
        input_ids.append(item["input_ids"] + [pad_idx] * src_pad)
        labels.append(item["labels"] + [pad_idx] * tgt_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "source_text": [item["source_text"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
    }


class TransliterationDataModule(pl.LightningDataModule):
    """Loads processed parquet splits and provides DataLoaders."""

    def __init__(
        self,
        processed_dir: str = "data/processed",
        model_name: str = "google/byt5-small",
        use_hf_tokenizer: bool = True,
        max_length: int = 512,
        batch_size: int = 16,
        num_workers: int = 4,
        prefix_to_tajik: str = "to_tajik: ",
        prefix_to_farsi: str = "to_farsi: ",
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.model_name = model_name
        self.use_hf_tokenizer = use_hf_tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefix_to_tajik = prefix_to_tajik
        self.prefix_to_farsi = prefix_to_farsi
        self.tokenizer = None
        self.char_vocab: CharVocab | None = None

    def setup(self, stage: str | None = None) -> None:
        train_df = pd.read_parquet(self.processed_dir / "train.parquet")
        val_df = pd.read_parquet(self.processed_dir / "val.parquet")
        test_df = pd.read_parquet(self.processed_dir / "test.parquet")

        if self.use_hf_tokenizer:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            self.char_vocab = CharVocab.from_dataframe(
                pd.concat([train_df, val_df, test_df], ignore_index=True)
            )

        kwargs = dict(
            tokenizer=self.tokenizer,
            char_vocab=self.char_vocab,
            max_length=self.max_length,
            prefix_to_tajik=self.prefix_to_tajik,
            prefix_to_farsi=self.prefix_to_farsi,
        )
        self.train_dataset = TransliterationDataset(train_df, **kwargs)
        self.val_dataset = TransliterationDataset(val_df, **kwargs)
        self.test_dataset = TransliterationDataset(test_df, **kwargs)

    def _collate(self, batch: list[dict]) -> dict:
        if self.use_hf_tokenizer:
            return _hf_collate_fn(batch, self.tokenizer.pad_token_id)
        return _char_collate_fn(batch)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
        )
