"""Download and preprocess all training/evaluation data."""

import subprocess
import unicodedata
from pathlib import Path

import pandas as pd

PARSTEXT_SOURCES = ["bbc", "dr", "jj"]
STIBIUMGHOST_REPO = "https://github.com/stibiumghost/tajik-to-persian-transliteration.git"
FLORES_DATASET = "openlanguagedata/flores_plus"
FLORES_TAJIK_COL = "tgk_Cyrl"
FLORES_PERSIAN_COL = "pes_Arab"


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _ngrams(text: str, order: int) -> set:
    return {text[i : i + order] for i in range(len(text) - order + 1)}


def _overlap_ratio(text_a: str, text_b: str, order: int) -> float:
    grams_a = _ngrams(text_a, order)
    grams_b = _ngrams(text_b, order)
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / min(len(grams_a), len(grams_b))


def _load_parstext(parstext_dir: Path) -> pd.DataFrame:
    rows = []
    for source in PARSTEXT_SOURCES:
        csv_path = parstext_dir / f"{source}.csv"
        if not csv_path.exists():
            continue
        for line in csv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            persian, tajik = _nfc(parts[0].strip()), _nfc(parts[1].strip())
            if persian and tajik:
                rows.append({"persian": persian, "tajik": tajik, "source": source})
    return pd.DataFrame(rows)


def _load_stibiumghost(stibiumghost_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in stibiumghost_dir.glob("data/**/*.csv"):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception:
            continue
        for col_pair in [("persian", "tajik"), ("fa", "tg"), ("farsi", "tajik")]:
            if col_pair[0] in df.columns and col_pair[1] in df.columns:
                for _, row in df.iterrows():
                    persian = _nfc(str(row[col_pair[0]]).strip())
                    tajik = _nfc(str(row[col_pair[1]]).strip())
                    if persian and tajik and persian != "nan" and tajik != "nan":
                        rows.append({"persian": persian, "tajik": tajik, "source": "stibiumghost"})
                break
    return pd.DataFrame(rows)


def _load_flores(ngram_order: int, threshold: float, train_persian: list) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(FLORES_DATASET, trust_remote_code=True)
    train_hashes: set[str] = set()
    for sent in train_persian:
        for order in range(2, ngram_order + 1):
            train_hashes.update(_ngrams(sent, order))

    rows = []
    for split_name in ["dev", "devtest"]:
        if split_name not in ds:
            continue
        for example in ds[split_name]:
            persian = _nfc(example.get(FLORES_PERSIAN_COL, "") or "")
            tajik = _nfc(example.get(FLORES_TAJIK_COL, "") or "")
            if not persian or not tajik:
                continue
            max_overlap = max(
                (_overlap_ratio(persian, train_sent, ngram_order) for train_sent in train_persian),
                default=0.0,
            )
            if max_overlap <= threshold:
                rows.append(
                    {
                        "persian": persian,
                        "tajik": tajik,
                        "source": f"flores_{split_name}",
                        "flores_split": split_name,
                    }
                )
    return pd.DataFrame(rows)


def _stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = df.sample(frac=1, random_state=seed)
    sources = rng["source"].unique()
    train_rows, val_rows, test_rows = [], [], []
    for source in sources:
        subset = rng[rng["source"] == source]
        n = len(subset)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_rows.append(subset.iloc[:n_train])
        val_rows.append(subset.iloc[n_train : n_train + n_val])
        test_rows.append(subset.iloc[n_train + n_val :])
    return (
        pd.concat(train_rows).reset_index(drop=True),
        pd.concat(val_rows).reset_index(drop=True),
        pd.concat(test_rows).reset_index(drop=True),
    )


def download_data(
    parstext_dir: str = "ParsText/data/aligned/csv",
    stibiumghost_dir: str = "data/raw/stibiumghost",
    processed_dir: str = "data/processed",
    train_split: float = 0.8,
    val_split: float = 0.1,
    ngram_dedup_threshold: float = 0.3,
    ngram_dedup_order: int = 8,
    seed: int = 42,
) -> None:
    """Download and preprocess all training data, saving parquet splits."""
    parstext_path = Path(parstext_dir)
    stibiumghost_path = Path(stibiumghost_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    print("Loading ParsText...")
    parstext_df = _load_parstext(parstext_path)
    print(f"  ParsText: {len(parstext_df)} pairs")

    if not stibiumghost_path.exists():
        print("Cloning stibiumghost corpus (sparse)...")
        stibiumghost_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "--filter=blob:none",
                "--sparse",
                STIBIUMGHOST_REPO,
                str(stibiumghost_path),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "data"],
            cwd=str(stibiumghost_path),
            check=True,
        )

    print("Loading stibiumghost corpus...")
    stibiumghost_df = _load_stibiumghost(stibiumghost_path)
    print(f"  stibiumghost: {len(stibiumghost_df)} pairs")

    combined_df = pd.concat([parstext_df, stibiumghost_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["persian", "tajik"])
    print(f"Combined training corpus: {len(combined_df)} pairs")

    train_df, val_df, test_df = _stratified_split(combined_df, train_split, val_split, seed)

    print(f"Loading FLORES-200 (dedup threshold={ngram_dedup_threshold})...")
    train_persian_sentences = train_df["persian"].tolist()
    flores_df = _load_flores(ngram_dedup_order, ngram_dedup_threshold, train_persian_sentences)
    print(f"  FLORES-200 after dedup: {len(flores_df)} pairs")

    train_df.to_parquet(processed_path / "train.parquet", index=False)
    val_df.to_parquet(processed_path / "val.parquet", index=False)
    test_df.to_parquet(processed_path / "test.parquet", index=False)
    flores_df.to_parquet(processed_path / "flores_ood.parquet", index=False)

    stats = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "flores_ood": len(flores_df),
    }
    pd.Series(stats).to_json(processed_path / "stats.json")
    print("Data saved:")
    for split, count in stats.items():
        print(f"  {split}: {count} pairs")
