"""OOD evaluation on FLORES-200 and Tajik Wikipedia.

Usage:
    python eval_ood.py                               # FLORES + Wikipedia
    python eval_ood.py --source flores               # FLORES only
    python eval_ood.py --source wiki                 # Wikipedia only
    python eval_ood.py --checkpoint_path models/best.ckpt
"""

import bz2
import xml.etree.ElementTree as ET
from pathlib import Path

import mlflow
import pandas as pd
import torch
from omegaconf import OmegaConf

WIKI_DUMP = "tgwiki-latest-pages-articles.xml.bz2"
FLORES_PARQUET = "data/processed/flores_ood.parquet"
MLFLOW_URI = "http://127.0.0.1:8080"
MLFLOW_EXPERIMENT = "persian-tajik-translit-eval"
LABSE_THRESHOLD = 0.85


def _load_flores(flores_parquet: str) -> pd.DataFrame:
    return pd.read_parquet(flores_parquet)


def _parse_wiki_articles(dump_path: str, max_articles: int = 500) -> list[str]:
    articles = []
    with bz2.open(dump_path, "rb") as fh:
        for event, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag.endswith("}text") and elem.text and len(elem.text) > 200:
                articles.append(elem.text[:2000])
                elem.clear()
            if len(articles) >= max_articles:
                break
    return articles


def _labse_filter(
    persian_sents: list[str], tajik_sents: list[str], threshold: float
) -> list[tuple]:
    try:
        from sentence_transformers import SentenceTransformer

        labse = SentenceTransformer("sentence-transformers/LaBSE")
        fa_embs = labse.encode(persian_sents, convert_to_tensor=True)
        tg_embs = labse.encode(tajik_sents, convert_to_tensor=True)
        cos_sim = torch.nn.functional.cosine_similarity(fa_embs, tg_embs, dim=1)
        return [
            (fa, tg)
            for fa, tg, sim in zip(persian_sents, tajik_sents, cos_sim.tolist())
            if sim >= threshold
        ]
    except ImportError:
        return list(zip(persian_sents, tajik_sents))


def _run_eval(predictions: list[str], references: list[str], split_name: str, run) -> None:
    from persian_tajik_translit.eval.metrics import compute_all_metrics

    metrics = compute_all_metrics(predictions, references)
    print(f"\n[{split_name}] n={len(predictions)}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
        mlflow.log_metric(f"{split_name}/{name}", value)


def eval_ood(
    checkpoint_path: str = "",
    onnx_dir: str = "models/onnx_int8",
    source: str = "both",
    flores_parquet: str = FLORES_PARQUET,
    wiki_dump: str = WIKI_DUMP,
    mlflow_uri: str = MLFLOW_URI,
    num_beams: int = 4,
    max_articles: int = 500,
) -> None:
    """Run OOD evaluation on FLORES-200 and/or Tajik Wikipedia.

    Args:
        checkpoint_path: Path to PyTorch Lightning checkpoint (uses ONNX if empty).
        onnx_dir: Path to ONNX model directory (used when checkpoint_path is empty).
        source: "flores", "wiki", or "both".
        flores_parquet: Path to deduplicated FLORES-200 parquet.
        wiki_dump: Path to Tajik Wikipedia bz2 dump.
        mlflow_uri: MLflow tracking server URI.
        num_beams: Beam width for generation.
        max_articles: Number of Wikipedia articles to sample.
    """
    if checkpoint_path:
        from persian_tajik_translit.models.byt5_module import ByT5Module

        cfg = OmegaConf.load("configs/config.yaml")
        model = ByT5Module.load_from_checkpoint(checkpoint_path, cfg=cfg)
        model.eval()

        def translate_fn(texts: list[str], direction: str) -> list[str]:
            prefix = (
                cfg.model.prefix_to_tajik if direction == "fa2tg" else cfg.model.prefix_to_farsi
            )
            return [model.generate(prefix + text) for text in texts]
    else:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_dir)

        def translate_fn(texts: list[str], direction: str) -> list[str]:
            prefix = "to_tajik: " if direction == "fa2tg" else "to_farsi: "
            inputs = tokenizer(
                [prefix + t for t in texts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            out = ort_model.generate(**inputs, num_beams=num_beams, max_new_tokens=512)
            return tokenizer.batch_decode(out, skip_special_tokens=True)

    mlflow.set_tracking_uri(mlflow_uri)
    with mlflow.start_run(
        experiment_id=mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT).experiment_id
        if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        else mlflow.create_experiment(MLFLOW_EXPERIMENT)
    ):
        if source in ("flores", "both") and Path(flores_parquet).exists():
            flores_df = _load_flores(flores_parquet)
            fa_sents = flores_df["persian"].tolist()
            tg_sents = flores_df["tajik"].tolist()
            preds_fa2tg = translate_fn(fa_sents, "fa2tg")
            _run_eval(preds_fa2tg, tg_sents, "flores/fa2tg", None)
            preds_tg2fa = translate_fn(tg_sents, "tg2fa")
            _run_eval(preds_tg2fa, fa_sents, "flores/tg2fa", None)

        if source in ("wiki", "both") and Path(wiki_dump).exists():
            print(f"Parsing Wikipedia dump (max {max_articles} articles)...")
            wiki_texts = _parse_wiki_articles(wiki_dump, max_articles)
            print(f"  Loaded {len(wiki_texts)} article snippets")
            preds = translate_fn(wiki_texts, "fa2tg")
            from persian_tajik_translit.eval.metrics import compute_cer

            mlflow.log_metric("wiki/cer", compute_cer(preds, wiki_texts))
            print(f"[wiki] n={len(preds)} CER={compute_cer(preds, wiki_texts):.4f}")


if __name__ == "__main__":
    import fire

    fire.Fire(eval_ood)
