"""Evaluation metrics for Tajik↔Persian transliteration."""

import editdistance
import sacrebleu

TAJIK_VOWELS = set("аоуиэёю")
PERSIAN_VOWELS = set("اوی")


def compute_chrf_pp(predictions: list[str], references: list[str]) -> float:
    """chrF++ with β=2, char-order=6, word-order=2 (matches Merchant et al. 2025)."""
    score = sacrebleu.corpus_chrf(
        predictions,
        [references],
        beta=2,
        char_order=6,
        word_order=2,
    )
    return float(score.score)


def compute_sequence_accuracy(predictions: list[str], references: list[str]) -> float:
    """Fraction of outputs character-identical to reference."""
    if not predictions:
        return 0.0
    return sum(pred == ref for pred, ref in zip(predictions, references)) / len(predictions)


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Character Error Rate: mean normalised Levenshtein distance."""
    if not predictions:
        return 0.0
    total = sum(
        editdistance.eval(pred, ref) / max(len(ref), 1)
        for pred, ref in zip(predictions, references)
    )
    return total / len(predictions)


def compute_levenshtein_ratio(predictions: list[str], references: list[str]) -> float:
    """Mean Levenshtein ratio (1 − CER), penalises structural mismatches."""
    return 1.0 - compute_cer(predictions, references)


def _vowel_f1_for_pair(pred: str, ref: str, vowel_set: set) -> tuple[int, int, int]:
    """Return (true_positive, false_positive, false_negative) for vowel positions."""
    true_pos = sum(p == r and r in vowel_set for p, r in zip(pred, ref))
    false_pos = sum(p in vowel_set and r not in vowel_set for p, r in zip(pred, ref))
    false_neg = sum(p not in vowel_set and r in vowel_set for p, r in zip(pred, ref))
    return true_pos, false_pos, false_neg


def compute_vowel_f1(
    predictions: list[str], references: list[str], direction: str = "fa2tg"
) -> float:
    """Vowel-class weighted F1 targeting the dominant failure mode.

    Args:
        direction: "fa2tg" checks Tajik vowels {а,о,у,и,э};
                   "tg2fa" checks Persian vowel letters {ا,و,ی}.
    """
    vowel_set = TAJIK_VOWELS if direction == "fa2tg" else PERSIAN_VOWELS
    total_tp = total_fp = total_fn = 0
    for pred, ref in zip(predictions, references):
        tp, fp, fn = _vowel_f1_for_pair(pred, ref, vowel_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_all_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute all five metrics at once."""
    return {
        "chrf_pp": compute_chrf_pp(predictions, references),
        "seq_acc": compute_sequence_accuracy(predictions, references),
        "cer": compute_cer(predictions, references),
        "lev_ratio": compute_levenshtein_ratio(predictions, references),
        "vowel_f1": compute_vowel_f1(predictions, references),
    }
