"""Evaluate alignment on the INFACT benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, save_dataframe, save_json

logger = logging.getLogger(__name__)

ALLOWED_VERDICTS = [
    "True",
    "False",
    "Mixed",
    "Mostly True",
    "Mostly False",
    "Unverifiable",
]

NUANCED_LABELS = {"Mostly True", "Mostly False", "Mixed"}


def compute_verdict_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Compute accuracy and F1 metrics for verdict predictions."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=ALLOWED_VERDICTS, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=ALLOWED_VERDICTS, zero_division=0)),
    }
    return metrics


def compute_nuance_collapse(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Compute the rate where nuanced gold labels are predicted as coarse labels."""
    total = 0
    collapsed = 0
    for gold, pred in zip(y_true, y_pred):
        if gold in NUANCED_LABELS:
            total += 1
            if pred in {"True", "False", "Unverifiable"}:
                collapsed += 1
    rate = float(collapsed / total) if total else 0.0
    return {"nuance_collapse_rate": rate, "nuanced_total": float(total), "nuanced_collapsed": float(collapsed)}


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i, tok_a in enumerate(a, start=1):
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[len(a), len(b)])


def _rouge_l_f1(pred: str, ref: str) -> float:
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_rouge_scores(preds: Iterable[str], refs: Iterable[str]) -> dict[str, float]:
    """Compute average ROUGE-L F1 for paired predictions and references."""
    scores = [
        _rouge_l_f1(p or "", r or "")
        for p, r in zip(preds, refs)
    ]
    return {
        "rouge_l_f1_mean": float(np.mean(scores)) if scores else 0.0,
        "rouge_l_f1_std": float(np.std(scores)) if scores else 0.0,
    }


EVIDENCE_PATTERNS = {
    "Law": [
        r"\blegea\b",
        r"\bart\.?\s*\d+",
        r"\bconstitut",
        r"\bordonan",
        r"\bOUG\b",
        r"\bhotarare\b",
        r"\bdecret\b",
        r"\bregulament\b",
    ],
    "Statistics": [
        r"\bprocent\b",
        r"%",
        r"\bstatistic\b",
        r"\bsondaj\b",
        r"\bdate\b",
        r"\bcifra\b",
        r"\bnumar\b",
        r"\bmiliard\b",
        r"\bmilion\b",
    ],
    "Authority": [
        r"\bminister\b",
        r"\bguvern\b",
        r"\bparlament\b",
        r"\binstitut\b",
        r"\bcomisia\b",
        r"\bcurtea\b",
        r"\buniversit\b",
        r"\boffic(i)?al\b",
        r"\beurostat\b",
        r"\bexper\w+",
        r"\bONU\b",
        r"\bUE\b",
    ],
    "Source/URL": [
        r"https?://\S+",
        r"\bsursa\b",
        r"\bfacebook\b",
        r"\btwitter\b",
        r"\bsite\b",
        r"\bpagina\b",
        r"\barticol\b",
        r"\blink\b",
    ],
    "Time": [
        r"\b(19|20)\d{2}\b",
        r"\bdata\b",
        r"\banul\b",
        r"\bluna\b",
        r"\bazi\b",
        r"\bieri\b",
        r"\bmaine\b",
        r"\bianuar|febru|mart|april|mai|iun|iul|aug|sept|oct|nov|dec",
    ],
}


def extract_evidence_types(text: str) -> set[str]:
    """Extract evidence categories present in the text using regex rules."""
    if not isinstance(text, str) or not text.strip():
        return set()
    matches: set[str] = set()
    for category, patterns in EVIDENCE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                matches.add(category)
                break
    return matches


def compute_evidence_overlap(preds: Iterable[str], refs: Iterable[str]) -> dict[str, float]:
    """Compute overlap between evidence types in predictions and references."""
    overlaps = []
    counts = []
    empty_refs = 0
    for pred, ref in zip(preds, refs):
        pred_types = extract_evidence_types(pred or "")
        ref_types = extract_evidence_types(ref or "")
        if not ref_types:
            empty_refs += 1
            continue
        overlap = pred_types.intersection(ref_types)
        overlaps.append(len(overlap) / len(ref_types))
        counts.append(len(overlap))

    return {
        "overlap_ratio_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "overlap_ratio_std": float(np.std(overlaps)) if overlaps else 0.0,
        "overlap_count_mean": float(np.mean(counts)) if counts else 0.0,
        "gold_empty_rate": float(empty_refs / max(1, empty_refs + len(overlaps))),
    }


def save_report(report: dict[str, Any], path: str) -> None:
    """Save a JSON report."""
    save_json(report, path)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Mistral-Large-2411 alignment.")
    parser.add_argument(
        "--input_path",
        default="data/infact.tsv",
        help="Path to INFACT TSV (default: data/infact.tsv)",
    )
    parser.add_argument(
        "--jsonl_path",
        default="results/llm_outputs/mistral_large_2411_outputs.jsonl",
    )
    parser.add_argument(
        "--report_path",
        default="results/reports/mistral_large_2411_alignment_report.json",
    )
    parser.add_argument(
        "--summary_path",
        default="results/tables/mistral_large_2411_alignment_summary.csv",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df = pd.read_csv(args.input_path, sep="\t", dtype={"record_id": "int64"})
    outputs = _read_jsonl(args.jsonl_path)
    out_df = pd.DataFrame(outputs)

    if out_df.empty:
        raise ValueError(f"No outputs found in {args.jsonl_path}")

    out_df["record_id"] = out_df["record_id"].astype("int64")
    out_df = out_df.drop_duplicates(subset=["record_id"]).copy()

    merged = df.merge(out_df, on="record_id", how="left", suffixes=("", "_pred"))
    logger.info("Merged %d dataset rows with %d outputs", len(df), len(out_df))

    gold = merged["verdict_normalized"].fillna("")
    pred = merged["verdict"].fillna("")

    valid_mask = gold.isin(ALLOWED_VERDICTS) & pred.isin(ALLOWED_VERDICTS)
    valid_gold = gold[valid_mask].tolist()
    valid_pred = pred[valid_mask].tolist()

    parse_ok_rate = float(merged["parse_ok"].fillna(False).mean())

    verdict_metrics = compute_verdict_metrics(valid_gold, valid_pred) if valid_gold else {
        "accuracy": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
    }
    nuance_metrics = compute_nuance_collapse(valid_gold, valid_pred) if valid_gold else {
        "nuance_collapse_rate": 0.0,
        "nuanced_total": 0.0,
        "nuanced_collapsed": 0.0,
    }

    rouge_conclusion = compute_rouge_scores(
        merged["explanation"].fillna("").tolist(),
        merged["conclusion"].fillna("").tolist(),
    )
    rouge_verification = compute_rouge_scores(
        merged["explanation"].fillna("").tolist(),
        merged["verification"].fillna("").tolist(),
    )

    overlap_conclusion = compute_evidence_overlap(
        merged["explanation"].fillna("").tolist(),
        merged["conclusion"].fillna("").tolist(),
    )
    overlap_verification = compute_evidence_overlap(
        merged["explanation"].fillna("").tolist(),
        merged["verification"].fillna("").tolist(),
    )

    report = {
        "input_path": args.input_path,
        "jsonl_path": args.jsonl_path,
        "rows_total": int(len(df)),
        "rows_with_outputs": int(merged["verdict"].notna().sum()),
        "parse_ok_rate": parse_ok_rate,
        "verdict_metrics": verdict_metrics,
        "nuance_collapse": nuance_metrics,
        "rouge_l_vs_conclusion": rouge_conclusion,
        "rouge_l_vs_verification": rouge_verification,
        "evidence_overlap_vs_conclusion": overlap_conclusion,
        "evidence_overlap_vs_verification": overlap_verification,
    }

    save_report(report, args.report_path)

    summary_row = {
        **verdict_metrics,
        **{f"nuance_{k}": v for k, v in nuance_metrics.items()},
        "rouge_l_conclusion": rouge_conclusion["rouge_l_f1_mean"],
        "rouge_l_verification": rouge_verification["rouge_l_f1_mean"],
        "evidence_overlap_conclusion": overlap_conclusion["overlap_ratio_mean"],
        "evidence_overlap_verification": overlap_verification["overlap_ratio_mean"],
        "parse_ok_rate": parse_ok_rate,
        "rows_total": int(len(df)),
        "rows_used": int(len(valid_gold)),
    }

    summary_df = pd.DataFrame([summary_row])
    ensure_dir(Path(args.summary_path).parent)
    save_dataframe(summary_df, args.summary_path, index=False, fmt="csv")

    logger.info("Saved report to %s", args.report_path)
    logger.info("Saved summary to %s", args.summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
