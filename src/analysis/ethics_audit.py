"""
ethics_audit.py
---------------
Bias and ethics auditing for the INFACT corpus.

Audits the dataset and model predictions for:

1. **Demographic and topical representation bias** — are certain domains or
   authors over- or under-represented?
2. **Label imbalance** — are verdict classes severely skewed?
3. **Temporal bias** — does coverage change significantly over time?
4. **Author bias** — do specific authors receive disproportionately negative
   verdicts?
5. **Model fairness (optional)** — given a set of predictions, compute
   per-group performance disparities.

All findings are written to a JSON report in ``results/reports/``.

Example usage
-------------
    from src.data.load_dataset import load_infact
    from src.data.label_mapping import apply_label_mapping
    from src.analysis.ethics_audit import run_ethics_audit

    df = load_infact("data/infact_dataset.tsv")
    df = apply_label_mapping(df)
    report = run_ethics_audit(df, output_dir="results/reports")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Imbalance threshold: flag if the most common class is this many times more
# frequent than the least common.
IMBALANCE_RATIO_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# Individual audit functions
# ---------------------------------------------------------------------------

def audit_label_imbalance(
    df: pd.DataFrame,
    label_col: str = "verdict_normalized",
) -> dict:
    """Assess class imbalance in the verdict distribution.

    Parameters
    ----------
    df:
        INFACT DataFrame.
    label_col:
        Column containing verdict labels.

    Returns
    -------
    dict
        ``counts``, ``proportions``, ``imbalance_ratio``, and a ``warning``
        flag if the ratio exceeds :data:`IMBALANCE_RATIO_THRESHOLD`.
    """
    if label_col not in df.columns:
        return {"error": f"Column '{label_col}' not found."}

    counts = df[label_col].value_counts()
    proportions = (counts / counts.sum()).round(4)
    ratio = float(counts.iloc[0] / counts.iloc[-1]) if len(counts) > 1 else 1.0

    return {
        "counts": counts.to_dict(),
        "proportions": proportions.to_dict(),
        "imbalance_ratio": round(ratio, 2),
        "warning": ratio >= IMBALANCE_RATIO_THRESHOLD,
    }


def audit_domain_representation(df: pd.DataFrame) -> dict:
    """Assess whether any domain is over- or under-represented.

    A domain is flagged as under-represented if its share of records is below
    2 % of the corpus.

    Parameters
    ----------
    df:
        INFACT DataFrame.

    Returns
    -------
    dict
        Domain proportions and lists of over- / under-represented domains.
    """
    if "domain_claim" not in df.columns:
        return {"error": "Column 'domain_claim' not found."}

    counts = df["domain_claim"].value_counts()
    proportions = (counts / counts.sum()).round(4)
    underrepresented = proportions[proportions < 0.02].index.tolist()
    overrepresented = proportions[proportions > 0.20].index.tolist()

    return {
        "proportions": proportions.to_dict(),
        "underrepresented_domains": underrepresented,
        "overrepresented_domains": overrepresented,
    }


def audit_author_verdict_bias(
    df: pd.DataFrame,
    label_col: str = "verdict_normalized",
    min_claims: int = 5,
) -> dict:
    """Identify authors with disproportionately skewed verdict distributions.

    Parameters
    ----------
    df:
        INFACT DataFrame.
    label_col:
        Column containing verdict labels.
    min_claims:
        Minimum number of claims an author must have to be included.

    Returns
    -------
    dict
        Per-author false-rate and a list of flagged authors (false rate > 80 %).
    """
    if "author_claim" not in df.columns or label_col not in df.columns:
        return {"error": "Required columns not found."}

    # Compute false-verdict rate per author
    df_clean = df.dropna(subset=["author_claim", label_col]).copy()
    df_clean["is_false"] = df_clean[label_col].isin(["False", "Mostly False"]).astype(int)

    author_stats = (
        df_clean.groupby("author_claim")
        .agg(n_claims=("is_false", "count"), false_rate=("is_false", "mean"))
        .query(f"n_claims >= {min_claims}")
        .round(4)
    )
    flagged = author_stats[author_stats["false_rate"] > 0.80].index.tolist()

    return {
        "n_authors_analysed": int(len(author_stats)),
        "flagged_authors_false_rate_gt_80pct": flagged,
        "top_authors_by_false_rate": author_stats.nlargest(10, "false_rate").to_dict("index"),
    }


def audit_temporal_bias(df: pd.DataFrame) -> dict:
    """Check whether temporal coverage is balanced year-over-year.

    A year is flagged if it contributes fewer than 5 % of total records.

    Parameters
    ----------
    df:
        INFACT DataFrame with a ``date_verified`` datetime column.

    Returns
    -------
    dict
        Yearly counts, proportions, and a list of sparse years.
    """
    if "date_verified" not in df.columns:
        return {"error": "Column 'date_verified' not found."}

    yearly = df["date_verified"].dt.year.value_counts().sort_index()
    proportions = (yearly / yearly.sum()).round(4)
    sparse_years = proportions[proportions < 0.05].index.tolist()

    return {
        "yearly_counts": {int(k): int(v) for k, v in yearly.items()},
        "yearly_proportions": {int(k): float(v) for k, v in proportions.items()},
        "sparse_years_lt_5pct": [int(y) for y in sparse_years],
    }


def audit_model_fairness(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str,
) -> dict:
    """Compute per-group accuracy to detect model performance disparities.

    Parameters
    ----------
    df:
        DataFrame with ground-truth labels, predictions, and group column.
    y_true_col:
        Column with ground-truth labels.
    y_pred_col:
        Column with predicted labels.
    group_col:
        Column defining demographic/topical groups.

    Returns
    -------
    dict
        Per-group accuracy and maximum accuracy gap between groups.
    """
    for col in (y_true_col, y_pred_col, group_col):
        if col not in df.columns:
            return {"error": f"Column '{col}' not found."}

    df_clean = df.dropna(subset=[y_true_col, y_pred_col, group_col]).copy()
    df_clean["correct"] = (df_clean[y_true_col] == df_clean[y_pred_col]).astype(int)
    group_acc = df_clean.groupby(group_col)["correct"].mean().round(4)
    accuracy_gap = float(group_acc.max() - group_acc.min())

    return {
        "per_group_accuracy": group_acc.to_dict(),
        "accuracy_gap": accuracy_gap,
        "warning": accuracy_gap > 0.15,
    }


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_ethics_audit(
    df: pd.DataFrame,
    label_col: str = "verdict_normalized",
    output_dir: str = "results/reports",
    y_true_col: Optional[str] = None,
    y_pred_col: Optional[str] = None,
    group_col: Optional[str] = None,
) -> dict:
    """Run the full ethics and bias audit suite.

    Parameters
    ----------
    df:
        INFACT DataFrame (label mapping recommended).
    label_col:
        Column containing normalised verdict labels.
    output_dir:
        Directory where the JSON report is saved.
    y_true_col:
        (Optional) Ground-truth label column for fairness audit.
    y_pred_col:
        (Optional) Predicted label column for fairness audit.
    group_col:
        (Optional) Group column for fairness audit.

    Returns
    -------
    dict
        Aggregated ethics audit report.
    """
    report: dict = {
        "label_imbalance": audit_label_imbalance(df, label_col=label_col),
        "domain_representation": audit_domain_representation(df),
        "author_verdict_bias": audit_author_verdict_bias(df, label_col=label_col),
        "temporal_bias": audit_temporal_bias(df),
    }

    if y_true_col and y_pred_col and group_col:
        report["model_fairness"] = audit_model_fairness(df, y_true_col, y_pred_col, group_col)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "ethics_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Saved ethics audit report to %s", report_path)

    # Print high-level warnings
    print("\n--- Ethics Audit Summary ---")
    li = report.get("label_imbalance", {})
    if li.get("warning"):
        print(
            f"  ⚠ Class imbalance ratio: {li['imbalance_ratio']:.1f}x "
            "(consider oversampling or class-weighted training)"
        )
    else:
        print(f"  ✓ Class imbalance ratio: {li.get('imbalance_ratio', 'N/A'):.1f}x")

    dr = report.get("domain_representation", {})
    underrep = dr.get("underrepresented_domains", [])
    if underrep:
        print(f"  ⚠ Under-represented domains (<2 %): {underrep[:5]}")

    ab = report.get("author_verdict_bias", {})
    flagged = ab.get("flagged_authors_false_rate_gt_80pct", [])
    if flagged:
        print(f"  ⚠ Authors with >80 % false verdicts: {flagged[:5]}")
    print()

    return report


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    from src.data.label_mapping import apply_label_mapping
    from src.data.load_dataset import load_infact, validate_dataset

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/infact_dataset.tsv"
    dataset = load_infact(data_path)
    validate_dataset(dataset)
    dataset = apply_label_mapping(dataset)
    run_ethics_audit(dataset)
