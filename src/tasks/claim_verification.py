"""Classical TF-IDF + Logistic Regression baselines for claim verification."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.io import save_figure, save_markdown, save_table, setup_logging
from src.utils.metrics import (
    accuracy,
    classification_report_df,
    confusion_matrix_fig,
    macro_f1,
    weighted_f1,
)

logger = setup_logging(__name__)

EXPERIMENTS: dict[str, list[str]] = {
    "A": ["claim_text"],
    "B": ["claim_text", "context"],
    "C": ["claim_text", "context", "verification_scope"],
}


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def prepare_features(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    """Concatenate *text_cols* with ' [SEP] ' separator into a single Series."""
    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].fillna("").apply(lambda row: " [SEP] ".join(row.values.astype(str)), axis=1)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def train_test_split_stratified(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    dev_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, dev_df, test_df) using stratified splitting.

    Falls back to non-stratified splits when the dataset is too small.
    """
    n_classes = df[label_col].nunique()
    # Need at least n_classes samples in the test set for stratification
    min_test_samples = max(n_classes, int(len(df) * test_size))
    use_stratify = len(df) >= max(n_classes * 5, 10)

    train_tmp, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col] if use_stratify else None,
        random_state=seed,
    )
    # dev_size is fraction of the *original* data
    dev_fraction_of_train = dev_size / (1.0 - test_size)
    n_dev = max(1, int(len(train_tmp) * dev_fraction_of_train))
    use_stratify_dev = len(train_tmp) >= max(train_tmp[label_col].nunique() * 3, 6)
    train_df, dev_df = train_test_split(
        train_tmp,
        test_size=n_dev,
        stratify=train_tmp[label_col] if use_stratify_dev else None,
        random_state=seed,
    )
    return train_df, dev_df, test_df


# ---------------------------------------------------------------------------
# Model pipeline
# ---------------------------------------------------------------------------


def build_pipeline(seed: int = 42) -> Pipeline:
    """Return a TF-IDF + Logistic Regression sklearn Pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=50_000,
                    sublinear_tf=True,
                    strip_accents=None,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------


def run_experiment(
    df: pd.DataFrame,
    text_cols: list[str],
    label_col: str,
    seed: int = 42,
) -> dict:
    """Train and evaluate one experiment variant.

    Returns a metrics dict with macro_f1, weighted_f1, accuracy, and per-class
    report DataFrame.
    """
    # Filter rows that have a valid label
    valid = df[df[label_col].notna() & (df[label_col] != "unknown")].copy()
    if len(valid) < 10:
        logger.warning("Too few valid samples (%d) for experiment.", len(valid))
        return {"error": "too_few_samples"}

    # Ensure enough samples per class for stratification
    class_counts = valid[label_col].value_counts()
    min_count = class_counts.min()
    if min_count < 3:
        # Drop rare classes
        keep = class_counts[class_counts >= 3].index
        valid = valid[valid[label_col].isin(keep)]
        logger.warning(
            "Dropped rare classes with < 3 samples. Remaining classes: %s",
            list(keep),
        )

    X = prepare_features(valid, text_cols)
    y = valid[label_col]
    labels = sorted(y.unique().tolist())

    train_df, dev_df, test_df = train_test_split_stratified(
        valid, label_col, test_size=0.2, dev_size=0.1, seed=seed
    )

    X_train = prepare_features(train_df, text_cols)
    X_test = prepare_features(test_df, text_cols)
    y_train = train_df[label_col]
    y_test = test_df[label_col]

    pipeline = build_pipeline(seed=seed)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "macro_f1": macro_f1(y_test, y_pred),
        "weighted_f1": weighted_f1(y_test, y_pred),
        "accuracy": accuracy(y_test, y_pred),
        "labels": labels,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "per_class_report": classification_report_df(y_test, y_pred, labels=labels),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ---------------------------------------------------------------------------
# Full baseline run
# ---------------------------------------------------------------------------


def run_baselines(
    df: pd.DataFrame,
    output_dir: str = "results",
    seed: int = 42,
) -> pd.DataFrame:
    """Run experiments A/B/C on epistemic_outcome and save all outputs.

    Saves:
      - results/tables/baseline_metrics.csv
      - results/tables/per_class_report_{exp}.csv  (per experiment)
      - results/figures/confusion_matrix_{exp}.png (per experiment)
      - results/reports/baseline_report.md

    Returns a summary metrics DataFrame.
    """
    tables_dir = Path(output_dir) / "tables"
    figures_dir = Path(output_dir) / "figures"
    reports_dir = Path(output_dir) / "reports"
    for d in (tables_dir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    label_col = "epistemic_outcome"
    summary_rows = []
    report_lines = ["# Baseline Classification Report\n\n"]

    for exp_id, text_cols in EXPERIMENTS.items():
        logger.info("Running experiment %s with columns: %s", exp_id, text_cols)
        results = run_experiment(df, text_cols, label_col, seed=seed)

        if "error" in results:
            report_lines.append(f"## Experiment {exp_id}\n\nError: {results['error']}\n\n")
            continue

        summary_rows.append(
            {
                "experiment": exp_id,
                "features": " + ".join(text_cols),
                "macro_f1": round(results["macro_f1"], 4),
                "weighted_f1": round(results["weighted_f1"], 4),
                "accuracy": round(results["accuracy"], 4),
                "n_train": results["n_train"],
                "n_test": results["n_test"],
            }
        )

        # Per-class report
        save_table(results["per_class_report"], tables_dir / f"per_class_report_{exp_id}.csv")

        # Confusion matrix
        labels = results["labels"]
        cm_fig = confusion_matrix_fig(
            results["y_test"],
            results["y_pred"],
            labels=labels,
            title=f"Experiment {exp_id} – Confusion Matrix",
        )
        save_figure(cm_fig, figures_dir / f"confusion_matrix_{exp_id}.png")
        plt.close(cm_fig)

        report_lines += [
            f"## Experiment {exp_id}: `{'` + `'.join(text_cols)}`\n\n",
            f"| Metric | Score |\n|---|---|\n",
            f"| Macro F1 | {results['macro_f1']:.4f} |\n",
            f"| Weighted F1 | {results['weighted_f1']:.4f} |\n",
            f"| Accuracy | {results['accuracy']:.4f} |\n",
            f"| Train samples | {results['n_train']} |\n",
            f"| Test samples | {results['n_test']} |\n\n",
            "### Per-class Results\n",
            results["per_class_report"].to_markdown(index=False) + "\n\n",
        ]

    metrics_df = pd.DataFrame(summary_rows)
    if not metrics_df.empty:
        save_table(metrics_df, tables_dir / "baseline_metrics.csv")
        report_lines += [
            "## Summary Table\n",
            metrics_df.to_markdown(index=False) + "\n",
        ]

    save_markdown("".join(report_lines), reports_dir / "baseline_report.md")
    logger.info("Baselines complete. Results in %s/", output_dir)
    return metrics_df
