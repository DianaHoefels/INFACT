"""
evaluation.py
-------------
Shared evaluation utilities for INFACT experiments.

Provides functions for:
- Computing classification metrics (accuracy, F1, precision, recall)
- Generating and saving confusion matrices
- Producing and saving classification reports

Example usage
-------------
    from src.experiments.evaluation import evaluate_predictions, plot_confusion_matrix

    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 2, 2, 0, 1]
    metrics = evaluate_predictions(y_true, y_pred, label_names=["True", "Mixed", "False"])
    plot_confusion_matrix(y_true, y_pred, label_names=["True", "Mixed", "False"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    label_names: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    tag: str = "evaluation",
) -> dict:
    """Compute and optionally save classification evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    label_names:
        Human-readable label names for the report.
    output_dir:
        If provided, saves the classification report JSON to this directory.
    tag:
        String prefix for saved files.

    Returns
    -------
    dict
        Dictionary with ``accuracy``, ``f1_macro``, ``f1_weighted``,
        ``precision_macro``, ``recall_macro``, and the full
        ``classification_report`` string.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    report_str = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "classification_report": report_str,
    }

    logger.info("Accuracy: %.4f | F1-macro: %.4f | F1-weighted: %.4f", acc, f1_macro, f1_weighted)
    print(report_str)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / f"{tag}_metrics.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(
                {k: v for k, v in metrics.items() if k != "classification_report"},
                fh,
                indent=2,
            )
        logger.info("Saved metrics to %s", report_path)

    return metrics


def plot_confusion_matrix(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    label_names: Optional[list[str]] = None,
    output_dir: str = "results/figures",
    tag: str = "confusion_matrix",
    show: bool = False,
    normalize: Optional[str] = "true",
) -> Path:
    """Plot and save a confusion matrix.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    label_names:
        Human-readable label names.
    output_dir:
        Directory where the figure is saved.
    tag:
        Filename prefix.
    show:
        Whether to call ``plt.show()`` interactively.
    normalize:
        Normalisation method passed to ``ConfusionMatrixDisplay``.
        ``"true"`` normalises by row (recall per class).

    Returns
    -------
    Path
        Path to the saved figure.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix ({normalize or 'raw'})", fontsize=14)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / f"{tag}.png"
    fig.savefig(save_path, dpi=150)
    logger.info("Saved confusion matrix to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    # Minimal smoke-test with synthetic data
    rng = np.random.default_rng(0)
    labels = [0, 1, 2, 3, 4]
    y_true = rng.choice(labels, size=100)
    y_pred = rng.choice(labels, size=100)

    label_names = ["True", "Mostly True", "Mixed", "Mostly False", "False"]
    temp_dir = Path(tempfile.gettempdir()) / "infact_evaluation"
    metrics = evaluate_predictions(y_true, y_pred, label_names=label_names, output_dir=str(temp_dir))
    plot_confusion_matrix(y_true, y_pred, label_names=label_names, output_dir=str(temp_dir))
    print("Accuracy:", metrics["accuracy"])
