"""Classification metrics helpers."""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_report_df(
    y_true,
    y_pred,
    labels=None,
) -> pd.DataFrame:
    """Return sklearn classification_report as a tidy DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    df.index.name = "class"
    return df.reset_index()


def macro_f1(y_true, y_pred) -> float:
    """Return macro-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def weighted_f1(y_true, y_pred) -> float:
    """Return weighted-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def accuracy(y_true, y_pred) -> float:
    """Return accuracy."""
    return float(accuracy_score(y_true, y_pred))


def confusion_matrix_fig(
    y_true,
    y_pred,
    labels: list[str],
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Return a matplotlib Figure containing a seaborn annotated heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor="grey",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return fig
