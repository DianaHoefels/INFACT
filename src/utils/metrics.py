"""
metrics.py
----------
Evaluation metric utilities shared across INFACT experiments.

Wraps scikit-learn and custom metrics with consistent interfaces and
provides helpers for aggregating results across multiple runs or folds.

Example usage
-------------
    from src.utils.metrics import compute_metrics, format_metric_table

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 2, 0, 1, 1]
    metrics = compute_metrics(y_true, y_pred)
    print(format_metric_table([metrics]))
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    label_names: Optional[list[str]] = None,
    prefix: str = "",
) -> dict:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.
    label_names:
        Optional human-readable label names (used only for display).
    prefix:
        String prefix added to each metric key (e.g., ``"test_"``).

    Returns
    -------
    dict
        Metrics: ``accuracy``, ``f1_macro``, ``f1_weighted``,
        ``precision_macro``, ``recall_macro``.
    """
    p = prefix
    return {
        f"{p}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{p}f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{p}f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{p}precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{p}recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def aggregate_cv_metrics(fold_metrics: list[dict]) -> dict:
    """Aggregate per-fold metrics into mean ± std.

    Parameters
    ----------
    fold_metrics:
        List of metric dictionaries, one per fold.

    Returns
    -------
    dict
        For each metric key, includes ``{key}_mean`` and ``{key}_std``.
    """
    if not fold_metrics:
        return {}

    keys = fold_metrics[0].keys()
    result: dict = {}
    for key in keys:
        values = [m[key] for m in fold_metrics if key in m]
        arr = np.array(values, dtype=float)
        result[f"{key}_mean"] = float(arr.mean())
        result[f"{key}_std"] = float(arr.std())

    return result


def format_metric_table(
    rows: list[dict],
    index_key: Optional[str] = None,
    float_fmt: str = ".4f",
) -> str:
    """Format a list of metric dictionaries as a pretty-printed table.

    Parameters
    ----------
    rows:
        List of metric dictionaries.
    index_key:
        If provided, use the value at this key as the row index.
    float_fmt:
        Format string for floating-point values.

    Returns
    -------
    str
        String representation of the table.
    """
    if not rows:
        return "(empty)"
    df = pd.DataFrame(rows)
    if index_key and index_key in df.columns:
        df = df.set_index(index_key)
    return df.to_string(float_format=lambda x: f"{x:{float_fmt}}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    y_pred = rng.integers(0, 5, size=200)

    metrics = compute_metrics(y_true, y_pred)
    print("Single-run metrics:")
    print(metrics)

    fold_results = [compute_metrics(rng.integers(0, 5, 100), rng.integers(0, 5, 100)) for _ in range(5)]
    agg = aggregate_cv_metrics(fold_results)
    print("\nAggregated CV metrics:")
    print(agg)

    print("\nFormatted table:")
    print(format_metric_table(fold_results))
