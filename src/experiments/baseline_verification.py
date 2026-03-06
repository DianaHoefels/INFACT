"""
baseline_verification.py
------------------------
Baseline claim verification experiments for the INFACT corpus.

Implements a TF-IDF + machine-learning pipeline that treats fact-checking
as a multi-class text classification problem.  Supported models:

* Logistic Regression
* Support Vector Machine (linear kernel)
* Multinomial Naive Bayes
* Random Forest

The pipeline uses ``claim_text`` (and optionally ``context``) as features and
``label_id`` (produced by :mod:`src.data.label_mapping`) as the target.

Example usage
-------------
    from src.data.load_dataset import load_infact
    from src.data.label_mapping import apply_label_mapping
    from src.experiments.baseline_verification import run_baseline

    df = load_infact("data/infact_dataset.tsv")
    df = apply_label_mapping(df)
    results = run_baseline(df, output_dir="results/tables")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

CLASSIFIERS: dict = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "svm": LinearSVC(max_iter=2000, class_weight="balanced", random_state=42),
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}


def build_text_features(
    df: pd.DataFrame,
    use_context: bool = False,
) -> pd.Series:
    """Combine ``claim_text`` and optionally ``context`` into a single string.

    Parameters
    ----------
    df:
        INFACT DataFrame.
    use_context:
        If ``True``, concatenate ``context`` to ``claim_text``.

    Returns
    -------
    pd.Series
        Series of combined text strings.
    """
    texts = df["claim_text"].fillna("")
    if use_context and "context" in df.columns:
        texts = texts + " " + df["context"].fillna("")
    return texts


def build_pipeline(classifier_name: str) -> Pipeline:
    """Construct a TF-IDF + classifier :class:`sklearn.pipeline.Pipeline`.

    Parameters
    ----------
    classifier_name:
        One of the keys in :data:`CLASSIFIERS`.

    Returns
    -------
    sklearn.pipeline.Pipeline

    Raises
    ------
    KeyError
        If *classifier_name* is not recognised.
    """
    if classifier_name not in CLASSIFIERS:
        raise KeyError(
            f"Unknown classifier '{classifier_name}'. "
            f"Choose from: {list(CLASSIFIERS.keys())}"
        )
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            ("clf", CLASSIFIERS[classifier_name]),
        ]
    )


def run_cross_validation(
    texts: pd.Series,
    labels: pd.Series,
    pipeline: Pipeline,
    n_splits: int = 5,
) -> dict:
    """Run stratified k-fold cross-validation.

    Parameters
    ----------
    texts:
        Text feature series.
    labels:
        Numeric label series.
    pipeline:
        Sklearn pipeline to evaluate.
    n_splits:
        Number of CV folds.

    Returns
    -------
    dict
        Mean and standard deviation of accuracy, macro-F1, and weighted-F1.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        texts,
        labels,
        cv=cv,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        n_jobs=-1,
    )
    return {
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
        "f1_macro_std": float(np.std(scores["test_f1_macro"])),
        "f1_weighted_mean": float(np.mean(scores["test_f1_weighted"])),
        "f1_weighted_std": float(np.std(scores["test_f1_weighted"])),
    }


def run_baseline(
    df: pd.DataFrame,
    classifier_names: Optional[list[str]] = None,
    use_context: bool = False,
    n_splits: int = 5,
    output_dir: str = "results/tables",
) -> pd.DataFrame:
    """Run baseline verification experiments and save results.

    Parameters
    ----------
    df:
        INFACT DataFrame with ``claim_text`` and ``label_id`` columns.
    classifier_names:
        List of classifiers to evaluate.  Defaults to all classifiers in
        :data:`CLASSIFIERS`.
    use_context:
        Whether to include ``context`` in the text features.
    n_splits:
        Number of CV folds.
    output_dir:
        Directory where the results CSV is saved.

    Returns
    -------
    pd.DataFrame
        Summary table of cross-validation scores for each classifier.
    """
    if classifier_names is None:
        classifier_names = list(CLASSIFIERS.keys())

    if "label_id" not in df.columns:
        raise ValueError("'label_id' column not found. Run apply_label_mapping() first.")

    # Drop rows with missing labels or text
    df_clean = df.dropna(subset=["claim_text", "label_id"]).copy()
    texts = build_text_features(df_clean, use_context=use_context)
    labels = df_clean["label_id"].astype(int)

    logger.info(
        "Running baseline experiments on %d samples, %d classes.",
        len(labels),
        labels.nunique(),
    )

    rows = []
    for name in classifier_names:
        logger.info("Evaluating classifier: %s", name)
        pipeline = build_pipeline(name)
        scores = run_cross_validation(texts, labels, pipeline, n_splits=n_splits)
        scores["classifier"] = name
        rows.append(scores)
        logger.info(
            "  Accuracy: %.3f ± %.3f  |  F1-macro: %.3f ± %.3f",
            scores["accuracy_mean"],
            scores["accuracy_std"],
            scores["f1_macro_mean"],
            scores["f1_macro_std"],
        )

    results_df = pd.DataFrame(rows).set_index("classifier")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "baseline_cv_results.csv"
    results_df.to_csv(save_path)
    logger.info("Saved results to %s", save_path)

    print("\nBaseline cross-validation results:")
    print(results_df.to_string())
    return results_df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    from src.data.label_mapping import apply_label_mapping
    from src.data.load_dataset import load_infact, validate_dataset

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/infact_dataset.tsv"
    dataset = load_infact(data_path)
    validate_dataset(dataset)
    dataset = apply_label_mapping(dataset)
    run_baseline(dataset)
