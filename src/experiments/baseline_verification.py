from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import resample

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


def oversample_minority(
    X: pd.Series,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Randomly oversample minority classes to match the majority class size.

    Parameters
    ----------
    X:
        Training texts.
    y:
        Training labels.
    random_state:
        Random seed.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Oversampled texts and labels.
    """
    df = pd.DataFrame({"text": X.reset_index(drop=True), "label": y.reset_index(drop=True)})
    max_count = df["label"].value_counts().max()

    frames = []
    for cls, subset in df.groupby("label", sort=False):
        frames.append(
            resample(
                subset,
                replace=True,
                n_samples=max_count,
                random_state=random_state,
            )
        )

    balanced = (
        pd.concat(frames, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return balanced["text"], balanced["label"]


def build_text_features(
    df: pd.DataFrame,
    use_context: bool = False,
) -> pd.Series:
    """
    Combine claim_text and optionally context into a single text field.
    """
    texts = df["claim_text"].fillna("")
    if use_context and "context" in df.columns:
        texts = texts + " " + df["context"].fillna("")
    return texts.str.strip()


def build_pipeline(classifier_name: str) -> Pipeline:
    """
    Build TF-IDF + classifier pipeline.
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
            ("clf", clone(CLASSIFIERS[classifier_name])),
        ]
    )


def run_cross_validation(
    texts: pd.Series,
    labels: pd.Series,
    classifier_name: str,
    n_splits: int = 5,
    oversample: bool = False,
    random_state: int = 42,
) -> dict:
    """
    Run stratified k-fold cross-validation with optional oversampling applied
    ONLY to the training fold.

    Parameters
    ----------
    texts:
        Input texts.
    labels:
        Integer label IDs.
    classifier_name:
        Classifier key from CLASSIFIERS.
    n_splits:
        Number of CV folds.
    oversample:
        Whether to oversample minority classes in each training fold.
    random_state:
        Random seed.

    Returns
    -------
    dict
        Mean and std for accuracy, macro-F1, and weighted-F1.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    acc_scores = []
    macro_scores = []
    weighted_scores = []

    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(texts, labels), start=1):
        X_train = texts.iloc[train_idx]
        y_train = labels.iloc[train_idx]
        X_test = texts.iloc[test_idx]
        y_test = labels.iloc[test_idx]

        if oversample:
            X_train, y_train = oversample_minority(X_train, y_train, random_state=random_state)

        pipeline = build_pipeline(classifier_name)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        macro_scores.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_scores.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        logger.info(
            "Fold %d | %s | acc=%.3f | macro-F1=%.3f | weighted-F1=%.3f",
            fold_id,
            classifier_name,
            acc_scores[-1],
            macro_scores[-1],
            weighted_scores[-1],
        )

    return {
        "accuracy_mean": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
        "f1_macro_mean": float(np.mean(macro_scores)),
        "f1_macro_std": float(np.std(macro_scores)),
        "f1_weighted_mean": float(np.mean(weighted_scores)),
        "f1_weighted_std": float(np.std(weighted_scores)),
    }


def run_baseline(
    df: pd.DataFrame,
    classifier_names: Optional[list[str]] = None,
    use_context: bool = False,
    n_splits: int = 5,
    output_dir: str = "results/tables",
    oversample: bool = False,
) -> pd.DataFrame:
    """
    Run baseline verification experiments and save cross-validation results.

    Notes
    -----
    Oversampling, if enabled, is applied only within each training fold,
    preventing leakage into validation folds.
    """
    if classifier_names is None:
        classifier_names = list(CLASSIFIERS.keys())

    if "label_id" not in df.columns:
        raise ValueError("'label_id' column not found. Run apply_label_mapping() first.")

    df_clean = df.dropna(subset=["claim_text", "label_id"]).copy()
    texts = build_text_features(df_clean, use_context=use_context)
    labels = df_clean["label_id"].astype(int)

    class_counts = labels.value_counts()
    rare_classes = class_counts[class_counts < n_splits].index
    if len(rare_classes) > 0:
        logger.warning(
            "Dropping %d class(es) with fewer than %d samples for CV: %s",
            len(rare_classes),
            n_splits,
            rare_classes.tolist(),
        )
        mask = ~labels.isin(rare_classes)
        texts = texts[mask]
        labels = labels[mask]

    logger.info(
        "Running baseline experiments on %d samples, %d classes.",
        len(labels),
        labels.nunique(),
    )

    rows = []
    for name in classifier_names:
        logger.info("Evaluating classifier: %s", name)
        scores = run_cross_validation(
            texts=texts,
            labels=labels,
            classifier_name=name,
            n_splits=n_splits,
            oversample=oversample,
            random_state=42,
        )
        scores["classifier"] = name
        rows.append(scores)

        logger.info(
            "  Accuracy: %.3f ± %.3f | F1-macro: %.3f ± %.3f | F1-weighted: %.3f ± %.3f",
            scores["accuracy_mean"],
            scores["accuracy_std"],
            scores["f1_macro_mean"],
            scores["f1_macro_std"],
            scores["f1_weighted_mean"],
            scores["f1_weighted_std"],
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

    run_baseline(
        dataset,
        use_context=False,
        n_splits=5,
        oversample=False,  # safer default
    )