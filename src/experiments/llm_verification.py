"""
llm_verification.py
-------------------
LLM-based claim verification experiments for the INFACT corpus.

Uses HuggingFace ``transformers`` to fine-tune or perform zero-shot / few-shot
inference with pre-trained multilingual language models.

Two modes are supported:

1. **Zero-shot** — format a claim as a prompt and ask a generative model to
   predict its truthfulness.
2. **Fine-tuning** — fine-tune a sequence-classification model (e.g.,
   ``bert-base-multilingual-cased``) on the training split of INFACT.

Example usage
-------------
    from src.data.load_dataset import load_infact
    from src.data.label_mapping import apply_label_mapping
    from src.experiments.llm_verification import run_finetuning

    df = load_infact("data/infact_dataset.tsv")
    df = apply_label_mapping(df)
    run_finetuning(df, model_name="bert-base-multilingual-cased")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def _try_import_torch():
    """Lazily import torch so the module can be imported without GPU."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def build_hf_dataset(
    df: pd.DataFrame,
    text_col: str = "claim_text",
    label_col: str = "label_id",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split *df* into train/test HuggingFace ``Dataset`` objects.

    Parameters
    ----------
    df:
        INFACT DataFrame with text and label columns.
    text_col:
        Name of the column containing input text.
    label_col:
        Name of the column containing integer label IDs.
    test_size:
        Fraction of data to use for testing.
    random_state:
        Random seed for reproducible splits.

    Returns
    -------
    tuple[datasets.Dataset, datasets.Dataset]
        ``(train_dataset, test_dataset)``

    Raises
    ------
    ImportError
        If the ``datasets`` package is not installed.
    """
    try:
        from datasets import Dataset
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError("Install 'datasets' with: pip install datasets") from exc

    df_clean = df.dropna(subset=[text_col, label_col]).copy()
    df_clean = df_clean[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    )
    df_clean["label"] = df_clean["label"].astype(int)

    train_df, test_df = train_test_split(
        df_clean, test_size=test_size, stratify=df_clean["label"], random_state=random_state
    )
    return Dataset.from_pandas(train_df, preserve_index=False), Dataset.from_pandas(
        test_df, preserve_index=False
    )


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------

def tokenize_dataset(dataset, tokenizer, max_length: int = 256):
    """Tokenize a HuggingFace ``Dataset`` using *tokenizer*.

    Parameters
    ----------
    dataset:
        HuggingFace ``Dataset`` with a ``"text"`` column.
    tokenizer:
        HuggingFace tokenizer.
    max_length:
        Maximum token sequence length.

    Returns
    -------
    datasets.Dataset
        Tokenized dataset.
    """
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_fn, batched=True)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def run_finetuning(
    df: pd.DataFrame,
    model_name: str = "bert-base-multilingual-cased",
    num_labels: Optional[int] = None,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    output_dir: str = "results",
    test_size: float = 0.2,
) -> dict:
    """Fine-tune a HuggingFace sequence-classification model on INFACT.

    Parameters
    ----------
    df:
        INFACT DataFrame with ``claim_text`` and ``label_id`` columns.
    model_name:
        HuggingFace model identifier.
    num_labels:
        Number of output labels.  Inferred from ``label_id`` if not provided.
    num_epochs:
        Number of training epochs.
    batch_size:
        Training and evaluation batch size.
    learning_rate:
        AdamW learning rate.
    max_length:
        Maximum token sequence length.
    output_dir:
        Root directory for saving checkpoints and results.
    test_size:
        Fraction of data for the test split.

    Returns
    -------
    dict
        Evaluation metrics from the final epoch.
    """
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("Install 'transformers' and 'torch': pip install transformers torch") from exc

    if "label_id" not in df.columns:
        raise ValueError("'label_id' column not found. Run apply_label_mapping() first.")

    if num_labels is None:
        num_labels = int(df["label_id"].dropna().nunique())

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Building train/test datasets (%d total samples)", len(df))
    train_ds, test_ds = build_hf_dataset(df, test_size=test_size)
    train_ds = tokenize_dataset(train_ds, tokenizer, max_length=max_length)
    test_ds = tokenize_dataset(test_ds, tokenizer, max_length=max_length)

    logger.info("Loading model: %s (num_labels=%d)", model_name, num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    ckpt_dir = Path(output_dir) / "checkpoints" / model_name.replace("/", "_")
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=str(ckpt_dir / "logs"),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    logger.info("Starting fine-tuning …")
    trainer.train()
    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)

    results_path = Path(output_dir) / "tables" / "llm_finetuning_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(eval_results, fh, indent=2)
    logger.info("Saved fine-tuning results to %s", results_path)

    return eval_results


# ---------------------------------------------------------------------------
# Zero-shot inference
# ---------------------------------------------------------------------------

def run_zero_shot(
    df: pd.DataFrame,
    candidate_labels: Optional[list[str]] = None,
    model_name: str = "facebook/bart-large-mnli",
    batch_size: int = 8,
    output_dir: str = "results/tables",
) -> pd.DataFrame:
    """Run zero-shot classification on INFACT claims.

    Parameters
    ----------
    df:
        INFACT DataFrame with a ``claim_text`` column.
    candidate_labels:
        List of label strings for zero-shot classification.
    model_name:
        HuggingFace zero-shot classification pipeline model.
    batch_size:
        Number of claims processed per batch.
    output_dir:
        Directory where predictions CSV is saved.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``claim_text``, ``predicted_label``, and ``score`` columns.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise ImportError("Install 'transformers': pip install transformers") from exc

    if candidate_labels is None:
        candidate_labels = ["true", "false", "misleading", "unverifiable", "mixed"]

    logger.info("Loading zero-shot pipeline: %s", model_name)
    classifier = hf_pipeline("zero-shot-classification", model=model_name)

    df_clean = df.dropna(subset=["claim_text"]).copy()
    texts = df_clean["claim_text"].tolist()

    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot inference"):
        batch = texts[i : i + batch_size]
        results = classifier(batch, candidate_labels)
        for res in results:
            top_label = res["labels"][0]
            top_score = res["scores"][0]
            predictions.append({"predicted_label": top_label, "score": top_score})

    pred_df = pd.DataFrame(predictions)
    df_clean = df_clean.reset_index(drop=True)
    output_df = pd.concat([df_clean[["claim_text"]], pred_df], axis=1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "zero_shot_predictions.csv"
    output_df.to_csv(save_path, index=False)
    logger.info("Saved zero-shot predictions to %s", save_path)

    return output_df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    from src.data.label_mapping import apply_label_mapping
    from src.data.load_dataset import load_infact, validate_dataset

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/infact_dataset.tsv"
    model = sys.argv[2] if len(sys.argv) > 2 else "bert-base-multilingual-cased"

    dataset = load_infact(data_path)
    validate_dataset(dataset)
    dataset = apply_label_mapping(dataset)
    run_finetuning(dataset, model_name=model)
