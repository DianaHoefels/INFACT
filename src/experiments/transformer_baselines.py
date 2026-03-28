"""
transformer_baselines.py
------------------------
Transformer-based baselines for INFACT claim verification.

Implements stratified k-fold cross-validation for two encoder models:
- XLM-RoBERTa-base (multilingual)
- Romanian BERT (dumitrescustefan/bert-base-romanian-cased-v1)

Each model is evaluated on two input variants:
1) claim_text + context
2) claim_text + context + verification_scope

Class-weighted loss is used to mitigate label imbalance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformerConfig:
    name: str
    model_id: str
    include_scope: bool


TRANSFORMER_CONFIGS: list[TransformerConfig] = [
    TransformerConfig(
        name="XLM-RoBERTa (claim+context)",
        model_id="xlm-roberta-base",
        include_scope=False,
    ),
    TransformerConfig(
        name="XLM-RoBERTa (claim+context+scope)",
        model_id="xlm-roberta-base",
        include_scope=True,
    ),
    TransformerConfig(
        name="Romanian BERT (claim+context)",
        model_id="dumitrescustefan/bert-base-romanian-cased-v1",
        include_scope=False,
    ),
    TransformerConfig(
        name="Romanian BERT (claim+context+scope)",
        model_id="dumitrescustefan/bert-base-romanian-cased-v1",
        include_scope=True,
    ),
    TransformerConfig(
        name="Romanian BERT uncased v1 (claim+context)",
        model_id="dumitrescustefan/bert-base-romanian-uncased-v1",
        include_scope=False,
    ),
    TransformerConfig(
        name="Romanian BERT uncased v1 (claim+context+scope)",
        model_id="dumitrescustefan/bert-base-romanian-uncased-v1",
        include_scope=True,
    ),
    TransformerConfig(
        name="Romanian BERT cased v2 (claim+context)",
        model_id="dumitrescustefan/bert-base-romanian-cased-v2",
        include_scope=False,
    ),
    TransformerConfig(
        name="Romanian BERT cased v2 (claim+context+scope)",
        model_id="dumitrescustefan/bert-base-romanian-cased-v2",
        include_scope=True,
    ),
]


def _try_import_transformers():
    try:
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "Install transformer dependencies with: pip install transformers torch datasets"
        ) from exc


def build_text_inputs(
    df: pd.DataFrame,
    include_scope: bool = False,
    sep_token: str = "[SEP]",
    add_tags: bool = True,
) -> pd.Series:
    """Concatenate claim text, context, and (optionally) verification scope.

    Tags help the model disambiguate the role of each segment.
    """
    claim = df["claim_text"].fillna("")
    context = df["context"].fillna("")
    if add_tags:
        pieces = (
            "CLAIM: " + claim + f" {sep_token} " + "CONTEXT: " + context
        )
    else:
        pieces = claim + f" {sep_token} " + context
    if include_scope and "verification_scope" in df.columns:
        scope = df["verification_scope"].fillna("")
        if add_tags:
            pieces = pieces + f" {sep_token} " + "SCOPE: " + scope
        else:
            pieces = pieces + f" {sep_token} " + scope
    return pieces.str.replace(r"\s+", " ", regex=True).str.strip()


def oversample_training_fold(train_df: pd.DataFrame, label_col: str = "label_id") -> pd.DataFrame:
    """Oversample minority classes in a training fold to match the majority count."""
    max_count = train_df[label_col].value_counts().max()
    frames = []
    for _, subset in train_df.groupby(label_col, sort=False):
        frames.append(
            subset.sample(n=max_count, replace=True, random_state=42)
        )
    return pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=42)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def _build_trainer(
    model_id: str,
    num_labels: int,
    class_weights: np.ndarray,
    train_dataset,
    eval_dataset,
    output_dir: Path,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    gradient_accumulation_steps: int,
    seed: int,
):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    )

    weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(logits.device))
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    import inspect

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "save_strategy": "no",
        "logging_strategy": "epoch",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "report_to": "none",
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer


def run_transformer_baselines(
    df: pd.DataFrame,
    configs: Iterable[TransformerConfig] = TRANSFORMER_CONFIGS,
    n_splits: int = 5,
    num_epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    max_length: int = 384,
    gradient_accumulation_steps: int = 4,
    output_dir: str = "results/tables",
    seed: int = 42,
    max_samples: Optional[int] = None,
    oversample: bool = True,
    merge_unverifiable: bool = True,
    drop_empty_context: bool = True,
    min_claim_tokens: int = 3,
    add_tags: bool = True,
) -> pd.DataFrame:
    """Run transformer baselines with stratified cross-validation.

    Returns a DataFrame with mean/std metrics per model configuration.
    """
    _try_import_transformers()

    if "label_id" not in df.columns:
        raise ValueError("'label_id' column not found. Run apply_label_mapping() first.")

    df_clean = df.dropna(subset=["claim_text", "label_id"]).copy()
    if merge_unverifiable and "verdict_normalized" in df_clean.columns:
        mixed_id = df_clean.loc[
            df_clean["verdict_normalized"] == "Mixed", "label_id"
        ].dropna()
        if not mixed_id.empty:
            df_clean.loc[
                df_clean["verdict_normalized"] == "Unverifiable", "label_id"
            ] = int(mixed_id.iloc[0])

    if drop_empty_context and "context" in df_clean.columns:
        df_clean = df_clean[df_clean["context"].fillna("").str.strip().ne("")]

    if min_claim_tokens > 0:
        token_counts = df_clean["claim_text"].fillna("").str.split().str.len()
        df_clean = df_clean[token_counts >= min_claim_tokens]
    if max_samples:
        df_clean = df_clean.sample(n=min(max_samples, len(df_clean)), random_state=seed)

    labels = df_clean["label_id"].astype(int).reset_index(drop=True)
    num_labels = int(labels.nunique())

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows: list[dict] = []
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for config in configs:
        logger.info("Running config: %s", config.name)
        try:
            from transformers import AutoConfig

            AutoConfig.from_pretrained(config.model_id)
        except OSError as exc:
            logger.warning(
                "Skipping config '%s' (model_id=%s): %s",
                config.name,
                config.model_id,
                exc,
            )
            continue
        fold_metrics: list[dict] = []

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(df_clean, labels), start=1):
            train_df = df_clean.iloc[train_idx].reset_index(drop=True)
            test_df = df_clean.iloc[test_idx].reset_index(drop=True)

            from datasets import Dataset

            tokenizer_sep = "[SEP]"
            try:
                from transformers import AutoTokenizer

                tokenizer_sep = AutoTokenizer.from_pretrained(config.model_id).sep_token or "[SEP]"
            except Exception:  # noqa: BLE001
                pass

            if oversample:
                train_df = oversample_training_fold(train_df, label_col="label_id")

            train_texts = build_text_inputs(
                train_df,
                include_scope=config.include_scope,
                sep_token=tokenizer_sep,
                add_tags=add_tags,
            )
            test_texts = build_text_inputs(
                test_df,
                include_scope=config.include_scope,
                sep_token=tokenizer_sep,
                add_tags=add_tags,
            )

            train_dataset = Dataset.from_pandas(
                pd.DataFrame({"text": train_texts, "labels": train_df["label_id"].astype(int)})
            )
            eval_dataset = Dataset.from_pandas(
                pd.DataFrame({"text": test_texts, "labels": test_df["label_id"].astype(int)})
            )

            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(num_labels),
                y=train_df["label_id"].astype(int).to_numpy(),
            )

            fold_dir = out_root / "checkpoints" / f"{config.model_id.replace('/', '_')}_fold{fold_id}"
            trainer = _build_trainer(
                model_id=config.model_id,
                num_labels=num_labels,
                class_weights=class_weights,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=fold_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_length=max_length,
                gradient_accumulation_steps=gradient_accumulation_steps,
                seed=seed,
            )

            logger.info("Fold %d/%d", fold_id, n_splits)
            trainer.train()
            metrics = trainer.evaluate()
            fold_metrics.append(metrics)

        metrics_df = pd.DataFrame(fold_metrics)
        row = {
            "model": config.name,
            "accuracy_mean": float(metrics_df["eval_accuracy"].mean()),
            "accuracy_std": float(metrics_df["eval_accuracy"].std()),
            "f1_macro_mean": float(metrics_df["eval_f1_macro"].mean()),
            "f1_macro_std": float(metrics_df["eval_f1_macro"].std()),
            "f1_weighted_mean": float(metrics_df["eval_f1_weighted"].mean()),
            "f1_weighted_std": float(metrics_df["eval_f1_weighted"].std()),
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_path = out_root / "transformer_cv_results.csv"
    results_df.to_csv(results_path, index=False)

    json_path = out_root / "transformer_cv_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    logger.info("Saved transformer baseline results to %s", results_path)
    return results_df


if __name__ == "__main__":
    import argparse

    from src.data_preprocessing.label_mapping import apply_label_mapping
    from src.data_preprocessing.load_dataset import load_infact, validate_dataset

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run transformer baselines for INFACT")
    parser.add_argument("--data_path", default="data/infact_dataset.tsv")
    parser.add_argument("--output_dir", default="results/tables")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--merge_unverifiable", action="store_true")
    parser.add_argument("--keep_empty_context", action="store_true")
    parser.add_argument("--min_claim_tokens", type=int, default=3)
    parser.add_argument("--disable_tags", action="store_true")

    args = parser.parse_args()

    dataset = load_infact(args.data_path)
    validate_dataset(dataset)
    dataset = apply_label_mapping(dataset)

    run_transformer_baselines(
        dataset,
        n_splits=args.n_splits,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        oversample=args.oversample,
        merge_unverifiable=args.merge_unverifiable,
        drop_empty_context=not args.keep_empty_context,
        min_claim_tokens=args.min_claim_tokens,
        add_tags=not args.disable_tags,
    )
