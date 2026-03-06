"""Utilities for loading and processing the INFACT dataset."""

import json
import os
from pathlib import Path
from typing import Iterator


# Valid verdict labels in INFACT
LABELS = ["TRUE", "FALSE", "PARTIALLY_TRUE", "UNVERIFIABLE"]

# Mapping from label string to integer index
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Default data directory (relative to this file's location)
_DATA_DIR = Path(__file__).parent.parent / "data"


def get_data_dir() -> Path:
    """Return the path to the data directory."""
    return _DATA_DIR


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSON Lines file and return a list of dictionaries.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        A list of dicts, one per line in the file.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    instances: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                instances.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return instances


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Lazily iterate over a JSON Lines file.

    Yields one dictionary per non-empty line. Useful for large files
    that do not fit in memory.

    Args:
        path: Path to the ``.jsonl`` file.

    Yields:
        Parsed JSON object for each line.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc


def load_split(split: str, data_dir: str | Path | None = None) -> list[dict]:
    """Load one of the canonical dataset splits.

    Args:
        split: One of ``"train"``, ``"dev"``, or ``"test"``.
        data_dir: Directory containing the split files. Defaults to
            the ``data/`` directory at the root of this repository.

    Returns:
        A list of instance dicts.

    Raises:
        ValueError: If *split* is not a recognised split name.
        FileNotFoundError: If the corresponding file does not exist.
    """
    valid_splits = ("train", "dev", "test")
    if split not in valid_splits:
        raise ValueError(f"Unknown split '{split}'. Must be one of {valid_splits}.")

    data_dir = Path(data_dir) if data_dir is not None else get_data_dir()
    path = data_dir / f"{split}.jsonl"
    return load_jsonl(path)


def get_labels(instances: list[dict]) -> list[str]:
    """Extract the label field from a list of instances.

    Args:
        instances: List of INFACT instance dicts.

    Returns:
        List of label strings in the same order as *instances*.
    """
    return [inst["label"] for inst in instances]


def encode_labels(labels: list[str]) -> list[int]:
    """Convert a list of label strings to integer indices.

    Args:
        labels: List of label strings (e.g. ``["TRUE", "FALSE", ...]``).

    Returns:
        List of integer label indices.

    Raises:
        KeyError: If an unrecognised label is encountered.
    """
    try:
        return [LABEL2ID[label] for label in labels]
    except KeyError as exc:
        raise KeyError(f"Unknown label: {exc}. Valid labels are {LABELS}.") from exc


def dataset_statistics(instances: list[dict]) -> dict:
    """Compute basic statistics for a list of INFACT instances.

    Args:
        instances: List of INFACT instance dicts.

    Returns:
        A dict with keys ``total``, ``label_counts``, and ``domain_counts``.
    """
    label_counts: dict[str, int] = {label: 0 for label in LABELS}
    domain_counts: dict[str, int] = {}

    for inst in instances:
        label = inst.get("label", "UNKNOWN")
        label_counts[label] = label_counts.get(label, 0) + 1

        domain = inst.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    return {
        "total": len(instances),
        "label_counts": label_counts,
        "domain_counts": domain_counts,
    }
