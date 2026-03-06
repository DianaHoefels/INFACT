"""Utility helpers: logging, file I/O, seed control."""
from __future__ import annotations

import csv
import logging
import os
import random
from pathlib import Path

import numpy as np


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a named logger with a stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_output_dirs(base: str = "results") -> None:
    """Create the standard results sub-directories if they don't exist."""
    for sub in ("tables", "figures", "reports"):
        Path(base, sub).mkdir(parents=True, exist_ok=True)


def save_table(df, path: str | os.PathLike) -> None:
    """Save a DataFrame as a UTF-8 CSV file."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
    except Exception as exc:
        logging.getLogger(__name__).error("save_table failed for %s: %s", path, exc)


def save_tsv(df, path: str | os.PathLike) -> None:
    """Save a DataFrame as a UTF-8 TSV file."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE)
    except Exception as exc:
        logging.getLogger(__name__).error("save_tsv failed for %s: %s", path, exc)


def save_markdown(text: str, path: str | os.PathLike) -> None:
    """Write a string to a Markdown (.md) file with UTF-8 encoding."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    except Exception as exc:
        logging.getLogger(__name__).error("save_markdown failed for %s: %s", path, exc)


def save_figure(fig, path: str | os.PathLike) -> None:
    """Save a matplotlib Figure as a PNG file."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    except Exception as exc:
        logging.getLogger(__name__).error("save_figure failed for %s: %s", path, exc)


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, random, and (optionally) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
