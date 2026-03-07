"""
label_mapping.py
----------------
Label engineering for epistemic outcomes in the INFACT dataset.

Raw verdict labels (``verdict_original``) are mapped to a normalised
coarse-grained scheme that groups semantically similar verdicts together and
assigns numeric codes suitable for machine-learning pipelines.

Epistemic outcome groups
~~~~~~~~~~~~~~~~~~~~~~~~
* **True**        — claim is supported by evidence
* **Mostly True** — claim is substantially supported with minor caveats
* **Mixed**       — claim contains both accurate and inaccurate elements
* **Mostly False**— claim is substantially inaccurate with minor truth
* **False**       — claim is not supported by evidence
* **Unverifiable**— claim cannot be verified due to lack of evidence
* **Other**       — does not fit any of the above categories

Example usage
-------------
    from src.data.label_mapping import apply_label_mapping, LABEL_TO_ID

    df = load_infact("data/infact_dataset.tsv")
    df = apply_label_mapping(df)
    print(df[["verdict_original", "verdict_normalized", "label_id"]].head())
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw-to-normalised mapping
# Extend this dictionary as new verdict strings are encountered in the corpus.
# ---------------------------------------------------------------------------
VERDICT_NORMALIZATION: dict[str, str] = {
    # True variants
    "adevarat": "True",
    "adevărat": "True",
    "corect": "True",
    "true": "True",
    "real": "True",
    # Mostly True variants
    "in mare parte adevarat": "Mostly True",
    "în mare parte adevărat": "Mostly True",
    "mostly true": "Mostly True",
    "partial adevarat": "Mostly True",
    "parțial adevărat": "Mostly True",
    "partial adevărat": "Mostly True",  # diacritic-stripped variant of "parțial"
    # Mixed variants — includes "trunchiat" (truncated/cherry-picked claims)
    "mixt": "Mixed",
    "mixed": "Mixed",
    "partial": "Mixed",
    "parțial": "Mixed",
    "half true": "Mixed",
    "half-true": "Mixed",
    "trunchiat": "Mixed",
    # Mostly False variants
    "in mare parte fals": "Mostly False",
    "în mare parte fals": "Mostly False",
    "mostly false": "Mostly False",
    "partial fals": "Mostly False",
    "parțial fals": "Mostly False",
    # False variants
    "fals": "False",
    "false": "False",
    "incorect": "False",
    "mincinos": "False",
    # Unverifiable variants — includes claims that cannot be publicly verified
    # "numai cu sprijin instituțional" (only with institutional support) is mapped
    # here because public verification is not feasible; the claim requires
    # privileged access to official data.
    "neverificabil": "Unverifiable",
    "unverifiable": "Unverifiable",
    "nu se poate verifica": "Unverifiable",
    "insufficient evidence": "Unverifiable",
    "dovezi insuficiente": "Unverifiable",
    "imposibil de verificat": "Unverifiable",
    "numai cu sprijin instituțional": "Unverifiable",
}

# Ordered list of normalised labels (defines the canonical label space)
LABEL_ORDER: list[str] = [
    "True",
    "Mostly True",
    "Mixed",
    "Mostly False",
    "False",
    "Unverifiable",
    "Other",
]

LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL: dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}

# Binary mapping: True-leaning vs False-leaning (collapses to 2 classes)
BINARY_MAP: dict[str, int] = {
    "True": 1,
    "Mostly True": 1,
    "Mixed": 0,
    "Mostly False": 0,
    "False": 0,
    "Unverifiable": 0,
    "Other": 0,
}


def normalize_verdict(raw: Optional[str]) -> str:
    """Normalise a single raw verdict string to its canonical label.

    Parameters
    ----------
    raw:
        The raw verdict string from the dataset.  Case and leading/trailing
        whitespace are ignored.

    Returns
    -------
    str
        One of the values in :data:`LABEL_ORDER`.
    """
    if pd.isna(raw) or raw is None:
        return "Other"
    key = str(raw).strip().lower()
    return VERDICT_NORMALIZATION.get(key, "Other")


def apply_label_mapping(
    df: pd.DataFrame,
    source_col: str = "verdict_original",
    target_col: str = "verdict_normalized",
    id_col: str = "label_id",
    binary_col: str = "label_binary",
) -> pd.DataFrame:
    """Add normalised label and numeric ID columns to *df*.

    Parameters
    ----------
    df:
        DataFrame that must contain *source_col*.
    source_col:
        Name of the column with raw verdict strings.
    target_col:
        Name of the new column for normalised verdict labels.
    id_col:
        Name of the new column for integer label IDs.
    binary_col:
        Name of the new column for binary labels (1 = True-leaning, 0 = False-leaning).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with three new columns appended.
    """
    df = df.copy()
    df[target_col] = df[source_col].apply(normalize_verdict)
    df[id_col] = df[target_col].map(LABEL_TO_ID)
    df[binary_col] = df[target_col].map(BINARY_MAP)

    mapping_counts = df[target_col].value_counts()
    logger.info("Label distribution after mapping:\n%s", mapping_counts.to_string())

    unmapped = (df[target_col] == "Other").sum()
    if unmapped > 0:
        logger.warning(
            "%d records mapped to 'Other'. Consider extending VERDICT_NORMALIZATION.", unmapped
        )

    return df


def get_label_statistics(df: pd.DataFrame, label_col: str = "verdict_normalized") -> dict:
    """Compute label distribution statistics.

    Parameters
    ----------
    df:
        DataFrame with a normalised label column.
    label_col:
        Name of the normalised label column.

    Returns
    -------
    dict
        Dictionary with ``counts``, ``proportions``, and ``n_classes``.
    """
    counts = df[label_col].value_counts()
    proportions = df[label_col].value_counts(normalize=True).round(4)
    return {
        "counts": counts.to_dict(),
        "proportions": proportions.to_dict(),
        "n_classes": int(len(counts)),
    }


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)

    from src.data.load_dataset import load_infact, validate_dataset

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/infact_dataset.tsv"
    dataset = load_infact(data_path)
    validate_dataset(dataset)
    dataset = apply_label_mapping(dataset)
    stats = get_label_statistics(dataset)
    print(json.dumps(stats, indent=2))
