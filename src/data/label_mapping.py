"""
label_mapping.py — Label engineering for epistemic outcomes in INFACT.

The INFACT corpus uses a Romanian verdict vocabulary.  This module provides
helpers to normalise those verdicts into binary (true / false) and coarse
multi-class (true / partial / false / other) representations suitable for
downstream machine-learning experiments.

Example usage::

    from src.data.label_mapping import map_verdict_to_binary, encode_labels
    import pandas as pd

    df = pd.DataFrame({"verdict_original": ["adevarat", "fals", "partial adevarat"]})
    df["label_binary"] = df["verdict_original"].map(map_verdict_to_binary)
    df["label_encoded"] = encode_labels(df["label_binary"])
    print(df)
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Vocabulary mappings
# ---------------------------------------------------------------------------

#: Mapping from normalised Romanian verdict string to a binary True/False
#: label.  Verdicts not present in this dictionary are mapped to ``None``.
BINARY_VERDICT_MAP: Dict[str, Optional[bool]] = {
    "adevarat": True,
    "adevărat": True,
    "corect": True,
    "real": True,
    "fals": False,
    "neadevarat": False,
    "neadevărat": False,
    "incorect": False,
    "partial adevarat": None,
    "parțial adevărat": None,
    "partial": None,
    "neverificabil": None,
    "imposibil de verificat": None,
    "exagerat": None,
    "inselator": None,
    "înșelător": None,
}

#: Mapping from normalised Romanian verdict string to a coarse four-way label.
MULTICLASS_VERDICT_MAP: Dict[str, str] = {
    "adevarat": "true",
    "adevărat": "true",
    "corect": "true",
    "real": "true",
    "fals": "false",
    "neadevarat": "false",
    "neadevărat": "false",
    "incorect": "false",
    "partial adevarat": "partial",
    "parțial adevărat": "partial",
    "partial": "partial",
    "exagerat": "partial",
    "neverificabil": "other",
    "imposibil de verificat": "other",
    "inselator": "other",
    "înșelător": "other",
}

#: Numeric encoding for the four-way label set.
MULTICLASS_LABEL_ENCODING: Dict[str, int] = {
    "true": 0,
    "partial": 1,
    "false": 2,
    "other": 3,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def map_verdict_to_binary(verdict: str) -> Optional[bool]:
    """Map a single Romanian verdict string to a binary True/False label.

    The verdict is normalised (stripped, lower-cased) before lookup.
    Verdicts that do not belong unambiguously to either the *true* or *false*
    class are returned as ``None``.

    Parameters
    ----------
    verdict:
        Raw verdict string (e.g. ``'Adevarat'``, ``'FALS'``).

    Returns
    -------
    bool or None
        ``True`` for confirmed claims, ``False`` for debunked claims,
        ``None`` for ambiguous / unverifiable verdicts.
    """
    if not isinstance(verdict, str):
        return None
    return BINARY_VERDICT_MAP.get(verdict.strip().lower())


def map_verdict_to_multiclass(verdict: str) -> Optional[str]:
    """Map a single Romanian verdict string to a coarse four-way label.

    The four classes are: ``'true'``, ``'partial'``, ``'false'``, ``'other'``.

    Parameters
    ----------
    verdict:
        Raw verdict string.

    Returns
    -------
    str or None
        One of ``'true'``, ``'partial'``, ``'false'``, ``'other'``, or
        ``None`` if the verdict is not recognised.
    """
    if not isinstance(verdict, str):
        return None
    return MULTICLASS_VERDICT_MAP.get(verdict.strip().lower())


def encode_labels(
    series: pd.Series,
    encoding: Optional[Dict[str, int]] = None,
) -> pd.Series:
    """Replace string label values with integer codes.

    Parameters
    ----------
    series:
        Series of string labels (e.g. ``'true'``, ``'false'``, …).
    encoding:
        Dictionary mapping label strings to integers.  Defaults to
        :data:`MULTICLASS_LABEL_ENCODING`.

    Returns
    -------
    pd.Series
        Integer-encoded series (``NaN`` for labels absent from *encoding*).
    """
    if encoding is None:
        encoding = MULTICLASS_LABEL_ENCODING
    return series.map(encoding)


def get_label_distribution(df: pd.DataFrame, column: str) -> pd.Series:
    """Return the value-count distribution of *column* in *df*.

    Parameters
    ----------
    df:
        DataFrame containing the label column.
    column:
        Name of the column to inspect.

    Returns
    -------
    pd.Series
        Counts indexed by label value, sorted descending.

    Raises
    ------
    KeyError
        If *column* is not present in *df*.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    return df[column].value_counts()


def apply_binary_mapping(df: pd.DataFrame, source_col: str = "verdict_original",
                         target_col: str = "label_binary") -> pd.DataFrame:
    """Add a binary label column to *df* derived from *source_col*.

    Parameters
    ----------
    df:
        INFACT DataFrame.
    source_col:
        Name of the column containing raw verdict strings.
    target_col:
        Name of the new column to create.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *target_col* appended.
    """
    out = df.copy()
    out[target_col] = out[source_col].apply(map_verdict_to_binary)
    return out


def apply_multiclass_mapping(df: pd.DataFrame,
                              source_col: str = "verdict_original",
                              target_col: str = "label_multiclass") -> pd.DataFrame:
    """Add a multiclass label column to *df* derived from *source_col*.

    Parameters
    ----------
    df:
        INFACT DataFrame.
    source_col:
        Name of the column containing raw verdict strings.
    target_col:
        Name of the new column to create.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *target_col* appended.
    """
    out = df.copy()
    out[target_col] = out[source_col].apply(map_verdict_to_multiclass)
    return out
