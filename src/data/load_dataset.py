"""
load_dataset.py — Data loading and validation for the INFACT corpus.

The INFACT corpus is a Romanian institutional fact-checking dataset stored as
a tab-separated values (TSV) file with the following columns:

    record_id, source_url, date_verified, author_claim, source_outlet,
    claim_text, context, verification_scope, verification, conclusion,
    domain_claim, verdict_original

Example usage::

    from src.data.load_dataset import load_infact_dataset, validate_dataset

    df = load_infact_dataset("data/infact.tsv")
    validate_dataset(df)
    print(df.head())
"""

import pandas as pd

REQUIRED_COLUMNS = [
    "record_id",
    "source_url",
    "date_verified",
    "author_claim",
    "source_outlet",
    "claim_text",
    "context",
    "verification_scope",
    "verification",
    "conclusion",
    "domain_claim",
    "verdict_original",
]


def load_infact_dataset(filepath: str, sep: str = "\t") -> pd.DataFrame:
    """Load the INFACT corpus from a delimited text file.

    Parameters
    ----------
    filepath:
        Path to the TSV (or CSV) file containing the INFACT corpus.
    sep:
        Column delimiter used in the file.  Defaults to ``'\\t'`` for TSV.

    Returns
    -------
    pd.DataFrame
        Raw corpus as a DataFrame, one row per fact-checked claim.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist on disk.
    ValueError
        If the file is empty or cannot be parsed as a delimited text file.
    """
    df = pd.read_csv(filepath, sep=sep, dtype=str)
    if df.empty:
        raise ValueError(f"Dataset loaded from '{filepath}' is empty.")
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Check that *df* contains all columns required by the INFACT schema.

    Parameters
    ----------
    df:
        DataFrame to validate.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}"
        )


def filter_by_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Return only rows whose *domain_claim* matches *domain* (case-insensitive).

    Parameters
    ----------
    df:
        INFACT DataFrame (must contain a ``domain_claim`` column).
    domain:
        Domain string to filter on (e.g. ``'politics'``).

    Returns
    -------
    pd.DataFrame
        Filtered view of *df*.
    """
    return df[df["domain_claim"].str.lower() == domain.lower()].reset_index(
        drop=True
    )


def filter_by_date_range(
    df: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    """Return rows whose *date_verified* falls within [*start*, *end*].

    Parameters
    ----------
    df:
        INFACT DataFrame (must contain a ``date_verified`` column).
    start:
        Inclusive lower bound in ``YYYY-MM-DD`` format.
    end:
        Inclusive upper bound in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Filtered view of *df*.
    """
    dates = pd.to_datetime(df["date_verified"], errors="coerce")
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return df[mask].reset_index(drop=True)


def drop_missing_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where *claim_text* is null or blank.

    Parameters
    ----------
    df:
        INFACT DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with blank-claim rows removed.
    """
    mask = df["claim_text"].notna() & (df["claim_text"].str.strip() != "")
    return df[mask].reset_index(drop=True)
