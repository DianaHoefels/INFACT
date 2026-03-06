"""
text_processing.py
------------------
Text pre-processing utilities for the INFACT pipeline.

Provides:
- Lowercasing and Unicode normalisation
- Punctuation removal
- Romanian and English stop-word filtering
- Simple tokenisation
- Batch processing for DataFrames

Example usage
-------------
    from src.utils.text_processing import preprocess_text, preprocess_series

    clean = preprocess_text("Poate că afirmația este adevărată.", remove_stopwords=True)
    print(clean)
    # "poate afirmatie adevarata"

    import pandas as pd
    series = pd.Series(["Afirmație falsă.", "Probabil adevărat."])
    cleaned = preprocess_series(series)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-word lists (small built-in defaults; replace with full lists as needed)
# ---------------------------------------------------------------------------

ROMANIAN_STOPWORDS: frozenset[str] = frozenset(
    [
        "și", "sau", "dar", "că", "care", "cu", "de", "din", "în", "la",
        "pe", "pentru", "prin", "după", "spre", "până", "între", "despre",
        "ca", "să", "nu", "este", "sunt", "era", "au", "am", "ai", "fi",
        "fost", "va", "vor", "ar", "cel", "cei", "cea", "cele", "un", "o",
        "al", "ai", "ale", "lui", "ei", "lor", "eu", "tu", "el", "ea",
        "noi", "voi", "ei", "ele", "se", "îl", "îi", "le", "îmi",
        "îți", "i", "a", "e", "ne", "v",
    ]
)

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    [
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "not", "no", "it", "its", "this", "that", "these", "those",
        "he", "she", "they", "we", "you", "i", "me", "my", "his", "her",
    ]
)

ALL_STOPWORDS: frozenset[str] = ROMANIAN_STOPWORDS | ENGLISH_STOPWORDS


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def normalise_unicode(text: str) -> str:
    """Apply Unicode NFC normalisation to *text*.

    Parameters
    ----------
    text:
        Input string.

    Returns
    -------
    str
        NFC-normalised string.
    """
    return unicodedata.normalize("NFC", text)


def remove_diacritics(text: str) -> str:
    """Strip diacritic characters from *text* (e.g. ă → a, î → i).

    Parameters
    ----------
    text:
        Input string.

    Returns
    -------
    str
        ASCII-folded string.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """Remove URLs, punctuation, and excess whitespace from *text*.

    Parameters
    ----------
    text:
        Input string.

    Returns
    -------
    str
        Cleaned string (lowercase).
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenise(text: str) -> list[str]:
    """Split *text* into tokens on whitespace boundaries.

    Parameters
    ----------
    text:
        Input string (should already be cleaned/lowercased).

    Returns
    -------
    list[str]
        List of token strings.
    """
    return text.split()


def remove_stopwords(tokens: list[str], stopwords: Optional[frozenset[str]] = None) -> list[str]:
    """Filter stopwords from a token list.

    Parameters
    ----------
    tokens:
        List of tokens.
    stopwords:
        Set of stopword strings.  Defaults to :data:`ALL_STOPWORDS`.

    Returns
    -------
    list[str]
        Filtered token list.
    """
    if stopwords is None:
        stopwords = ALL_STOPWORDS
    return [t for t in tokens if t not in stopwords]


def preprocess_text(
    text: str,
    lowercase: bool = True,
    strip_diacritics: bool = False,
    remove_stopwords_flag: bool = False,
    stopwords: Optional[frozenset[str]] = None,
) -> str:
    """Full pre-processing pipeline for a single text string.

    Parameters
    ----------
    text:
        Raw input text.
    lowercase:
        Whether to lowercase the output (default: ``True``).
    strip_diacritics:
        Whether to remove diacritics (default: ``False`` — preserves Romanian
        characters by default).
    remove_stopwords_flag:
        Whether to remove stop words.
    stopwords:
        Custom stop-word set.  Defaults to :data:`ALL_STOPWORDS`.

    Returns
    -------
    str
        Pre-processed text as a single space-joined string.
    """
    text = normalise_unicode(text)
    if strip_diacritics:
        text = remove_diacritics(text)
    text = clean_text(text)  # lowercases as a side effect
    if not lowercase:
        pass  # clean_text already lowercases; re-casing not implemented
    tokens = tokenise(text)
    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens, stopwords)
    return " ".join(tokens)


def preprocess_series(
    series: pd.Series,
    **kwargs,
) -> pd.Series:
    """Apply :func:`preprocess_text` to every element of a Series.

    Parameters
    ----------
    series:
        Series of raw text strings.
    **kwargs:
        Keyword arguments forwarded to :func:`preprocess_text`.

    Returns
    -------
    pd.Series
        Series of pre-processed strings.
    """
    return series.fillna("").apply(lambda t: preprocess_text(t, **kwargs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    samples = [
        "Poate că afirmația este adevărată.",
        "Conform declarațiilor oficiale, rata inflației a crescut cu 5%.",
        "This is absolutely FALSE and misleading!",
    ]

    for s in samples:
        print(f"Original : {s}")
        print(f"Processed: {preprocess_text(s, remove_stopwords_flag=True)}")
        print()
