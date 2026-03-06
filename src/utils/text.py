"""Romanian text utilities: word-counting, diacritic normalisation, lexicons."""
from __future__ import annotations

import re
import unicodedata

import numpy as np

# ---------------------------------------------------------------------------
# Diacritic normalisation
# ---------------------------------------------------------------------------

# Old-style cedilla variants → new-style comma-below variants
_DIACRITIC_MAP = str.maketrans(
    {
        "\u015e": "\u0218",  # Ş → Ș
        "\u015f": "\u0219",  # ş → ș
        "\u0162": "\u021a",  # Ţ → Ț
        "\u0163": "\u021b",  # ţ → ț
    }
)


def normalize_diacritics(text: str) -> str:
    """Normalise Romanian diacritics to the new-style comma-below variants.

    Converts old-style cedilla forms (ş, ţ, Ş, Ţ) to the correct comma-below
    forms (ș, ț, Ș, Ț) used by modern Romanian orthography.
    """
    if not isinstance(text, str):
        return ""
    return text.translate(_DIACRITIC_MAP)


# ---------------------------------------------------------------------------
# Word counting
# ---------------------------------------------------------------------------


def word_count(text) -> int:
    """Return the number of whitespace-delimited tokens in *text*.

    Returns 0 for None, NaN, or empty strings.
    """
    if text is None:
        return 0
    try:
        if np.isnan(text):  # handles float NaN passed directly
            return 0
    except (TypeError, ValueError):
        pass
    text = str(text).strip()
    if not text:
        return 0
    return len(text.split())


# ---------------------------------------------------------------------------
# Romanian lexicons
# ---------------------------------------------------------------------------

HEDGE_MARKERS: list[str] = [
    "se pare",
    "probabil",
    "posibil",
    "pare că",
    "ar putea",
    "ar fi",
    "poate că",
    "eventual",
    "presupun",
    "presupunem",
    "nu este clar",
    "nu este sigur",
    "incert",
    "incertitudine",
    "s-ar putea",
    "pare-se",
    "după unii",
    "unii susțin",
    "unii afirmă",
    "există dubii",
    "în opinia unor",
    "nu am certitudine",
    "nu se știe",
    "rămâne de văzut",
    "aproximativ",
    "cam",
    "undeva",
]

CERTAINTY_MARKERS: list[str] = [
    "cert",
    "sigur",
    "clar",
    "evident",
    "fără îndoială",
    "în mod cert",
    "cu certitudine",
    "indiscutabil",
    "incontestabil",
    "demonstrat",
    "dovedit",
    "confirmat",
    "absolut",
    "neîndoielnic",
    "fără dubiu",
    "fără echivoc",
    "cu siguranță",
    "categoric",
    "în mod sigur",
    "este clar că",
    "se confirmă că",
    "este demonstrat",
]

AUTHORITY_MARKERS: list[str] = [
    "potrivit",
    "conform",
    "după",
    "declarat de",
    "declară",
    "afirmă",
    "susține",
    "arată",
    "precizează",
    "menționează",
    "anunță",
    "informează",
    "transmite",
    "notifică",
    "expert",
    "specialist",
    "oficial",
    "autoritate",
    "instituție",
    "declarație",
    "raport",
    "studiu",
    "cercetare",
    "anchetă",
    "investigație",
    "date oficiale",
    "statistici oficiale",
]

MODAL_MARKERS: list[str] = [
    "trebuie",
    "ar trebui",
    "poate",
    "ar putea",
    "se poate",
    "se cuvine",
    "este necesar",
    "este obligatoriu",
    "se impune",
    "se recomandă",
    "este posibil",
    "este permis",
    "este interzis",
    "nu se poate",
    "nu trebuie",
    "ar fi bine",
    "ar fi necesar",
    "ar fi posibil",
]


# ---------------------------------------------------------------------------
# Marker rate computation
# ---------------------------------------------------------------------------


def compute_marker_rates(text, markers: list[str]) -> float:
    """Return occurrences of *markers* per 100 words in *text*.

    Matching is case-insensitive and diacritics are normalised before matching.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    norm = normalize_diacritics(text).lower()
    wc = word_count(text)
    if wc == 0:
        return 0.0
    count = sum(
        len(re.findall(re.escape(normalize_diacritics(m).lower()), norm))
        for m in markers
    )
    return (count / wc) * 100.0


def compute_confidence_index(text) -> float:
    """Return a confidence index: (certainty_rate – hedge_rate) normalised to [−1, 1].

    A positive value indicates more certain language; negative indicates more hedged.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    certainty = compute_marker_rates(text, CERTAINTY_MARKERS)
    hedge = compute_marker_rates(text, HEDGE_MARKERS)
    raw = certainty - hedge
    # Normalise: clamp to ±10 per-100-word range then scale to [−1, 1]
    return float(np.clip(raw / 10.0, -1.0, 1.0))
