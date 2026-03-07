"""
test_label_mapping.py
----------------------
Unit tests for src/data/label_mapping.py.

Covers:
- normalize_verdict(): Romanian and English variants, case insensitivity,
  whitespace stripping, None/NaN, unmapped values
- apply_label_mapping(): new columns created, correct IDs, binary labels,
  missing source column raises KeyError
- get_label_statistics(): counts, proportions, n_classes
"""

import pytest
import pandas as pd
import numpy as np

from src.data.label_mapping import (
    BINARY_MAP,
    ID_TO_LABEL,
    LABEL_ORDER,
    LABEL_TO_ID,
    VERDICT_NORMALIZATION,
    apply_label_mapping,
    get_label_statistics,
    normalize_verdict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def verdicts_df():
    """DataFrame with one row per known raw verdict."""
    rows = [{"verdict_original": raw} for raw in VERDICT_NORMALIZATION]
    return pd.DataFrame(rows)


@pytest.fixture
def mapped_df(verdicts_df):
    """verdicts_df with label mapping already applied."""
    return apply_label_mapping(verdicts_df)


# ---------------------------------------------------------------------------
# TestLabelMapping
# ---------------------------------------------------------------------------


class TestLabelMapping:
    # --- normalize_verdict ---

    def test_normalize_true_romanian(self):
        assert normalize_verdict("adevarat") == "True"

    def test_normalize_true_romanian_diacritic(self):
        assert normalize_verdict("adevărat") == "True"

    def test_normalize_true_english(self):
        assert normalize_verdict("true") == "True"

    def test_normalize_mostly_true_romanian(self):
        assert normalize_verdict("în mare parte adevărat") == "Mostly True"

    def test_normalize_mostly_true_english(self):
        assert normalize_verdict("mostly true") == "Mostly True"

    def test_normalize_mixed_mixt(self):
        assert normalize_verdict("mixt") == "Mixed"

    def test_normalize_mixed_half_true(self):
        assert normalize_verdict("half true") == "Mixed"

    def test_normalize_mixed_trunchiat(self):
        assert normalize_verdict("trunchiat") == "Mixed"

    def test_normalize_mostly_false_romanian(self):
        assert normalize_verdict("în mare parte fals") == "Mostly False"

    def test_normalize_mostly_false_english(self):
        assert normalize_verdict("mostly false") == "Mostly False"

    def test_normalize_false_romanian(self):
        assert normalize_verdict("fals") == "False"

    def test_normalize_false_english(self):
        assert normalize_verdict("false") == "False"

    def test_normalize_unverifiable_romanian(self):
        assert normalize_verdict("neverificabil") == "Unverifiable"

    def test_normalize_unverifiable_english(self):
        assert normalize_verdict("unverifiable") == "Unverifiable"

    def test_normalize_unverifiable_numai(self):
        assert normalize_verdict("numai cu sprijin instituțional") == "Unverifiable"

    def test_normalize_case_insensitive(self):
        assert normalize_verdict("ADEVARAT") == "True"
        assert normalize_verdict("Fals") == "False"
        assert normalize_verdict("MIXED") == "Mixed"

    def test_normalize_strips_whitespace(self):
        assert normalize_verdict("  adevarat  ") == "True"
        assert normalize_verdict("\tfals\n") == "False"

    def test_normalize_none_returns_other(self):
        assert normalize_verdict(None) == "Other"

    def test_normalize_nan_returns_other(self):
        assert normalize_verdict(float("nan")) == "Other"

    def test_normalize_pandas_na_returns_other(self):
        assert normalize_verdict(pd.NA) == "Other"

    def test_normalize_empty_string_returns_other(self):
        assert normalize_verdict("") == "Other"

    def test_normalize_unknown_string_returns_other(self):
        assert normalize_verdict("completely unknown verdict") == "Other"

    def test_normalize_all_known_verdicts(self):
        """Every key in VERDICT_NORMALIZATION must map to a value in LABEL_ORDER."""
        for raw, expected in VERDICT_NORMALIZATION.items():
            assert normalize_verdict(raw) == expected
            assert expected in LABEL_ORDER

    # --- apply_label_mapping ---

    def test_apply_adds_normalized_column(self, mapped_df):
        assert "verdict_normalized" in mapped_df.columns

    def test_apply_adds_label_id_column(self, mapped_df):
        assert "label_id" in mapped_df.columns

    def test_apply_adds_binary_column(self, mapped_df):
        assert "label_binary" in mapped_df.columns

    def test_apply_does_not_mutate_input(self, verdicts_df):
        original_cols = list(verdicts_df.columns)
        apply_label_mapping(verdicts_df)
        assert list(verdicts_df.columns) == original_cols

    def test_apply_normalized_values_in_label_order(self, mapped_df):
        assert set(mapped_df["verdict_normalized"]).issubset(set(LABEL_ORDER))

    def test_apply_label_ids_are_integers(self, mapped_df):
        assert pd.api.types.is_integer_dtype(mapped_df["label_id"])

    def test_apply_binary_values_are_zero_or_one(self, mapped_df):
        assert set(mapped_df["label_binary"]).issubset({0, 1})

    def test_apply_true_verdict_id(self):
        df = pd.DataFrame({"verdict_original": ["adevarat"]})
        result = apply_label_mapping(df)
        assert result["label_id"].iloc[0] == LABEL_TO_ID["True"]

    def test_apply_false_verdict_binary(self):
        df = pd.DataFrame({"verdict_original": ["fals"]})
        result = apply_label_mapping(df)
        assert result["label_binary"].iloc[0] == 0

    def test_apply_true_verdict_binary(self):
        df = pd.DataFrame({"verdict_original": ["adevarat"]})
        result = apply_label_mapping(df)
        assert result["label_binary"].iloc[0] == 1

    def test_apply_mostly_true_binary(self):
        df = pd.DataFrame({"verdict_original": ["mostly true"]})
        result = apply_label_mapping(df)
        assert result["label_binary"].iloc[0] == 1

    def test_apply_unverifiable_binary(self):
        df = pd.DataFrame({"verdict_original": ["neverificabil"]})
        result = apply_label_mapping(df)
        assert result["label_binary"].iloc[0] == 0

    def test_apply_other_binary(self):
        df = pd.DataFrame({"verdict_original": ["unknown"]})
        result = apply_label_mapping(df)
        assert result["label_binary"].iloc[0] == 0

    def test_apply_custom_column_names(self):
        df = pd.DataFrame({"raw_verdict": ["adevarat", "fals"]})
        result = apply_label_mapping(
            df,
            source_col="raw_verdict",
            target_col="norm",
            id_col="lid",
            binary_col="bin",
        )
        assert "norm" in result.columns
        assert "lid" in result.columns
        assert "bin" in result.columns

    def test_apply_nan_verdict_maps_to_other(self):
        df = pd.DataFrame({"verdict_original": [None, float("nan")]})
        result = apply_label_mapping(df)
        assert (result["verdict_normalized"] == "Other").all()

    def test_apply_label_id_to_label_roundtrip(self, mapped_df):
        """label_id should round-trip back to verdict_normalized via ID_TO_LABEL."""
        for _, row in mapped_df.iterrows():
            assert ID_TO_LABEL[row["label_id"]] == row["verdict_normalized"]

    def test_apply_missing_source_column_raises(self):
        df = pd.DataFrame({"other_col": ["value"]})
        with pytest.raises(KeyError):
            apply_label_mapping(df)

    # --- get_label_statistics ---

    def test_statistics_returns_dict(self, mapped_df):
        stats = get_label_statistics(mapped_df)
        assert isinstance(stats, dict)

    def test_statistics_has_counts(self, mapped_df):
        stats = get_label_statistics(mapped_df)
        assert "counts" in stats
        assert isinstance(stats["counts"], dict)

    def test_statistics_has_proportions(self, mapped_df):
        stats = get_label_statistics(mapped_df)
        assert "proportions" in stats

    def test_statistics_has_n_classes(self, mapped_df):
        stats = get_label_statistics(mapped_df)
        assert "n_classes" in stats
        assert isinstance(stats["n_classes"], int)

    def test_statistics_counts_sum(self, mapped_df):
        stats = get_label_statistics(mapped_df)
        assert sum(stats["counts"].values()) == len(mapped_df)

    def test_statistics_proportions_sum_to_one(self, mapped_df):
        stats = get_label_statistics(mapped_df)
        total = sum(stats["proportions"].values())
        assert abs(total - 1.0) < 0.01

    def test_statistics_n_classes_correct(self):
        df = pd.DataFrame({"verdict_normalized": ["True", "False", "True", "Mixed"]})
        stats = get_label_statistics(df)
        assert stats["n_classes"] == 3

    def test_statistics_single_class(self):
        df = pd.DataFrame({"verdict_normalized": ["True", "True", "True"]})
        stats = get_label_statistics(df)
        assert stats["n_classes"] == 1
        assert stats["proportions"]["True"] == pytest.approx(1.0)

    def test_statistics_custom_label_col(self, mapped_df):
        stats = get_label_statistics(mapped_df, label_col="verdict_normalized")
        assert "counts" in stats

    # --- LABEL_TO_ID / ID_TO_LABEL consistency ---

    def test_label_to_id_covers_all_labels(self):
        assert set(LABEL_TO_ID.keys()) == set(LABEL_ORDER)

    def test_id_to_label_is_inverse_of_label_to_id(self):
        for label, idx in LABEL_TO_ID.items():
            assert ID_TO_LABEL[idx] == label

    def test_binary_map_covers_all_labels(self):
        assert set(BINARY_MAP.keys()) == set(LABEL_ORDER)
