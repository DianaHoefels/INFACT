"""Unit tests for src/data/label_mapping.py."""

import pandas as pd
import pytest

from src.data.label_mapping import (
    BINARY_VERDICT_MAP,
    MULTICLASS_LABEL_ENCODING,
    MULTICLASS_VERDICT_MAP,
    apply_binary_mapping,
    apply_multiclass_mapping,
    encode_labels,
    get_label_distribution,
    map_verdict_to_binary,
    map_verdict_to_multiclass,
)


# ---------------------------------------------------------------------------
# map_verdict_to_binary
# ---------------------------------------------------------------------------


class TestMapVerdictToBinary:
    def test_true_verdicts(self):
        for verdict in ("adevarat", "adevărat", "corect", "real"):
            assert map_verdict_to_binary(verdict) is True, verdict

    def test_false_verdicts(self):
        for verdict in ("fals", "neadevarat", "neadevărat", "incorect"):
            assert map_verdict_to_binary(verdict) is False, verdict

    def test_none_verdicts(self):
        for verdict in (
            "partial adevarat", "parțial adevărat", "partial",
            "neverificabil", "imposibil de verificat", "exagerat",
            "inselator", "înșelător",
        ):
            assert map_verdict_to_binary(verdict) is None, verdict

    def test_case_insensitive(self):
        assert map_verdict_to_binary("Adevarat") is True
        assert map_verdict_to_binary("FALS") is False
        assert map_verdict_to_binary("Partial Adevarat") is None

    def test_strips_whitespace(self):
        assert map_verdict_to_binary("  adevarat  ") is True
        assert map_verdict_to_binary("  fals  ") is False

    def test_unknown_verdict_returns_none(self):
        assert map_verdict_to_binary("unknown_verdict") is None

    def test_non_string_returns_none(self):
        assert map_verdict_to_binary(None) is None
        assert map_verdict_to_binary(42) is None
        assert map_verdict_to_binary(True) is None

    def test_empty_string_returns_none(self):
        assert map_verdict_to_binary("") is None


# ---------------------------------------------------------------------------
# map_verdict_to_multiclass
# ---------------------------------------------------------------------------


class TestMapVerdictToMulticlass:
    def test_true_class(self):
        for verdict in ("adevarat", "adevărat", "corect", "real"):
            assert map_verdict_to_multiclass(verdict) == "true", verdict

    def test_false_class(self):
        for verdict in ("fals", "neadevarat", "neadevărat", "incorect"):
            assert map_verdict_to_multiclass(verdict) == "false", verdict

    def test_partial_class(self):
        for verdict in ("partial adevarat", "parțial adevărat", "partial", "exagerat"):
            assert map_verdict_to_multiclass(verdict) == "partial", verdict

    def test_other_class(self):
        for verdict in ("neverificabil", "imposibil de verificat", "inselator", "înșelător"):
            assert map_verdict_to_multiclass(verdict) == "other", verdict

    def test_case_insensitive(self):
        assert map_verdict_to_multiclass("FALS") == "false"
        assert map_verdict_to_multiclass("Adevarat") == "true"

    def test_strips_whitespace(self):
        assert map_verdict_to_multiclass("  fals  ") == "false"

    def test_unknown_returns_none(self):
        assert map_verdict_to_multiclass("completely_unknown") is None

    def test_non_string_returns_none(self):
        assert map_verdict_to_multiclass(None) is None
        assert map_verdict_to_multiclass(0) is None


# ---------------------------------------------------------------------------
# encode_labels
# ---------------------------------------------------------------------------


class TestEncodeLabels:
    def _series(self, values):
        return pd.Series(values)

    def test_encodes_all_four_classes(self):
        s = self._series(["true", "partial", "false", "other"])
        result = encode_labels(s)
        assert list(result) == [0, 1, 2, 3]

    def test_unknown_label_maps_to_nan(self):
        s = self._series(["true", "unknown"])
        result = encode_labels(s)
        assert result.iloc[0] == 0
        assert pd.isna(result.iloc[1])

    def test_custom_encoding(self):
        custom = {"yes": 1, "no": 0}
        s = self._series(["yes", "no", "yes"])
        result = encode_labels(s, encoding=custom)
        assert list(result) == [1, 0, 1]

    def test_returns_series(self):
        s = self._series(["true", "false"])
        result = encode_labels(s)
        assert isinstance(result, pd.Series)

    def test_empty_series(self):
        s = self._series([])
        result = encode_labels(s)
        assert len(result) == 0

    def test_default_encoding_matches_constant(self):
        for label, code in MULTICLASS_LABEL_ENCODING.items():
            s = self._series([label])
            assert encode_labels(s).iloc[0] == code


# ---------------------------------------------------------------------------
# get_label_distribution
# ---------------------------------------------------------------------------


class TestGetLabelDistribution:
    def _make_df(self, labels):
        return pd.DataFrame({"label": labels})

    def test_counts_labels(self):
        df = self._make_df(["true", "true", "false", "other"])
        dist = get_label_distribution(df, "label")
        assert dist["true"] == 2
        assert dist["false"] == 1
        assert dist["other"] == 1

    def test_returns_series(self):
        df = self._make_df(["true", "false"])
        assert isinstance(get_label_distribution(df, "label"), pd.Series)

    def test_sorted_descending(self):
        df = self._make_df(["true"] * 5 + ["false"] * 3 + ["other"])
        dist = get_label_distribution(df, "label")
        assert dist.iloc[0] >= dist.iloc[1] >= dist.iloc[2]

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        with pytest.raises(KeyError, match="nonexistent"):
            get_label_distribution(df, "nonexistent")

    def test_empty_dataframe(self):
        df = pd.DataFrame({"label": pd.Series([], dtype=str)})
        dist = get_label_distribution(df, "label")
        assert len(dist) == 0


# ---------------------------------------------------------------------------
# apply_binary_mapping
# ---------------------------------------------------------------------------


class TestApplyBinaryMapping:
    def _make_df(self, verdicts):
        return pd.DataFrame({"verdict_original": verdicts})

    def test_adds_label_binary_column(self):
        df = self._make_df(["adevarat", "fals"])
        result = apply_binary_mapping(df)
        assert "label_binary" in result.columns

    def test_correct_binary_values(self):
        df = self._make_df(["adevarat", "fals", "partial"])
        result = apply_binary_mapping(df)
        assert result["label_binary"].iloc[0] is True
        assert result["label_binary"].iloc[1] is False
        assert result["label_binary"].iloc[2] is None

    def test_does_not_mutate_input(self):
        df = self._make_df(["adevarat"])
        _ = apply_binary_mapping(df)
        assert "label_binary" not in df.columns

    def test_custom_source_and_target_columns(self):
        df = pd.DataFrame({"my_verdict": ["fals", "adevarat"]})
        result = apply_binary_mapping(df, source_col="my_verdict", target_col="my_label")
        assert "my_label" in result.columns
        assert result["my_label"].iloc[0] == False  # noqa: E712

    def test_unknown_verdict_maps_to_none(self):
        df = self._make_df(["unknown_verdict"])
        result = apply_binary_mapping(df)
        assert result["label_binary"].iloc[0] is None


# ---------------------------------------------------------------------------
# apply_multiclass_mapping
# ---------------------------------------------------------------------------


class TestApplyMulticlassMapping:
    def _make_df(self, verdicts):
        return pd.DataFrame({"verdict_original": verdicts})

    def test_adds_label_multiclass_column(self):
        df = self._make_df(["adevarat", "fals"])
        result = apply_multiclass_mapping(df)
        assert "label_multiclass" in result.columns

    def test_correct_multiclass_values(self):
        df = self._make_df(["adevarat", "fals", "partial", "neverificabil"])
        result = apply_multiclass_mapping(df)
        assert result["label_multiclass"].iloc[0] == "true"
        assert result["label_multiclass"].iloc[1] == "false"
        assert result["label_multiclass"].iloc[2] == "partial"
        assert result["label_multiclass"].iloc[3] == "other"

    def test_does_not_mutate_input(self):
        df = self._make_df(["fals"])
        _ = apply_multiclass_mapping(df)
        assert "label_multiclass" not in df.columns

    def test_custom_columns(self):
        df = pd.DataFrame({"raw": ["fals"]})
        result = apply_multiclass_mapping(df, source_col="raw", target_col="cls")
        assert "cls" in result.columns
        assert result["cls"].iloc[0] == "false"

    def test_unknown_verdict_maps_to_none(self):
        df = self._make_df(["completely_unknown"])
        result = apply_multiclass_mapping(df)
        assert result["label_multiclass"].iloc[0] is None


# ---------------------------------------------------------------------------
# Vocabulary consistency checks
# ---------------------------------------------------------------------------


class TestVocabularyConsistency:
    def test_binary_map_has_only_bool_or_none_values(self):
        for key, val in BINARY_VERDICT_MAP.items():
            assert val is None or isinstance(val, bool), (
                f"BINARY_VERDICT_MAP['{key}'] has unexpected type {type(val)}"
            )

    def test_multiclass_map_values_are_valid_classes(self):
        valid = {"true", "partial", "false", "other"}
        for key, val in MULTICLASS_VERDICT_MAP.items():
            assert val in valid, (
                f"MULTICLASS_VERDICT_MAP['{key}'] = '{val}' is not a valid class"
            )

    def test_multiclass_encoding_covers_all_classes(self):
        classes_in_map = set(MULTICLASS_VERDICT_MAP.values())
        for cls in classes_in_map:
            assert cls in MULTICLASS_LABEL_ENCODING, (
                f"Class '{cls}' is in MULTICLASS_VERDICT_MAP but not in MULTICLASS_LABEL_ENCODING"
            )

    def test_encoding_values_are_unique(self):
        codes = list(MULTICLASS_LABEL_ENCODING.values())
        assert len(codes) == len(set(codes)), "Duplicate encoding values found"
