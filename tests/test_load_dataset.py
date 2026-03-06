"""Unit tests for src/data/load_dataset.py."""

import io
import textwrap

import pandas as pd
import pytest

from src.data.load_dataset import (
    REQUIRED_COLUMNS,
    drop_missing_claims,
    filter_by_date_range,
    filter_by_domain,
    load_infact_dataset,
    validate_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TSV_HEADER = "\t".join(REQUIRED_COLUMNS)

SAMPLE_ROW = "\t".join(
    [
        "1",                          # record_id
        "http://example.com",         # source_url
        "2023-01-15",                 # date_verified
        "Ion Popescu",                # author_claim
        "Digi24",                     # source_outlet
        "Salariul minim va creste.",  # claim_text
        "Context text here.",         # context
        "national",                   # verification_scope
        "Verificat",                  # verification
        "Partial adevarat.",          # conclusion
        "economics",                  # domain_claim
        "partial adevarat",           # verdict_original
    ]
)

SAMPLE_TSV = f"{TSV_HEADER}\n{SAMPLE_ROW}\n"


def _make_df(**kwargs) -> pd.DataFrame:
    """Return a one-row DataFrame with all required columns, optionally overriding values."""
    defaults = dict(zip(REQUIRED_COLUMNS, SAMPLE_ROW.split("\t")))
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# load_infact_dataset
# ---------------------------------------------------------------------------


class TestLoadInfactDataset:
    def test_loads_valid_tsv(self, tmp_path):
        path = tmp_path / "infact.tsv"
        path.write_text(SAMPLE_TSV, encoding="utf-8")
        df = load_infact_dataset(str(path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_all_required_columns_present(self, tmp_path):
        path = tmp_path / "infact.tsv"
        path.write_text(SAMPLE_TSV, encoding="utf-8")
        df = load_infact_dataset(str(path))
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Column '{col}' missing after load"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_infact_dataset(str(tmp_path / "nonexistent.tsv"))

    def test_raises_on_empty_file(self, tmp_path):
        path = tmp_path / "empty.tsv"
        # A file with only a header produces an empty DataFrame
        path.write_text(TSV_HEADER + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_infact_dataset(str(path))

    def test_loads_multiple_rows(self, tmp_path):
        rows = "\n".join([SAMPLE_ROW, SAMPLE_ROW, SAMPLE_ROW])
        path = tmp_path / "multi.tsv"
        path.write_text(f"{TSV_HEADER}\n{rows}\n", encoding="utf-8")
        df = load_infact_dataset(str(path))
        assert len(df) == 3

    def test_custom_separator(self, tmp_path):
        csv_header = ",".join(REQUIRED_COLUMNS)
        csv_row = ",".join(
            [
                "2", "http://b.com", "2023-02-01", "Maria Ion", "ProTV",
                "Alt claim", "Context", "local", "Verificat", "Fals.",
                "politics", "fals",
            ]
        )
        path = tmp_path / "infact.csv"
        path.write_text(f"{csv_header}\n{csv_row}\n", encoding="utf-8")
        df = load_infact_dataset(str(path), sep=",")
        assert len(df) == 1
        assert df.iloc[0]["verdict_original"] == "fals"

    def test_values_loaded_as_strings(self, tmp_path):
        path = tmp_path / "infact.tsv"
        path.write_text(SAMPLE_TSV, encoding="utf-8")
        df = load_infact_dataset(str(path))
        # dtype may be 'object' or pandas StringDtype depending on version
        assert pd.api.types.is_string_dtype(df["record_id"])


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------


class TestValidateDataset:
    def test_passes_with_all_columns(self):
        df = _make_df()
        # Should not raise
        validate_dataset(df)

    def test_raises_on_missing_column(self):
        df = _make_df()
        df = df.drop(columns=["verdict_original"])
        with pytest.raises(ValueError, match="verdict_original"):
            validate_dataset(df)

    def test_raises_listing_all_missing_columns(self):
        df = pd.DataFrame({"some_col": [1, 2]})
        with pytest.raises(ValueError) as exc_info:
            validate_dataset(df)
        msg = str(exc_info.value)
        # All required columns should be mentioned
        for col in REQUIRED_COLUMNS:
            assert col in msg

    def test_extra_columns_are_allowed(self):
        df = _make_df()
        df["extra_column"] = "extra"
        validate_dataset(df)  # should not raise

    def test_empty_dataframe_with_correct_columns(self):
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        validate_dataset(df)  # should not raise


# ---------------------------------------------------------------------------
# filter_by_domain
# ---------------------------------------------------------------------------


class TestFilterByDomain:
    def _make_multi_domain_df(self) -> pd.DataFrame:
        rows = [
            _make_df(domain_claim="Politics"),
            _make_df(domain_claim="economics"),
            _make_df(domain_claim="POLITICS"),
            _make_df(domain_claim="health"),
        ]
        return pd.concat(rows, ignore_index=True)

    def test_filters_by_domain(self):
        df = self._make_multi_domain_df()
        result = filter_by_domain(df, "politics")
        assert len(result) == 2

    def test_filter_is_case_insensitive(self):
        df = self._make_multi_domain_df()
        assert len(filter_by_domain(df, "POLITICS")) == 2
        assert len(filter_by_domain(df, "Politics")) == 2
        assert len(filter_by_domain(df, "politics")) == 2

    def test_returns_empty_for_unknown_domain(self):
        df = self._make_multi_domain_df()
        result = filter_by_domain(df, "nonexistent")
        assert len(result) == 0

    def test_returns_dataframe(self):
        df = self._make_multi_domain_df()
        result = filter_by_domain(df, "economics")
        assert isinstance(result, pd.DataFrame)

    def test_index_is_reset(self):
        df = self._make_multi_domain_df()
        result = filter_by_domain(df, "politics")
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# filter_by_date_range
# ---------------------------------------------------------------------------


class TestFilterByDateRange:
    def _make_dated_df(self) -> pd.DataFrame:
        rows = [
            _make_df(date_verified="2022-06-01"),
            _make_df(date_verified="2023-01-15"),
            _make_df(date_verified="2023-06-30"),
            _make_df(date_verified="2024-03-10"),
        ]
        return pd.concat(rows, ignore_index=True)

    def test_keeps_rows_within_range(self):
        df = self._make_dated_df()
        result = filter_by_date_range(df, "2023-01-01", "2023-12-31")
        assert len(result) == 2

    def test_inclusive_bounds(self):
        df = self._make_dated_df()
        result = filter_by_date_range(df, "2023-01-15", "2023-06-30")
        assert len(result) == 2

    def test_returns_empty_when_no_match(self):
        df = self._make_dated_df()
        result = filter_by_date_range(df, "2025-01-01", "2025-12-31")
        assert len(result) == 0

    def test_index_is_reset(self):
        df = self._make_dated_df()
        result = filter_by_date_range(df, "2023-01-01", "2023-12-31")
        assert list(result.index) == list(range(len(result)))

    def test_invalid_dates_coerced_to_nat(self):
        rows = [
            _make_df(date_verified="not-a-date"),
            _make_df(date_verified="2023-05-01"),
        ]
        df = pd.concat(rows, ignore_index=True)
        result = filter_by_date_range(df, "2023-01-01", "2023-12-31")
        # Only the valid date row should remain
        assert len(result) == 1


# ---------------------------------------------------------------------------
# drop_missing_claims
# ---------------------------------------------------------------------------


class TestDropMissingClaims:
    def test_drops_null_claim_text(self):
        df = pd.concat(
            [_make_df(claim_text=None), _make_df(claim_text="Valid claim")],
            ignore_index=True,
        )
        result = drop_missing_claims(df)
        assert len(result) == 1
        assert result.iloc[0]["claim_text"] == "Valid claim"

    def test_drops_whitespace_only_claim_text(self):
        df = pd.concat(
            [_make_df(claim_text="   "), _make_df(claim_text="Real claim")],
            ignore_index=True,
        )
        result = drop_missing_claims(df)
        assert len(result) == 1

    def test_keeps_valid_claims(self):
        df = _make_df(claim_text="A valid fact-check claim.")
        result = drop_missing_claims(df)
        assert len(result) == 1

    def test_returns_empty_if_all_missing(self):
        df = pd.concat(
            [_make_df(claim_text=None), _make_df(claim_text="")],
            ignore_index=True,
        )
        result = drop_missing_claims(df)
        assert len(result) == 0

    def test_index_is_reset(self):
        df = pd.concat(
            [_make_df(claim_text=None), _make_df(claim_text="Claim A"),
             _make_df(claim_text="Claim B")],
            ignore_index=True,
        )
        result = drop_missing_claims(df)
        assert list(result.index) == [0, 1]
