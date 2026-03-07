"""
test_load_dataset.py
---------------------
Unit tests for src/data/load_dataset.py.

Covers:
- load_infact(): happy path, duplicate removal, date parsing, missing file
- validate_dataset(): passes with all columns, raises on missing columns
- get_dataset_summary(): correct counts, date range, missing values
"""

import pytest
import pandas as pd

from src.data.load_dataset import (
    REQUIRED_COLUMNS,
    get_dataset_summary,
    load_infact,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ROW = {
    "record_id": "1",
    "source_url": "http://example.com",
    "date_verified": "01.01.2023",
    "author_claim": "Author A",
    "source_outlet": "Outlet X",
    "claim_text": "The sky is blue.",
    "context": "Science",
    "verification_scope": "National",
    "verification": "Checked",
    "conclusion": "Supported",
    "domain_claim": "Science",
    "verdict_original": "adevarat",
}


@pytest.fixture
def sample_df():
    """A minimal valid INFACT DataFrame with two rows."""
    row1 = dict(SAMPLE_ROW)
    row2 = dict(SAMPLE_ROW, record_id="2", verdict_original="fals")
    return pd.DataFrame([row1, row2])


@pytest.fixture
def tsv_file(tmp_path, sample_df):
    """Write sample_df to a TSV file and return its path."""
    path = tmp_path / "infact_test.tsv"
    sample_df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def tsv_file_with_duplicates(tmp_path, sample_df):
    """TSV file that contains a duplicate row."""
    df_with_dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    path = tmp_path / "infact_dup.tsv"
    df_with_dup.to_csv(path, sep="\t", index=False)
    return path


# ---------------------------------------------------------------------------
# TestLoadDataset
# ---------------------------------------------------------------------------


class TestLoadDataset:
    # --- load_infact ---

    def test_load_infact_returns_dataframe(self, tsv_file):
        df = load_infact(tsv_file)
        assert isinstance(df, pd.DataFrame)

    def test_load_infact_row_count(self, tsv_file, sample_df):
        df = load_infact(tsv_file)
        assert len(df) == len(sample_df)

    def test_load_infact_columns_present(self, tsv_file):
        df = load_infact(tsv_file)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_load_infact_date_parsed(self, tsv_file):
        df = load_infact(tsv_file)
        assert pd.api.types.is_datetime64_any_dtype(df["date_verified"])

    def test_load_infact_date_value(self, tsv_file):
        df = load_infact(tsv_file)
        assert df["date_verified"].iloc[0] == pd.Timestamp("2023-01-01")

    def test_load_infact_drop_duplicates_true(self, tsv_file_with_duplicates):
        df = load_infact(tsv_file_with_duplicates, drop_duplicates=True)
        # Original sample had 2 unique rows; the duplicate should be removed.
        assert len(df) == 2

    def test_load_infact_drop_duplicates_false(self, tsv_file_with_duplicates):
        df = load_infact(tsv_file_with_duplicates, drop_duplicates=False)
        # All 3 rows should be preserved (2 unique + 1 duplicate).
        assert len(df) == 3

    def test_load_infact_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_infact("/nonexistent/path/file.tsv")

    def test_load_infact_invalid_date_coerced(self, tmp_path):
        """Rows with unparseable dates should become NaT."""
        row = dict(SAMPLE_ROW, date_verified="not-a-date")
        df_bad = pd.DataFrame([row])
        path = tmp_path / "bad_date.tsv"
        df_bad.to_csv(path, sep="\t", index=False)
        df = load_infact(path)
        assert pd.isna(df["date_verified"].iloc[0])

    # --- validate_dataset ---

    def test_validate_dataset_passes(self, sample_df):
        """No exception raised for a fully valid DataFrame."""
        validate_dataset(sample_df)

    def test_validate_dataset_raises_on_missing_column(self, sample_df):
        df_missing = sample_df.drop(columns=["verdict_original"])
        with pytest.raises(ValueError, match="verdict_original"):
            validate_dataset(df_missing)

    def test_validate_dataset_extra_columns(self, sample_df):
        """Validation passes when extra required columns are present."""
        sample_df = sample_df.copy()
        sample_df["extra_col"] = "value"
        validate_dataset(sample_df, extra_columns=["extra_col"])

    def test_validate_dataset_raises_on_missing_extra_column(self, sample_df):
        with pytest.raises(ValueError, match="extra_col"):
            validate_dataset(sample_df, extra_columns=["extra_col"])

    def test_validate_dataset_empty_dataframe(self):
        """An empty DataFrame with correct columns should pass validation."""
        df_empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
        validate_dataset(df_empty)

    # --- get_dataset_summary ---

    def test_summary_n_records(self, sample_df):
        summary = get_dataset_summary(sample_df)
        assert summary["n_records"] == len(sample_df)

    def test_summary_n_columns(self, sample_df):
        summary = get_dataset_summary(sample_df)
        assert summary["n_columns"] == len(sample_df.columns)

    def test_summary_verdict_counts(self, sample_df):
        summary = get_dataset_summary(sample_df)
        assert "adevarat" in summary["verdict_counts"]
        assert "fals" in summary["verdict_counts"]

    def test_summary_domain_counts(self, sample_df):
        summary = get_dataset_summary(sample_df)
        assert "Science" in summary["domain_counts"]

    def test_summary_missing_per_column(self, sample_df):
        summary = get_dataset_summary(sample_df)
        assert isinstance(summary["missing_per_column"], dict)

    def test_summary_date_range(self, tsv_file):
        df = load_infact(tsv_file)
        summary = get_dataset_summary(df)
        assert "date_range" in summary
        assert summary["date_range"]["min"] is not None
        assert summary["date_range"]["max"] is not None

    def test_summary_no_verdict_column(self):
        df = pd.DataFrame({"col1": [1, 2]})
        summary = get_dataset_summary(df)
        assert summary["verdict_counts"] == {}

    def test_summary_no_domain_column(self):
        df = pd.DataFrame({"col1": [1, 2]})
        summary = get_dataset_summary(df)
        assert summary["domain_counts"] == {}

    def test_summary_all_dates_null(self):
        df = pd.DataFrame({"date_verified": pd.to_datetime([pd.NaT, pd.NaT])})
        summary = get_dataset_summary(df)
        assert summary["date_range"]["min"] is None
        assert summary["date_range"]["max"] is None

    def test_summary_empty_dataframe(self):
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        summary = get_dataset_summary(df)
        assert summary["n_records"] == 0
