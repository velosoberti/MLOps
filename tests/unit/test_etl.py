"""Unit tests for ETL functions.

Tests cover:
- extract() with sample CSV data
- transform() produces correct predictor/target split
- create_timestamps() adds correct timestamp column
- create_patient_ids() adds sequential IDs
"""

import os

import pandas as pd
import pytest

from etl_functions.etl import (
    create_patient_ids,
    create_timestamps,
    extract,
    save_parquet,
    transform,
)


class TestExtract:
    """Tests for the extract() function."""

    def test_extract_reads_csv_file(self, sample_diabetes_data: pd.DataFrame, tmp_path):
        """Test that extract() correctly reads a CSV file."""
        csv_path = tmp_path / "test_diabetes.csv"
        sample_diabetes_data.to_csv(csv_path, index=False)

        result = extract(input_path=str(csv_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_diabetes_data)
        assert list(result.columns) == list(sample_diabetes_data.columns)

    def test_extract_raises_on_missing_file(self):
        """Test that extract() raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            extract(input_path="/nonexistent/path/file.csv")


class TestTransform:
    """Tests for the transform() function."""

    def test_transform_splits_predictor_and_target(self, sample_diabetes_data: pd.DataFrame):
        """Test that transform() correctly splits data into predictor and target."""
        predictor, target = transform(sample_diabetes_data)

        assert isinstance(predictor, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert "Outcome" not in predictor.columns
        assert target.name == "Outcome"

    def test_transform_preserves_row_count(self, sample_diabetes_data: pd.DataFrame):
        """Test that transform() preserves the number of rows."""
        predictor, target = transform(sample_diabetes_data)

        assert len(predictor) == len(sample_diabetes_data)
        assert len(target) == len(sample_diabetes_data)

    def test_transform_predictor_has_correct_columns(self, sample_diabetes_data: pd.DataFrame):
        """Test that predictor has all columns except Outcome."""
        predictor, _ = transform(sample_diabetes_data)

        expected_columns = [c for c in sample_diabetes_data.columns if c != "Outcome"]
        assert list(predictor.columns) == expected_columns

    def test_transform_target_values_match(self, sample_diabetes_data: pd.DataFrame):
        """Test that target values match the original Outcome column."""
        _, target = transform(sample_diabetes_data)

        pd.testing.assert_series_equal(
            target.reset_index(drop=True), sample_diabetes_data["Outcome"].reset_index(drop=True)
        )


class TestCreateTimestamps:
    """Tests for the create_timestamps() function."""

    def test_create_timestamps_adds_column(self, sample_diabetes_data: pd.DataFrame):
        """Test that create_timestamps() adds event_timestamp column."""
        predictor, target = transform(sample_diabetes_data)
        pred_with_ts, target_with_ts = create_timestamps(predictor, target)

        assert "event_timestamp" in pred_with_ts.columns
        assert "event_timestamp" in target_with_ts.columns

    def test_create_timestamps_preserves_row_count(self, sample_diabetes_data: pd.DataFrame):
        """Test that create_timestamps() preserves row count."""
        predictor, target = transform(sample_diabetes_data)
        pred_with_ts, target_with_ts = create_timestamps(predictor, target)

        assert len(pred_with_ts) == len(predictor)
        assert len(target_with_ts) == len(target)

    def test_create_timestamps_are_datetime(self, sample_diabetes_data: pd.DataFrame):
        """Test that timestamps are datetime type."""
        predictor, target = transform(sample_diabetes_data)
        pred_with_ts, _ = create_timestamps(predictor, target)

        assert pd.api.types.is_datetime64_any_dtype(pred_with_ts["event_timestamp"])


class TestCreatePatientIds:
    """Tests for the create_patient_ids() function."""

    def test_create_patient_ids_adds_column(self, sample_diabetes_data: pd.DataFrame):
        """Test that create_patient_ids() adds patient_id column."""
        predictor, target = transform(sample_diabetes_data)
        pred_with_ts, target_with_ts = create_timestamps(predictor, target)
        pred_with_ids, target_with_ids = create_patient_ids(pred_with_ts, target_with_ts)

        assert "patient_id" in pred_with_ids.columns
        assert "patient_id" in target_with_ids.columns

    def test_create_patient_ids_are_sequential(self, sample_diabetes_data: pd.DataFrame):
        """Test that patient IDs are sequential starting from 0."""
        predictor, target = transform(sample_diabetes_data)
        pred_with_ts, target_with_ts = create_timestamps(predictor, target)
        pred_with_ids, _ = create_patient_ids(pred_with_ts, target_with_ts)

        expected_ids = list(range(len(sample_diabetes_data)))
        assert list(pred_with_ids["patient_id"]) == expected_ids

    def test_create_patient_ids_match_in_both_dataframes(self, sample_diabetes_data: pd.DataFrame):
        """Test that patient IDs match between predictor and target."""
        predictor, target = transform(sample_diabetes_data)
        pred_with_ts, target_with_ts = create_timestamps(predictor, target)
        pred_with_ids, target_with_ids = create_patient_ids(pred_with_ts, target_with_ts)

        pd.testing.assert_series_equal(
            pred_with_ids["patient_id"].reset_index(drop=True), target_with_ids["patient_id"].reset_index(drop=True)
        )


class TestSaveParquet:
    """Tests for the save_parquet() function."""

    def test_save_parquet_creates_files(self, sample_diabetes_data: pd.DataFrame, tmp_path):
        """Test that save_parquet() creates parquet files."""
        predictor, target = transform(sample_diabetes_data)
        pred_with_ts, target_with_ts = create_timestamps(predictor, target)
        pred_with_ids, target_with_ids = create_patient_ids(pred_with_ts, target_with_ts)

        output_dir = str(tmp_path / "output")
        save_parquet(pred_with_ids, target_with_ids, output_dir=output_dir)

        assert os.path.exists(os.path.join(output_dir, "predictor.parquet"))
        assert os.path.exists(os.path.join(output_dir, "target.parquet"))

    def test_save_parquet_files_are_readable(self, sample_diabetes_data: pd.DataFrame, tmp_path):
        """Test that saved parquet files can be read back."""
        predictor, target = transform(sample_diabetes_data)
        pred_with_ts, target_with_ts = create_timestamps(predictor, target)
        pred_with_ids, target_with_ids = create_patient_ids(pred_with_ts, target_with_ts)

        output_dir = str(tmp_path / "output")
        save_parquet(pred_with_ids, target_with_ids, output_dir=output_dir)

        loaded_predictor = pd.read_parquet(os.path.join(output_dir, "predictor.parquet"))
        loaded_target = pd.read_parquet(os.path.join(output_dir, "target.parquet"))

        assert len(loaded_predictor) == len(pred_with_ids)
        assert len(loaded_target) == len(target_with_ids)
