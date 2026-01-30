"""Unit tests for training module functions.

Tests cover:
- Data preparation functions with mock data
- Model evaluation metrics calculation
"""

import pandas as pd

from src.training import (
    calculate_accuracy,
    prepare_features,
)


class TestPrepareFeatures:
    """Tests for the prepare_features() function."""

    def test_prepare_features_separates_target(self, sample_diabetes_data: pd.DataFrame):
        """Test that prepare_features correctly separates features and target."""
        X, y = prepare_features(sample_diabetes_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "Outcome" not in X.columns
        assert y.name == "Outcome"

    def test_prepare_features_preserves_row_count(self, sample_diabetes_data: pd.DataFrame):
        """Test that prepare_features preserves the number of rows."""
        X, y = prepare_features(sample_diabetes_data)

        assert len(X) == len(sample_diabetes_data)
        assert len(y) == len(sample_diabetes_data)

    def test_prepare_features_drops_specified_columns(self, sample_diabetes_data: pd.DataFrame):
        """Test that prepare_features drops specified columns."""
        # Add columns that should be dropped
        data_with_extra = sample_diabetes_data.copy()
        data_with_extra["event_timestamp"] = pd.Timestamp.now()
        data_with_extra["patient_id"] = range(len(data_with_extra))

        X, y = prepare_features(data_with_extra)

        assert "event_timestamp" not in X.columns
        assert "patient_id" not in X.columns
        assert "Outcome" not in X.columns

    def test_prepare_features_custom_columns_to_drop(self, sample_diabetes_data: pd.DataFrame):
        """Test that prepare_features respects custom columns_to_drop."""
        X, y = prepare_features(sample_diabetes_data, columns_to_drop=["Pregnancies", "Age"])

        assert "Pregnancies" not in X.columns
        assert "Age" not in X.columns
        assert "Glucose" in X.columns  # Other columns should remain

    def test_prepare_features_custom_target_column(self):
        """Test that prepare_features works with custom target column."""
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "custom_target": [0, 1, 0]})

        X, y = prepare_features(data, target_column="custom_target", columns_to_drop=[])

        assert "custom_target" not in X.columns
        assert y.name == "custom_target"


class TestCalculateAccuracy:
    """Tests for the calculate_accuracy() function."""

    def test_calculate_accuracy_perfect_predictions(self):
        """Test accuracy calculation with perfect predictions."""
        y_true = pd.Series([0, 1, 0, 1, 1])
        y_pred = pd.Series([0, 1, 0, 1, 1])

        accuracy = calculate_accuracy(y_true, y_pred)

        assert accuracy == 1.0

    def test_calculate_accuracy_all_wrong(self):
        """Test accuracy calculation with all wrong predictions."""
        y_true = pd.Series([0, 0, 0, 0, 0])
        y_pred = pd.Series([1, 1, 1, 1, 1])

        accuracy = calculate_accuracy(y_true, y_pred)

        assert accuracy == 0.0

    def test_calculate_accuracy_partial(self):
        """Test accuracy calculation with partial correct predictions."""
        y_true = pd.Series([0, 1, 0, 1, 0])
        y_pred = pd.Series([0, 1, 1, 0, 0])  # 3 correct out of 5

        accuracy = calculate_accuracy(y_true, y_pred)

        assert accuracy == 0.6

    def test_calculate_accuracy_returns_float(self):
        """Test that calculate_accuracy returns a float."""
        y_true = pd.Series([0, 1, 0])
        y_pred = pd.Series([0, 1, 1])

        accuracy = calculate_accuracy(y_true, y_pred)

        assert isinstance(accuracy, float)
