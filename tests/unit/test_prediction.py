"""Unit tests for prediction module functions.

Tests cover:
- Feature preparation for prediction
- Feast feature name generation
"""

import pandas as pd
import pytest

from src.prediction import (
    get_feast_features,
    prepare_features_for_prediction,
)


class TestGetFeastFeatures:
    """Tests for the get_feast_features() function."""

    def test_get_feast_features_returns_list(self):
        """Test that get_feast_features returns a list."""
        features = get_feast_features()

        assert isinstance(features, list)
        assert len(features) > 0

    def test_get_feast_features_format(self):
        """Test that features have correct format with feature view prefix."""
        features = get_feast_features()

        for feature in features:
            assert ":" in feature
            parts = feature.split(":")
            assert len(parts) == 2


class TestPrepareFeaturesForPrediction:
    """Tests for the prepare_features_for_prediction() function."""

    def test_prepare_features_drops_patient_id(self):
        """Test that prepare_features_for_prediction drops patient_id column."""
        df = pd.DataFrame({"patient_id": [1, 2, 3], "BMI": [25.0, 30.0, 22.0], "Insulin": [100, 150, 80]})

        X = prepare_features_for_prediction(df)

        assert "patient_id" not in X.columns
        assert "BMI" in X.columns
        assert "Insulin" in X.columns

    def test_prepare_features_with_model_features(self):
        """Test that features are selected and ordered by model_features."""
        df = pd.DataFrame({"patient_id": [1, 2], "BMI": [25.0, 30.0], "Insulin": [100, 150], "SkinThickness": [20, 25]})
        model_features = ["Insulin", "BMI"]  # Different order

        X = prepare_features_for_prediction(df, model_features=model_features)

        assert list(X.columns) == model_features

    def test_prepare_features_raises_on_missing_features(self):
        """Test that ValueError is raised when required features are missing."""
        df = pd.DataFrame({"patient_id": [1, 2], "BMI": [25.0, 30.0]})
        model_features = ["BMI", "Insulin", "MissingFeature"]

        with pytest.raises(ValueError) as exc_info:
            prepare_features_for_prediction(df, model_features=model_features)

        assert "Features faltando" in str(exc_info.value)

    def test_prepare_features_without_model_features_sorts_alphabetically(self):
        """Test that features are sorted alphabetically when model_features is None."""
        df = pd.DataFrame({"patient_id": [1, 2], "Zebra": [1, 2], "Apple": [3, 4], "Mango": [5, 6]})

        X = prepare_features_for_prediction(df, model_features=None)

        assert list(X.columns) == ["Apple", "Mango", "Zebra"]

    def test_prepare_features_preserves_values(self):
        """Test that feature values are preserved correctly."""
        df = pd.DataFrame({"patient_id": [1, 2], "BMI": [25.5, 30.2], "Insulin": [100, 150]})
        model_features = ["BMI", "Insulin"]

        X = prepare_features_for_prediction(df, model_features=model_features)

        assert X["BMI"].tolist() == [25.5, 30.2]
        assert X["Insulin"].tolist() == [100, 150]
