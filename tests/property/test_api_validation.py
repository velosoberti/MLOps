"""Property-based tests for API request validation.

# Feature: ml-pipeline-refactor-cicd, Property 3: API Request Validation

**Validates: Requirements 6.2**

This module tests that the Flask API correctly validates prediction requests
and returns appropriate error responses for invalid inputs.
"""

import os
import sys

# Ensure proper import path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from typing import Any  # noqa: E402

from hypothesis import assume, given  # noqa: E402
from hypothesis import settings as hypothesis_settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# Import api.models directly now that the folder is renamed
from api.models import (
    validate_batch_request,
    validate_prediction_request,
)

# Strategy for valid numeric values (positive floats)
valid_numeric = st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)

# Strategy for invalid (non-numeric) values
# Note: Pydantic coerces numeric strings to floats, so we exclude them
invalid_values = st.one_of(
    st.text(min_size=1, max_size=10).filter(lambda s: not _is_numeric_string(s)),
    st.lists(st.integers(), min_size=0, max_size=3),
    st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), min_size=0, max_size=2),
    st.none(),
)


def _is_numeric_string(s: str) -> bool:
    """Check if a string can be converted to a float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# Required feature names
REQUIRED_FEATURES = ["Glucose", "BMI", "DiabetesPedigreeFunction", "Insulin", "SkinThickness"]


class TestAPIRequestValidation:
    """Property 3: API Request Validation.

    *For any* prediction request with invalid or missing required fields,
    the Flask API SHALL return a 400 status code with an error message
    that identifies the specific validation failure.
    """

    @given(
        glucose=valid_numeric,
        bmi=valid_numeric,
        dpf=valid_numeric,
        insulin=valid_numeric,
        skin=valid_numeric,
    )
    @hypothesis_settings(max_examples=100)
    def test_valid_request_passes_validation(
        self, glucose: float, bmi: float, dpf: float, insulin: float, skin: float
    ) -> None:
        """Test that valid requests with all required fields pass validation.

        **Validates: Requirements 6.2**
        """
        data = {
            "Glucose": glucose,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Insulin": insulin,
            "SkinThickness": skin,
        }

        is_valid, error_message, request = validate_prediction_request(data)

        assert is_valid, f"Valid request should pass validation, got error: {error_message}"
        assert error_message is None
        assert request is not None
        assert request.Glucose == glucose
        assert bmi == request.BMI

    @given(missing_field=st.sampled_from(REQUIRED_FEATURES))
    @hypothesis_settings(max_examples=100)
    def test_missing_required_field_returns_error(self, missing_field: str) -> None:
        """Test that missing required fields are detected and reported.

        **Validates: Requirements 6.2**
        """
        # Create a complete valid request
        data = {
            "Glucose": 100.0,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "Insulin": 50.0,
            "SkinThickness": 20.0,
        }

        # Remove one required field
        del data[missing_field]

        is_valid, error_message, request = validate_prediction_request(data)

        assert not is_valid, f"Request missing '{missing_field}' should fail validation"
        assert error_message is not None
        assert request is None
        # Error message should mention the missing field or indicate validation failure
        assert (
            "missing" in error_message.lower()
            or "required" in error_message.lower()
            or "validation" in error_message.lower()
        )

    @given(
        field_to_invalidate=st.sampled_from(REQUIRED_FEATURES),
        invalid_value=invalid_values,
    )
    @hypothesis_settings(max_examples=100)
    def test_invalid_field_type_returns_error(self, field_to_invalidate: str, invalid_value: Any) -> None:
        """Test that invalid field types are detected and reported.

        **Validates: Requirements 6.2**
        """
        # Skip None values as they're handled by missing field test
        assume(invalid_value is not None)

        # Create a complete valid request
        data = {
            "Glucose": 100.0,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "Insulin": 50.0,
            "SkinThickness": 20.0,
        }

        # Replace one field with invalid value
        data[field_to_invalidate] = invalid_value

        is_valid, error_message, request = validate_prediction_request(data)

        assert not is_valid, (
            f"Request with invalid '{field_to_invalidate}' value ({type(invalid_value).__name__}) "
            f"should fail validation"
        )
        assert error_message is not None
        assert request is None

    @given(negative_value=st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False))
    @hypothesis_settings(max_examples=100)
    def test_negative_values_return_error(self, negative_value: float) -> None:
        """Test that negative values (where positive required) are detected.

        **Validates: Requirements 6.2**
        """
        data = {
            "Glucose": negative_value,  # Should be >= 0
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "Insulin": 50.0,
            "SkinThickness": 20.0,
        }

        is_valid, error_message, request = validate_prediction_request(data)

        assert not is_valid, "Request with negative Glucose should fail validation"
        assert error_message is not None
        assert request is None

    def test_empty_request_returns_error(self) -> None:
        """Test that empty request body returns appropriate error.

        **Validates: Requirements 6.2**
        """
        is_valid, error_message, request = validate_prediction_request({})

        assert not is_valid
        assert error_message is not None
        assert (
            "empty" in error_message.lower()
            or "required" in error_message.lower()
            or "missing" in error_message.lower()
        )
        assert request is None

    def test_none_request_returns_error(self) -> None:
        """Test that None request body returns appropriate error.

        **Validates: Requirements 6.2**
        """
        is_valid, error_message, request = validate_prediction_request(None)

        assert not is_valid
        assert error_message is not None
        assert "empty" in error_message.lower()
        assert request is None

    def test_non_dict_request_returns_error(self) -> None:
        """Test that non-dict request body returns appropriate error.

        **Validates: Requirements 6.2**
        """
        is_valid, error_message, request = validate_prediction_request([1, 2, 3])

        assert not is_valid
        assert error_message is not None
        assert "object" in error_message.lower() or "json" in error_message.lower()
        assert request is None


class TestBatchRequestValidation:
    """Tests for batch prediction request validation."""

    @given(num_instances=st.integers(min_value=1, max_value=10))
    @hypothesis_settings(max_examples=50)
    def test_valid_batch_request_passes_validation(self, num_instances: int) -> None:
        """Test that valid batch requests pass validation.

        **Validates: Requirements 6.2**
        """
        instances = [
            {
                "Glucose": 100.0 + i,
                "BMI": 25.0,
                "DiabetesPedigreeFunction": 0.5,
                "Insulin": 50.0,
                "SkinThickness": 20.0,
            }
            for i in range(num_instances)
        ]

        data = {"instances": instances}

        is_valid, error_message, request = validate_batch_request(data)

        assert is_valid, f"Valid batch request should pass validation, got error: {error_message}"
        assert error_message is None
        assert request is not None
        assert len(request.instances) == num_instances

    def test_missing_instances_field_returns_error(self) -> None:
        """Test that missing 'instances' field returns appropriate error.

        **Validates: Requirements 6.2**
        """
        data = {"data": [{"Glucose": 100}]}  # Wrong field name

        is_valid, error_message, request = validate_batch_request(data)

        assert not is_valid
        assert error_message is not None
        assert "instances" in error_message.lower()
        assert request is None

    def test_empty_instances_list_returns_error(self) -> None:
        """Test that empty instances list returns appropriate error.

        **Validates: Requirements 6.2**
        """
        data = {"instances": []}

        is_valid, error_message, request = validate_batch_request(data)

        assert not is_valid
        assert error_message is not None
        assert (
            "empty" in error_message.lower()
            or "non-empty" in error_message.lower()
            or "at least" in error_message.lower()
        )
        assert request is None

    def test_too_many_instances_returns_error(self) -> None:
        """Test that exceeding max instances returns appropriate error.

        **Validates: Requirements 6.2**
        """
        # Create 1001 instances (exceeds 1000 limit)
        instances = [{"Glucose": 100} for _ in range(1001)]
        data = {"instances": instances}

        is_valid, error_message, request = validate_batch_request(data)

        assert not is_valid
        assert error_message is not None
        assert "1000" in error_message or "maximum" in error_message.lower()
        assert request is None
