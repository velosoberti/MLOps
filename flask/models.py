"""Pydantic models for Flask API request/response validation.

This module provides Pydantic models for validating prediction requests
and structuring API responses.
"""

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single prediction request model.

    Validates that all required features are present and have valid numeric values.
    """

    Glucose: float = Field(..., ge=0, description="Plasma glucose concentration")
    BMI: float = Field(..., ge=0, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Diabetes pedigree function")
    Insulin: float = Field(..., ge=0, description="2-Hour serum insulin")
    SkinThickness: float = Field(..., ge=0, description="Triceps skin fold thickness")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"Glucose": 148, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Insulin": 0, "SkinThickness": 35}
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Prediction response model."""

    score: float = Field(..., ge=0, le=1, description="Prediction probability score")
    prediction: str = Field(..., description="Prediction label (diabetes/no_diabetes)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    model_version: int | None = Field(None, description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model.

    Contains a list of prediction instances to process.
    """

    instances: list[dict[str, Any]] = Field(
        ..., min_length=1, max_length=1000, description="List of prediction instances"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "instances": [
                        {
                            "Glucose": 148,
                            "BMI": 33.6,
                            "DiabetesPedigreeFunction": 0.627,
                            "Insulin": 0,
                            "SkinThickness": 35,
                        },
                        {
                            "Glucose": 85,
                            "BMI": 26.6,
                            "DiabetesPedigreeFunction": 0.351,
                            "Insulin": 0,
                            "SkinThickness": 29,
                        },
                    ]
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""

    predictions: list[dict[str, Any]] = Field(..., description="List of predictions")
    total: int = Field(..., ge=0, description="Total successful predictions")
    errors: list[dict[str, Any]] | None = Field(None, description="List of errors for failed instances")
    model_version: int | None = Field(None, description="Model version used")
    timestamp: str = Field(..., description="Batch processing timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Health check timestamp")
    services: dict[str, Any] = Field(..., description="Service status details")


class ModelInfoResponse(BaseModel):
    """Model information response model."""

    model_name: str = Field(..., description="Name of the model")
    model_version: int | None = Field(None, description="Model version")
    features: list[str] | None = Field(None, description="Expected feature names")
    feature_count: int = Field(..., ge=0, description="Number of features")
    loaded_at: str | None = Field(None, description="Model load timestamp")
    mlflow_tracking_uri: str = Field(..., description="MLflow tracking URI")
    experiment_id: str = Field(..., description="MLflow experiment ID")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str | None = Field(None, description="Error timestamp")
    expected_features: list[str] | None = Field(None, description="Expected features (for validation errors)")
    received_features: list[str] | None = Field(None, description="Received features (for validation errors)")


def validate_prediction_request(data: dict[str, Any]) -> tuple[bool, str | None, PredictionRequest | None]:
    """Validate prediction request data using Pydantic.

    Args:
        data: Input data dictionary to validate.

    Returns:
        Tuple of (is_valid, error_message, validated_request).
        If valid, error_message is None and validated_request contains the model.
        If invalid, error_message contains the validation error and validated_request is None.
    """
    if not data:
        return False, "Request body is empty", None

    if not isinstance(data, dict):
        return False, "Request body must be a JSON object", None

    try:
        request = PredictionRequest(**data)
        return True, None, request
    except Exception as e:
        # Extract field-specific error messages
        error_str = str(e)
        if "Field required" in error_str:
            # Extract missing field name
            import re

            match = re.search(r"'(\w+)'", error_str)
            if match:
                return False, f"Missing required field: {match.group(1)}", None
        return False, f"Validation error: {error_str}", None


def validate_batch_request(data: dict[str, Any]) -> tuple[bool, str | None, BatchPredictionRequest | None]:
    """Validate batch prediction request data using Pydantic.

    Args:
        data: Input data dictionary to validate.

    Returns:
        Tuple of (is_valid, error_message, validated_request).
    """
    if not data:
        return False, "Request body is empty", None

    if not isinstance(data, dict):
        return False, "Request body must be a JSON object", None

    try:
        request = BatchPredictionRequest(**data)
        return True, None, request
    except Exception as e:
        error_str = str(e)
        if "'instances'" in error_str and "Field required" in error_str:
            return False, "Request body must contain 'instances' (list of objects)", None
        if "at least 1 item" in error_str.lower():
            return False, "'instances' must be a non-empty list", None
        if "at most 1000" in error_str.lower():
            return False, "Maximum 1000 instances per batch", None
        return False, f"Validation error: {error_str}", None
