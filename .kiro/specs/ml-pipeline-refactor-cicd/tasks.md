# Implementation Plan: ML Pipeline Refactoring and CI/CD

## Overview

This implementation plan breaks down the refactoring and CI/CD setup into incremental tasks. Each task builds on previous work, ensuring the codebase remains functional throughout the refactoring process. The plan prioritizes configuration management first (to unblock other changes), followed by code quality improvements, testing infrastructure, and finally CI/CD workflows.

## Tasks

- [x] 1. Set up configuration management
  - [x] 1.1 Create config module with Pydantic settings
    - Create `config/__init__.py` and `config/settings.py`
    - Define Settings class with all configuration fields
    - Add environment variable loading with `ML_` prefix
    - Add validation for required fields
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [x] 1.2 Write property test for configuration loading
    - **Property 1: Configuration Loading with Fallbacks**
    - Test that env vars override defaults correctly
    - Test that missing optional fields use defaults
    - **Validates: Requirements 1.1**
  
  - [x] 1.3 Create .env.example template
    - Document all configuration variables
    - Provide example values for development
    - _Requirements: 1.3_

- [x] 2. Refactor ETL functions to use configuration
  - [x] 2.1 Update etl_functions/etl.py
    - Import settings from config module
    - Replace hardcoded INPUT_FILE and OUTPUT_FILE paths
    - Add type annotations to all functions
    - Add docstrings to all functions
    - _Requirements: 1.4, 2.1, 2.5_
  
  - [x] 2.2 Write unit tests for ETL functions
    - Test extract() with sample CSV data
    - Test transform() produces correct predictor/target split
    - Test create_timestamps() adds correct timestamp column
    - Test create_patient_ids() adds sequential IDs
    - _Requirements: 3.1_

- [x] 3. Refactor src modules to use configuration
  - [x] 3.1 Update src/training.py
    - Import settings from config module
    - Replace hardcoded MLFLOW_TRACKING_URI, FEAST_REPO_PATH
    - Add type annotations to all functions
    - _Requirements: 1.4, 2.1_
  
  - [x] 3.2 Update src/prediction.py
    - Import settings from config module
    - Replace hardcoded paths and URIs
    - Add type annotations to all functions
    - _Requirements: 1.4, 2.1_
  
  - [x] 3.3 Update src/api_consctructor.py
    - Import settings from config module
    - Replace hardcoded MLFLOW_TRACKING_URI, MODEL_NAME
    - Add type annotations to ModelManager and InputValidator
    - _Requirements: 1.4, 2.1, 6.1_
  
  - [x] 3.4 Write unit tests for training and prediction logic
    - Test data preparation functions with mock data
    - Test model evaluation metrics calculation
    - _Requirements: 3.3_

- [x] 4. Checkpoint - Verify configuration refactoring
  - Ensure all modules import successfully
  - Verify no hardcoded paths remain in refactored files
  - Ask the user if questions arise

- [x] 5. Refactor Flask API
  - [x] 5.1 Create application factory pattern
    - Update flask/api.py with create_app() factory
    - Accept Settings as optional parameter for testing
    - Initialize ModelManager with injected config
    - _Requirements: 6.1_
  
  - [x] 5.2 Write property test for API configuration injection
    - **Property 2: API Configuration Injection**
    - Test that different configs produce different behaviors
    - **Validates: Requirements 6.1**
  
  - [x] 5.3 Add Pydantic request validation models
    - Create PredictionRequest model with field validation
    - Create BatchPredictionRequest model
    - Update endpoints to use Pydantic validation
    - _Requirements: 6.2_
  
  - [ ] 5.4 Write property test for API request validation
    - **Property 3: API Request Validation**
    - Test that invalid inputs return 400 with specific errors
    - **Validates: Requirements 6.2**
  
  - [x] 5.5 Write unit tests for Flask API endpoints
    - Test /health endpoint returns correct structure
    - Test /predict with valid input returns prediction
    - Test /predict with invalid input returns 400
    - Test /predict/batch with multiple instances
    - Test /model/info returns model metadata
    - _Requirements: 3.2_

- [x] 6. Checkpoint - Verify API refactoring
  - Ensure API starts without errors
  - Verify all endpoints respond correctly
  - Ask the user if questions arise

- [x] 7. Refactor Airflow DAGs
  - [x] 7.1 Update airflow/dags/etl.py
    - Import settings from config module
    - Add proper error handling with descriptive messages
    - _Requirements: 7.1, 7.2_
  
  - [x] 7.2 Update airflow/dags/train.py
    - Import settings from config module
    - Replace hardcoded paths in task functions
    - _Requirements: 7.1_
  
  - [x] 7.3 Update airflow/dags/predict.py
    - Import settings from config module
    - Replace hardcoded paths in task functions
    - _Requirements: 7.1_
  
  - [x] 7.4 Update airflow/dags/feature_store.py
    - Import settings from config module
    - Replace hardcoded REPO_PATH, ENTITY_PATH, DATASET_OUTPUT_PATH
    - _Requirements: 7.1_
  
  - [x] 7.5 Write DAG import tests
    - Test that all DAGs can be imported without errors
    - Test that DAG task dependencies are correctly defined
    - _Requirements: 7.3, 7.4, 8.4_

- [x] 8. Set up development dependencies
  - [x] 8.1 Update pyproject.toml with dev dependencies
    - Add pytest, pytest-cov, hypothesis to dev dependencies
    - Add ruff, mypy to dev dependencies
    - Add pydantic-settings to main dependencies
    - _Requirements: 5.1, 5.4_
  
  - [x] 8.2 Create ruff configuration
    - Add ruff configuration to pyproject.toml
    - Configure line length, select rules
    - _Requirements: 2.2_
  
  - [x] 8.3 Create mypy configuration
    - Add mypy configuration to pyproject.toml
    - Enable strict mode for src/ and etl_functions/
    - _Requirements: 2.3_

- [x] 9. Set up test infrastructure
  - [x] 9.1 Create tests/conftest.py with shared fixtures
    - Add sample_diabetes_data fixture
    - Add test_settings fixture
    - Add mock_mlflow_client fixture
    - _Requirements: 3.5_
  
  - [x] 9.2 Create pytest configuration
    - Add pytest configuration to pyproject.toml
    - Configure coverage settings
    - _Requirements: 3.4_

- [x] 10. Checkpoint - Verify test infrastructure
  - Run pytest to ensure test discovery works
  - Verify fixtures are accessible
  - Ask the user if questions arise

- [x] 11. Create CI workflow
  - [x] 11.1 Create .github/workflows/ci.yml
    - Add lint job with ruff check and format
    - Add type-check job with mypy
    - Add test job with pytest and coverage
    - Configure dependency caching with uv
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.5_
  
  - [x] 11.2 Add lock file validation
    - Add step to verify uv.lock is in sync
    - _Requirements: 5.2_

- [x] 12. Create CD workflow
  - [x] 12.1 Create .github/workflows/cd.yml
    - Add job to run full test suite on main
    - Add job to validate Airflow DAG syntax
    - Add job to build Docker image
    - _Requirements: 8.1, 8.2, 8.4_
  
  - [x] 12.2 Create Dockerfile for Flask API
    - Create multi-stage Dockerfile
    - Use uv for dependency installation
    - Configure health check
    - _Requirements: 8.2_

- [x] 13. Fix linting and type errors
  - [x] 13.1 Run ruff and fix all errors
    - Fix import ordering
    - Fix unused imports
    - Fix code style issues
    - _Requirements: 2.2_
  
  - [x] 13.2 Run mypy and fix type errors
    - Add missing type annotations
    - Fix type mismatches
    - Add py.typed marker
    - _Requirements: 2.1, 2.3_

- [x] 14. Final checkpoint - Full validation
  - Run complete CI workflow locally
  - Verify all tests pass
  - Verify coverage meets 70% threshold
  - Ask the user if questions arise

## Notes

- All tasks are required for comprehensive coverage
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
