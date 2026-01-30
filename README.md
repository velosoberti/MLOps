# 🚀 MLOps End-to-End Pipeline for Diabetes Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc)](https://dvc.org/)
[![Feast](https://img.shields.io/badge/Feast-Feature%20Store-FF6B6B)](https://feast.dev/)
[![Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?logo=apache-airflow)](https://airflow.apache.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask)](https://flask.palletsprojects.com/)
[![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/features/actions)

A **production-ready MLOps pipeline** demonstrating industry best practices for machine learning operations. This project covers the complete ML lifecycle from data versioning to model deployment with automated CI/CD.

---

## 📚 Table of Contents

1. [System Overview](#-system-overview)
2. [Quick Start Guide](#-quick-start-guide)
3. [Step-by-Step Tutorial](#-step-by-step-tutorial)
   - [Step 1: Environment Setup](#step-1-environment-setup)
   - [Step 2: Data Versioning with DVC](#step-2-data-versioning-with-dvc)
   - [Step 3: Feature Store with Feast](#step-3-feature-store-with-feast)
   - [Step 4: Experiment Tracking with MLflow](#step-4-experiment-tracking-with-mlflow)
   - [Step 5: Pipeline Orchestration with Airflow](#step-5-pipeline-orchestration-with-airflow)
   - [Step 6: Model Serving with Flask API](#step-6-model-serving-with-flask-api)
   - [Step 7: CI/CD with GitHub Actions](#step-7-cicd-with-github-actions)
4. [Configuration Management](#-configuration-management)
5. [Project Structure](#-project-structure)
6. [Troubleshooting](#-troubleshooting)
7. [Contributing](#-contributing)

---

## 🎯 System Overview

This MLOps pipeline implements a diabetes prediction system with the following components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Pipeline Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   DVC    │───▶│   ETL    │───▶│  Feast   │───▶│ Training │              │
│  │  (Data)  │    │(Airflow) │    │(Features)│    │ (MLflow) │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │                                               │                      │
│       │                                               ▼                      │
│       │                                         ┌──────────┐                │
│       │                                         │  Model   │                │
│       │                                         │ Registry │                │
│       │                                         └──────────┘                │
│       │                                               │                      │
│       ▼                                               ▼                      │
│  ┌──────────┐                                   ┌──────────┐                │
│  │  GitHub  │◀──────────────────────────────────│Flask API │                │
│  │ Actions  │         CI/CD Pipeline            │(Serving) │                │
│  └──────────┘                                   └──────────┘                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Component | Tool | Purpose |
|-----------|------|---------|
| Data Versioning | DVC | Track and version datasets |
| Feature Store | Feast | Centralized feature management |
| Experiment Tracking | MLflow | Track experiments and model registry |
| Orchestration | Airflow | Schedule and monitor pipelines |
| Model Serving | Flask | REST API for predictions |
| CI/CD | GitHub Actions | Automated testing and deployment |

---

## ⚡ Quick Start Guide

```bash
# 1. Clone and setup
git clone https://github.com/velosoberti/MLOps_projects.git
cd MLOps_projects

# 2. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 3. Create environment and install dependencies
uv venv
source .venv/bin/activate
uv sync

# 4. Copy and configure environment
cp .env.example .env
# Edit .env with your paths (see Configuration section)

# 5. Pull data with DVC
dvc pull

# 6. Start MLflow (Terminal 1)
mlflow ui --host 0.0.0.0 --port 5000

# 7. Start Airflow (Terminal 2)
cd airflow && docker compose up -d

# 8. Start Flask API (Terminal 3)
python flask/api.py
```

Access the services:
- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080 (user: `airflow`, pass: `airflow`)
- **Flask API**: http://localhost:5005

---

## 📖 Step-by-Step Tutorial

### Step 1: Environment Setup

#### Prerequisites
- Python 3.10+
- Docker and Docker Compose
- Git
- 4GB RAM minimum (8GB recommended)

#### Installation

```bash
# Install uv (recommended package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Clone the repository
git clone https://github.com/velosoberti/MLOps_projects.git
cd MLOps_projects

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install all dependencies
uv sync

# Install development dependencies (for testing)
uv sync --dev
```

#### Configure Environment Variables

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your actual paths
nano .env  # or use your preferred editor
```

**Important variables to set:**
```bash
# Update these paths to match your system
ML_DATA_BASE_PATH=/path/to/MLOps_projects/data
ML_FEAST_REPO_PATH=/path/to/MLOps_projects/feature_store/feature_repo
ML_ARTIFACTS_PATH=/path/to/MLOps_projects/data/artifacts
ML_INPUT_FILE=/path/to/MLOps_projects/data/diabetes.csv
```

---

### Step 2: Data Versioning with DVC

DVC (Data Version Control) tracks your datasets like Git tracks code.

#### Initialize DVC (first time only)

```bash
# Initialize DVC in your project
dvc init

# Add a remote storage (optional but recommended)
# Local storage example:
dvc remote add -d myremote /path/to/dvc-storage

# S3 example:
# dvc remote add -d myremote s3://mybucket/dvc-storage

# Google Drive example:
# dvc remote add -d myremote gdrive://folder_id
```

#### Track Your Data

```bash
# Add data file to DVC tracking
dvc add data/diabetes.csv

# This creates data/diabetes.csv.dvc (metadata file)
# Commit the .dvc file to Git
git add data/diabetes.csv.dvc data/.gitignore
git commit -m "Track diabetes dataset with DVC"

# Push data to remote storage
dvc push
```

#### Pull Data (when cloning the repo)

```bash
# Download the data from remote storage
dvc pull

# Check status
dvc status
```

#### Version Your Data

```bash
# After modifying your dataset
dvc add data/diabetes.csv
git add data/diabetes.csv.dvc
git commit -m "Update dataset v2"
dvc push

# Restore a previous version
git checkout <commit-hash> data/diabetes.csv.dvc
dvc checkout
```

**Key DVC Commands:**
| Command | Description |
|---------|-------------|
| `dvc init` | Initialize DVC in project |
| `dvc add <file>` | Track a file with DVC |
| `dvc push` | Upload data to remote |
| `dvc pull` | Download data from remote |
| `dvc status` | Check data status |
| `dvc checkout` | Restore data to match .dvc files |

---

### Step 3: Feature Store with Feast

Feast provides a centralized feature store for ML features.

#### Understanding Feast Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Feast Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │   Parquet   │────▶│   Offline   │────▶│  Training   │   │
│  │   Files     │     │   Store     │     │   Data      │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
│         │                                                    │
│         │ materialize                                        │
│         ▼                                                    │
│  ┌─────────────┐     ┌─────────────┐                        │
│  │   Online    │────▶│  Real-time  │                        │
│  │   Store     │     │ Predictions │                        │
│  │  (SQLite)   │     │             │                        │
│  └─────────────┘     └─────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Setup Feast

```bash
# Navigate to feature store directory
cd feature_store/feature_repo

# Apply feature definitions (creates registry)
feast apply

# Expected output:
# Created entity patient_id
# Created feature view predictors_df_feature_view
# Created feature view ptarget_df_feature_view
```

#### Define Features (example_repo.py)

The feature definitions are in `feature_store/feature_repo/example_repo.py`:

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

# Define entity (the key for looking up features)
patient = Entity(
    name="patient_id",
    value_type=ValueType.INT64,
    description="Patient ID"
)

# Define feature view (group of related features)
df1_feature_view = FeatureView(
    name="predictors_df_feature_view",
    entities=[patient],
    schema=[
        Field(name='Glucose', dtype=Int64),
        Field(name='BMI', dtype=Float64),
        Field(name='DiabetesPedigreeFunction', dtype=Float64),
        Field(name='Insulin', dtype=Int64),
        Field(name='SkinThickness', dtype=Int64),
        # ... more features
    ],
    source=FileSource(
        path='/path/to/data/artifacts/predictor.parquet',
        event_timestamp_column='event_timestamp'
    ),
    online=True,  # Enable online serving
)
```

#### Materialize Features (Offline → Online)

```bash
# Materialize features to online store for real-time serving
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

# This copies recent features from parquet files to SQLite for fast lookups
```

#### Retrieve Features

**For Training (Historical Features):**
```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_store/feature_repo")

# Entity dataframe with patient IDs and timestamps
entity_df = pd.DataFrame({
    "patient_id": [1, 2, 3],
    "event_timestamp": pd.to_datetime(["2024-01-01"] * 3)
})

# Get historical features for training
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "predictors_df_feature_view:Glucose",
        "predictors_df_feature_view:BMI",
        "predictors_df_feature_view:Insulin",
    ]
).to_df()
```

**For Inference (Online Features):**
```python
# Get features for real-time prediction
online_features = store.get_online_features(
    features=[
        "predictors_df_feature_view:Glucose",
        "predictors_df_feature_view:BMI",
    ],
    entity_rows=[{"patient_id": 123}]
).to_dict()
```

**Key Feast Commands:**
| Command | Description |
|---------|-------------|
| `feast apply` | Apply feature definitions |
| `feast materialize-incremental <timestamp>` | Update online store |
| `feast feature-views list` | List all feature views |
| `feast entities list` | List all entities |

---

### Step 4: Experiment Tracking with MLflow

MLflow tracks experiments, parameters, metrics, and models.

#### Start MLflow Server

```bash
# Start the MLflow tracking server
mlflow ui --host 0.0.0.0 --port 5000

# Access at http://localhost:5000
```

#### Track Experiments in Code

```python
import mlflow
from sklearn.linear_model import LogisticRegression

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create or get experiment
mlflow.set_experiment("diabetes-prediction")

# Start a run
with mlflow.start_run(run_name="logistic-regression-v1"):
    # Log parameters
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 100)
    
    # Train model
    model = LogisticRegression(solver="lbfgs", max_iter=100)
    model.fit(X_train, y_train)
    
    # Log metrics
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts (plots, files)
    mlflow.log_artifact("confusion_matrix.png")
```

#### Register Models

```bash
# Register a model from a run
mlflow models register -m "runs:/<run_id>/model" -n "diabete_model"

# Or in Python:
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="diabete_model"
)
```

#### Load Models for Inference

```python
# Load latest version
model = mlflow.sklearn.load_model("models:/diabete_model/latest")

# Load specific version
model = mlflow.sklearn.load_model("models:/diabete_model/3")

# Make predictions
predictions = model.predict(X_new)
```

**Key MLflow Commands:**
| Command | Description |
|---------|-------------|
| `mlflow ui` | Start tracking UI |
| `mlflow experiments list` | List experiments |
| `mlflow runs list --experiment-id <id>` | List runs |
| `mlflow models list` | List registered models |

---

### Step 5: Pipeline Orchestration with Airflow

Airflow schedules and monitors your ML pipelines.

#### Start Airflow

```bash
cd airflow

# First time: Initialize the database
docker compose up airflow-init

# Start all services
docker compose up -d

# Check logs
docker compose logs -f airflow-webserver

# Access UI at http://localhost:8080
# Username: airflow
# Password: airflow
```

#### Understanding the DAGs

This project includes 4 DAGs:

**1. ETL Pipeline (`etl_pipeline_final`)** - Daily
```
Extract CSV → Transform → Add Timestamps → Add Patient IDs → Save Parquet
```

**2. Feature Store (`feature_store_cre`)** - Daily
```
Load Entities → Get Historical Features → Save Training Dataset
```

**3. Training Pipeline (`ml_training_pipeline`)** - Weekly
```
Setup MLflow → Load Data → Split Data → Train → Evaluate → Log to MLflow
```

**4. Prediction Pipeline (`ml_prediction_pipeline`)** - Daily
```
Materialize Features → Find Patients → Fetch Features → Load Model → Predict → Save
```

#### Execute DAGs

**Via UI:**
1. Go to http://localhost:8080
2. Enable the DAG (toggle switch)
3. Click "Trigger DAG" (play button)

**Via CLI:**
```bash
# Trigger a DAG
docker compose exec airflow-scheduler airflow dags trigger etl_pipeline_final

# List DAG runs
docker compose exec airflow-scheduler airflow dags list-runs -d etl_pipeline_final

# Check task status
docker compose exec airflow-scheduler airflow tasks list ml_training_pipeline
```

#### Recommended Execution Order (First Time)

```bash
# 1. Run ETL to generate parquet files
docker compose exec airflow-scheduler airflow dags trigger etl_pipeline_final

# 2. Wait for completion, then create feature store dataset
docker compose exec airflow-scheduler airflow dags trigger feature_store_cre

# 3. Train the model
docker compose exec airflow-scheduler airflow dags trigger ml_training_pipeline

# 4. Run predictions
docker compose exec airflow-scheduler airflow dags trigger ml_prediction_pipeline
```

**Key Airflow Commands:**
| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Airflow |
| `docker compose down` | Stop Airflow |
| `docker compose logs -f` | View logs |
| `airflow dags trigger <dag_id>` | Trigger a DAG |
| `airflow dags list` | List all DAGs |

---

### Step 6: Model Serving with Flask API

The Flask API serves predictions in real-time.

#### Start the API

```bash
# Make sure MLflow is running first!
python flask/api.py

# API available at http://localhost:5005
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/model/reload` | Reload model |

#### Make Predictions

**Health Check:**
```bash
curl http://localhost:5005/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:5005/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Glucose": 148,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Insulin": 0,
    "SkinThickness": 35
  }'
```

**Response:**
```json
{
  "score": 0.6523,
  "prediction": "diabetes",
  "confidence": 0.6523,
  "model_version": 3,
  "timestamp": "2025-01-30T10:30:00"
}
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5005/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"Glucose": 148, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Insulin": 0, "SkinThickness": 35},
      {"Glucose": 85, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Insulin": 94, "SkinThickness": 28}
    ]
  }'
```

**Python Client:**
```python
import requests

response = requests.post(
    "http://localhost:5005/predict",
    json={
        "Glucose": 148,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Insulin": 0,
        "SkinThickness": 35
    }
)
print(response.json())
```

---

### Step 7: CI/CD with GitHub Actions

Automated testing and deployment with GitHub Actions.

#### CI Workflow (`.github/workflows/ci.yml`)

Runs on every pull request:
- **Linting**: `ruff check` and `ruff format`
- **Type Checking**: `mypy`
- **Testing**: `pytest` with coverage

```yaml
# Triggered on pull requests
on:
  pull_request:
    branches: [main, develop]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --dev
      - run: uv run pytest tests/ --cov
```

#### CD Workflow (`.github/workflows/cd.yml`)

Runs on merge to main:
- Full test suite
- DAG validation
- Docker image build

#### Run CI Locally

```bash
# Run linting
uv run ruff check .
uv run ruff format --check .

# Run type checking
uv run mypy src/ etl_functions/ flask/ config/

# Run tests with coverage
uv run pytest tests/ -v --cov
```

---

## ⚙️ Configuration Management

All configuration is managed via environment variables with the `ML_` prefix.

#### Configuration File (`.env`)

```bash
# MLflow
ML_MLFLOW_TRACKING_URI=http://127.0.0.1:5000/
ML_MLFLOW_EXPERIMENT_ID=467326610704772702

# Feast
ML_FEAST_REPO_PATH=/home/user/MLOps_projects/feature_store/feature_repo
ML_FEAST_FEATURE_VIEW=predictors_df_feature_view

# Data Paths
ML_DATA_BASE_PATH=/home/user/MLOps_projects/data
ML_ARTIFACTS_PATH=/home/user/MLOps_projects/data/artifacts
ML_INPUT_FILE=/home/user/MLOps_projects/data/diabetes.csv

# Model
ML_MODEL_NAME=diabete_model

# API
ML_API_HOST=0.0.0.0
ML_API_PORT=5005
ML_API_DEBUG=false

# Logging
ML_LOG_LEVEL=INFO
```

#### Access Configuration in Code

```python
from config.settings import settings

# Use configuration values
print(settings.mlflow_tracking_uri)
print(settings.feast_repo_path)
print(settings.get_input_file_path())
```

---

## 📁 Project Structure

```
MLOps_projects/
├── .github/workflows/       # CI/CD pipelines
│   ├── ci.yml              # Continuous Integration
│   └── cd.yml              # Continuous Deployment
├── airflow/                 # Orchestration
│   ├── dags/               # Airflow DAGs
│   │   ├── etl.py          # ETL pipeline
│   │   ├── train.py        # Training pipeline
│   │   ├── predict.py      # Prediction pipeline
│   │   └── feature_store.py # Feature store pipeline
│   └── docker-compose.yaml # Airflow Docker setup
├── config/                  # Configuration
│   └── settings.py         # Pydantic settings
├── data/                    # Data files
│   ├── diabetes.csv        # Raw dataset
│   ├── diabetes.csv.dvc    # DVC tracking
│   └── artifacts/          # Processed data
├── etl_functions/           # ETL utilities
│   └── etl.py              # Extract, Transform, Load
├── feature_store/           # Feast feature store
│   └── feature_repo/       # Feature definitions
├── flask/                   # REST API
│   ├── api.py              # Flask application
│   └── models.py           # Pydantic models
├── mlflow/                  # MLflow data
│   ├── mlruns/             # Experiment tracking
│   └── mlartifacts/        # Model artifacts
├── src/                     # Core ML code
│   ├── training.py         # Training functions
│   └── prediction.py       # Prediction functions
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── property/           # Property-based tests
│   └── integration/        # Integration tests
├── .env.example            # Environment template
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

---

## 🔧 Troubleshooting

#### MLflow Connection Error
```bash
# Make sure MLflow is running
mlflow ui --host 0.0.0.0 --port 5000

# Check if port is in use
lsof -i :5000
```

#### Feast Feature Not Found
```bash
# Re-apply feature definitions
cd feature_store/feature_repo
feast apply

# Re-materialize features
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

#### Airflow DAG Not Showing
```bash
# Check for syntax errors
docker compose exec airflow-scheduler python -c "import airflow.dags.etl"

# Refresh DAGs
docker compose exec airflow-scheduler airflow dags list
```

#### API Model Not Loaded
```bash
# Check MLflow has registered models
mlflow models list

# Verify model name matches configuration
echo $ML_MODEL_NAME
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `uv run pytest tests/ -v`
4. Run linting: `uv run ruff check . && uv run ruff format .`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Luis Veloso**
- GitHub: [@velosoberti](https://github.com/velosoberti)
- LinkedIn: [velosoberti](https://www.linkedin.com/in/velosoberti/)

---

**Last Updated:** January 30, 2026
