# 🚀 MLOps - Complete Machine Learning Pipeline for Diabetes Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc)](https://dvc.org/)
[![Feast](https://img.shields.io/badge/Feast-Feature%20Store-FF6B6B)](https://feast.dev/)
[![Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?logo=apache-airflow)](https://airflow.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-EDA-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask)](https://flask.palletsprojects.com/)

## 📋 About the Project

This project demonstrates a **complete and professional end-to-end MLOps pipeline implementation** for diabetes prediction, integrating industry-leading tools and practices. The pipeline covers everything from data versioning to model deployment, with continuous prediction monitoring.

### 🎯 Objectives

- ✅ **Data versioning** with DVC
- ✅ **Centralized Feature Store** with Feast (Online + Offline Store)
- ✅ **Interactive exploratory analysis** with Streamlit
- ✅ **Automated pipeline orchestration** with Airflow
- ✅ **Experiment tracking** and model registry with MLflow
- ✅ **Professional REST API** for serving predictions with Flask
- ✅ **Interactive Dashboard** for monitoring and visualization
- ✅ **Containerization** with Docker

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML PROJECT DASHBOARD                               │
│                         http://localhost:8086                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │  Home   │  │   API   │  │   EDA   │  │ MLflow  │  │ Dataset │           │
│  │         │  │ Predict │  │Streamlit│  │  Runs   │  │  View   │           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
└───────┼────────────┼────────────┼────────────┼────────────┼─────────────────┘
        │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Flask     │ │  Streamlit  │ │   MLflow    │ │   Feast     │ │    DVC      │
│   API       │ │    EDA      │ │   Server    │ │   Store     │ │   Data      │
│  :5005      │ │   :8501     │ │   :5000     │ │   Local     │ │  Version    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
        │                              │               │
        └──────────────────────────────┼───────────────┘
                                       ▼
                            ┌─────────────────────┐
                            │      Airflow        │
                            │   Orchestration     │
                            │      :8080          │
                            └─────────────────────┘
```

### Directory Structure

```
MLOps_projects/
├── 📁 airflow/                     # Orchestration (Docker)
│   ├── dags/                       # Airflow DAGs
│   │   ├── etl.py                  # Daily ETL pipeline
│   │   ├── feature_store.py        # Training dataset creation
│   │   ├── train.py                # Training pipeline
│   │   └── predict.py              # Prediction pipeline
│   ├── docker-compose.yaml         # Docker configuration
│   ├── Dockerfile                  # Custom image
│   └── requirements.txt            # Airflow dependencies
│
├── 📁 dashboard/                   # Web Dashboard
│   ├── index.html                  # Main HTML
│   ├── css/styles.css              # Styles
│   ├── js/                         # JavaScript modules
│   └── server/server.py            # Backend server
│
├── 📁 data/                        # Data Layer
│   ├── diabetes.csv                # Original dataset
│   ├── diabetes.csv.dvc            # DVC metadata
│   └── artifacts/                  # Processed artifacts
│
├── 📁 eda_streamlit/               # Exploratory Analysis
│   └── eda.py                      # Streamlit dashboard
│
├── 📁 feature_store/               # Feature Store (Feast)
│   └── feature_repo/               # Feast Repository
│       ├── feature_store.yaml      # Feast configuration
│       └── example_repo.py         # Feature definitions
│
├── 📁 flask/                       # REST API
│   ├── api.py                      # Flask server
│   └── request.py                  # Test client
│
├── 📁 framework/                   # Reusable ML Framework
│   ├── training.py                 # Training functions
│   ├── prediction.py               # Prediction functions
│   └── api_constructor.py          # API builder
│
├── 📁 mlflow/                      # Tracking and Registry
│   ├── mlruns/                     # Local experiments
│   └── mlartifacts/                # Model artifacts
│
├── 📄 requirements.txt             # Project dependencies
├── 📄 pyproject.toml               # Project configuration
└── 📄 README.md                    # This file
```

---

## 📦 Prerequisites

- **Python 3.10** (required - not 3.11+)
- **Docker and Docker Compose**
- **Git**
- **4GB RAM minimum** (8GB recommended)
- **5GB disk space**

---

## 🚀 Complete Installation Guide (From Zero)

### Step 1: Linux Environment Setup (Windows Users)

If you're on Windows, install WSL (Windows Subsystem for Linux):

```bash
# Install WSL and Ubuntu
wsl --install -d Ubuntu

# Restart your computer after installation
# Then open Ubuntu terminal
```

### Step 2: Install UV Package Manager

```bash
# Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Step 3: Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/velosoberti/MLOps_projects.git
cd MLOps_projects

# Create virtual environment with Python 3.10
uv venv --python 3.10
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt

# Verify installation
python --version  # Should show Python 3.10.x
```

---

## 🔧 Service Configuration & Startup

### Service Ports Overview

| Service | Port | URL |
|---------|------|-----|
| **Dashboard** | 8086 | http://localhost:8086 |
| **MLflow** | 5000 | http://localhost:5000 |
| **Flask API** | 5005 | http://localhost:5005 |
| **Streamlit EDA** | 8501 | http://localhost:8501 |
| **Airflow** | 8080 | http://localhost:8080 |

---

## 1️⃣ DVC - Data Version Control

DVC tracks and versions your data files, similar to how Git tracks code.

### Initialize DVC (First Time Only)

```bash
# Initialize DVC in the project
dvc init

# Configure local cache (already done in this project)
dvc remote add -d localcache ./dvc_cache
```

### Pull Versioned Data

```bash
# Download the versioned dataset
dvc pull

# Verify the data exists
ls -la data/diabetes.csv

# Check DVC status
dvc status
```

### Update Data (When You Modify the Dataset)

```bash
# After modifying diabetes.csv
dvc add data/diabetes.csv

# Commit the changes
git add data/diabetes.csv.dvc
git commit -m "Update dataset"

# Push data to remote storage
dvc push
```

### Useful DVC Commands

```bash
# Check what's tracked
dvc list . --dvc-only

# View file info
dvc diff

# Restore previous version
git checkout <commit-hash> data/diabetes.csv.dvc
dvc checkout
```

---

## 2️⃣ MLflow - Experiment Tracking & Model Registry

MLflow tracks experiments, logs metrics, and stores trained models.

### Start MLflow Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start MLflow UI on port 5000
mlflow ui --host 0.0.0.0 --port 5000
```

**Access:** http://localhost:5000

### MLflow Directory Structure

```
mlflow/
├── mlruns/           # Experiment runs and metrics
│   └── <experiment_id>/
│       └── <run_id>/
│           ├── metrics/
│           ├── params/
│           └── artifacts/
└── mlartifacts/      # Model artifacts
```

### Useful MLflow Commands

```bash
# List all experiments
mlflow experiments list

# List runs for an experiment
mlflow runs list --experiment-id <EXPERIMENT_ID>

# Register a model from a run
mlflow models register -m "runs:/<run_id>/model" -n "diabete_model"

# Serve a model directly
mlflow models serve -m "models:/diabete_model/latest" -p 5001
```

### View Registered Models

1. Open http://localhost:5000
2. Click on "Models" tab
3. View model versions and stages

---

## 3️⃣ Feast - Feature Store

Feast manages features for ML models with both offline (training) and online (serving) stores.

### Navigate to Feature Store

```bash
cd feature_store/feature_repo
```

### Apply Feature Definitions

```bash
# Register entities and feature views
feast apply
```

Expected output:
```
Created entity patient_id
Created feature view predictors_df_feature_view
Created feature view ptarget_df_feature_view
```

### Materialize Features to Online Store

```bash
# Materialize features for serving (run after ETL pipeline)
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### Verify Feature Store

```bash
# List feature views
feast feature-views list

# List entities
feast entities list

# Describe a feature view
feast feature-views describe predictors_df_feature_view
```

### Feature Store Configuration

The `feature_store.yaml` defines:
```yaml
project: feature_store
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
registry: data/registry.db
```

### Return to Project Root

```bash
cd ../..
```

---

## 4️⃣ Airflow - Pipeline Orchestration

Airflow orchestrates the ML pipelines using Docker containers.

### Navigate to Airflow Directory

```bash
cd airflow
```

### Create Environment File

```bash
# Create .env file with required variables
echo "AIRFLOW_UID=$(id -u)" > .env
echo "WEBSERVER_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(16))')" >> .env
```

### Initialize Airflow (First Time Only)

```bash
# Initialize the database and create admin user
docker compose up airflow-init
```

Wait for the message: `airflow-init exited with code 0`

### Start Airflow Services

```bash
# Start all services in background
docker compose up -d

# Check services are running
docker compose ps
```

**Access:** http://localhost:8080  
**Credentials:** `airflow` / `airflow`

### Available DAGs

| DAG Name | Schedule | Description |
|----------|----------|-------------|
| `etl_pipeline_final` | Daily | Process raw data → Parquet files |
| `feature_store_cre` | Daily | Create training dataset from Feast |
| `ml_training_pipeline` | Weekly | Train model and register in MLflow |
| `ml_prediction_pipeline` | Daily | Make batch predictions |

### Run DAGs from Terminal

```bash
# Trigger a DAG manually
docker compose exec airflow-scheduler airflow dags trigger etl_pipeline_final

# Check DAG run status
docker compose exec airflow-scheduler airflow dags list-runs -d etl_pipeline_final

# List all DAGs
docker compose exec airflow-scheduler airflow dags list
```

### Run DAGs from UI

1. Open http://localhost:8080
2. Login with `airflow` / `airflow`
3. Find the DAG in the list
4. Toggle the DAG to "ON" (unpause)
5. Click the "Play" button → "Trigger DAG"
6. Monitor progress in the "Graph" or "Grid" view

### Recommended Execution Order (First Time)

Run these DAGs in order:

1. **`etl_pipeline_final`** - Creates predictor.parquet and target.parquet
2. **`feature_store_cre`** - Creates training dataset
3. **`ml_training_pipeline`** - Trains and registers model
4. **`ml_prediction_pipeline`** - Makes predictions

### View Airflow Logs

```bash
# View webserver logs
docker compose logs -f airflow-webserver

# View scheduler logs
docker compose logs -f airflow-scheduler

# Access scheduler container
docker compose exec airflow-scheduler bash
```

### Stop Airflow

```bash
# Stop services (keep data)
docker compose down

# Stop and remove all data (clean restart)
docker compose down -v
```

### Return to Project Root

```bash
cd ..
```

---

## 5️⃣ Flask API - Model Serving

The Flask API serves predictions in real-time.

### Start Flask API

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the API server
cd flask
python api.py
```

**Access:** http://localhost:5005

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/model/reload` | Reload model |

### Test the API

```bash
# Health check
curl http://localhost:5005/health

# Single prediction
curl -X POST http://localhost:5005/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Glucose": 148,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Insulin": 0,
    "SkinThickness": 35
  }'

# Model info
curl http://localhost:5005/model/info
```

### Example Response

```json
{
  "score": 0.6523,
  "prediction": "diabetes",
  "confidence": 0.6523,
  "model_version": 1,
  "timestamp": "2026-02-02T10:30:00"
}
```

### Return to Project Root

```bash
cd ..
```

---

## 6️⃣ Streamlit - Exploratory Data Analysis

Streamlit provides an interactive EDA dashboard.

### Start Streamlit

```bash
# Activate virtual environment
source .venv/bin/activate

# Start Streamlit
cd eda_streamlit
streamlit run eda.py --server.port 8501
```

**Access:** http://localhost:8501

### Features

- 📊 Basic dataset information
- 🔍 Missing values analysis
- 📈 Statistical summaries
- 📊 Distribution plots
- 📦 Boxplots for outlier detection
- 🔗 Correlation matrix
- 🎯 Target variable analysis

### Return to Project Root

```bash
cd ..
```

---

## 7️⃣ Dashboard - Web Interface

The dashboard provides a unified interface to monitor all services and make predictions.

### Start Dashboard Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the dashboard
cd dashboard
python server/server.py --port 8086
```

**Access:** http://localhost:8086

### Dashboard Panels

| Panel | Description |
|-------|-------------|
| **Home** | Overview with quick links to all services |
| **API** | 🧪 Interactive prediction form + API documentation |
| **EDA** | Embedded Streamlit dashboard (requires Streamlit running) |
| **MLflow** | Experiment tracking information |
| **Database** | DVC versioning info (md5 hash, file size) |
| **Dataset** | Browse and paginate diabetes.csv data |
| **Feature Store** | Feast configuration, entities, and feature views |

### API Panel Features

The API panel includes:

1. **Flask API Overview** - Base URL and link to open API directly
2. **Interactive Prediction Form** - Test the model with custom values
3. **Example Patient Selector** - Pre-loaded test cases
4. **API Endpoints Documentation** - All available endpoints with examples
5. **Input Features Reference** - Expected fields and their descriptions

### Return to Project Root

```bash
cd ..
```

---

## 8️⃣ Making Predictions - Multiple Methods

There are **3 ways** to make predictions in this project:

### Method 1: Dashboard Web Form (Recommended for Testing)

The dashboard provides an interactive form to test predictions visually.

**Prerequisites:** Flask API must be running on port 5005

1. Open http://localhost:8086
2. Click on **"API"** in the sidebar
3. Use the **"Try the API"** section:

**Using Pre-defined Examples:**
- Select from dropdown: "Alto Risco", "Médio Risco", "Baixo Risco", or "Valores Normais"
- Click **"Make Prediction"**

**Using Custom Values:**
- Fill in the form fields:
  - **Glucose**: Plasma glucose concentration (mg/dL) - Range: 0-300
  - **BMI**: Body mass index - Range: 10-70
  - **DiabetesPedigreeFunction**: Diabetes pedigree score - Range: 0-3
  - **Insulin**: 2-Hour serum insulin (mu U/ml) - Range: 0-900
  - **SkinThickness**: Triceps skin fold thickness (mm) - Range: 0-100
- Click **"Make Prediction"**

**Result Display:**
- ✅ Green card = No diabetes (low probability)
- ⚠️ Yellow card = Diabetes (high probability)
- Shows: Score, Confidence level, Model version, Interpretation

### Method 2: Manual Prediction Script (Command Line)

The `manual_pred.py` script provides a command-line interface for predictions.

**Prerequisites:** Flask API must be running on port 5005

```bash
# Activate virtual environment
source .venv/bin/activate

# Interactive menu mode
python manual_pred.py

# Run pre-defined examples
python manual_pred.py --example

# Custom prediction mode
python manual_pred.py --custom

# Load from JSON file
python manual_pred.py --file

# Show help
python manual_pred.py --help
```

**Pre-defined Example Patients:**

| Example | Glucose | BMI | DiabetesPedigree | Insulin | SkinThickness |
|---------|---------|-----|------------------|---------|---------------|
| Alto Risco | 180 | 38.5 | 0.85 | 150 | 40 |
| Médio Risco | 120 | 30.2 | 0.45 | 80 | 28 |
| Baixo Risco | 85 | 24.5 | 0.25 | 40 | 20 |
| Valores Normais | 95 | 26.0 | 0.30 | 50 | 25 |

**Programmatic Usage:**

```python
from framework.manual_predict_by_api_request import PredictionClient

# Create client
client = PredictionClient()

# Single prediction
result = client.predict_single({
    "Glucose": 148,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Insulin": 0,
    "SkinThickness": 35
})
print(result)

# Batch prediction
results = client.predict_batch([
    {"Glucose": 180, "BMI": 38.5, "DiabetesPedigreeFunction": 0.85, "Insulin": 150, "SkinThickness": 40},
    {"Glucose": 85, "BMI": 24.5, "DiabetesPedigreeFunction": 0.25, "Insulin": 40, "SkinThickness": 20}
])
print(results)
```

### Method 3: Direct API Calls (curl/Postman)

Make HTTP requests directly to the Flask API.

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

**Batch Prediction:**
```bash
curl -X POST http://localhost:5005/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"Glucose": 180, "BMI": 38.5, "DiabetesPedigreeFunction": 0.85, "Insulin": 150, "SkinThickness": 40},
      {"Glucose": 85, "BMI": 24.5, "DiabetesPedigreeFunction": 0.25, "Insulin": 40, "SkinThickness": 20}
    ]
  }'
```

**Response Format:**
```json
{
  "score": 0.6523,
  "prediction": "diabetes",
  "confidence": 0.6523,
  "model_version": 1,
  "timestamp": "2026-02-02T10:30:00"
}
```

### Understanding the Results

| Field | Description |
|-------|-------------|
| **score** | Probability of diabetes (0.0 to 1.0) |
| **prediction** | "diabetes" if score ≥ 0.5, else "no_diabetes" |
| **confidence** | How confident the model is (higher = more certain) |
| **model_version** | Version of the model used |

**Confidence Interpretation:**
- **> 80%**: High confidence - Result is reliable
- **60-80%**: Medium confidence - Consider additional tests
- **< 60%**: Low confidence - Result is uncertain

---

## 🎯 Quick Start - Run Everything

Open **5 terminal windows** and run each service:

### Terminal 1: MLflow
```bash
cd MLOps_projects
source .venv/bin/activate
mlflow ui --host 0.0.0.0 --port 5000
```

### Terminal 2: Airflow
```bash
cd MLOps_projects/airflow
docker compose up
```

### Terminal 3: Flask API
```bash
cd MLOps_projects
source .venv/bin/activate
cd flask
python api.py
```

### Terminal 4: Streamlit
```bash
cd MLOps_projects
source .venv/bin/activate
cd eda_streamlit
streamlit run eda.py --server.port 8501
```

### Terminal 5: Dashboard
```bash
cd MLOps_projects
source .venv/bin/activate
cd dashboard
python server/server.py --port 8086
```

### Access All Services

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:8086 |
| **MLflow** | http://localhost:5000 |
| **Flask API** | http://localhost:5005 |
| **Streamlit** | http://localhost:8501 |
| **Airflow** | http://localhost:8080 |

---

## 🔄 Complete Workflow Example

### 1. Initialize Data Pipeline

```bash
# Pull data with DVC
dvc pull

# Start MLflow (Terminal 1)
mlflow ui --host 0.0.0.0 --port 5000

# Start Airflow (Terminal 2)
cd airflow && docker compose up -d
```

### 2. Run ETL Pipeline

```bash
# Via terminal
docker compose exec airflow-scheduler airflow dags trigger etl_pipeline_final

# Or via UI: http://localhost:8080 → etl_pipeline_final → Trigger
```

### 3. Setup Feature Store

```bash
# Apply Feast definitions
cd feature_store/feature_repo
feast apply
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
cd ../..

# Or run the DAG
docker compose exec airflow-scheduler airflow dags trigger feature_store_cre
```

### 4. Train Model

```bash
# Via Airflow
docker compose exec airflow-scheduler airflow dags trigger ml_training_pipeline

# Check MLflow for the new run: http://localhost:5000
```

### 5. Start Serving

```bash
# Start Flask API (Terminal 3)
cd flask && python api.py

# Start Dashboard (Terminal 4)
cd dashboard && python server/server.py --port 8086
```

### 6. Make Predictions

```bash
# Via curl
curl -X POST http://localhost:5005/predict \
  -H "Content-Type: application/json" \
  -d '{"Glucose": 148, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Insulin": 0, "SkinThickness": 35}'

# Or via Dashboard: http://localhost:8086 → API panel
```

---

## 🛠️ Troubleshooting

### MLflow Not Starting
```bash
# Check if port is in use
lsof -i :5000

# Kill existing process
kill -9 <PID>
```

### Airflow Container Issues
```bash
# Reset everything
cd airflow
docker compose down -v
docker compose up airflow-init
docker compose up -d
```

### Flask API CORS Errors
The API includes CORS headers. If issues persist:
```bash
# Restart the Flask API
cd flask
python api.py
```

### Feast Materialization Fails
```bash
# Ensure parquet files exist
ls -la data/artifacts/

# Re-run ETL pipeline first
docker compose exec airflow-scheduler airflow dags trigger etl_pipeline_final
```

### Dashboard Not Loading
```bash
# Hard refresh browser
Ctrl+Shift+R (or Cmd+Shift+R on Mac)

# Restart server
cd dashboard
python server/server.py --port 8086
```

---

## 📚 Additional Documentation

- [DVC Documentation](https://dvc.org/doc)
- [Feast Documentation](https://docs.feast.dev/)
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Luis Veloso**

- GitHub: [@velosoberti](https://github.com/velosoberti)
- LinkedIn: [velosoberti](https://www.linkedin.com/in/velosoberti/)

---

## 📈 Project Status

🟢 **Active Development** - This project is actively maintained and updated regularly.

**Last Updated:** February 2026
