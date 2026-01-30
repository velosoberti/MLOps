"""
Feature Store DAG for creating and persisting Feast datasets.

This DAG creates a saved dataset in Feast by:
1. Loading entity data (target.parquet) with patient IDs and timestamps
2. Retrieving historical features from the Feast feature view
3. Persisting the dataset to parquet for training

Configuration is loaded from the config module via environment variables.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from airflow.decorators import dag, task
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

from config.settings import settings

logger = logging.getLogger(__name__)


def _get_entity_path() -> str:
    """Get the path to the entity file (target.parquet).
    
    Returns:
        Path to the target.parquet file.
    """
    return str(Path(settings.get_artifacts_path()) / "target.parquet")


def _get_dataset_output_path() -> str:
    """Get the path for the output dataset.
    
    Returns:
        Path for the saved dataset parquet file.
    """
    if settings.output_dir:
        return str(Path(settings.output_dir) / "my_training_dataset.parquet")
    return str(Path("feature_store/data") / "my_training_dataset.parquet")


@dag(
    dag_id="feature_store_cre",
    schedule_interval="@daily",
    start_date=datetime(2023, 8, 1),
    catchup=False,
    tags=["mlops", "feast", "feature_store"],
)
def feature_store_dag() -> None:
    """Feature store dataset creation DAG."""

    @task()
    def create_dataset_task() -> str:
        """Create and persist a Feast dataset.
        
        Retrieves historical features from Feast and saves them as a
        registered dataset for training.
        
        Returns:
            Path to the saved dataset file.
            
        Raises:
            FileNotFoundError: If entity file does not exist.
            ValueError: If Feast feature retrieval fails.
        """
        repo_path = settings.feast_repo_path
        entity_path = _get_entity_path()
        dataset_output_path = _get_dataset_output_path()
        feature_view = settings.feast_feature_view
        dataset_name = settings.feast_dataset_name
        
        logger.info(f"Feature store configuration:")
        logger.info(f"  Repo path: {repo_path}")
        logger.info(f"  Entity path: {entity_path}")
        logger.info(f"  Output path: {dataset_output_path}")
        logger.info(f"  Feature view: {feature_view}")
        
        # Validate repo path
        if not repo_path:
            raise ValueError(
                "Feast repository path not configured. "
                "Set ML_FEAST_REPO_PATH environment variable."
            )
        
        if not os.path.exists(repo_path):
            raise FileNotFoundError(
                f"Feast repository not found at '{repo_path}'. "
                f"Please verify ML_FEAST_REPO_PATH configuration."
            )
        
        # Initialize Feast store
        try:
            store = FeatureStore(repo_path=repo_path)
        except Exception as e:
            logger.error(f"Failed to initialize Feast store: {e}")
            raise ValueError(
                f"Failed to initialize Feast store at '{repo_path}': {e}"
            ) from e
        
        # Load entity data
        logger.info(f"Loading entities from: {entity_path}")
        if not os.path.exists(entity_path):
            raise FileNotFoundError(
                f"Entity file not found at '{entity_path}'. "
                f"Please run the ETL pipeline first to generate target.parquet."
            )
        
        try:
            entity_df = pd.read_parquet(entity_path)
            logger.info(f"Loaded {len(entity_df)} entities")
        except Exception as e:
            logger.error(f"Failed to read entity file: {e}")
            raise ValueError(
                f"Failed to read entity file '{entity_path}': {e}"
            ) from e
        
        # Retrieve historical features
        logger.info("Retrieving historical features...")
        try:
            retrieval_job = store.get_historical_features(
                entity_df=entity_df,
                features=[
                    f"{feature_view}:DiabetesPedigreeFunction",
                    f"{feature_view}:BMI",
                    f"{feature_view}:SkinThickness",
                    f"{feature_view}:Insulin",
                    f"{feature_view}:Glucose",
                ],
            )
        except Exception as e:
            logger.error(f"Failed to retrieve historical features: {e}")
            raise ValueError(
                f"Failed to retrieve historical features from '{feature_view}': {e}. "
                f"Please verify the feature view exists and has data."
            ) from e
        
        # Create output directory
        output_dir = os.path.dirname(dataset_output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Persist dataset via Feast
        logger.info("Persisting dataset via Feast...")
        try:
            store.create_saved_dataset(
                from_=retrieval_job,
                name=dataset_name,
                storage=SavedDatasetFileStorage(path=dataset_output_path),
            )
        except Exception as e:
            logger.error(f"Failed to create saved dataset: {e}")
            raise ValueError(
                f"Failed to create saved dataset '{dataset_name}': {e}"
            ) from e
        
        logger.info(f"Dataset saved and registered at: {dataset_output_path}")
        return dataset_output_path

    # Execute task
    create_dataset_task()


fs_dag = feature_store_dag()