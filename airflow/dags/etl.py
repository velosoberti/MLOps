"""
ETL Pipeline DAG for diabetes data processing.

This DAG extracts data from CSV, transforms it into predictor/target format,
adds timestamps and patient IDs, and saves the results as parquet files.
"""
import logging
from typing import Any, Dict

import pandas as pd
from airflow.decorators import dag, task

from config.settings import settings
from etl_functions.etl import (
    create_patient_ids,
    create_timestamps,
    extract,
    save_parquet,
    transform,
)

logger = logging.getLogger(__name__)


@dag(
    dag_id="etl_pipeline_final",
    schedule_interval="@daily",
    start_date=pd.Timestamp(2023, 8, 1),
    catchup=False,
    tags=["etl", "diabetes", "data_processing"],
)
def etl_flow() -> None:
    """ETL pipeline for diabetes data processing."""

    @task()
    def task_extract() -> pd.DataFrame:
        """Extract data from the configured input file.
        
        Returns:
            DataFrame with raw diabetes data.
            
        Raises:
            FileNotFoundError: If the input file does not exist.
            pd.errors.EmptyDataError: If the input file is empty.
        """
        input_path = settings.get_input_file_path()
        logger.info(f"Extracting data from: {input_path}")
        try:
            df = extract(input_path)
            logger.info(f"Successfully extracted {len(df)} rows")
            return df
        except FileNotFoundError as e:
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(
                f"ETL extract failed: Input file '{input_path}' does not exist. "
                f"Please verify ML_INPUT_FILE or ML_DATA_BASE_PATH configuration."
            ) from e
        except pd.errors.EmptyDataError as e:
            logger.error(f"Input file is empty: {input_path}")
            raise pd.errors.EmptyDataError(
                f"ETL extract failed: Input file '{input_path}' is empty."
            ) from e

    @task(multiple_outputs=True)
    def task_transform(df: pd.DataFrame) -> Dict[str, Any]:
        """Transform raw data into predictor and target DataFrames.
        
        Args:
            df: Raw input DataFrame.
            
        Returns:
            Dictionary with 'predictor' and 'target' keys.
            
        Raises:
            KeyError: If required columns are missing.
            ValueError: If transformation produces invalid data.
        """
        logger.info(f"Transforming data with {len(df)} rows")
        try:
            predictor, target = transform(df)
            logger.info(
                f"Transform complete: predictor shape={predictor.shape}, "
                f"target length={len(target)}"
            )
            return {"predictor": predictor, "target": target}
        except KeyError as e:
            logger.error(f"Missing required column during transform: {e}")
            raise KeyError(
                f"ETL transform failed: Missing required column {e}. "
                f"Expected columns include 'Outcome' for target variable."
            ) from e
        except Exception as e:
            logger.error(f"Transform failed with unexpected error: {e}")
            raise ValueError(
                f"ETL transform failed: {str(e)}"
            ) from e

    @task(multiple_outputs=True)
    def task_timestamps(predictor: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """Add timestamp column to predictor and target DataFrames.
        
        Args:
            predictor: Predictor DataFrame.
            target: Target Series.
            
        Returns:
            Dictionary with updated 'predictor' and 'target'.
        """
        logger.info("Adding timestamps to data")
        try:
            p, t = create_timestamps(predictor, target)
            logger.info("Timestamps added successfully")
            return {"predictor": p, "target": t}
        except Exception as e:
            logger.error(f"Failed to add timestamps: {e}")
            raise ValueError(
                f"ETL timestamp creation failed: {str(e)}"
            ) from e

    @task(multiple_outputs=True)
    def task_patient_ids(predictor: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """Add patient ID column to predictor and target DataFrames.
        
        Args:
            predictor: Predictor DataFrame.
            target: Target Series.
            
        Returns:
            Dictionary with updated 'predictor' and 'target'.
        """
        logger.info("Adding patient IDs to data")
        try:
            p, t = create_patient_ids(predictor, target)
            logger.info(f"Patient IDs added: {len(p)} records")
            return {"predictor": p, "target": t}
        except Exception as e:
            logger.error(f"Failed to add patient IDs: {e}")
            raise ValueError(
                f"ETL patient ID creation failed: {str(e)}"
            ) from e

    @task()
    def task_save(predictor: pd.DataFrame, target: pd.Series) -> None:
        """Save predictor and target DataFrames as parquet files.
        
        Args:
            predictor: Predictor DataFrame to save.
            target: Target Series to save.
            
        Raises:
            PermissionError: If output directory is not writable.
            OSError: If disk space is insufficient.
        """
        output_dir = settings.get_artifacts_path()
        logger.info(f"Saving parquet files to: {output_dir}")
        try:
            save_parquet(predictor, target, output_dir)
            logger.info("Parquet files saved successfully")
        except PermissionError as e:
            logger.error(f"Permission denied writing to: {output_dir}")
            raise PermissionError(
                f"ETL save failed: Cannot write to '{output_dir}'. "
                f"Please check directory permissions."
            ) from e
        except OSError as e:
            logger.error(f"OS error saving parquet files: {e}")
            raise OSError(
                f"ETL save failed: {str(e)}. "
                f"Please verify ML_ARTIFACTS_PATH configuration and disk space."
            ) from e

    # --- FLOW DEFINITION ---

    # 1. Extract
    df = task_extract()

    # 2. Transform (output is a dict)
    data_transformed = task_transform(df)

    # 3. Create Timestamps (uses keys from previous dict)
    data_with_time = task_timestamps(
        predictor=data_transformed["predictor"],
        target=data_transformed["target"],
    )

    # 4. Create Patient IDs
    data_final = task_patient_ids(
        predictor=data_with_time["predictor"],
        target=data_with_time["target"],
    )

    # 5. Save
    task_save(
        predictor=data_final["predictor"],
        target=data_final["target"],
    )


dag_instance = etl_flow()