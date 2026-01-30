"""ETL functions for diabetes data processing.

This module provides functions for extracting, transforming, and loading
diabetes data for the ML pipeline. All paths are configurable via the
config module.
"""

import os

import pandas as pd

from config.settings import settings


def extract(input_path: str | None = None) -> pd.DataFrame:
    """Extract data from CSV file.

    Reads the diabetes dataset from a CSV file. The file path can be
    overridden via the input_path parameter, otherwise uses the
    configured path from settings.

    Args:
        input_path: Optional override for input file path. If not provided,
            uses settings.get_input_file_path().

    Returns:
        DataFrame containing the raw diabetes data.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    path = input_path or settings.get_input_file_path()
    return pd.read_csv(path)


def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Transform raw data into predictor features and target variable.

    Separates the input DataFrame into predictor features (all columns
    except 'Outcome') and the target variable ('Outcome' column).

    Args:
        df: Raw input DataFrame containing diabetes data with an 'Outcome' column.

    Returns:
        A tuple containing:
            - predictor: DataFrame with all feature columns (excluding 'Outcome')
            - target: Series containing the 'Outcome' values

    Raises:
        KeyError: If 'Outcome' column is not present in the DataFrame.
    """
    predictor = df.loc[:, df.columns != "Outcome"]
    target = df["Outcome"]
    return predictor, target


def create_timestamps(predictor: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add event timestamps to predictor and target data.

    Creates a date range ending at the current timestamp with daily frequency,
    and appends it as an 'event_timestamp' column to both predictor and target.
    This is required for Feast feature store compatibility.

    Args:
        predictor: DataFrame containing predictor features.
        target: Series containing target values.

    Returns:
        A tuple containing:
            - predictor: DataFrame with added 'event_timestamp' column
            - target: DataFrame with added 'event_timestamp' column
    """
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=len(predictor), freq="D").to_frame(
        name="event_timestamp", index=False
    )

    predictor = pd.concat([predictor, timestamps], axis=1)
    target_df = pd.concat([target.to_frame(), timestamps], axis=1)
    return predictor, target_df


def create_patient_ids(predictor: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add sequential patient IDs to predictor and target data.

    Creates a sequential patient_id column (0 to n-1) and appends it
    to both predictor and target DataFrames. This serves as the entity
    identifier for the Feast feature store.

    Args:
        predictor: DataFrame containing predictor features.
        target: DataFrame containing target values.

    Returns:
        A tuple containing:
            - predictor: DataFrame with added 'patient_id' column
            - target: DataFrame with added 'patient_id' column
    """
    data_len = len(predictor)
    ids_list = list(range(data_len))
    patient_ids = pd.DataFrame(ids_list, columns=["patient_id"])

    predictor = pd.concat([predictor, patient_ids], axis=1)
    target = pd.concat([target, patient_ids], axis=1)
    return predictor, target


def save_parquet(predictor: pd.DataFrame, target: pd.DataFrame, output_dir: str | None = None) -> None:
    """Save predictor and target DataFrames to parquet files.

    Saves the processed data to parquet format in the specified output
    directory. Creates the directory if it doesn't exist.

    Args:
        predictor: DataFrame containing predictor features to save.
        target: DataFrame containing target values to save.
        output_dir: Optional override for output directory. If not provided,
            uses settings.get_artifacts_path().
    """
    output_path = output_dir or settings.get_artifacts_path()
    os.makedirs(output_path, exist_ok=True)

    predictor.to_parquet(path=f"{output_path}/predictor.parquet", engine="pyarrow")
    target.to_parquet(path=f"{output_path}/target.parquet", engine="pyarrow")
