from typing import Any

import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

store = FeatureStore(repo_path="/home/luisveloso/MLOps_projects/feature_store/feature_repo")
entity_df = pd.read_parquet("/home/luisveloso/MLOps_projects/data/artifacts/target.parquet")


def trainin_data() -> pd.DataFrame:
    """Get training data from feature store."""
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "predictors_df_feature_view:DiabetesPedigreeFunction",
            "predictors_df_feature_view:BMI",
            "predictors_df_feature_view:SkinThickness",
            "predictors_df_feature_view:Insulin",
            "predictors_df_feature_view:Glucose",
        ],
    )
    return training_data.to_df()


def dataset(training_data: Any) -> None:
    """Create saved dataset in feature store."""
    store.create_saved_dataset(
        from_=trainin_data,  # type: ignore[arg-type]
        name="my_training_dataset-test",
        storage=SavedDatasetFileStorage(
            path="/home/luisveloso/MLOps_projects/feature_store/feature_repo/data/my_training_dataset_test.parquet"
        ),
    )
