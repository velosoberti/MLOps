

from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
    ValueType
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64

patient = Entity(name="patient_id", value_type=ValueType.INT64,
                 description = "Patient ID")



# Predictor View
file_source1 = FileSource(
    path = '/home/luisveloso/MLOps_projects/data/artifacts/predictor.parquet',
    event_timestamp_column = 'event_timestamp'
)

df1_feature_view = FeatureView(
    name = "predictors_df_feature_view",
    ttl = timedelta(seconds=86400*2), #importante para online store
    entities= [patient],
    schema= [
        Field(name = 'Pregnancies',              dtype=Int64),
        Field(name = 'Glucose',                  dtype=Int64),
        Field(name = 'BloodPressure',            dtype=Int64),
        Field(name = 'SkinThickness',            dtype=Int64),
        Field(name = 'Insulin',                  dtype=Int64),
        Field(name = 'BMI',                      dtype=Float64),
        Field(name = 'DiabetesPedigreeFunction', dtype=Float64),
        Field(name = 'Age',                      dtype=Int64)
    ],
    source= file_source1,
    online=True,
    tags={}
)


# Target View
file_source2 = FileSource(
    path = '/home/luisveloso/MLOps_projects/data/artifacts/target.parquet',
    event_timestamp_column = 'event_timestamp'
)

df1_target_view = FeatureView(
    name = "ptarget_df_feature_view",
    ttl = timedelta(seconds=86400*2), #importante para online store
    entities= [patient],
    schema= [
        Field(name = 'Outcome',              dtype=Int64)
    ],
    source= file_source2,
    online=True,
    tags={}
)