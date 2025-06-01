import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import mlflow
import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.schedules import IntervalSchedule
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

def get_training_validation_dates():
    """Calculate training and validation dates dynamically."""
    today = datetime.today()
    train_date = today - timedelta(days=60)  # Two months ago
    val_date = today - timedelta(days=30)  # One month ago
    return train_date.year, train_date.month, val_date.year, val_date.month

@task(retries=2, retry_delay_seconds=1, log_prints=True)
def read_dataframe(color: str, year: int, month: int):
    """Read taxi trip data."""
    filename = f"{color}_tripdata_{year}-{month:02d}.parquet"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
    print(f"Reading from cloud: {url}")
    df = pd.read_parquet(url)

    # Compute duration
    if color == "green":
        df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    elif color == "yellow":
        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
        df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime

    df["duration"] = df["duration"].apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    return df

def get_features(df):
    """Extract categorical and numerical features."""
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    numerical = []
    print(f'Using categorical features: {categorical}')
    print(f'Using numerical features: {numerical}')
    return df, categorical + numerical

@task(log_prints=True)
def create_X(df: pd.DataFrame, dv: DictVectorizer = None) -> tuple:
    """Transform features into a format suitable for ML models."""
    df, colnames = get_features(df)
    dicts = df[colnames].to_dict(orient="records")

    if dv is None:
        print("Fitting and transforming with DictVectorizer")
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        print("Transforming with DictVectorizer")
        X = dv.transform(dicts)

    return X, dv

@task(log_prints=True)
def train_model(modeltype: str, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, dv: DictVectorizer):
    """Train a regression model for taxi duration prediction."""
    print(f"Training model of type {modeltype}")
    with mlflow.start_run() as run:
        if modeltype == "LR":
            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_val)
            mlflow.sklearn.log_model(regressor, artifact_path="models_mlflow")

        elif modeltype == "XGB":
            import xgboost as xgb
            train = xgb.DMatrix(X_train, label=y_train)
            valid = xgb.DMatrix(X_val, label=y_val)

            best_params = {
                "learning_rate": 0.09585,
                "max_depth": 30,
                "min_child_weight": 1.06,
                "objective": "reg:linear",
                "reg_alpha": 0.018,
                "reg_lambda": 0.0116,
                "seed": 42,
            }

            mlflow.log_params(best_params)

            booster = xgb.train(
                params=best_params, dtrain=train,
                num_boost_round=30, evals=[(valid, "validation")],
                early_stopping_rounds=50
            )

            y_pred = booster.predict(valid)
            mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        else:
            raise ValueError(f"Unknown model type: {modeltype}")

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        return run.info.run_id

@flow
def nyc_taxi_train_pipeline(color: str, modeltype: str):
    """Main training pipeline that automatically retrieves the correct dates."""
    train_year, train_month, val_year, val_month = get_training_validation_dates()

    print(f"Training a {modeltype} model for {color} taxis using {train_year}-{train_month} data")

    df_train = read_dataframe(color, train_year, train_month)
    df_val = read_dataframe(color, val_year, val_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    run_id = train_model(modeltype, X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id

# Scheduling the pipeline to run monthly
schedule = IntervalSchedule(interval=timedelta(days=30))

@flow(schedule=schedule)
def scheduled_nyc_taxi_pipeline():
    """Scheduled Prefect pipeline that runs automatically every month."""
    nyc_taxi_train_pipeline(color="yellow", modeltype="LR")
