#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error


import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(color, year, month, read_local=False, categorical=['PULocationID', 'DOLocationID']):
    filename = f'{color}_tripdata_{year}-{month:02d}.parquet'
    
    if read_local:
        print(f'reading local: {filename}')
        url = f'./data_raw/{filename}'
    else:
        print(f'reading from cloud: {filename}')
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}'
    
    df = pd.read_parquet(url)

    if color=='green':
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    elif color=='yellow':
        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # categorical = ['PU_DO']
    df[categorical] = df[categorical].astype(str)

    # df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

def create_X(df, dv=None, categorical=['PULocationID', 'DOLocationID']):
    # categorical = ['PU_DO']
    # numerical = ['trip_distance']
    numerical = []

    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        print('Fitting and Transforming with DictVectorizer')
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        print('Transforming with DictVectorizer')
        X = dv.transform(dicts)

    return X, dv


def train_model(modeltype, X_train, y_train, X_val, y_val, dv):
    print (f'Training model of type {modeltype}')
    with mlflow.start_run() as run:

        if modeltype == 'LR':
            from sklearn.linear_model import LinearRegression
            
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_val)
            mlflow.sklearn.log_model(regressor, artifact_path='models')
            print(f'Intercept of the model: {regressor.intercept_}')

        elif modeltype == 'XGB':
            import xgboost as xgb
            train = xgb.DMatrix(X_train, label=y_train)
            valid = xgb.DMatrix(X_val, label=y_val)

            best_params = {
                'learning_rate': 0.09585355369315604,
                'max_depth': 30,
                'min_child_weight': 1.060597050922164,
                'objective': 'reg:linear',
                'reg_alpha': 0.018060244040060163,
                'reg_lambda': 0.011658731377413597,
                'seed': 42
            }

            mlflow.log_params(best_params)

            booster = xgb.train(
                params=best_params,
                dtrain=train,
                num_boost_round=30,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )

            y_pred = booster.predict(valid)
            mlflow.xgboost.log_model(booster, artifact_path='models_mlflow')
        else:
            raise ValueError(f'Unknown model type: {modeltype}. Allowed: [LR, XGB]')

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        return run.info.run_id



def run(color, year, month, modeltype, read_local):
    df_train = read_dataframe(color=color, year=year, month=month, read_local=read_local)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(color=color, year=next_year, month=next_month, read_local=read_local)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(modeltype, X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--color', type=str, required=True, help='Color of the taxi (green/yellow)')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    parser.add_argument('--modeltype', type=str, required=True, help='Model type (LR/XGB)')    
    args = parser.parse_args()

    print ('Run params: {}'.format(args))
    run_id = run(
        color=args.color,
        year=args.year,
        month=args.month,
        modeltype=args.modeltype,
        read_local=False)




