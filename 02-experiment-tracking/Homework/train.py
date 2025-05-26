import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import click
import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random_forest_experiment")

def load_pickle(file):

    with open(file,'rb') as f:
        return pickle.load(f)
    
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):

     with mlflow.start_run():
        
        mlflow.sklearn.autolog()
        mlflow.set_tag("developer","Vishnuvardhan Reddy")

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        print("started")
        rf.fit(X_train, y_train)
        print("Fiting")
        y_pred = rf.predict(X_valid)
        print("Over")

        mse = mean_squared_error(y_valid, y_pred)
        rmse = np.sqrt(mse)


if __name__=="__main__":
    
    
    run_train()
