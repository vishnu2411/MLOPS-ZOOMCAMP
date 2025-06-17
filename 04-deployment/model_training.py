import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

def get_ny_taxi_data(color, y, m):
    print(f'Fetching {color} taxi trip data for {y}-{m}')
    fpath = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{y}-{m}.parquet'
    df = pd.read_parquet(fpath)
    print(f'Dataset contains {df.shape[0]} records and {df.shape[1]} columns')
    return df

def calc_duration(df):
    df['duration_td'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration_td.apply(lambda dtd: dtd.total_seconds()/60)
    return df

def remove_outliers(df):
    df_cl = df.loc[(df.duration >= 1) & (df.duration <= 60)].copy()
    print(f'{df_cl.shape[0]/df.shape[0]*100}% of records are kept after outlier removal')
    return df_cl

def fit_transform_X(df, categorical, numerical, dv=None):
    df[categorical] = df[categorical].astype(str)
    dicts = df[categorical+numerical].to_dict(orient='records')
    if not dv:
        print(f'fit-transforming the features')
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
        print(f'{len(dv.feature_names_)} features will be used')
        return X, dv
    else:
        print(f'transforming the features')
        X = dv.transform(dicts)
        print(f'{len(dv.feature_names_)} features will be used')
        return X

def prep_y(df, target):
    y_train = df[target].values
    return y_train

def train_regressor(X_train, y_train, type='LR'):
    print(f'Training regressor of type {type}')
    if type == 'LR':
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
    return regressor

def predict_evaluate(regressor, X, y, plot=False, return_pred=False):
    y_pred = regressor.predict(X)
    # rmse = mean_squared_error(y, y_pred, squared=False)
    if y is not None:
        rmse = root_mean_squared_error(y, y_pred)
        print(f'RMSE: {rmse}')
    if plot:
        sns.histplot(y_pred,label='prediction')
        if y is not None:
            sns.histplot(y, label='actual')
        plt.legend()
    if return_pred:
        return y_pred

def train_pipeline():
    # Train (Let's make sure that the pipeline produces the same results as before)
    df_train = get_ny_taxi_data('yellow', '2023', '01')
    df_train = calc_duration(df_train)
    df_train = remove_outliers(df_train)
    X_train, dv = fit_transform_X(df_train, categorical=['PULocationID', 'DOLocationID'], numerical=[], dv=None)  # numerical=['trip_distance']
    y_train = prep_y(df_train, 'duration')
    regressor = train_regressor(X_train, y_train, 'LR')
    predict_evaluate(regressor, X_train, y_train)

    modelfile = '/home/vrunbuntu/anaconda3/MLOPS-ZOOMCAMP/MLOPS-ZOOMCAMP/04-deployment/model.bin'
    with open (modelfile, "wb") as f_out:
        pickle.dump((dv, regressor), f_out)

def validation_pipeline(dv, model):
    # Validation
    df_val = get_ny_taxi_data('yellow', '2023', '02')
    df_val = calc_duration(df_val)
    df_val = remove_outliers(df_val)
    X_val = fit_transform_X(df_val, categorical=['PULocationID', 'DOLocationID'], numerical=[], dv=dv)  # 'trip_distance'
    y_val = prep_y(df_val, 'duration')
    predict_evaluate(regressor=model, X=X_val, y=y_val)
