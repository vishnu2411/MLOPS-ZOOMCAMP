import os

import pandas as pd
import pickle


class TaxiDurationPredictor():

    def __init__(
            self,
            modeldir):
        
        model_fpath = os.path.join(modeldir,'model.bin')
        with open(model_fpath, 'rb') as f_in:
            dv, model = pickle.load(f_in)

        self.dv = dv
        self.model = model
        self.categorical = ['PULocationID', 'DOLocationID']

    def read_data(self, filename):
        df = pd.read_parquet(filename)

        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[self.categorical] = df[self.categorical].fillna(-1).astype('int').astype('str')
        
        return df

    def store_output(self, y_pred, year, month):
        df_result = pd.DataFrame({'duration': y_pred})
        df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df_result.index.astype('str')
        df_result.to_parquet(
            'predictions_{}-{:02d}.parquet'.format(year, month),
            engine='pyarrow',
            compression=None,
            index=False
        )

    def run(self, year, month):
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
        print(f'reading from cloud: {url}')
        df = self.read_data(url)

        dicts = df[self.categorical].to_dict(orient='records')
        X_val = self.dv.transform(dicts)
        y_pred = self.model.predict(X_val)

        print(f'predicted mean duration: {y_pred.mean()}')

        self.store_output(y_pred, year, month)


if __name__ == '__main__':
    predictor = TaxiDurationPredictor('./04-deployment')
    predictor.run(2023, 4)
