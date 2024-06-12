import click
import os
import mlflow
import logging
import tempfile
import pretty_errors
import yaml
from minio import  Minio
from utils import truth_checker 
# get environment variables
from dotenv import load_dotenv
load_dotenv()
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_CLIENT_URL = os.environ.get("MINIO_CLIENT_URL")
MINIO_SSL = truth_checker(os.environ.get("MINIO_SSL"))
client = Minio(MINIO_CLIENT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, secure=MINIO_SSL)


@click.command()
@click.option("--pyfunc-model-folder",
              type=str,
              default="s3://mlflow-bucket/2/33d85746285c42a7b3ef403eb2f5c95f/artifacts/pyfunc_model")
@click.option("--forecast-horizon",
              type=str,
              default="960")
@click.option("--series-uri",
              type=str,
              default="ENG/series.csv")
@click.option("--future-covariates-uri",
              type=str,
              default="None")
@click.option("--past-covariates-uri",
              type=str,
              default="None")
@click.option("--roll-size",
              type=str,
              default="96")
@click.option("--batch-size",
              type=str,
              default="1")
@click.option("--multiple",
              type=str,
              default="False")
@click.option("--weather-covariates",
              type=str,
              default="None")
@click.option("--resolution",
              type=str,
              default="15")
@click.option("--ts-id-pred",
              type=str,
              default="None")

def MLflowDartsModelPredict(pyfunc_model_folder, forecast_horizon, series_uri, future_covariates_uri, past_covariates_uri, roll_size, batch_size, multiple, weather_covariates, resolution, ts_id_pred):
    """This is the main function for predicting MLflow pyfunc models. The inputs are csv file uris (online or local) and integers. 
    The csv files are dowloaded and the converted to darts.TimeSeries and finally given to the loaded models and for prediction according to their 
    specification"""

    with mlflow.start_run(run_name='inference') as mlrun:

        weather_covariates = yaml.safe_load(weather_covariates)
        print(weather_covariates)
        input = {
            "n": forecast_horizon,
            "series_uri": series_uri,
            "roll_size": roll_size,
            "future_covariates_uri": future_covariates_uri,
            "past_covariates_uri": past_covariates_uri,
            "batch_size": batch_size,
            "multiple": multiple,
            "weather_covariates": weather_covariates,
            "resolution": resolution,
            "ts_id_pred": ts_id_pred,
        }

        # Load model as a PyFuncModel.
        print("\nLoading pyfunc model...")
        loaded_model = mlflow.pyfunc.load_model(client, pyfunc_model_folder)

        # Predict on a Pandas DataFrame.
        print("\nPyfunc model prediction...")
        predictions = loaded_model.predict(input)

        # Store CSV of predictions: first locally and then to MLflow server
        infertmpdir = tempfile.mkdtemp()
        predictions.to_csv(os.path.join(infertmpdir, 'predictions.csv'))
        mlflow.log_artifacts(infertmpdir)

#TODO: Input should be newly ingested time series data passing through the load_raw data step and the etl step. How can this be done?
# Maybe I need a second pipeline (inference pipeline) that goes like that: load_raw_data -> etl -> inference for a specific registered MLflow model"""
if __name__ == '__main__':
    print("\n=========== INFERENCE =============")
    logging.info("\n=========== INFERENCE =============")
    MLflowDartsModelPredict()
