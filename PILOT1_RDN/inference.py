from utils import none_checker, load_local_csv_as_darts_timeseries, download_online_file
import click
import os
import mlflow
import darts
import logging
import tempfile

# get environment variables
from dotenv import load_dotenv
load_dotenv()
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

@click.command()
@click.option("--pyfunc-model-folder",
              type=str,
              default="s3://mlflow-bucket/2/33d85746285c42a7b3ef403eb2f5c95f/artifacts/pyfunc_model")
@click.option("--forecast-horizon",
              type=str,
              default="96")
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
def MLflowDartsModelPredict(pyfunc_model_folder, forecast_horizon, series_uri, future_covariates_uri, past_covariates_uri, roll_size, batch_size):
    """This is the main function for predicting MLflow pyfunc models. The inputs are csv file uris (online or local) and integers. 
    The csv files are dowloaded and the converted to darts.TimeSeries and finally given to the loaded models and for prediction according to their 
    specification"""
    # Parse arguments
    batch_size = int(batch_size)
    roll_size = int(roll_size)
    forecast_horizon = int(forecast_horizon)

    with mlflow.start_run(run_name='inference') as mlrun:

        # in case of remote series uri
        if 's3://mlflow-bucket/' in series_uri:
            series_uri = series_uri.replace("s3:/", S3_ENDPOINT_URL)
            download_file_path = download_online_file(
                series_uri, dst_filename="load.csv")
            series_uri = download_file_path
        series = load_local_csv_as_darts_timeseries(
            local_path=series_uri,
            name='series',
            time_col='Date',
            last_date=None)

        if none_checker(future_covariates_uri) is not None:
            future_covariates = darts.TimeSeries.from_csv(
                future_covariates_uri, time_col='Date')
        else:
            future_covariates = None
        if none_checker(past_covariates_uri) is not None:
            past_covariates = darts.TimeSeries.from_csv(
                past_covariates_uri, time_col='Date')
        else:
            past_covariates = None

        input = {
            "n": forecast_horizon,
            "history": series,
            "roll_size": roll_size,
            "future_covariates": future_covariates,
            "past_covariates": past_covariates,
            "batch_size": batch_size,
        }
        print("\nPyfunc model prediction...")

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(pyfunc_model_folder)

        # Predict on a Pandas DataFrame.
        predictions = loaded_model.predict(input)
        print(predictions)

        infertmpdir = tempfile.mkdtemp()
        predictions.to_csv(os.path.join(infertmpdir, 'predictions.csv'))
        mlflow.log_artifacts(infertmpdir)


#TODO: Input should be newly ingested time series data passing through the load_raw data step and the etl step. How can this be done?
# Maybe I need a second pipeline (inference pipeline) that goes like that: load_raw_data -> etl -> inference for a specific registered MLflow model"""
if __name__ == '__main__':
    print("\n=========== INFERENCE =============")
    logging.info("\n=========== INFERENCE =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    MLflowDartsModelPredict()
