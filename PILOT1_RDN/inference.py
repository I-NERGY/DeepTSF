from sys import version_info
import cloudpickle
from utils import load_model, load_scaler, none_checker, load_local_csv_as_darts_timeseries, download_online_file
import click
import os
import mlflow, darts
import logging
import numpy as np

# get environment variables
from dotenv import load_dotenv
load_dotenv()
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

mlflow_serve_conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
                'cloudpickle=={}'.format(cloudpickle.__version__),
                'logging=={}'.format(logging.__version__),
                'mlflow=={}'.format(mlflow.__version__),
                'darts=={}'.format(darts.__version__),
                'click=={}'.format(click.__version__),
                'python-dotenv',
            ],
        },
    ],
    'name': 'darts_infer_env'
}


class _MLflowDartsModelWrapper:
    def __init__(self, darts_model, transformer=None):
        self.model = darts_model
        self.transformer = transformer

    def predict(self, model_input):
        """ 
        :param model_input: Dict
        {"n": int, "history": darts.TimeSeries, "past_covariates": darts.TimeSeries, "future_covariates": darts.TimeSeries, "roll_size":int, "batch_size": int}
        """
        # Does it work?
        n = model_input["n"] if model_input["n"] is not None else self.model.output_chunk_length
        roll_size = model_input["roll_size"] if model_input["roll_size"] is not None else self.model.output_chunk_length
        history = model_input["history"]
        past_covariates = model_input["past_covariates"]
        future_covariates = model_input["future_covariates"]
        batch_size = model_input["batch_size"] if model_input["batch_size"] is not None else 1

        # Transform to darts
        history = history
        if past_covariates is not None:
            past_covariates = past_covariates
        if future_covariates is not None:
            future_covariates = future_covariates

        # Transform
        if self.transformer is not None:
            history = self.transformer.transform(history)

        # Predict
        predictions = self.model.predict(
            n=n,
            roll_size=roll_size,
            series=history,
            future_covariates=future_covariates,
            past_covariates=past_covariates,
            batch_size=batch_size)

        # Untransform
        if self.transformer is not None:
            predictions = self.transformer.inverse_transform(predictions)

        # Return as DataFrame
            return predictions.pd_dataframe()

def _load_pyfunc(model_file):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    ## load model from MLflow or local folder
    
    print(f"Inside _load_pyfunc: {model_file}")
    model = load_model(model_uri=model_file, mode="local", model_type="pl")
    scaler = load_scaler(scaler_uri=f"{model_file}/scaler_series.pkl", mode="local")

    return _MLflowDartsModelWrapper(model, scaler)


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
    # Parse arguments
    batch_size = int(batch_size)
    roll_size = int(roll_size)
    forecast_horizon = int(forecast_horizon)

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
    # local_dir = tempfile.mkdtemp()
    # client = mlflow.tracking.MlflowClient()
    # client.download_artifacts(
    #     run_id=train_run_id,
    #     dst_path=local_dir
    # )
    # logged_model = os.path.join(
    #     local_dir, 'pyfunc_model')
    # features = os.path.join(
    #     local_dir, 'features')
    # series = darts.TimeSeries.from_csv(f'{features}/series.csv')
    # logged_model = 's3://mlflow-bucket/2/9cb056d7c21844b482462bfe941ed9b7/artifacts/pyfunc_model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(pyfunc_model_folder)

    # Predict on a Pandas DataFrame.
    print(loaded_model.predict(input))

#TODO: Input should be newly ingested time series data passing through the load_raw data step and the etl step. How can this be done?
# Maybe I need a second pipeline (inference pipeline) that goes like that: load_raw_data -> etl -> inference for a specific registered MLflow model"""
if __name__ == '__main__':
    print("\n=========== INFERENCE =============")
    logging.info("\n=========== INFERENCE =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    MLflowDartsModelPredict()
