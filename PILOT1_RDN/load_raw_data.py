"""
Downloads the RDN dataset and saves it as an artifact. ALso need to include interaction with weather apis here.
"""
import requests
import tempfile
import os
import zipfile
import mlflow
import click
from utils import read_config
import logging
import pandas as pd
import darts

# get environment variables
from dotenv import load_dotenv
load_dotenv()

# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
os.environ["MLFLOW_TRACKING_URI"] = read_config('config.yml', 'mlflow_settings')['mlflow_tracking_uri']
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

@click.command(
    help="Downloads the RDN series and saves it as an mlflow artifact "
    "called 'load_x_y.csv'."
)
@click.option("--local-path", 
    type=str, 
    default="../../RDN/Load_Data",
    help="Local folder to draw the file from")
@click.option("--fname", 
    type=str, 
    default="load_2018_2021.csv",
    help="Name of the local file inside the local folder")
@click.option("--url", default=None)
def load_raw_data(local_path, fname, url):
    local_path = local_path.replace('/', os.path.sep)
    with mlflow.start_run() as mlrun:
        if url is None or url == "None":
            local_path = local_path.replace("'", "") if "'" in local_path else local_path
            series_filename = os.path.join(local_path, fname)
            print(series_filename)
            # series = pd.read_csv(series_filename,  index_col=0, parse_dates=True, squeeze=True)
            # darts_series = darts.TimeSeries.from_series(series, freq=f'{timestep}min')
            print(f'Uploading timeseries: {series_filename}')
            mlflow.log_artifact(series_filename, fname)
        else:
            pass
        ## TODO: Read from APi


# check for stream to csv: https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/load_raw_data.py

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    logging.info("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    load_raw_data()
