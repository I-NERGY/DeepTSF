"""
Downloads the RDN dataset and saves it as an artifact. ALso need to include interaction with weather apis here.
"""
import requests
import tempfile
import os
import mlflow
import click
from utils import ConfigParser
import logging
import pandas as pd
import csv
from utils import download_online_file
import shutil
import pretty_errors
import uuid

# get environment variables
from dotenv import load_dotenv
load_dotenv()

# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

@click.command(
    help="Downloads the RDN series and saves it as an mlflow artifact "
    "called 'load_x_y.csv'."
    )
# TODO: Update that to accept url as input instead of local file
@click.option("--series-csv", 
    type=str, 
    default="../../RDN/Load_Data/2018-2021-global-load.csv",
    help="Local time series csv file"
    )
@click.option("--series-uri", 
    default="online_artifact",
    help="Remote time series csv file. If set, it overwrites the local value."
    )
def load_raw_data(series_csv, series_uri):

    if series_uri != "online_artifact":
        download_file_path = download_online_file(series_uri, dst_filename="series.csv")
        series_csv = download_file_path
    
    series_csv = series_csv.replace('/', os.path.sep)
    fname = series_csv.split(os.path.sep)[-1]
    local_path = series_csv.split(os.path.sep)[:-1]
    
    with mlflow.start_run(run_name='load_data', nested=True) as mlrun:
        
        local_path = local_path.replace("'", "") if "'" in local_path else local_path
        series_filename = os.path.join(*local_path, fname)
        # series = pd.read_csv(series_filename,  index_col=0, parse_dates=True, squeeze=True)
        # darts_series = darts.TimeSeries.from_series(series, freq=f'{timestep}min')
        print(f'\nUploading timeseries to MLflow server: {series_filename}')
        logging.info(f'\nUploading timeseries to MLflow server: {series_filename}')
        mlflow.log_artifact(series_filename, "raw_data")

        ## TODO: Read from APi

        # set mlflow tags for next steps
        mlflow.set_tag("run_id", mlrun.info.run_id)
        mlflow.set_tag('dataset_uri', f'{mlrun.info.artifact_uri}/raw_data/{fname}')

        return


# check for stream to csv: https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/load_raw_data.py

if __name__ == "__main__":
    print("\n=========== LOAD DATA =============")
    logging.info("\n=========== LOAD DATA =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
    logging.info("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
    load_raw_data()
