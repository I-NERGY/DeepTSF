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
from datetime import datetime
from utils import download_online_file
import shutil
import pretty_errors
import uuid
from exceptions import DatesNotInOrder
from exceptions import WrongColumnNames
from utils import truth_checker
import tempfile

# get environment variables
from dotenv import load_dotenv
load_dotenv()

# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

def read_and_validate_input(series_csv: str = "../../RDN/Load_Data/2009-2019-global-load.csv",
                            day_first: bool = True,
                            multiple: bool = False):
    """
    Validates the input after read_csv is called and
    throws apropriate exception if it detects an error

    Parameters
    ----------
    series_csv
        The path to the local file of the series to be validated
    day_first
        Whether to read the csv assuming day comes before the month
    multiple
        Whether to train on multiple timeseries

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe from series_csv
    """
    ts = pd.read_csv(series_csv,
                     sep=None,
                     header=0,
                     index_col=0,
                     parse_dates=True,
                     dayfirst=day_first,
                     engine='python')
    if not multiple:
        if not ts.index.sort_values().equals(ts.index):
            raise DatesNotInOrder()
        elif not (len(ts.columns) == 1 and ts.columns[0] == 'Load' and ts.index.name == 'Date'):
            raise WrongColumnNames(list(ts.columns) + [ts.index.name])
    else:
        pass
    return ts

@click.command(
    help="Downloads the RDN series and saves it as an mlflow artifact "
    "called 'load_x_y.csv'."
    )
# TODO: Update that to accept url as input instead of local file
@click.option("--series-csv",
    type=str,
    default="../../RDN/Load_Data/2009-2019-global-load.csv",
    help="Local time series csv file"
    )
@click.option("--series-uri",
    default="online_artifact",
    help="Remote time series csv file. If set, it overwrites the local value."
    )
@click.option("--day-first",
    type=str,
    default="true",
    help="Whether the date has the day before the month")
@click.option("--multiple",
    type=str,
    default="false",
    help="Whether to train on multiple timeseries")

def load_raw_data(series_csv, series_uri, day_first, multiple):

    if series_uri != "online_artifact":
        download_file_path = download_online_file(series_uri, dst_filename="series.csv")
        series_csv = download_file_path

    series_csv = series_csv.replace('/', os.path.sep)
    fname = series_csv.split(os.path.sep)[-1]
    local_path = series_csv.split(os.path.sep)[:-1]

    day_first = truth_checker(day_first)

    multiple = truth_checker(multiple)

    with mlflow.start_run(run_name='load_data', nested=True) as mlrun:

        print(f'Validating timeseries on local file: {series_csv}')
        logging.info(f'Validating timeseries on local file: {series_csv}')
        ts = read_and_validate_input(series_csv, day_first, multiple=multiple)


        local_path = local_path.replace("'", "") if "'" in local_path else local_path
        series_filename = os.path.join(*local_path, fname)
        # series = pd.read_csv(series_filename,  index_col=0, parse_dates=True, squeeze=True)
        # darts_series = darts.TimeSeries.from_series(series, freq=f'{timestep}min')
        print(f'\nUploading timeseries to MLflow server: {series_filename}')
        logging.info(f'\nUploading timeseries to MLflow server: {series_filename}')

        tmpdir = tempfile.mkdtemp()
        ts_filename = os.path.join(tmpdir, fname)
        ts.to_csv(ts_filename, index=True)
        mlflow.log_artifact(ts_filename, "raw_data")

        ## TODO: Read from APi

        # set mlflow tags for next steps
        ##TODO fix this
        if multiple:
#            mlflow.set_tag("dataset_start", datetime.strftime(ts["Day"][0], "%Y%m%d"))
#            mlflow.set_tag("dataset_end", datetime.strftime(ts["Day"][-1], "%Y%m%d"))
            pass
        else:
            mlflow.set_tag("dataset_start", datetime.strftime(ts.index[0], "%Y%m%d"))
            mlflow.set_tag("dataset_end", datetime.strftime(ts.index[-1], "%Y%m%d"))
        mlflow.set_tag("run_id", mlrun.info.run_id)

        mlflow.set_tag("stage", "load_raw_data")
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
