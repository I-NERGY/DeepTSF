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
import numpy as np
import csv
from datetime import datetime
from utils import download_online_file
import shutil
import pretty_errors
import uuid
from exceptions import WrongIDs, EmptyDataframe, DifferentComponentDimensions, WrongColumnNames, DatesNotInOrder
from utils import truth_checker
import tempfile

# get environment variables
from dotenv import load_dotenv
load_dotenv()

# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

MONGO_URL = os.environ.get("MONGO_URL")


def read_and_validate_input(series_csv: str = "../../RDN/Load_Data/2009-2019-global-load.csv",
                            day_first: bool = True,
                            multiple: bool = False,
                            resolution: int = 15,
                            from_mongo: bool = False):
    """
    Validates the input after read_csv is called and
    throws apropriate exception if it detects an error

    Multiple timeseries file format (along with example values):

    Index | Day         | ID | Source  | Source Code  | Timeseries ID | 00:00:00 | 00:00:00 + resolution | ... | 24:00:00 - resolution
    0     | 2015-04-09  | 0  | Portugal| PT           |  0            | 5248     | 5109                  | ... | 5345
    1     | 2015-04-09  | 1  | Spain   | ES           |  1            | 25497    | 23492                 | ... | 25487
    .
    .

    Each ID corresponds to a component of a timeseries in the file. Timeseries ID column is not complulsory, 
    and shows the timeseries to which each component belongs. If Timeseries ID is not present, it is assumed that 
    each component represents one seperate series. Columns can be in any order and ID must be unique for every 
    series component in the file. Also, all the above column names must be present in the file, and the hour 
    columns must be consequtive and separated by resolution minutes. However they can be in any order and still 
    be considered valid. The lines can also be at any order as long as the Day column is increasing for each source. 

    Parameters
    ----------
    series_csv
        The path to the local file of the series to be validated
    day_first
        Whether to read the csv assuming day comes before the month
    multiple
        Whether to train on multiple timeseries
    resolution
        The resolution of the dataset

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe from series_csv
    """
    ts = pd.read_csv(series_csv,
                     sep=None,
                     header=0,
                     index_col=0,
                     parse_dates=(['Day'] if multiple else True),
                     dayfirst=day_first,
                     engine='python')
    #print(ts)
    #ts.to_csv("temp")
    #if ts is empty, raise exception
    if ts.empty:
        raise EmptyDataframe(from_mongo)
    if not multiple:
        #Check that dates are in order. If month is used before day and day_first is set to True, this is not the case.
        if not ts.index.sort_values().equals(ts.index):
            raise DatesNotInOrder()
        #Check that column Date is used as index, and that Load is the only other column in the csv
        elif not (len(ts.columns) == 1 and ts.columns[0] == 'Load' and ts.index.name == 'Date'):
            raise WrongColumnNames([ts.index.name] + list(ts.columns), 2, ['Load', 'Date'])
    else:
        if "Timeseries ID" not in ts.columns:
            ts["Timeseries ID"] = ts["ID"]
        if "Source Code" not in ts.columns:
            ts["Source Code"] = ts["ID"]
        if "Source" not in ts.columns:
            ts["Source"] = ts["ID"]
        des_columns = list(map(str, ['Day', 'ID', 'Source', 'Source Code', 'Timeseries ID'] + [(pd.Timestamp("00:00:00") + i*pd.DateOffset(minutes=resolution)).time() for i in range(60*24//resolution)]))
        #Check that all columns 'Day', 'ID', 'Source', 'Source Code' and the time columns exist in any order.
        if not set(des_columns) == set(list(ts.columns)):
            raise WrongColumnNames(list(ts.columns), len(des_columns), des_columns)
        #Check that all dates for each source are sorted
        for id in np.unique(ts["ID"]):
            if not ts.loc[ts["ID"] == id]["Day"].sort_values().equals(ts.loc[ts["ID"] == id]["Day"]):
                #(ts.loc[ts["ID"] == id]["Day"].sort_values()).to_csv("test.csv")
                raise DatesNotInOrder(id)
        #Check that all timeseries in a multiple timeseries file have the same number of components
        if len(set(len(np.unique(ts.loc[ts["Timeseries ID"] == ts_id]["ID"])) for ts_id in np.unique(ts["Timeseries ID"]))) != 1:
            raise DifferentComponentDimensions()
            
        
    return ts

from pymongo import MongoClient
import pandas as pd

client = MongoClient(MONGO_URL)


def unfold_timeseries(lds):
    new_loads = {'Date': [], 'Load': []}
    prev_date = ''
    #print(lds)
    for l in reversed(list(lds)):
        if prev_date != l['date']:
            for key in l:
                if key != '_id' and key != 'date':
                    new_date = l['date'] + ' ' + key
                    new_loads['Date'].append(new_date)
                    new_loads['Load'].append(l[key])
        prev_date = l['date']
    return new_loads


def load_data_to_csv(tmpdir, mongo_name):
    db = client['inergy_prod_db']
    if mongo_name == "asm_historical_smart_meters_uc6":
        df = list(map(lambda x: pd.DataFrame(x.find()).drop(columns={'_id', ''}, errors='ignore'), 
                 [db[mongo_name + "_current"], db[mongo_name + "_voltage"], db[mongo_name + "_power"]]))
    else:
        collection = db[mongo_name]
        df = pd.DataFrame(collection.find()).drop(columns={'_id', ''}, errors='ignore')
    if mongo_name == "asm_historical_smart_meters_uc7":
        #Rename to ts id
        df["Source"] = df["id"] + " " + df["energy_type"]
        cols_to_drop = {'date', 'id', 'energy_type'}
    elif mongo_name == "asm_historical_smart_meters_uc6":
        df[2] = df[2].loc[df[2]["id"] != "W1"]
        df[0]["Source"] = df[0]["id"] + " " + df[0]["phase"]
        df[1]["Source"] = df[1]["id"] + " " + df[1]["voltage_type"]
        df[2]["Source"] = df[2]["id"] + " " + df[2]["power_type"]
        cols_to_drop = {'date', 'id', 'power_type', 'voltage_type', 'phase'}
        df = pd.concat(df) 

    elif mongo_name == "asm_historical_smart_meters_uc6_current":
        df["Source"] = df["id"] + " " + df["phase"]
        cols_to_drop = {'date', 'id', 'phase'}
    elif mongo_name == "asm_historical_smart_meters_uc6_voltage":
        df["Source"] = df["id"] + " " + df["voltage_type"]
        cols_to_drop = {'date', 'id', 'voltage_type'}
    elif mongo_name == "asm_historical_smart_meters_uc6_power":
        #df.index = list(range(len(df)))
        df["Source"] = df["id"] + " " + df["power_type"]
        cols_to_drop = {'date', 'id', 'power_type'}
    else:
        df = unfold_timeseries(collection.find().sort('_id', -1))
        df = pd.DataFrame.from_dict(new_loads)
        df.to_csv(f'{tmpdir}/load.csv', index=False)
        client.close()
        return
    df["Source Code"] = df["Source"]
    
    unique_ts = pd.unique(df["Source"])
    name_to_ID = {}
    for i in range(len(unique_ts)):
        name_to_ID[unique_ts[i]] = i
    df["ID"] = df["Source"].apply(lambda x: name_to_ID[x])

    if mongo_name == "asm_historical_smart_meters_uc6":
        unique_ts = pd.unique(df["id"])
        name_to_ID = {}
        for i in range(len(unique_ts)):
            name_to_ID[unique_ts[i]] = i
        df["Timeseries ID"] = df["id"].apply(lambda x: name_to_ID[x])

    df["Day"] = df["date"]
    #print("1", df)
    df = df.drop_duplicates(subset=["Day", "ID"]).\
            sort_values(by=["Day", "ID"], ignore_index=True).\
            drop(columns=cols_to_drop)
    #print("2", df)
    df.to_csv(f'{tmpdir}/load.csv', index=True)
    client.close()
    return

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
@click.option("--resolution",
    default="15",
    type=str,
    help="The resolution of the dataset in minutes."
)
@click.option("--from-mongo",
    default="false",
    type=str,
    help="Whether to read the dataset from mongodb."
)
@click.option("--mongo-name",
    default="rdn_load_data",
    type=str,
    help="Which mongo file to read."
)

def load_raw_data(series_csv, series_uri, day_first, multiple, resolution, from_mongo, mongo_name):

    from_mongo = truth_checker(from_mongo)
    tmpdir = tempfile.mkdtemp()

    if from_mongo:
        load_data_to_csv(tmpdir, mongo_name)
        series_csv = f'{tmpdir}/load.csv'

    elif series_uri != "online_artifact":
        download_file_path = download_online_file(series_uri, dst_filename="series.csv")
        series_csv = download_file_path

    series_csv = series_csv.replace('/', os.path.sep)
    fname = series_csv.split(os.path.sep)[-1]
    local_path = series_csv.split(os.path.sep)[:-1]

    day_first = truth_checker(day_first)

    multiple = truth_checker(multiple)

    resolution = int(resolution)

    with mlflow.start_run(run_name='load_data', nested=True) as mlrun:

        print(f'Validating timeseries on local file: {series_csv}')
        logging.info(f'Validating timeseries on local file: {series_csv}')
        ts = read_and_validate_input(series_csv, day_first, multiple=multiple, resolution=resolution, from_mongo=from_mongo)


        local_path = local_path.replace("'", "") if "'" in local_path else local_path
        series_filename = os.path.join(*local_path, fname)
        # series = pd.read_csv(series_filename,  index_col=0, parse_dates=True, squeeze=True)
        # darts_series = darts.TimeSeries.from_series(series, freq=f'{timestep}min')
        print(f'\nUploading timeseries to MLflow server: {series_filename}')
        logging.info(f'\nUploading timeseries to MLflow server: {series_filename}')

        ts_filename = os.path.join(tmpdir, fname)
        ts.to_csv(ts_filename, index=True)
        mlflow.log_artifact(ts_filename, "raw_data")

        ## TODO: Read from APi

        # set mlflow tags for next steps
        if multiple:
            mlflow.set_tag("dataset_start", datetime.strftime(ts["Day"].iloc[0], "%Y%m%d"))
            mlflow.set_tag("dataset_end", datetime.strftime(ts["Day"].iloc[-1], "%Y%m%d"))
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
