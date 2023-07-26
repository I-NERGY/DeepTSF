"""
Downloads the RDN dataset and saves it as an artifact. ALso need to include interaction with weather apis here.
"""
import requests
import tempfile
import os
import mlflow
import click
import sys
sys.path.append('..')

from utils import ConfigParser
import logging
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from utils import download_online_file, multiple_ts_file_to_dfs, multiple_dfs_to_ts_file
import shutil
import pretty_errors
import uuid
from exceptions import WrongIDs, EmptyDataframe, DifferentComponentDimensions, WrongColumnNames, DatesNotInOrder
from utils import truth_checker, none_checker
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
                            from_mongo: bool = False,
                            covariates: str = "series"):
    """
    Validates the input after read_csv is called and throws apropriate exception if it detects an error.
    
    The checks that are performed are the following:

    For all timeseries:
        - The dataframe can not be empty
        - All the dates must be sorted

    For non multiple timeseries:
        - Column Date must be used as index 
        - If the timeseries is the main dataset, Load must be the only other column in the dataframe
        - If the timeseries is a covariates timeseries, there must be only one column in the dataframe
          named arbitrarily

    For multiple timeseries:
        - Columns 'Day', 'ID', and the time columns exist in any order
        - Only the permitted column names exist in the dataframe (see Multiple timeseries file format bellow)
        - All timeseries in the dataframe have the same number of components

    In case of a multiple timeseries, its resolution is also infered, and stored in mlflow as a tag. If we
    have a single timeseries, then mlflow stores the resolution given by the user

    Multiple timeseries file format (along with example values):

    Index | Day         | ID | Source  | Source Code  | Timeseries ID | 00:00:00 | 00:00:00 + resolution | ... | 24:00:00 - resolution
    0     | 2015-04-09  | 0  | Portugal| PT           |  0            | 5248     | 5109                  | ... | 5345
    1     | 2015-04-09  | 1  | Spain   | ES           |  1            | 25497    | 23492                 | ... | 25487
    .
    .
    The columns that can be present in the csv have the following meaning:
        - Index: Simply a monotonic integer range
        - Day: The Day each row is referring to
        - ID: Each ID corresponds to a component of a timeseries in the file. 
              This ID must be unique for each timeseries component in the file.
        - Source (Optional): The Source of the timeseries. If referring to country
              loads it can be the country name. If it is not provided, it is set
              to ID.
        - Source Code (Optional): The Source Code of the timeseries. If referring to country
              loads it can be the country code. In this case, this will be used to obtain 
              the country holidays for the imputation function as well as the time covariates.
              If it is not provided, it is set to ID.
        - Timeseries ID (Optional): Timeseries ID column is not compulsory, and shows the 
              timeseries to which each component belongs. If Timeseries ID is not present, it is 
              assumed that each component represents one seperate series (the column is set to ID).
        - Time columns: Columns that store the Load of each compontent. They must be consequtive and 
              separated by resolution minutes. They should start at 00:00:00, and end at 24:00:00 - 
              resolution
        
    Columns can be in any order and the file will be considered valid.

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
    from_mongo
        Whether the dataset was from MongoDB
    covariates
        If the function is called for the main dataset, then this equal to "series".
        If it is called for the past / future covariate series, then it is equal to
        "past" / "future" respectively. 

    Returns
    -------
    (pandas.DataFrame, int)
        A tuple consisting of the resulting dataframe from series_csv as well as the resolution
    """

    ts = pd.read_csv(series_csv,
                     sep=None,
                     header=0,
                     index_col=0,
                     parse_dates=(['Day'] if multiple else True),
                     dayfirst=day_first,
                     engine='python')
    
    #Dataframe can not be empty
    if ts.empty:
        raise EmptyDataframe(from_mongo)
    
    if not multiple:
        #Check that dates are in order. If month is used before day and day_first is set to True, this is not the case.
        if not ts.index.sort_values().equals(ts.index):
            raise DatesNotInOrder()
        #Check that column Date is used as index, and that Load is the only other column in the csv for the series csv
        elif covariates == "series" and not (len(ts.columns) == 1 and ts.columns[0] == 'Load' and ts.index.name == 'Date'):
            raise WrongColumnNames([ts.index.name] + list(ts.columns), 2, ['Date', 'Load'])
        #Check that column Date is used as index, and that there is only other column in the csv for the covariates csvs
        elif covariates != "series" and not (len(ts.columns) == 1 and ts.index.name == 'Date'):
            raise WrongColumnNames([ts.index.name] + list(ts.columns), 2, ['Date', '<Value Column Name>'])
    else:
        #If columns don't exist set defaults
        if "Timeseries ID" not in ts.columns:
            ts["Timeseries ID"] = ts["ID"]
        if "Source Code" not in ts.columns:
            ts["Source Code"] = ts["ID"]
        if "Source" not in ts.columns:
            ts["Source"] = ts["ID"]

        #Infering resolution for multiple timeseries
        times = []
        for elem in list(ts.columns):
            try:
                times.append(pd.Timestamp(elem))
            except:
                pass
        times.sort()
        resolution = (times[1] - times[0]).seconds // 60

        des_columns = list(map(str, ['Day', 'ID', 'Source', 'Source Code', 'Timeseries ID'] + [(pd.Timestamp("00:00:00") + i*pd.DateOffset(minutes=resolution)).time() for i in range(60*24//resolution)]))
        #Check that all columns 'Day', 'ID', 'Source', 'Source Code' 'Timeseries ID' and the time columns exist in any order.
        if not set(des_columns) == set(list(ts.columns)):
            raise WrongColumnNames(list(ts.columns), len(des_columns), des_columns)
        #Check that all dates for each source component are sorted
        for id in np.unique(ts["ID"]):
            if not ts.loc[ts["ID"] == id]["Day"].sort_values().equals(ts.loc[ts["ID"] == id]["Day"]):
                #(ts.loc[ts["ID"] == id]["Day"].sort_values()).to_csv("test.csv")
                raise DatesNotInOrder(id)
        
        #Check that all timeseries in a multiple timeseries file have the same number of components
        if len(set(len(np.unique(ts.loc[ts["Timeseries ID"] == ts_id]["ID"])) for ts_id in np.unique(ts["Timeseries ID"]))) != 1:
            raise DifferentComponentDimensions()

    mlflow.set_tag(f'infered_resolution_{covariates}', resolution)
            
        
    return ts, resolution

def make_multiple(ts_covs, series_csv, day_first, inf_resolution):
    """
    In case covariates.

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
    from_mongo
        Whether the dataset was from MongoDB
    covariates
        If the function is called for the main dataset, then this equal to "series".
        If it is called for the past / future covariate series, then it is equal to
        "past" / "future" respectively. 

    Returns
    -------
    (pandas.DataFrame, int)
        A tuple consisting of the resulting dataframe from series_csv as well as the resolution
    """


    if series_csv != None:
        ts_list, _, _, _, ts_id_l = multiple_ts_file_to_dfs(series_csv, day_first, inf_resolution)

        ts_list_covs = [[ts_covs] for _ in range(len(ts_list))]
        id_l_covs = [[str(list(ts_covs.columns)[0]) + "_" + ts_id_l[i]] for i in range(len(ts_list))]
    else:
        ts_list_covs = [[ts_covs]]
        id_l_covs = [[list(ts_covs.columns)[0]]]
    return multiple_dfs_to_ts_file(ts_list_covs, id_l_covs, id_l_covs, id_l_covs, id_l_covs, "", save=False)


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
    collection = db[mongo_name]
    df = pd.DataFrame(collection.find()).drop(columns={'_id', ''}, errors='ignore')
    df = unfold_timeseries(collection.find().sort('_id', -1))
    df = pd.DataFrame.from_dict(df)
    df.to_csv(f'{tmpdir}/load.csv', index=False)
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
@click.option("--past-covs-csv",
    type=str,
    default="None",
    help="Local past covaraites csv file"
    )
@click.option("--past-covs-uri",
    default="None",
    help="Remote past covariates csv file. If set, it overwrites the local value."
    )
@click.option("--future-covs-csv",
    type=str,
    default="None",
    help="Local future covaraites csv file"
    )
@click.option("--future-covs-uri",
    default="None",
    help="Remote future covariates csv file. If set, it overwrites the local value."
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

def load_raw_data(series_csv, series_uri, past_covs_csv, past_covs_uri, future_covs_csv, future_covs_uri, day_first, multiple, resolution, from_mongo, mongo_name):

    from_mongo = truth_checker(from_mongo)
    tmpdir = tempfile.mkdtemp()

    past_covs_csv = none_checker(past_covs_csv)
    past_covs_uri = none_checker(past_covs_uri)
    future_covs_csv = none_checker(future_covs_csv)
    future_covs_uri = none_checker(future_covs_uri)

    if from_mongo:
        load_data_to_csv(tmpdir, mongo_name)
        series_csv = f'{tmpdir}/load.csv'

    elif series_uri != "online_artifact":
        download_file_path = download_online_file(series_uri, dst_filename="series.csv")
        series_csv = download_file_path

    if past_covs_uri != None:
        download_file_path = download_online_file(past_covs_uri, dst_filename="past_covs.csv")
        past_covs_csv = download_file_path

    if future_covs_uri != None:
        download_file_path = download_online_file(future_covs_uri, dst_filename="future_covs.csv")
        future_covs_csv = download_file_path

    series_csv = series_csv.replace('/', os.path.sep)
    fname = series_csv.split(os.path.sep)[-1]
    local_path = series_csv.split(os.path.sep)[:-1]

    day_first = truth_checker(day_first)

    multiple = truth_checker(multiple)

    resolution = int(resolution)

    with mlflow.start_run(run_name='load_data', nested=True) as mlrun:

        ts, _ = read_and_validate_input(series_csv, day_first, multiple=multiple, resolution=resolution, from_mongo=from_mongo)

        print(f'Validating timeseries on local file: {series_csv}')
        logging.info(f'Validating timeseries on local file: {series_csv}')

        local_path = local_path.replace("'", "") if "'" in local_path else local_path
        series_filename = os.path.join(*local_path, fname)
        # series = pd.read_csv(series_filename,  index_col=0, parse_dates=True, squeeze=True)
        # darts_series = darts.TimeSeries.from_series(series, freq=f'{timestep}min')
        print(f'\nUploading timeseries to MLflow server: {series_filename}')
        logging.info(f'\nUploading timeseries to MLflow server: {series_filename}')

        ts_filename = os.path.join(tmpdir, fname)
        ts.to_csv(ts_filename, index=True)
        mlflow.log_artifact(ts_filename, "raw_data")

        if past_covs_csv != None:
            past_covs_csv = past_covs_csv.replace('/', os.path.sep)
            past_covs_fname = past_covs_csv.split(os.path.sep)[-1]
            local_path_past_covs = past_covs_csv.split(os.path.sep)[:-1]

            if multiple:
                try:
                    ts_past_covs, _ = read_and_validate_input(past_covs_csv,
                                                              day_first,
                                                              multiple=True,
                                                              resolution=resolution,
                                                              from_mongo=from_mongo,
                                                              covariates="past")
                except:
                    ts_past_covs, inf_resolution = read_and_validate_input(past_covs_csv,
                                                                           day_first,
                                                                           multiple=False,
                                                                           resolution=resolution,
                                                                           from_mongo=from_mongo,
                                                                           covariates="past")
                    ts_past_covs = make_multiple(ts_past_covs,
                                                 series_csv,
                                                 day_first,
                                                 str(inf_resolution))
                    
            else:
                ts_past_covs, _ = read_and_validate_input(past_covs_csv,
                                                       day_first,
                                                       multiple=multiple,
                                                       resolution=resolution,
                                                       from_mongo=from_mongo,
                                                       covariates="past")
                
                #TODO Infer resolution on single timeseries also
                ts_past_covs = make_multiple(ts_past_covs, 
                                             None, 
                                             day_first, 
                                             resolution)
                
            local_path_past_covs = local_path_past_covs.replace("'", "") if "'" in local_path_past_covs else local_path_past_covs
            past_covs_filename = os.path.join(*local_path_past_covs, past_covs_fname)

            print(f'\nUploading past covariates timeseries to MLflow server: {past_covs_filename}')
            logging.info(f'\nUploading past covariates timeseries to MLflow server: {past_covs_filename}')


            ts_past_covs_filename = os.path.join(tmpdir, past_covs_fname)
            ts_past_covs.to_csv(ts_past_covs_filename, index=True)
            mlflow.log_artifact(ts_past_covs_filename, "past_covs_data")
            mlflow.set_tag('past_covs_uri', f'{mlrun.info.artifact_uri}/past_covs_data/{past_covs_fname}')
        else:
            mlflow.set_tag(f'infered_resolution_past', "None")
            mlflow.set_tag('past_covs_uri', "None")

        if future_covs_csv != None:
            future_covs_csv = future_covs_csv.replace('/', os.path.sep)
            future_covs_fname = future_covs_csv.split(os.path.sep)[-1]
            local_path_future_covs = future_covs_csv.split(os.path.sep)[:-1]

            if multiple:
                try:
                    ts_future_covs, _ = read_and_validate_input(future_covs_csv,
                                                              day_first,
                                                              multiple=True,
                                                              resolution=resolution,
                                                              from_mongo=from_mongo,
                                                              covariates="future")
                except:
                    ts_future_covs, inf_resolution = read_and_validate_input(future_covs_csv,
                                                                           day_first,
                                                                           multiple=False,
                                                                           resolution=resolution,
                                                                           from_mongo=from_mongo,
                                                                           covariates="future")
                    ts_future_covs = make_multiple(ts_future_covs,
                                                 series_csv,
                                                 day_first,
                                                 str(inf_resolution))
                    
            else:
                ts_future_covs, _ = read_and_validate_input(future_covs_csv,
                                                       day_first,
                                                       multiple=multiple,
                                                       resolution=resolution,
                                                       from_mongo=from_mongo,
                                                       covariates="future")
                
                ts_future_covs = make_multiple(ts_future_covs, 
                                             None, 
                                             day_first, 
                                             resolution)
                
            local_path_future_covs = local_path_future_covs.replace("'", "") if "'" in local_path_future_covs else local_path_future_covs
            future_covs_filename = os.path.join(*local_path_future_covs, future_covs_fname)

            print(f'\nUploading future covariates timeseries to MLflow server: {future_covs_filename}')
            logging.info(f'\nUploading future covariates timeseries to MLflow server: {future_covs_filename}')


            ts_future_covs_filename = os.path.join(tmpdir, future_covs_fname)
            ts_future_covs.to_csv(ts_future_covs_filename, index=True)
            mlflow.log_artifact(ts_future_covs_filename, "future_covs_data")
            mlflow.set_tag('future_covs_uri', f'{mlrun.info.artifact_uri}/future_covs_data/{future_covs_fname}')
        else:
            mlflow.set_tag(f'infered_resolution_future', "None")
            mlflow.set_tag('future_covs_uri', "None")

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
