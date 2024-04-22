from enum import Enum
import uvicorn
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Depends
import pandas as pd
import mlflow
from utils import ConfigParser
import tempfile
from uc2.load_raw_data import read_and_validate_input
from exceptions import DatetimesNotInOrder, WrongColumnNames
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient
from utils import load_artifacts, to_seconds, change_form, make_time_list
import psutil, nvsmi
import os
from dotenv import load_dotenv
from fastapi import APIRouter
from app.auth import admin_validator, scientist_validator, engineer_validator, common_validator, oauth2_scheme
from app.config import settings
import pymongo
from pymongo import MongoClient
import datetime
from typing import Tuple
from math import nan
import bson

load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
user = os.environ.get('MONGO_USER')
password = os.environ.get('MONGO_PASS')
address = os.environ.get('MONGO_ADDRESS')
database = os.environ.get('MONGO_DB_NAME')
mongo_collection_uc7 = os.environ.get('MONGO_COLLECTION_UC7')
mongo_collection_uc2 = os.environ.get('MONGO_COLLECTION_UC2')
mongo_collection_uc6 = os.environ.get('MONGO_COLLECTION_UC6')
mongo_url = f"mongodb://{user}:{password}@{address}"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# allows automated type check with pydantic
# class ModelName(str, Enum):

tags_metadata = [
    {"name": "MLflow Info", "description": "REST APIs for retrieving elements from MLflow"},
    {"name": "Metrics and models retrieval", "description": "REST APIs for retrieving available metrics, alongside models and their respective hyperparameters"},
    {"name": "Experimentation Pipeline", "description": "REST APIs for setting up and running the experimentation pipeline"},
    {"name": "Model Evaluation", "description": "REST APIs for retrieving model evaluation results"},
    {"name": "System Monitoring", "description": "REST APIs for monitoring the host machine of the API"},
    {"name": "MongoDB integration", "description": "REST APIs for retrieving datastes from the I-NERGY MongoDB"},
]

metrics = [
    {"metric_name": "mape", "search_term": "mape"},
    {"metric_name": "mase", "search_term": "mase"},
    {"metric_name": "mae", "search_term": "mae"},
    {"metric_name": "rmse", "search_term": "rmse"},
    {"metric_name": "smape", "search_term": "smape"},
    {"metric_name": "nrmse_max", "search_term": "nrmse_max"},
    {"metric_name": "nrmse_mean", "search_term": "nrmse_mean"}]

class DateLimits(int, Enum):
    """This function will read the uploaded csv before running the pipeline and will decide which are the allowed values
    for: validation_start_date < test_start_date < test_end_date """
    @staticmethod
    def dict():
        return {"resolution": list(map(lambda c: c.value, ModelName))}

app = FastAPI(
    title="I-NERGY Load Forecasting Service API",
    description="Collection of REST APIs for Serving Execution of I-NERGY Load Forecasting Service",
    version="0.0.1",
    openapi_tags=tags_metadata,
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# creating routers
# admin validator passed as dependency
admin_router = APIRouter(
    dependencies=[Depends(admin_validator)]
)
# scientist validator passed as dependency
scientist_router = APIRouter(
    dependencies=[Depends(scientist_validator)]
)
engineer_router = APIRouter(
    dependencies=[Depends(engineer_validator)]
)
common_router = APIRouter(
    dependencies=[Depends(common_validator)]
)

if os.getenv("USE_KEYCLOAK", 'True') == 'False':
    admin_router.dependencies = []
    scientist_router.dependencies = []
    engineer_router.dependencies = []
    common_router.dependencies = []

# implement this method for login functionality
# @app.post('/token')
# def login(request: Request):
#     token = ''
#     return {"access_token": token, "token_type": "bearer"}
@scientist_router.get("/")
async def root():
    return {"message": "Congratulations! Your API is working as expected. Now head over to http://localhost:8080/docs"}


@scientist_router.get("/models/get_model_names/{resolution}/{multiple}", tags=['Metrics and models retrieval'])
async def get_model_names(resolution: str, multiple: bool):

    resolution = to_seconds(change_form(resolution, 'pandas_form'))

    default_input_chunk = int(60 * 60 / resolution * 168) if int(60 * 60 / resolution * 168) > 0 else 1
    default_output_chunk =  int(60 * 60 / resolution * 24) if int(60 * 60 / resolution * 24) > 0 else 1

    hparams_naive = [ 
        {"name": "days_seasonality", "type": "int", "description": "Period of sNaive model (in days)", 'min': 1, 'max': 366, 'default': 1}   
        ]

    hparams_nbeats = [    
        {"name": "input_chunk_length", "type": "int", "description": "Lookback window length", 'min': 1, 'max': 1000, 'default': default_input_chunk},
        {"name": "output_chunk_length", "type": "int", "description": "Forecast horizon length", 'min': 1, 'max': 1000, 'default': default_output_chunk},
        {"name": "num_stacks", "type": "int", "description": "Number of stacks", 'min': 1, 'max': 10, 'default': 2},
        {"name": "num_blocks", "type": "int", "description": "Number of blocks", 'min': 1, 'max': 10, 'default': 3},
        {"name": "num_layers", "type": "int", "description": "Number of layers", 'min': 1, 'max': 10, 'default': 1},
        {"name": "layer_widths", "type": "int", "description": "Width of layers", 'min': 1, 'max': 512, 'default': 64},
        {"name": "dropout", "type": "float", "description": "Fraction of neurons affected by dropout", 'min': 0, 'max': 1, 'default': 0.0},
        {"name": "n_epochs", "type": "int", "description": "Epochs threshold", 'min': 1, 'max': 1000, 'default': 300},
        {"name": "expansion_coefficient_dim", "type": "int", "description": "Dimension of expansion coefficient", 'min': 1, 'max': 10, 'default': 5},
        {"name": "random_state", "type": "int", "description": "Randomness of neural weight initialization", 'min': 0, 'max': 10000, 'default': 42},
        {"name": "batch_size", "type": "int", "description": "Batch size", 'min': 1, 'max': 1024, 'default': 16},
        ]

    hparams_nhits = [
        {"name": "input_chunk_length", "type": "int", "description": "Lookback window length", 'min': 1, 'max': 1000, 'default': 120, 'default': default_input_chunk},
        {"name": "output_chunk_length", "type": "int", "description": "Forecast horizon length", 'min': 1, 'max': 1000, 'default': 24, 'default': default_output_chunk},
        {"name": "num_stacks", "type": "int", "description": "Number of stacks", 'min': 1, 'max': 1000, 'default': 2},
        {"name": "num_blocks", "type": "int", "description": "Number of blocks", 'min': 1, 'max': 1000, 'default': 3},
        {"name": "num_layers", "type": "int", "description": "Number of layers", 'min': 1, 'max': 1000, 'default': 1},
        {"name": "layer_widths", "type": "int", "description": "Width of layers", 'min': 1, 'max': 1000, 'default': 64},
        {"name": "dropout", "type": "float", "description": "Fraction of neurons affected by dropout", 'min': 0, 'max': 1, 'default': 0.0},
        {"name": "n_epochs", "type": "int", "description": "Epochs threshold", 'min': 0, 'max': 1000, 'default': 300},
        {"name": "random_state", "type": "int", "description": "Randomness of neural weight initialization", 'min': 0, 'max': 10000, 'default': 42},
        {"name": "batch_size", "type": "int", "description": "Batch size", 'min': 1, 'max': 1024, 'default': 16},
    ]

    hparams_transformer = [   
        {"name": "input_chunk_length", "type": "int", "description": "Lookback window length", 'min': 1, 'max': 1000, 'default': default_input_chunk},
        {"name": "output_chunk_length", "type": "int", "description": "Forecast horizon length", 'min': 1, 'max': 1000, 'default': default_output_chunk},
        {"name": "d_model", "type": "int", "description": "Number of encoder/decoder features", 'min': 1, 'max': 128, 'default': 16},
        {"name": "nhead", "type": "int", "description": "Number of attention heads", 'min': 1, 'max': 6, 'default': 2},
        {"name": "num_encoder_layers", "type": "int", "description": "Number of encoder layers", 'min': 1, 'max': 20, 'default': 1},
        {"name": "num_decoder_layers", "type": "int", "description": "Number of decoder layers", 'min': 1, 'max': 20, 'default': 1},
        {"name": "dim_feedforward", "type": "int", "description": "Dimension of the feedforward network model", 'min': 1, 'max': 1024, 'default': 64},
        {"name": "n_epochs", "type": "int", "description": "Epochs threshold", 'min': 1, 'max': 1000, 'default': 500},
        {"name": "random_state", "type": "int", "description": "Randomness of neural weight initialization", 'min': 0, 'max': 10000, 'default': 42},
        {"name": "batch_size", "type": "int", "description": "Batch size", 'min': 1, 'max': 1024, 'default': 16},
    ]

    hparams_rnn = [
        {"name": "input_chunk_length", "type": "int", "description": "Lookback window length", 'min': 1, 'max': 1000, 'default': default_input_chunk},
        {"name": "output_chunk_length", "type": "int", "description": "Forecast horizon length", 'min': 1, 'max': 1000, 'default': default_output_chunk},
        {"name": "model", "type": "str", "description": "Number of recurrent layers", 'range': ['RNN', 'LSTM', 'GRU'], 'default': 'LSTM'},
        {"name": "n_rnn_layers", "type": "int", "description": "Number of recurrent layers", 'min': 1, 'max': 5, 'default': 1},
        {"name": "hidden_dim", "type": "int", "description": "Hidden dimension size within each RNN layer", 'min': 1, 'max': 512, 'default': 8},
        # {"name": "learning rate", "type": "float", "description": "Learning rate", 'min': 0.000000001, 'max': 1, 'default': 0.0008},
        # {"name": "training_length", "type": "int", "description": "Training length", 'min': 1, 'max': 1000},
        {"name": "dropout", "type": "float", "description": "Fraction of neurons affected by dropout", 'min': 0, 'max': 1, 'default': 0.0},
        {"name": "n_epochs", "type": "int", "description": "Epochs threshold", 'min': 0, 'max': 100, 'default': 700},
        {"name": "random_state", "type": "int", "description": "Randomness of neural weight initialization", 'min': 0, 'max': 10000, 'default': 42},
        {"name": "batch_size", "type": "int", "description": "Batch size", 'min': 1, 'max': 1024, 'default': 16},
        ]

    hparams_tft = [    
        {"name": "input_chunk_length", "type": "int", "description": "Lookback window length", 'min': 1, 'max': 1000, 'default': default_input_chunk},
        {"name": "output_chunk_length", "type": "int", "description": "Forecast horizon length", 'min': 1, 'max': 1000, 'default': default_output_chunk},
        {"name": "lstm_layers", "type": "int", "description": "Number of LSTM layers", 'min': 1, 'max': 5,  'default': 1},
        {"name": "num_attention_heads", "type": "int", "description": "Number of attention heads", 'min': 1, 'max': 6, 'default': 1},
        {"name": "dropout", "type": "float", "description": "Fraction of neurons affected by dropout", 'min': 0, 'max': 1, 'default': 0.0},
        {"name": "n_epochs", "type": "int", "description": "Epochs threshold", 'min': 0, 'max': 100, 'default': 700},
        {"name": "random_state", "type": "int", "description": "Randomness of neural weight initialization", 'min': 0, 'max': 10000, 'default': 42},
        {"name": "batch_size", "type": "int", "description": "Batch size", 'min': 1, 'max': 1024, 'default': 16},
        ]

    hparams_tcn = [
        {"name": "input_chunk_length", "type": "int", "description": "Lookback window length", 'min': 1, 'max': 1000, 'default': default_input_chunk},
        {"name": "output_chunk_length", "type": "int", "description": "Forecast horizon length", 'min': 1, 'max': 1000, 'default': default_output_chunk},
        {"name": "kernel_size", "type": "int", "description": "Number of recurrent layers", 'min': 1, 'max': 10, 'default': 3},
        {"name": "num_filters", "type": "int", "description": "Number of recurrent layers", 'min': 1, 'max': 1000, 'default': 3},
        {"name": "dilation_base", "type": "int", "description": "Number of recurrent layers", 'min': 1, 'max': 1000, 'default': 2},
        {"name": "dropout", "type": "float", "description": "Fraction of neurons affected by dropout", 'min': 0, 'max': 1, 'default': 0.0},
        {"name": "n_epochs", "type": "int", "description": "Epochs threshold", 'min': 0, 'max': 100, 'default': 500},
        {"name": "random_state", "type": "int", "description": "Randomness of neural weight initialization", 'min': 0, 'max': 10000, 'default': 42},
        {"name": "batch_size", "type": "int", "description": "Batch size", 'min': 1, 'max': 1024, 'default': 16},
        {"name": "weight_norm", "type": "bool", "description": "Weight normalization", 'default': True},
    ]

    hparams_blockrnn = [    
        {"name": "input_chunk_length", "type": "int", "description": "Lookback window length", 'min': 1, 'max': 1000, 'default': default_input_chunk},
        {"name": "output_chunk_length", "type": "int", "description": "Forecast horizon length", 'min': 1, 'max': 1000, 'default': default_output_chunk},
        {"name": "model", "type": "str", "description": "Number of recurrent layers", 'range': ['RNN', 'LSTM', 'GRU'], 'default': 'LSTM'},
        {"name": "n_rnn_layers", "type": "int", "description": "Number of recurrent layers", 'min': 1, 'max': 5, 'default': 1},
        {"name": "hidden_dim", "type": "int", "description": "Hidden dimension size within each RNN layer", 'min': 1, 'max': 512, 'default': 8},
        # {"name": "learning rate", "type": "float", "description": "Learning rate", 'min': 0.000000001, 'max': 1, 'default': 0.0008},
        {"name": "dropout", "type": "float", "description": "Fraction of neurons affected by dropout", 'min': 0, 'max': 1, 'default': 0.0},
        {"name": "n_epochs", "type": "int", "description": "Epochs threshold", 'min': 0, 'max': 100, 'default': 700},
        {"name": "random_state", "type": "int", "description": "Randomness of neural weight initialization", 'min': 0, 'max': 10000, 'default': 42},
        {"name": "batch_size", "type": "int", "description": "Batch size", 'min': 1, 'max': 1024, 'default': 16},
    ]

    hparams_lgbm = [    
        {"name": "lags", "type": "int", "description": "Lookback window length", 'min': 1, 'max': 1000, 'default': default_input_chunk},
        {"name": "output_chunk_length", "type": "int", "description": "Forecast horizon length", 'min': 1, 'max': 1000, 'default': default_output_chunk},
        {"name": "random_state", "type": "int", "description": "Randomness of weight initialization", 'min': 0, 'max': 10000, 'default': 42},
        ]

    hparams_rf = [   
        {"name": "lags", "type": "int", "description": "Lookback window length", 'min': 1, 'max': 1000, 'default': default_input_chunk},
        {"name": "output_chunk_length", "type": "int", "description": "Forecast horizon length", 'min': 1, 'max': 1000, 'default': default_output_chunk},
        {"name": "random_state", "type": "int", "description": "Randomness of weight initialization", 'min': 0, 'max': 10000, 'default': 42},
        ]
    
    hparams_arima = [    
        {"name": "p", "type": "int", "description": "Order (number of time lags) of the autoregressive model (AR)", 'min': 0, 'max': 1000, 'default': 12},
        {"name": "d", "type": "int", "description": "Order of differentiation", 'min': 0, 'max': 1000, 'default': 1},
        {"name": "q", "type": "int", "description": "Size of the moving average window (MA)", 'min': 0, 'max': 1000, 'default': 0},
        {"name": "random_state", "type": "int", "description": "Random state", 'min': 0, 'max': 10000, 'default': 42},
        ]

    models = [
        {"model_name": "Naive", "hparams": hparams_naive},
        {"model_name": "NBEATS", "hparams": hparams_nbeats},
        {"model_name": "NHiTS", "hparams": hparams_nhits},
        {"model_name": "Transformer", "hparams": hparams_transformer},
        {"model_name": "RNN", "hparams": hparams_rnn},
        {"model_name": "TFT", "hparams": hparams_tft},
        {"model_name": "TCN", "hparams": hparams_tcn},
        {"model_name": "BlockRNN", "hparams": hparams_blockrnn},
        {"model_name": "LightGBM", "hparams": hparams_lgbm},
        {"model_name": "RandomForest", "hparams": hparams_rf},
        {"model_name": "ARIMA", "hparams": hparams_arima},
        ]
    
    # Multiple does not work with Naive
    if multiple:
        del models[0]
    
    return models


@engineer_router.get("/metrics/get_metric_names", tags=['Metrics and models retrieval'])
async def get_metric_names():
    return metrics


def csv_validator(fname: str, day_first: bool, multiple: bool, allow_empty_series=False, format='long'):

    fileExtension = fname.split(".")[-1].lower() == "csv"
    if not fileExtension:
        print("Unsupported file type provided. Please upload CSV file")
        raise HTTPException(status_code=415, detail="Unsupported file type provided. Please upload CSV file")
    try:
        ts, resolution = read_and_validate_input(series_csv=fname, day_first=day_first, multiple=multiple, allow_empty_series=allow_empty_series, format=format)
    except WrongColumnNames:
        print("There was an error validating the file. Please reupload CSV with correct column names")
        raise HTTPException(status_code=415, detail="There was an error validating the file. Please reupload CSV with correct column names")
    except DatetimesNotInOrder:
        print("There was an error validating the file. Datetimes are not in order")
        raise HTTPException(status_code=415, detail="There was an error validating the file. Datetimes are not in order")
    
    resolutions = make_time_list(resolution=resolution)    
    return ts, resolutions

@scientist_router.post('/upload/uploadCSVfile', tags=['Experimentation Pipeline'])
async def create_upload_csv_file(file: UploadFile = File(...), day_first: bool = Form(default=True), 
                                 multiple: bool = Form(default=False), format: str = Form(default=False)):

    # Store uploaded dataset to backend
    print("Uploading file...")
    try:
        # write locally
        local_dir = tempfile.mkdtemp()
        contents = await file.read()
        fname = os.path.join(local_dir, file.filename)
        with open(fname, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=415, detail="There was an error uploading the file")
        #return {"message": "There was an error uploading the file"}
    finally:
        print(f'\n{fname}\n')
        await file.close()

    # Validation
    print("Validating file...") 
    ts, resolutions = csv_validator(fname, day_first, multiple, format=format)

    if multiple:
        if format == "long":
            dataset_start_multiple = ts.iloc[0]['Datetime']
            dataset_end_multiple = ts.iloc[-1]['Datetime']
        else:
            dataset_start_multiple = ts.iloc[0]['Date']
            dataset_end_multiple = ts.iloc[-1]['Date']
    
    return {"message": "Validation successful", 
            "fname": fname,
            "dataset_start": datetime.datetime.strftime(ts.index[0], "%Y-%m-%d") if multiple==False else dataset_start_multiple,
            "allowed_validation_start": datetime.datetime.strftime(ts.index[0] + timedelta(days=10), "%Y-%m-%d") if multiple==False else dataset_start_multiple + timedelta(days=10),
            "dataset_end": datetime.datetime.strftime(ts.index[-1], "%Y-%m-%d") if multiple==False else dataset_end_multiple,
            "allowed_resolutions": resolutions,
            "ts_used_id": None,
            "evaluate_all_ts": True if multiple else None
            }

def store_df_to_csv(df, csv_name, index):
    local_dir = tempfile.mkdtemp()
    fname = f'{local_dir}/{csv_name}'
    df.to_csv(fname, index=index)
    return fname

def unfold_timeseries(lds):
    """
    Function that turns mongo data to dictionary form

    Parameters
    ----------
    lds
        pymongo.cursor.Cursor object that contaions the timeseries data
    Returns
    -------
    (dict)
        A dictionary that contaions the timeseries data
    """
    new_loads = {'Datetime': [], "Value": []}
    prev_date = ''
    for l in reversed(list(lds)):
        if prev_date != l['date']:
            for key in l:
                if key != '_id' and key != 'date':
                    new_date = l['date'] + ' ' + key
                    new_loads['Datetime'].append(new_date)
                    new_loads["Value"].append(l[key])
        prev_date = l['date']
    #print("outp", new_loads)
    return new_loads

@scientist_router.get('/db_integration/retrieve_dataset/uc2', tags=['MongoDB integration'])
async def retrieve_uc2_dataset():
    # Connect to DB and get file
    print("Connecting to DB to retrieve dataset...")
    # TODO: missing exception here
    client = MongoClient(mongo_url)
    db = client[database]
    # Get collection and store to dataframe
    collection = db[mongo_collection_uc2]
    df = unfold_timeseries(collection.find().sort('_id', -1))
    df = pd.DataFrame.from_dict(df)
    # Store dataset as csv in backend
    print("Storing dataset to DeepTSF backend...")
    fname = store_df_to_csv(df, 'uc2.csv', index=False)
    print(fname)
    # Close connection to DB
    print("Closing connection to DB...")
    client.close()
    # Validate_csv
    multiple = False
    ts, resolutions = csv_validator(fname, day_first=False, multiple=multiple, format='short')
    return {"message": "Validation successful",
        "fname": fname,
        "dataset_start": datetime.datetime.strftime(ts.index[0], "%Y-%m-%d") if multiple==False else ts.iloc[0]['Date'],
        "allowed_validation_start": datetime.datetime.strftime(ts.index[0] + timedelta(days=10), "%Y-%m-%d") if multiple==False else ts.iloc[0]['Date'] + timedelta(days=10),
        "dataset_end": datetime.datetime.strftime(ts.index[-1], "%Y-%m-%d") if multiple==False else ts.iloc[-1]['Date'],
        "allowed_resolutions": resolutions,
        "ts_used_id": None,
        "evaluate_all_ts": False,
        "uc": 2,
        "multiple": multiple
        }
'''
run experimentation pipeline body example (series_csv is returned from retrieve_dataset/uc2 endpoint):
{
        "experiment_name": "uc2",
        "rmv_outliers": true,
        "multiple": false,
        "series_csv": "/tmp/tmpxebh4mdj/uc2.csv", 
        "resolution": "15min",
        "resampling_agg_method": "averaging",
        "validation_start_date": "20220101",
        "test_start_date": "20220201",
        "test_end_date": "20220301",
        "model": "LightGBM",
        "forecast_horizon": "96",
        "hyperparams_entrypoint": "{lags: 24}",
        "ignore_previous_runs": true,
        "l_interpolation": true,
        "ts_used_id": "null",
	    "evaluate_all_ts": false,
        "uc": 2
}
New handled arguments:
- series_csv: returned from retrieve_dataset/uc2 endpoint
- ts_used_id: "null"
- evaluate_all_ts: false,
- uc: 2,
'''
@scientist_router.get('/db_integration/retrieve_dataset/uc6/', tags=['MongoDB integration'])
async def retrieve_uc6_dataset(series_name: str):
    if series_name not in ["W6 positive_active", "W4 positive_active", "W6 positive_reactive", "W4 positive_reactive"]:
        print('Series name must be one of: "W6 positive_active", "W4 positive_active", "W6 positive_reactive", "W4 positive_reactive"')
        raise HTTPException(status_code=415, detail='Series name must be one of: "W6 positive_active", "W4 positive_active", "W6 positive_reactive", "W4 positive_reactive"')
    # Connect to DB and get file
    print("Connecting to DB to retrieve dataset...")
    # TODO: missing exception here
    client = MongoClient(mongo_url)
    db = client[database]
    # Get collection and store to dataframe
    collection = db[mongo_collection_uc6]
    df = pd.DataFrame(collection.find()).drop(columns={'_id', ''}, errors='ignore')
    df["ID"] = df["id"] + " " + df["power_type"]
    cols_to_drop = {'date', 'id', 'power_type'}
    df["Date"] = df["date"]
    df["Timeseries ID"] = df["ID"]
    df = df.drop_duplicates(subset=["Date", "ID"]).\
            sort_values(by=["Date", "ID"], ignore_index=True).\
            drop(columns=cols_to_drop)
    # Store dataset as csv in backend
    print("Storing dataset to DeepTSF backend...")
    fname = store_df_to_csv(df, 'uc6.csv', index=True)
    print(fname)
    # Close connection to DB
    print("Closing connection to DB...")
    client.close()
    # Validate_csv
    multiple = True
    ts, resolutions = csv_validator(fname, day_first=False, multiple=multiple, format='short')
    return {"message": "Validation successful",
        "fname": fname,
        "dataset_start": datetime.datetime.strftime(ts.index[0], "%Y-%m-%d") if multiple==False else ts.iloc[0]['Date'],
        "allowed_validation_start": datetime.datetime.strftime(ts.index[0] + timedelta(days=10), "%Y-%m-%d") if multiple==False else ts.iloc[0]['Date'] + timedelta(days=10),
        "dataset_end": datetime.datetime.strftime(ts.index[-1], "%Y-%m-%d") if multiple==False else ts.iloc[-1]['Date'],
        "allowed_resolutions": resolutions,
        "ts_used_id": series_name,
        "evaluate_all_ts": False,
        "uc": 6,
        "multiple": multiple
        }
'''
run experimentation pipeline body example :
{
        "experiment_name": "uc6",
        "rmv_outliers": true,
        "multiple": true,
        "series_csv": "/tmp/tmpsxph8ydb/uc6.csv",
        "resolution": "5min",
        "resampling_agg_method": "averaging",
        "validation_start_date": "20220101",
        "test_start_date": "20220201",
        "test_end_date": "20220301",
        "model": "LightGBM",
        "forecast_horizon": "24",
        "hyperparams_entrypoint": "{lags: 24}",
        "ignore_previous_runs": true,
        "l_interpolation": true,
	    "ts_used_id": "W6 positive_active",
	    "evaluate_all_ts": false,
        "uc": 6
}
New handled arguments:
- series_csv: returned from retrieve_dataset/uc6 endpoint
- ts_used_id: "W6 positive_active" or "W4 positive_active" or "W6 positive_reactive" or "W4 positive_reactive"
- evaluate_all_ts: false,
- uc: 6
'''

'''
run experimentation pipeline body example :
{
        "experiment_name": "uc7",
        "rmv_outliers": true,
        "multiple": true,
        "series_csv": "/tmp/tmpsxph8ydb/uc7.csv",
        "resolution": "5min",
        "resampling_agg_method": "averaging",
        "validation_start_date": "20220101",
        "test_start_date": "20220201",
        "test_end_date": "20220301",
        "model": "LightGBM",
        "forecast_horizon": "24",
        "hyperparams_entrypoint": "{lags: 24}",
        "ignore_previous_runs": true,
        "l_interpolation": true,
	    "ts_used_id": "null",
	    "evaluate_all_ts": true,
        "uc": 7
}
New handled arguments:
- series_csv: returned from retrieve_dataset/uc7 endpoint
- ts_used_id: "null"
- evaluate_all_ts: true
- uc: 7
'''

@admin_router.get('/experimentation_pipeline/training/hyperparameter_entrypoints', tags=['Experimentation Pipeline'])
async def get_experimentation_pipeline_hparam_entrypoints():
    entrypoints = ConfigParser().read_entrypoints()
    return entrypoints

#@app.get('/experimentation_pipeline/etl/get_resolutions/')
#async def get_resolutions():
#    return ResolutionMinutes.dict()

@admin_router.get('/get_mlflow_tracking_uri', tags=['MLflow Info'])
async def get_mlflow_tracking_uri():
    return mlflow.tracking.get_tracking_uri()

def mlflow_run(params: dict, experiment_name: str, uc: str = "2"):
    # TODO: generalize to all use cases
    # TODO: run through dagster for orchestration and error inspection. enershare?
    # will need GraphQL client for Dagster as it is in another container...
    pipeline_run = mlflow.projects.run(
            uri=f"./uc{uc}/",
            experiment_name=experiment_name,
            entry_point="exp_pipeline",
            parameters=params,
            env_manager="local"
            )

@scientist_router.post('/experimentation_pipeline/run_all', tags=['Experimentation Pipeline'])
async def run_experimentation_pipeline(parameters: dict, background_tasks: BackgroundTasks):

    # if this key exists then I am on the "user uploaded dataset" case so I proceed to the changes of the other parameters in the dict
    try:
        uc = parameters['uc']  # Trying to access a key that doesn't exist
    except KeyError:
        uc = "2" # the default uc
        if parameters["multiple"]:
           parameters["ts_used_id"] = None
           parameters["eval_all_ts"] = True
           # this is the default use case for all other runs except uc7
        pass  

    print(parameters["hyperparams_entrypoint"])

    # fix TFT as no covariates come from front
    if parameters['model'] == "TFT":
        parameters["hyperparams_entrypoint"]["add_relative_index"] = 'True'
    # format hparams string
    hparam_str = str(parameters["hyperparams_entrypoint"])
    hparam_str = hparam_str.replace('"', '')
    hparam_str = hparam_str.replace("'", "")
    print(hparam_str)

    params = { 
        "rmv_outliers": parameters["rmv_outliers"],  
        "multiple": parameters["multiple"],
        "series_csv": parameters["series_csv"], # input: get value from @app.post('/upload/validateCSVfile/') | type: str | example: -
        "resolution": change_form(freq=parameters["resolution"], change_format_to="pandas_form"), # input: user | type: str | example: "15" | get allowed values from @app.get('/experimentation_pipeline/etl/get_resolutions/')
        "resampling_agg_method": parameters["resampling_agg_method"],
        "cut_date_val": parameters["validation_start_date"], # input: user | type: str | example: "20201101" | choose from calendar, should be > dataset_start and < dataset_end
        "cut_date_test": parameters["test_start_date"], # input: user | type: str | example: "20210101" | Choose from calendar, should be > cut_date_val and < dataset_end
        "test_end_date": parameters["test_end_date"],  # input: user | type: str | example: "20220101" | Choose from calendar, should be > cut_date_test and <= dataset_end, defaults to dataset_end
        "darts_model": parameters["model"], # input: user | type: str | example: "nbeats" | get values from @app.get("/models/get_model_names")
        "forecast_horizon": parameters["forecast_horizon"], # input: user | type: str | example: "96" | should be int > 0 (default 24 if resolution=60, 96 if resolution=15, 48 if resolution=30)
        "hyperparams_entrypoint": hparam_str,
        "ignore_previous_runs": parameters["ignore_previous_runs"],
        "imputation_method": parameters["imputation_method"],    
	    "ts_used_id": parameters["ts_used_id"], # uc2: None, uc6: 'W6 positive_active' or 'W6 positive_active' or 'W4 positive_reactive' or 'W4 positive_active', uc7: None 
        "eval_series": parameters["ts_used_id"], # same as above,
	    "evaluate_all_ts": parameters["evaluate_all_ts"],
        "format": parameters["format"] 	    
        # "country": parameters["country"], this should be given if we want to have advanced imputation
     }
    
    # TODO: generalize for all countries
    # if parameters["model"] != "NBEATS":
    #    params["time_covs"] = "PT"
    
    try:
        background_tasks.add_task(mlflow_run, params, parameters['experiment_name'], uc)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Could not initiate run. Check system logs")
    
    return {"message": "Experimentation pipeline initiated. Proceed to MLflow for details..."}


@engineer_router.get('/results/get_list_of_experiments', tags=['MLflow Info', 'Model Evaluation'])
async def get_list_of_mlflow_experiments():
    client = MlflowClient()
    experiments = client.list_experiments()
    experiment_names = [client.list_experiments()[i].name
                        for i in range(len(experiments))]
    experiment_ids = [client.list_experiments()[i].experiment_id
                      for i in range(len(experiments))]
    experiments = dict(zip(experiment_names, experiment_ids))
    experiments_response = [
        {"experiment_name": key, "experiment_id": experiments[key]}
        for key in experiments.keys()
    ]
    return experiments_response


@engineer_router.get('/results/get_best_run_id_by_mlflow_experiment/{experiment_id}/{metric}',
                     tags=['MLflow Info', 'Model Evaluation'])
async def get_best_run_id_by_mlflow_experiment(experiment_id: str, metric: str = 'mape'):
    df = mlflow.search_runs([experiment_id], order_by=[f"metrics.{metric} ASC"])
    if df.empty:
        raise HTTPException(status_code=404, detail="No run has any metrics")
    else:
       best_run_id = df.loc[0, 'run_id']
       return best_run_id


@engineer_router.get('/results/get_forecast_vs_actual/{run_id}/n_samples/{n}', tags=['MLflow Info', 'Model Evaluation'])
async def get_forecast_vs_actual(run_id: str, n: int):
    forecast = load_artifacts(
        run_id=run_id, src_path="eval_results/predictions.csv")
    forecast_df = pd.read_csv(forecast, index_col=0).iloc[-n:]
    actual = load_artifacts(
        run_id=run_id, src_path="eval_results/test.csv")
    actual_df = pd.read_csv(actual, index_col=0)[-n:]
    forecast_response = forecast_df.to_dict('split')
    actual_response = actual_df.to_dict('split')
    # unlist
    actual_response["data"] = [i[0] for i in actual_response["data"]]
    forecast_response["data"] = [i[0] for i in forecast_response["data"]]
    response = {"forecast": forecast_response,
                "actual":  actual_response}
    return response


@engineer_router.get('/results/get_metric_list/{run_id}', tags=['MLflow Info', 'Model Evaluation'])
async def get_metric_list(run_id: str):
    client = MlflowClient()
    metrix = client.get_run(run_id).data.metrics
    metrix_response = {"labels":[i for i in metrix.keys()], "data": [i for i in metrix.values()]}
    return metrix_response


@admin_router.get('/system_monitoring/get_cpu_usage', tags=['System Monitoring'])
async def get_cpu_usage():
    cpu_count_logical = psutil.cpu_count()
    cpu_count = psutil.cpu_count(logical=False)
    cpu_usage = psutil.cpu_percent(percpu=True)
    cpu_percentage_response = {'labels': [f'CPU {i}' for i in range(1, len(cpu_usage)+1)], 'data': cpu_usage}
    response = {'barchart_1': cpu_percentage_response,
                'text_1': cpu_count,
                'text_2': cpu_count_logical}
    return response


@admin_router.get('/system_monitoring/get_memory_usage', tags=['System Monitoring'])
async def get_memory_usage():
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    swap_memory_response = {
        'title': 'Swap memory usage (Mbytes)',
        'low': swap_memory.used // 1024**2,
        'high': swap_memory.total // 1024**2}
    virtual_memory_response = {
        'title': 'Virtual memory usage (Mbytes)',
        'low': virtual_memory.used // 1024**2,
        'high': virtual_memory.total // 1024**2}
    response = {
        'progressbar_1': virtual_memory_response,
        'progressbar_2': swap_memory_response}
    return response


@admin_router.get('/system_monitoring/get_gpu_usage', tags=['System Monitoring'])
async def get_gpu_usage():
    gpus_stats = nvsmi.get_gpus()
    response = {}
    for gpu_stats in gpus_stats:
        response[gpu_stats.id] = {
           "progressbar_1": {'title': "GPU utilization (%)", 'percent': gpu_stats.gpu_util}, 
           "progressbar_2": {'title': "GPU memory utilization (Mbytes)",
                            'low':  gpu_stats.mem_used,
                            'high':  gpu_stats.mem_total}}
    print(response)
    return response


@common_router.get("/user/info")
async def get_info(token: str = Depends(oauth2_scheme)):
    headers = {
        'accept': 'application/json',
        'cache-control': 'no-cache',
        'content-type': 'application/x-www-form-urlencoded',
    }
    data = {
        'client_id': settings.client_id,
        'client_secret': settings.client_secret,
        'token': token,
    }
    url = settings.token_issuer + '/introspect'
    response = httpx.post(url, headers=headers, data=data)
    return response.json()


app.include_router(admin_router)
app.include_router(scientist_router)
app.include_router(engineer_router)
if os.getenv("USE_KEYCLOAK", 'True') == 'True':
    app.include_router(common_router)

# if __name__ == "__main__":
#     uvicorn.run('api:app', reload=True)

# UC7 ommited for complexity purposes:

# class SmartMetersProcessor:
#     PRODUCTION_TAG_NAME = '2_8_0'
#     CONSUMPTION_TAG_NAME = '1_8_0'
#     TIME_INTERVALS = {
#         '30': [f"{hour:02d}:{minute:02d}:00" for hour in range(24) for minute in range(0, 60, 30)],
#         '60': [f"{hour:02d}:{minute:02d}:00" for hour in range(24) for minute in range(0, 60, 60)],
#     }
#     RESAMPLING_FREQS = {
#         '30': '30T',
#         '60': '1H',
#     }

#     DEEP_TSF_COLUMN_MAPPER = {
#         'device_id': 'Timeseries ID',
#         'tag_name': 'ID',
#         'date': 'Date',
#     }

#     def __init__(self, specs_df: pd.DataFrame, resolution: int = 60):
#         self.specs_df = specs_df
#         self.DEFAULT_TIME_INTERVALS = self.TIME_INTERVALS[str(resolution)]
#         self.DEFAULT_RESAMPLING_FREQ = self.RESAMPLING_FREQS[str(resolution)]

#     def retrieve_specs(self, device_id: str) -> Tuple[bool, float, bool, float]:
#         try:
#             smart_meter_specs = self.specs_df.loc[(self.specs_df['id'] == device_id)].iloc[0].to_dict()
#             production_max = smart_meter_specs['Production (kW)']
#             consumption_max = smart_meter_specs['Contractual power (kW)']
#             return production_max >= 0, production_max, consumption_max >= 0, consumption_max
#         except IndexError:
#             # iloc[0] index error when smart meter is missing from csv with specs.
#             print(f'Smart meter {device_id} does not exist in contract')
#             return False, nan, False, nan

#     @staticmethod
#     def remove_outliers(sm_df: pd.DataFrame, max_value: float, contract_exists: bool) -> None:
#         min_value = 0
#         max_value = max_value if contract_exists else 200
#         if contract_exists and max_value == 0:
#             sm_df['value'] = nan
#         else:
#             sm_df.loc[(sm_df['value'] < min_value) | (sm_df['value'] > max_value), 'value'] = nan

#     @staticmethod
#     def apply_naming_convention(smart_meter_name: str) -> str:
#         # Fix missing B in smart meters name
#         if smart_meter_name.startswith('BB'):
#             if not smart_meter_name.startswith('BBB'):
#                 smart_meter_name = 'B' + smart_meter_name
#         else:
#             raise Exception(f'Smart meter {smart_meter_name} does not follow the BBB naming convention')
#         return smart_meter_name

#     def smart_meters_load_forecasting_processing(self, data_cursor: pymongo.cursor,
#                                                  output_file_path: str) -> bson.objectid.ObjectId:
#         # create time intervals columns
#         columns = [self.DEEP_TSF_COLUMN_MAPPER["device_id"], self.DEEP_TSF_COLUMN_MAPPER["tag_name"],
#                    self.DEEP_TSF_COLUMN_MAPPER["date"]] + self.DEFAULT_TIME_INTERVALS

#         if not os.path.exists(output_file_path):
#             headers_df = pd.DataFrame(columns=columns)
#             headers_df.to_csv(output_file_path, mode='a', index=False)

#         last_document_id = None
#         for doc in data_cursor:
#             last_document_id = doc['_id']
#             smart_meter = self.apply_naming_convention(doc["device_id"])  # smart meter id
#             # fetch smart meters specs
#             supports_prod, prod_max, supports_cons, cons_max = self.retrieve_specs(device_id=smart_meter)
#             date = doc["date"]  # date of measurements
#             doc_df = pd.DataFrame(doc["meter"])  # time series data
#             doc_df["datetime"] = pd.to_datetime(date + ' ' + doc_df["time"])  # add column datetime
#             # drop time, quality, quality_detail, opc_quality columns
#             doc_df.drop(columns=['time', 'quality', 'quality_detail', 'opc_quality'], axis=1, inplace=True)
#             # group data by tag name to resample properly, remember that data at this point refers to a single smart meter and a single day
#             grouped = doc_df.groupby('tag_name')
#             # initialise empty DataFrame
#             resampled_df = pd.DataFrame()

#             for tag_name, group_data in grouped:
#                 # resample data within the tag_name group in 30 minutes intervals
#                 resampled_group = group_data.resample(self.DEFAULT_RESAMPLING_FREQ, on='datetime', closed='left',
#                                                       label='left').agg({'value': 'mean'})

#                 # create a full day index
#                 full_day_date_range = pd.date_range(start=(date + ' ' + '00:00:00'), end=(date + ' ' + '23:59:59'),
#                                                     freq=self.DEFAULT_RESAMPLING_FREQ)
#                 # reindex to expand to full day, even with NaNs
#                 resampled_group = resampled_group.reindex(full_day_date_range)
#                 resampled_group.reset_index(inplace=True)

#                 resampled_group["device_id"] = smart_meter  # add smart_meter id to data
#                 resampled_group["tag_name"] = tag_name  # add tag_name in data
#                 resampled_group["date"] = date  # add date in data
#                 resampled_group.rename(columns={'index': 'datetime'}, inplace=True)

#                 # remove outliers based on contractual power and production
#                 # in place operation
#                 if self.PRODUCTION_TAG_NAME in tag_name:
#                     self.remove_outliers(sm_df=resampled_group, max_value=prod_max, contract_exists=supports_prod)
#                 elif self.CONSUMPTION_TAG_NAME in tag_name:
#                     self.remove_outliers(sm_df=resampled_group, max_value=cons_max, contract_exists=supports_cons)
#                 # Pivot the DataFrame
#                 pivoted_df = resampled_group.pivot(index=['device_id', 'tag_name', 'date'], columns='datetime',
#                                                    values='value')
#                 pivoted_df.columns = pivoted_df.columns.strftime('%H:%M:%S')

#                 # Concatenate the resampled group with the overall resampled DataFrame
#                 resampled_df = pd.concat([resampled_df, pivoted_df])

#             # append data to monthly record
#             resampled_df = resampled_df.reset_index()
#             resampled_df.to_csv(output_file_path, mode='a', header=False, index=False)
#         return last_document_id
    
# @scientist_router.get('/db_integration/retrieve_dataset/uc7/', tags=['MongoDB integration'])
# async def retrieve_uc7_dataset(start_date:str, end_date: str):
#     # default resolution for uc7 initial dataset
#     resolution = 60
    
#     collection = os.environ.get('MONGO_COLLECTION_UC7')

#     # Connect to DB and get file
#     print("Connecting to DB to retrieve dataset...")
#     # TODO: missing exception here
#     client = MongoClient(mongo_url)
#     db = client[database]
#     # Get collection and store to dataframe
#     collection = db[mongo_collection_uc7]

#     sm_specs_df = pd.read_csv(
#         os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs/example_files/smart_meter_description.csv"))

#     smart_meters_processor = SmartMetersProcessor(
#         specs_df=sm_specs_df,
#         resolution=resolution
#     )

#     # output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'archive{start_date}_{end_date}.csv')
#     local_dir = tempfile.mkdtemp()
#     output_file_path = os.path.join(local_dir, f'archive_{start_date}_{end_date}.csv')
    
#     try:
#         start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
#         end_date = datetime.datetime.strptime(end_date, "%Y%m%d")

#         query = {"date": {"$gte": start_date.strftime("%Y-%m-%d"), "$lte": end_date.strftime("%Y-%m-%d")}}
#         # batch_size
#         batch_size = 150
#         last_id = None
#         while True:
#             # used as equivalent to skip option
#             if last_id:
#                 query["_id"] = {"$gt": last_id}

#             # query the collection
#             batch = collection.find(query).limit(batch_size)

#             if not batch.alive:
#                 break
#             # process batch
#             last_object_id = smart_meters_processor.smart_meters_load_forecasting_processing(
#                 data_cursor=batch,
#                 output_file_path=output_file_path
#             )
#             if last_object_id is None:
#                 break

#             # Update last_id for the next iteration
#             last_id = last_object_id

#         # drop duplicates and reorder
#         df = pd.read_csv(output_file_path)
#         df.drop_duplicates(inplace=True)
        
#         # only keep APIU for load forecasting (this is a requirement to avoid DifferentComponentDimensions error in Darts)
#         df = df[~df['ID'].str.contains("Ameno")].reset_index(drop=True)

#         # TODO: remove series that have been cut on NaNs
#         ## Look for 'Timeseries ID' that have NaNs on the end date and remove them entirely.
        
#         # sort dates
#         df = df.sort_values(by=['Date', 'Timeseries ID'], ascending=[True, True]).reset_index(drop=True)
        
#         # store to csv
#         df.to_csv(output_file_path, index=True)
        
#         print(df.head())
#         print(f'\nOutput csv path: {output_file_path}\n')

#     except ValueError:
#         raise HTTPException(
#             status_code=415,
#             detail="Error parsing dates. "
#                    "The appropriate date format is: YYYYMMDD. "
#                    f"The available date range is: 20210916 - {date.today().strftime('%Y%m%d')}"
#     )
#     finally:
#         client.close()

#     # Validate_csv
#     multiple = True
#     ts, resolutions = csv_validator(output_file_path, day_first=False, multiple=multiple, allow_empty_series=True)
#     return {"message": "Validation successful",
#         "fname": output_file_path,
#         "dataset_start": datetime.datetime.strftime(ts.index[0], "%Y-%m-%d") if multiple==False else ts.iloc[0]['Date'],
#         "allowed_validation_start": datetime.datetime.strftime(ts.index[0] + timedelta(days=10), "%Y-%m-%d") if multiple==False else ts.iloc[0]['Date'] + timedelta(days=10),
#         "dataset_end": datetime.datetime.strftime(ts.index[-1], "%Y-%m-%d") if multiple==False else ts.iloc[-1]['Date'],
#         "allowed_resolutions": resolutions,
#         "ts_used_id": None,
#         "evaluate_all_ts": True,
#         "uc": 7,
#         "multiple": multiple
#         }