from distutils.log import error
from enum import Enum
import uvicorn
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Depends
import pandas as pd
import mlflow
from utils import ConfigParser, truth_checker
import tempfile
import os
from uc2.load_raw_data import read_and_validate_input
from exceptions import DatetimesNotInOrder, WrongColumnNames
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient
from utils import load_artifacts
import psutil, nvsmi
import os
from dotenv import load_dotenv
from fastapi import APIRouter
from app.auth import admin_validator, scientist_validator, engineer_validator, common_validator, oauth2_scheme
from app.config import settings
import pretty_errors

load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
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
]

# metrics = [
#     {"metric_name": "mape", "search_term": "mape"},
#     {"metric_name": "mase", "search_term": "mase"},
#     {"metric_name": "mae", "search_term": "mae"},
#     {"metric_name": "rmse", "search_term": "rmse"},
#     {"metric_name": "smape", "search_term": "smape"}]
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
async def get_model_names(resolution: int, multiple: bool):

    default_input_chunk = int(60 / resolution * 168) if int(60 / resolution * 168) > 0 else 1
    default_output_chunk =  int(60 / resolution * 24) if int(60 / resolution * 24) > 0 else 1

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
        {"name": "random_state", "type": "int", "description": "Randomness of neural weight initialization", 'min': 0, 'max': 10000,'default': 42},
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
        {"name": "random_state", "type": "int", "description": "Randomness of neural weight initialization", 'min': 0, 'max': 10000},
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
        {"name": "learning rate", "type": "float", "description": "Learning rate", 'min': 0.000000001, 'max': 1, 'default': 0.0008},
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
        ]
    
    # Multiple does not work with Naive
    if multiple:
        del models[0]
    
    return models


@engineer_router.get("/metrics/get_metric_names", tags=['Metrics and models retrieval'])
async def get_metric_names():
    return metrics


@scientist_router.post('/upload/uploadCSVfile', tags=['Experimentation Pipeline'])
async def create_upload_csv_file(file: UploadFile = File(...), day_first: bool = Form(default=True), multiple: bool = Form(default=False)):
    # Loading
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
        await file.close()

    # Validation
    print("Validating file...")
    fileExtension = fname.split(".")[-1].lower() == "csv"
    if not fileExtension:
        raise HTTPException(
            status_code=415, detail="Unsupported file type provided. Please upload CSV file")
    try:
        ts, resolution_minutes = read_and_validate_input(series_csv=fname, day_first=day_first, multiple=multiple)
    except WrongColumnNames:
        raise HTTPException(status_code=415, detail="There was an error validating the file. Please reupload CSV with 2 columns with names: 'Datetime', 'Value'")
    except DatetimesNotInOrder:
        raise HTTPException(status_code=415, detail="There was an error validating the file. Datetimes are not in order")

    if resolution_minutes < 1 and resolution_minutes > 3600:
       raise HTTPException(status_code=415, detail="Dataset resolution should be between 1 and 180 minutes")
       # return {"message": "Dataset resolution should be between 1 and 180 minutes"}

    resolutions = []
    for m in range(1, 3601):
        if resolution_minutes == m:
            resolutions = [{"value": str(resolution_minutes), "default": True}]
        if resolution_minutes < m and m % resolution_minutes == 0: 
            resolutions.append({k: v for (k,v) in zip(["value", "default"],[str(m), False])})

    return {"message": "Validation successful",
            "fname": fname,
            "dataset_start": datetime.strftime(ts.index[0], "%Y-%m-%d") if multiple==False else ts.iloc[0]['Date'],
            "allowed_validation_start": datetime.strftime(ts.index[0] + timedelta(days=10), "%Y-%m-%d") if multiple==False else ts.iloc[0]['Date'] + timedelta(days=10),
            "dataset_end": datetime.strftime(ts.index[-1], "%Y-%m-%d") if multiple==False else ts.iloc[-1]['Date'],
            "allowed_resolutions": resolutions
            }

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

def mlflow_run(params: dict, experiment_name: str):
    # TODO: generalize to all use cases
    pipeline_run = mlflow.projects.run(
            uri="./uc2/",
            experiment_name=experiment_name,
            entry_point="exp_pipeline",
            parameters=params,
            env_manager="local"
            )

@scientist_router.post('/experimentation_pipeline/run_all', tags=['Experimentation Pipeline'])
async def run_experimentation_pipeline(parameters: dict, background_tasks: BackgroundTasks):

    hparam_str = str(parameters["hyperparams_entrypoint"])
    hparam_str = hparam_str.replace('"', '')
    hparam_str = hparam_str.replace("'", "")
    print(hparam_str)
    # parameters["hyperparams_entrypoint"] = { (key.replace('"', '')) : (val.replace('"', '') if isinstance(val, str) else val) for key, val in parameters["hyperparams_entrypoint"].items()}
    
    print(parameters["resampling_agg_method"])

    params = { 
        "rmv_outliers": parameters["rmv_outliers"],  
        "multiple": parameters["multiple"],
        "series_csv": parameters["series_csv"], # input: get value from @app.post('/upload/validateCSVfile/') | type: str | example: -
        "resolution": parameters["resolution"], # input: user | type: str | example: "15" | get allowed values from @app.get('/experimentation_pipeline/etl/get_resolutions/')
        "resampling_agg_method": parameters["resampling_agg_method"],
        "cut_date_val": parameters["validation_start_date"], # input: user | type: str | example: "20201101" | choose from calendar, should be > dataset_start and < dataset_end
        "cut_date_test": parameters["test_start_date"], # input: user | type: str | example: "20210101" | Choose from calendar, should be > cut_date_val and < dataset_end
        "test_end_date": parameters["test_end_date"],  # input: user | type: str | example: "20220101" | Choose from calendar, should be > cut_date_test and <= dataset_end, defaults to dataset_end
        "darts_model": parameters["model"], # input: user | type: str | example: "nbeats" | get values from @app.get("/models/get_model_names")
        "forecast_horizon": parameters["forecast_horizon"], # input: user | type: str | example: "96" | should be int > 0 (default 24 if resolution=60, 96 if resolution=15, 48 if resolution=30)
        "hyperparams_entrypoint": hparam_str,
        "ignore_previous_runs": parameters["ignore_previous_runs"],
        "l_interpolation": True,    
        "evaluate_all_ts": True,
        # "country": parameters["country"], this should be given if we want to have advanced imputation
     }
    
    # TODO: generalize for all countries
    # if parameters["model"] != "NBEATS":
    #    params["time_covs"] = "PT"
    
    try:
        background_tasks.add_task(mlflow_run, params, parameters['experiment_name'])
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
