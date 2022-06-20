from distutils.log import error
from enum import Enum
from fastapi import File, UploadFile, FastAPI
import pandas as pd
import mlflow
from utils import ConfigParser
import tempfile
import os

# allows automated type check with pydantic
class ModelName(str, Enum):
    nbeats = "nbeats"
    tcn = "tcn"
    rnn = "rnn"
    lgbm = "lightgbm"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, ModelName))
    
    @staticmethod
    def dict():
        return {"models": list(map(lambda c: c.value, ModelName))}

class ResolutionMinutes(int, Enum):
    a = 5
    b = 15
    c = 30
    d = 60

    @staticmethod
    def dict():
        return {"resolution": list(map(lambda c: c.value, ModelName))}

class DateLimits(int, Enum):
    """This function will read the uploaded csv before running the pipeline and will decide which are the allowed values
    for: validation_start_date < test_start_date < test_end_date """


    @staticmethod
    def dict():
        return {"resolution": list(map(lambda c: c.value, ModelName))}

app = FastAPI()

# <api>.<http_operation>.(<route>)
@app.get("/")
async def root():
    return {"message": "I-NERGY Load Forecasting API"}

@app.get("/models/get_model_names")
async def get_model_names():
    return ModelName.dict()

@app.get("/models/{model_name}/hparams")
async def get_model_hparams():
    return ModelName.dict()

@app.post('/uploadfile/')
async def create_upload_file(file: UploadFile = File(...), experiment_name: str = None):
    try:
        # write locally
        local_dir = tempfile.mkdtemp()
        contents = await file.read()
        fname = os.path.join(local_dir, file.filename)
        with open(fname, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        await file.close()

    # Validate 
    params = {
        "series_uri": "online_artifact",
        "series_csv": fname
    }

    load_raw_data_run = mlflow.run(
        uri=".",
        entry_point="load_raw_data",
        parameters=params,
        env_manager="local",
        experiment_name=experiment_name
    )

    if load_raw_data_run.info.status == "FAILED":
        return {"message": "Successfully uploaded file. However validation failed. Check file format!",
                "validator_run_id": mlflow.tracking.MlflowClient().get_run(load_raw_data_run.run_id),
                "experiment_name": experiment_name}

    elif load_raw_data_run.info.status == "FINISHED":
        return {"message": f"Successfuly uploaded and validated {file.filename}",
                "validator_run_id": mlflow.tracking.MlflowClient().get_run(load_raw_data_run.run_id),
                "experiment_name": experiment_name,
                "series_uri": load_raw_data_run.data.tags['dataset_uri'],
                "dataset_start": load_raw_data_run.data.tags['dataset_start'],
                "dataset_end": load_raw_data_run.data.tags['dataset_end']
                }

@app.get('/experimentation_pipeline/training/hyperparameter_entrypoints/')
async def get_experimentation_pipeline_hparam_entrypoints():
    return ConfigParser().read_entrypoints()

@app.get('/experimentation_pipeline/etl/get_resolutions/')
async def get_resolutions():
    return ResolutionMinutes.dict()

@app.post('/experimentation_pipeline/run_all')
async def run_experimentation_pipeline(parameters: dict):

    params = {
        # get it from create_upload_file
        "series_uri": parameters["series_uri"], # input: get value from from @app.post('/uploadfile/') | type: str | example: 
        "experiment_name": parameters["experiment_name"], # input: get it from @app.post('/uploadfile/') | type: str | example: 
        "resolution": parameters["resolution"], # input: user | type: str | example: "15" | get allowed values from @app.get('/experimentation_pipeline/etl/get_resolutions/')
        "cut_date_val": parameters["validation_start_date"], # input: user | type: str | example: 20201101 | choose from calendar, should be > dataset_start and < dataset_end
        "cut_date_test": parameters["test_start_date"], # input: user | type: str | example: "20210101" | Choose from calendar, should be > cut_date_val and < dataset_end
        "test_end_date": parameters["test_end_date"],  # input: user | type: str | example: "20220101" | Choose from calendar, should be > cut_date_test, defaults to dataset_end
        "darts_model": parameters["model"], # input: user | type: str | example: "nbeats" | get values from @app.get("/models/get_model_names")
        "forecast_horizon": parameters["forecast_horizon"], # input: user | type: str | example: "96" | should be int > 0
        "ignore_previous_runs": parameters["ignore_previous_runs"], # input: user | type: str | example: "true" | allowed values "true" or "false", defaults to false)
    }

    pipeline_run = mlflow.run(
        uri=".",
        entry_point="exp_pipeline",
        parameters=params,
        env_manager="local",
        experiment_name=parameters["experiment_name"]
        )

    return {"parent_run_id": mlflow.tracking.MlflowClient().get_run(pipeline_run.run_id)}

# # find child runs of father run
# query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
# results = mlflow.search_runs(filter_string=query)
