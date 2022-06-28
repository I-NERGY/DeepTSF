from distutils.log import error
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import pandas as pd
import mlflow
from utils import ConfigParser, truth_checker
import tempfile
import os
from load_raw_data import read_and_validate_input
from exceptions import DatesNotInOrder, WrongColumnNames
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

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
        return {"resolution": list(map(lambda c: c.value, ResolutionMinutes))}

class DateLimits(int, Enum):
    """This function will read the uploaded csv before running the pipeline and will decide which are the allowed values
    for: validation_start_date < test_start_date < test_end_date """


    @staticmethod
    def dict():
        return {"resolution": list(map(lambda c: c.value, ModelName))}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# <api>.<http_operation>.(<route>)
@app.get("/")
async def root():
    return {"message": "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs"}

@app.get("/models/get_model_names")
async def get_model_names():
    return ModelName.dict()

@app.post('/upload/uploadCSVfile/')
async def create_upload_csv_file(file: UploadFile = File(...), day_first: bool = Form(default=True)):
    print("Uploading file...")
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

    print("Validating file...")
    params = {
        "fname": fname,  # get from create_upload_csv_file()
        # get from ui (tickbox or radio button or smth...), allowed values: true or false, default: false
        "day_first": day_first
    }
    fileExtension = fname.split(".")[-1].lower() == "csv"
    if not fileExtension:
        raise HTTPException(
            status_code=415, detail="Unsupported file type provided. Please upload CSV file")
    try:
        ts = read_and_validate_input(fname, day_first)
    except DatesNotInOrder:
        return {"message": "There was an error validating the file. Dates are not in order",
                "fname": fname}
    except WrongColumnNames:
        return {"message": "There was an error validating the file. Please reupload CSV with 2 columns with names: 'Date', 'Load'"}
    return {"message": "Validation successful",
            "fname": fname,
            "dataset_start": datetime.strftime(ts.index[0], "%Y%m%d"),
            "dataset_end": datetime.strftime(ts.index[-1], "%Y%m%d")
            }

@app.get('/experimentation_pipeline/training/hyperparameter_entrypoints/')
async def get_experimentation_pipeline_hparam_entrypoints():
    return ConfigParser().read_entrypoints()

@app.get('/experimentation_pipeline/etl/get_resolutions/')
async def get_resolutions():
    return ResolutionMinutes.dict()

@app.post('/experimentation_pipeline/run_all')
async def run_experimentation_pipeline(parameters: dict):
    #TODO: korbakis how can you obtain these data?
    params = {
        "series_uri": parameters["series_uri"], # input: get value from @app.post('/upload/validateCSVfile/') | type: str | example: -
        "experiment_name": parameters["experiment_name"], # input: user | type: str | example: ml_experiment
        "resolution": parameters["resolution"], # input: user | type: str | example: "15" | get allowed values from @app.get('/experimentation_pipeline/etl/get_resolutions/')
        "cut_date_val": parameters["validation_start_date"], # input: user | type: str | example: "20201101" | choose from calendar, should be > dataset_start and < dataset_end
        "cut_date_test": parameters["test_start_date"], # input: user | type: str | example: "20210101" | Choose from calendar, should be > cut_date_val and < dataset_end
        "test_end_date": parameters["test_end_date"],  # input: user | type: str | example: "20220101" | Choose from calendar, should be > cut_date_test and <= dataset_end, defaults to dataset_end
        "darts_model": parameters["model"], # input: user | type: str | example: "nbeats" | get values from @app.get("/models/get_model_names")
        "forecast_horizon": parameters["forecast_horizon"], # input: user | type: str | example: "96" | should be int > 0 (default 24 if resolution=60, 96 if resolution=15, 48 if resolution=30)
        "hyperparams_entrypoint": parameters["hyperameters"], # input: user | type: str | example: "nbeats0_2" | get values from config.yaml headers
        "ignore_previous_runs": parameters["ignore_previous_runs"], # input: user | type: str | example: "true" | allowed values "true" or "false", defaults to false)
    }

    pipeline_run = mlflow.run(
        uri=".",
        entry_point="exp_pipeline",
        parameters=params,
        env_manager="local",
        experiment_name=parameters["experiment_name"]
        )

    # for now send them to MLflow to check their metrics.
    return {"parent_run_id": mlflow.tracking.MlflowClient().get_run(pipeline_run.run_id),
            "mlflow_tracking_uri": mlflow.tracking.get_tracking_uri()}

# # find child runs of father run
# query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
# results = mlflow.search_runs(filter_string=query)
