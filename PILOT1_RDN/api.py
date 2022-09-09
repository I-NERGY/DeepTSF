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
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient
from utils import load_artifacts

# allows automated type check with pydantic
#class ModelName(str, Enum):

models = [
    {"model_name": "NBEATS", "search_term": "nbeats"},
    {"model_name": "TCN", "search_term": "tcn"},
    {"model_name": "BlockRNN", "search_term": "blocklstm"},
    {"model_name": "LightGBM", "search_term": "lgbm"},
    {"model_name": "RandomForest", "search_term": "rf"}
    ]

metrics = [
    {"metric_name": "mape", "search_term": "mape"},
    {"metric_name": "mase", "search_term": "mase"},
    {"metric_name": "mae", "search_term": "mae"},
    {"metric_name": "rmse", "search_term": "rmse"},
    {"metric_name": "smape", "search_term": "smape"}]

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
    return models

@ app.get("/metrics/get_metric_names")
async def get_metric_names():
    return metrics

@app.post('/upload/uploadCSVfile')
async def create_upload_csv_file(file: UploadFile = File(...), day_first: bool = Form(default=True)):
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
        return {"message": "There was an error uploading the file"}
    finally:
        await file.close()

    # Validation
    print("Validating file...")
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

    resolution_minutes = int((ts.index[1] - ts.index[0]).total_seconds() // 60)
    if resolution_minutes < 1 and resolution_minutes > 180:
       return {"message": "Dataset resolution should be between 1 and 180 minutes"}

    resolutions = []
    for m in range(1, 181):
        if resolution_minutes == m:
            resolutions = [{"value": str(resolution_minutes), "display_value": str(resolution_minutes) + "(Current)"}]
        if resolution_minutes < m and m % 5 == 0: 
            resolutions.append({k: v for (k,v) in zip(["value", "display_value"],[str(m), str(m)])})
    
    return {"message": "Validation successful",
            "fname": fname,
            "dataset_start": datetime.strftime(ts.index[0], "%Y-%m-%d"),
            "allowed_validation_start": datetime.strftime(ts.index[0] + timedelta(days=10), "%Y-%m-%d"),
            "dataset_end": datetime.strftime(ts.index[-1], "%Y-%m-%d"),
            "allowed_resolutions": resolutions
            }

@app.get('/experimentation_pipeline/training/hyperparameter_entrypoints')
async def get_experimentation_pipeline_hparam_entrypoints():
    return ConfigParser().read_entrypoints()

#@app.get('/experimentation_pipeline/etl/get_resolutions/')
#async def get_resolutions():
#    return ResolutionMinutes.dict()

@app.get('/get_mlflow_tracking_uri')
async def get_mlflow_tracking_uri():
    return mlflow.tracking.get_tracking_uri()

@app.post('/experimentation_pipeline/run_all')
async def run_experimentation_pipeline(parameters: dict):
    params = {
        "series_csv": parameters["series_csv"], # input: get value from @app.post('/upload/validateCSVfile/') | type: str | example: -
        "resolution": parameters["resolution"], # input: user | type: str | example: "15" | get allowed values from @app.get('/experimentation_pipeline/etl/get_resolutions/')
        "cut_date_val": parameters["validation_start_date"], # input: user | type: str | example: "20201101" | choose from calendar, should be > dataset_start and < dataset_end
        "cut_date_test": parameters["test_start_date"], # input: user | type: str | example: "20210101" | Choose from calendar, should be > cut_date_val and < dataset_end
        "test_end_date": parameters["test_end_date"],  # input: user | type: str | example: "20220101" | Choose from calendar, should be > cut_date_test and <= dataset_end, defaults to dataset_end
        "darts_model": parameters["model"], # input: user | type: str | example: "nbeats" | get values from @app.get("/models/get_model_names")
        "forecast_horizon": parameters["forecast_horizon"], # input: user | type: str | example: "96" | should be int > 0 (default 24 if resolution=60, 96 if resolution=15, 48 if resolution=30)
        "hyperparams_entrypoint": parameters["hyperparams_entrypoint"], # input: user | type: str | example: "nbeats0_2" | get values from config.yaml headers
        "ignore_previous_runs": parameters["ignore_previous_runs"], # input: user | type: str | example: "true" | allowed values "true" or "false", defaults to false)
     }

    experiment_name = parameters["experiment_name"] # user | str

    # TODO: generalize for all countries
    if parameters["model"] == "NBEATS":
        params["time_covs"] = "PT"

    try: 
        pipeline_run = mlflow.projects.run(
            uri=".",
            experiment_name=experiment_name,
            entry_point="exp_pipeline",
            parameters=params,
            env_manager="local"
            )
    except Exception as e:
        return {"message": "Experimentation pipeline failed",
                "status": "Failed",
                # "parent_run_id": mlflow.tracking.MlflowClient().get_run(pipeline_run.run_id),
                "mlflow_tracking_uri": mlflow.tracking.get_tracking_uri(),
                "experiment_name": experiment_name,
                "experiment_id": MlflowClient().get_experiment_by_name(experiment_name).experiment_id
           }
    # for now send them to MLflow to check their metrics.
    return {"message": "Experimentation pipeline successful",
            "status": "Success",
            "parent_run_id": mlflow.tracking.MlflowClient().get_run(pipeline_run.run_id),
            "mlflow_tracking_uri": mlflow.tracking.get_tracking_uri(),
	    "experiment_name": experiment_name,
            "experiment_id": MlflowClient().get_experiment_by_name(experiment_name).experiment_id
	   }

@app.get('/results/get_list_of_experiments')
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

@app.get('/results/get_best_run_id_by_mlflow_experiment/{experiment_id}/{metric}')
async def get_best_run_id_by_mlflow_experiment(experiment_id: str, metric: str = 'mape'):
    df = mlflow.search_runs([experiment_id], order_by=[f"metrics.{metric} ASC"])
    best_run_id = df.loc[0, 'run_id']
    return best_run_id

@app.get('/results/get_forecast_vs_actual/{run_id}')
async def get_forecast_vs_actual(run_id: str):
    forecast = load_artifacts(
        run_id=run_id, src_path="eval_results/predictions.csv")
    forecast_df = pd.read_csv(forecast, index_col=0).iloc[-2000:-1000]
    actual = load_artifacts(
        run_id=run_id, src_path="eval_results/test.csv")
    actual_df = pd.read_csv(actual, index_col=0)[-2000:-1000]
    forecast_response = forecast_df.to_dict('split')
    actual_response = actual_df.to_dict('split')
    # unlist
    actual_response["data"] = [i[0] for i in actual_response["data"]]
    forecast_response["data"] = [i[0] for i in forecast_response["data"]]
    response = {"forecast": forecast_response,
                "actual":  actual_response}
    print(response)
    return response

@app.get('/results/get_metric_list/{run_id}')
async def get_metric_list(run_id: str):
    client = MlflowClient()
    metrix = client.get_run(run_id).data.metrics
    metrix_response = {"labels":[i for i in metrix.keys()], "data": [i for i in metrix.values()]}
    return metrix_response


    # # find child runs of father run
    # query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
    # results = mlflow.search_runs(filter_string=query)
    # graph_dict = {'labels':[], 'data':[]}
