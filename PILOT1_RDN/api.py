from distutils.log import error
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
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
import psutil, nvsmi

# allows automated type check with pydantic
#class ModelName(str, Enum):

tags_metadata = [
    {"name": "MLflow Info", "description": "REST APIs for retrieving elements from MLflow"},
    {"name": "Hardcoded Info", "description": "REST APIs for retrieving hard coded elements"},
    {"name": "Experimentation Pipeline", "description": "REST APIs for setting up and running the experimentation pipeline"},
    {"name": "Model Evaluation", "description": "REST APIs for retrieving model evaluation results"},
    {"name": "System Monitoring", "description": "REST APIs for monitoring the host machine of the API"},
]


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

# <api>.<http_operation>.(<route>)
@app.get("/")
async def root():
    return {"message": "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs"}

@app.get("/models/get_model_names", tags=['Hardcoded Info'])
async def get_model_names():
    return models

@ app.get("/metrics/get_metric_names", tags=['Hardcoded Info'])
async def get_metric_names():
    return metrics

@app.post('/upload/uploadCSVfile', tags=['Experimentation Pipeline'])
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
        ts = read_and_validate_input(fname, day_first)
    except WrongColumnNames:
        raise HTTPException(status_code=415, detail="There was an error validating the file. Please reupload CSV with 2 columns with names: 'Date', 'Load'")
    except DatesNotInOrder:
        raise HTTPException(status_code=415, detail="There was an error validating the file. Dates are not in order")
        # return {"message": "There was an error validating the file. Dates are not in order",
        #        "fname": fname}

    resolution_minutes = int((ts.index[1] - ts.index[0]).total_seconds() // 60)
    if resolution_minutes < 1 and resolution_minutes > 180:
       raise HTTPException(status_code=415, detail="Dataset resolution should be between 1 and 180 minutes")
       # return {"message": "Dataset resolution should be between 1 and 180 minutes"}

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

@app.get('/experimentation_pipeline/training/hyperparameter_entrypoints', tags=['Experimentation Pipeline'])
async def get_experimentation_pipeline_hparam_entrypoints():
    entrypoints = ConfigParser().read_entrypoints()
    return entrypoints

#@app.get('/experimentation_pipeline/etl/get_resolutions/')
#async def get_resolutions():
#    return ResolutionMinutes.dict()

@app.get('/get_mlflow_tracking_uri', tags=['MLflow Info'])
async def get_mlflow_tracking_uri():
    return mlflow.tracking.get_tracking_uri()

def mlflow_run(params: dict, experiment_name: str):
    pipeline_run = mlflow.projects.run(
            uri=".",
            experiment_name=experiment_name,
            entry_point="exp_pipeline",
            parameters=params,
            env_manager="local"
            )

@app.post('/experimentation_pipeline/run_all', tags=['Experimentation Pipeline'])
async def run_experimentation_pipeline(parameters: dict, background_tasks: BackgroundTasks):
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
        "time_covs": "None", # can give it as param to the pipeline and insert from front end: Need a list of all country codes to give to the user for selection
     }

    # TODO: generalize for all countries
    # if parameters["model"] != "NBEATS":
    #    params["time_covs"] = "PT"
    
    try:
        background_tasks.add_task(mlflow_run, params, parameters['experiment_name'])
    except Exception as e:
        raise HTTPException(status_code=404, detail="Could not initiate run. Check system logs")

#    try: 
#        pipeline_run = mlflow.projects.run(
#            uri=".",
#            experiment_name=experiment_name,
#            entry_point="exp_pipeline",
#            parameters=params,
#            env_manager="local"
#            )
#    except Exception as e:
#        return {"message": "Experimentation pipeline failed",
#                "status": "Failed",
                # "parent_run_id": mlflow.tracking.MlflowClient().get_run(pipeline_run.run_id),
#                "mlflow_tracking_uri": mlflow.tracking.get_tracking_uri(),
#                "experiment_name": experiment_name,
#                "experiment_id": MlflowClient().get_experiment_by_name(experiment_name).experiment_id
#           }
    # for now send them to MLflow to check their metrics.
#    return {"message": "Experimentation pipeline successful",
#            "status": "Success",
#            "parent_run_id": mlflow.tracking.MlflowClient().get_run(pipeline_run.run_id),
#            "mlflow_tracking_uri": mlflow.tracking.get_tracking_uri(),
#	    "experiment_name": experiment_name,
#            "experiment_id": MlflowClient().get_experiment_by_name(experiment_name).experiment_id
#	   }
    return {"message": "Experimentation pipeline initiated. Proceed to MLflow for details..."}

@app.get('/results/get_list_of_experiments', tags=['MLflow Info', 'Model Evaluation'])
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

@app.get('/results/get_best_run_id_by_mlflow_experiment/{experiment_id}/{metric}', tags=['MLflow Info', 'Model Evaluation'])
async def get_best_run_id_by_mlflow_experiment(experiment_id: str, metric: str = 'mape'):
    df = mlflow.search_runs([experiment_id], order_by=[f"metrics.{metric} ASC"])
    if df.empty:
        raise HTTPException(status_code=404, detail="No run has any metrics")
    else:
       best_run_id = df.loc[0, 'run_id']
       return best_run_id

@app.get('/results/get_forecast_vs_actual/{run_id}/n_samples/{n}', tags=['MLflow Info', 'Model Evaluation'])
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

@app.get('/results/get_metric_list/{run_id}', tags=['MLflow Info', 'Model Evaluation'])
async def get_metric_list(run_id: str):
    client = MlflowClient()
    metrix = client.get_run(run_id).data.metrics
    metrix_response = {"labels":[i for i in metrix.keys()], "data": [i for i in metrix.values()]}
    return metrix_response

@app.get('/system_monitoring/get_cpu_usage', tags=['System Monitoring'])
async def get_cpu_usage():
    cpu_count_logical = psutil.cpu_count()
    cpu_count = psutil.cpu_count(logical=False)
    cpu_usage = psutil.cpu_percent(percpu=True)
    cpu_percentage_response = {'labels': [f'CPU {i}' for i in range(1, len(cpu_usage)+1)], 'data': cpu_usage}
    response = {'barchart_1': cpu_percentage_response,
                'text_1': cpu_count,
                'text_2': cpu_count_logical}
    return response

@app.get('/system_monitoring/get_memory_usage', tags=['System Monitoring'])
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

@app.get('/system_monitoring/get_gpu_usage', tags=['System Monitoring'])
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
