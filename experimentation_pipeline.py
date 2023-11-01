"""
Downloads the REN dataset, ETLs (cleansing, resampling) it together with time covariates,
trains a darts model, and evaluates the model.
"""
import matplotlib.pyplot as plt
import tempfile
import pretty_errors
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, BlockRNNModel, NBEATSModel, LightGBMModel, RandomForest, TFTModel, TCNModel#, NHiTSModel, TransformerModel
import mlflow
import click
import os
import pretty_errors
from utils import download_online_file, check_mandatory
# from darts.utils.likelihood_models import ContinuousBernoulliLikelihood, GaussianLikelihood, DirichletLikelihood, ExponentialLikelihood, GammaLikelihood, GeometricLikelihood
import pretty_errors
import click
import os
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id
from utils import truth_checker, load_yaml_as_dict, download_online_file, ConfigParser, save_dict_as_yaml, none_checker
import optuna
import logging

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')
"""MLFLOW_TRACKING_INSECURE_TLS = os.environ.get('MLFLOW_TRACKING_INSECURE_TLS')
MLFLOW_TRACKING_USERNAME = os.environ.get('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.environ.get('MLFLOW_TRACKING_PASSWORD')

os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = MLFLOW_TRACKING_INSECURE_TLS
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)"""

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping " "(run_id=%s, status=%s)")
                % (run_info.run_id, run_info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                (
                    "Run matched, but has a different source code version, so skipping "
                    "(found=%s, expected=%s)"
                )
                % (previous_version, git_commit)
            )
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependent steps
def _get_or_run(entrypoint, parameters, git_commit, ignore_previous_run=True, use_cache=True):
    # TODO: this was removed to always run the pipeline from the beginning.
    if not ignore_previous_run:
        existing_run = _already_ran(entrypoint, parameters, git_commit)
        if use_cache and existing_run:
            print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
            return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, env_manager="local")
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


@click.command()
# load arguments
@click.option("--series-csv",
    type=str,
    default="None",
    help="Local timeseries file"
    )
@click.option("--series-uri",
    default="None",
    help="Online link to download the series from"
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
# etl arguments
@click.option("--resolution",
    default="None",
    type=str,
    help="Change the resolution of the dataset (minutes)."
)
@click.option('--year-range',
    default="None",
    type=str,
    help='Choose year range to include in the dataset.'
)
@click.option("--time-covs",
    default="false",
    type=str,
    help="Whether to add time covariates to the timeseries."
)
# training arguments
@click.option("--darts-model",
              type=click.Choice(
                  ['NBEATS',
                   'NHiTS',
                   'Transformer',
                   'RNN',
                   'TCN',
                   'BlockRNN',
                   'TFT',
                   'LightGBM',
                   'RandomForest',
                   'Naive',
                   'AutoARIMA']),
              multiple=False,
              default='None',
              help="The base architecture of the model to be trained"
              )
@click.option("--hyperparams-entrypoint", "-h",
              type=str,
              default='None',
              help=""" The entry point of config.yml under the 'hyperparams'
              one containing the desired hyperparameters for the selected model"""
              )
@click.option("--cut-date-val",
              type=str,
              default='None',
              help="Validation set start date [str: 'YYYYMMDD']"
              )
@click.option("--cut-date-test",
              type=str,
              default='None',
              help="Test set start date [str: 'YYYYMMDD']",
              )
@click.option("--test-end-date",
              type=str,
              default='None',
              help="Test set ending date [str: 'YYYYMMDD']",
              )
@click.option("--device",
              type=click.Choice(
                  ['gpu',
                   'cpu']),
              multiple=False,
              default='gpu',
              )
# eval
@click.option("--forecast-horizon",
              type=str,
              default="None")
@click.option("--stride",
              type=str,
              default="None")
@click.option("--retrain",
              type=str,
              default="false",
              help="Whether to retrain model during backtesting")
@click.option("--ignore-previous-runs",
              type=str,
              default="true",
              help="Whether to ignore previous step runs while running the pipeline")
@click.option("--scale",
              type=str,
              default="true",
              help="Whether to scale the target series")
@click.option("--scale-covs",
              type=str,
              default="true",
              help="Whether to scale the covariates")
@click.option("--day-first",
              type=str,
              default="true",
              help="Whether the date has the day before the month")
@click.option("--country",
              type=str,
              default="PT",
              help="The country code this dataset belongs to")

@click.option("--std-dev",
              type=str,
              default="4.5",
              help="The number to be multiplied with the standard deviation of \
                    each 1 month  period of the dataframe. The result is then used as \
                    a cut-off value as described above")

@click.option("--max-thr",
              type=str,
              default="-1",
              help="If there is a consecutive subseries of NaNs longer than max_thr, \
                    then it is not imputed and returned with NaN values. If -1, every value will\
                    be imputed regardless of how long the consecutive subseries of NaNs it belongs \
                    to is.")

@click.option("--a",
              type=str,
              default="0.3",
              help="The weight that shows how quickly simple interpolation's weight decreases as \
                    the distacne to the nearest non NaN value increases")

@click.option("--wncutoff",
              type=str,
              default="0.000694",
              help="Historical data will only take into account dates that have at most wncutoff distance \
                    from the current null value's WN(Week Number)")

@click.option("--ycutoff",
             type=str,
             default="3",
             help="Historical data will only take into account dates that have at most ycutoff distance \
                   from the current null value's year")

@click.option("--ydcutoff",
             type=str,
             default="30",
             help="Historical data will only take into account dates that have at most ydcutoff distance \
                   from the current null value's yearday")

@click.option("--shap-data-size",
             type=str,
             default="100",
             help="Size of shap dataset in samples")

@click.option("--analyze-with-shap",
             type=str,
             default="False",
             help="Whether to do SHAP analysis on the model. Only global forecasting models are supported")

@click.option("--multiple",
    type=str,
    default="false",
    help="Whether to train on multiple timeseries")

@click.option("--eval-series",
    type=str,
    default="None",
    help="On which timeseries to run the backtesting. Only for multiple timeseries")

@click.option("--n-trials",
    type=str,
    default="100",
    help="How many trials optuna will run")

@click.option("--opt-test",
    type=str,
    default="false",
    help="Whether we are running optuna")

@click.option("--from-database",
    default="false",
    type=str,
    help="Whether to read the dataset from the database."
)
@click.option("--database-name",
    default="rdn_load_data",
    type=str,
    help="Which database file to read."
)
@click.option("--num-workers",
        type=str,
        default="4",
        help="Number of threads that will be used by pytorch")

@click.option("--eval-method",
    type=click.Choice(
        ['ts_ID',
         'ID']),
    default="ts_ID",
    help="what ID type is speciffied in eval_series: if ts_ID is speciffied, then we look at Timeseries ID column. Else, we look at ID column ")
@click.option("--l-interpolation",
    type=str,
    default="false",
    help="Whether to only use linear interpolation")

@click.option("--rmv-outliers",
    type=str,
    default="true",
    help="Whether to remove outliers")

@click.option("--loss-function",
    type=click.Choice(
        ["mape",
         "smape",
         "mase", 
         "mae",
         "rmse"]),
    default="mape",
    help="Loss function to use for optuna")

@click.option("--evaluate-all-ts",
    type=str,
    default="false",
    help="Whether to validate the models for all timeseries, and return the mean of their metrics")

@click.option("--convert-to-local-tz",
    type=str,
    default="true",
    help="Whether to convert time")

@click.option("--grid-search",
    type=str,
    default="false",
    help="Whether to run a grid search or use tpe in optuna")

@click.option("--shap-input-length",
             type=str,
             default="None",
             help="The length of each sample of the dataset used during SHAP analysis. Is taken into account \
             only if not evaluating a model with input_chunk_length as one of its parameters. In the latter case, \
             shap_input_length=input_chunk_length")

@click.option("--ts-used-id",
    type=str,
    default="None",
    help="If not None, only ts with this id will be used for training and evaluation. Applicable only on multiple ts files")

@click.option("--m-mase",
    type=str,
    default="1",
    help="m to use for mase metric")

@click.option("--min-non-nan-interval",
    type=str,
    default="24",
    help="If after imputation there exist continuous intervals of non nan values that are smaller than min_non_nan_interval \
        time steps, these intervals are all replaced by nan values")

@click.option("--num-samples",
    type=str,
    default="1",
    help="Number of samples to use for evaluating/validating a probabilistic model's output")
@click.option("--resampling-agg-method",
    default="averaging",
    type=click.Choice(['averaging',
                       'summation',
                       'downsampling']),
    multiple=False,
    help="Method to use for resampling."
    )

def workflow(series_csv, series_uri, past_covs_csv, past_covs_uri, future_covs_csv, future_covs_uri, year_range, 
             resolution, time_covs, darts_model, hyperparams_entrypoint, cut_date_val, test_end_date, cut_date_test, device,
             forecast_horizon, stride, retrain, ignore_previous_runs, scale, scale_covs, day_first,
             country, std_dev, max_thr, a, wncutoff, ycutoff, ydcutoff, shap_data_size, analyze_with_shap,
             multiple, eval_series, n_trials, opt_test, from_database, database_name, num_workers, eval_method,
             l_interpolation, rmv_outliers, loss_function, evaluate_all_ts, convert_to_local_tz, grid_search, shap_input_length,
             ts_used_id, m_mase, min_non_nan_interval, num_samples, resampling_agg_method):

    disable_warnings(InsecureRequestWarning)
    
    if none_checker(series_uri) == None and not truth_checker(from_database) and none_checker(series_uri) == None:
        check_mandatory(series_csv, "series_csv", [["series_uri", "None"], ["from_database", "False"]])

    if none_checker(resolution) == None:
        check_mandatory(resolution, "resolution", [])

    if none_checker(darts_model) == None:
        check_mandatory(darts_model, "darts_model", [])
    
    if none_checker(hyperparams_entrypoint) == None:
        check_mandatory(hyperparams_entrypoint, "hyperparams_entrypoint", [])

    if none_checker(cut_date_val) == None:
        check_mandatory(cut_date_val, "cut_date_val", [])

    if none_checker(cut_date_test) == None:
        check_mandatory(cut_date_test, "cut_date_test", [])

    if none_checker(forecast_horizon) == None:
        check_mandatory(forecast_horizon, "forecast_horizon", [])

    if none_checker(eval_series) == None and truth_checker(multiple) and not truth_checker(evaluate_all_ts):
        check_mandatory(eval_series, "eval_series", [["multiple", "True"], ["evaluate_all_ts", "False"]])


    # Argument preprocessing
    ignore_previous_runs = truth_checker(ignore_previous_runs)
    opt_test = truth_checker(opt_test)
    from_database = truth_checker(from_database)
    time_covs = truth_checker(time_covs)


    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run(run_name=darts_model + '_pipeline') as active_run:
        mlflow.set_tag("stage", "main")

        # 1.Load Data
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)

        load_raw_data_params = {"series_csv": series_csv, 
                                "series_uri": series_uri,
                                "past_covs_csv": past_covs_csv, 
                                "past_covs_uri": past_covs_uri, 
                                "future_covs_csv": future_covs_csv, 
                                "future_covs_uri": future_covs_uri,
                                "day_first": day_first, 
                                "multiple": multiple, 
                                "from_database": from_database,
                                "database_name": database_name,
                                "resolution":resolution}
        
        load_raw_data_run = _get_or_run("load_raw_data", load_raw_data_params, git_commit, ignore_previous_runs)
        # series_uri = f"{load_raw_data_run.info.artifact_uri}/raw_data/series.csv" \
        #                 .replace("s3:/", S3_ENDPOINT_URL)
        load_data_series_uri = load_raw_data_run.data.tags['dataset_uri'].replace("s3:/", S3_ENDPOINT_URL)
        infered_resolution_series = load_raw_data_run.data.tags['infered_resolution_series']

        load_data_past_covs_uri = load_raw_data_run.data.tags['past_covs_uri'].replace("s3:/", S3_ENDPOINT_URL)
        infered_resolution_past = load_raw_data_run.data.tags['infered_resolution_past']

        load_data_future_covs_uri = load_raw_data_run.data.tags['future_covs_uri'].replace("s3:/", S3_ENDPOINT_URL)
        infered_resolution_future = load_raw_data_run.data.tags['infered_resolution_future']

        # 2. ETL
        etl_params = {"series_uri": load_data_series_uri,
                      "year_range": year_range,
                      "resolution": resolution,
                      "time_covs": time_covs,
                      "day_first": day_first,
                      "country": country,
                      "std_dev": std_dev,
                      "max_thr": max_thr,
                      "a": a,
                      "wncutoff": wncutoff,
                      "ycutoff": ycutoff,
                      "ydcutoff": ydcutoff,
                      "multiple": multiple,
                      "l_interpolation": l_interpolation,
                      "rmv_outliers": rmv_outliers,
                      "convert_to_local_tz": convert_to_local_tz,
                      "ts_used_id": ts_used_id,
                      "infered_resolution_series": infered_resolution_series,
                      "past_covs_uri": load_data_past_covs_uri,
                      "infered_resolution_past": infered_resolution_past,
                      "future_covs_uri": load_data_future_covs_uri,
                      "infered_resolution_future": infered_resolution_future,
                      "min_non_nan_interval": min_non_nan_interval,
                      "cut_date_val": cut_date_val,
                      "resampling_agg_method": resampling_agg_method}

        etl_run = _get_or_run("etl", etl_params, git_commit, ignore_previous_runs)

        etl_series_uri =  etl_run.data.tags["series_uri"].replace("s3:/", S3_ENDPOINT_URL)
        etl_past_covariates_uri =  etl_run.data.tags["past_covs_uri"].replace("s3:/", S3_ENDPOINT_URL)
        etl_future_covariates_uri =  etl_run.data.tags["future_covs_uri"].replace("s3:/", S3_ENDPOINT_URL)

        # weather_covariates_uri = ...

        # 3. Training or Optuna test
        if opt_test:
            optuna_params = {
                "series_uri": etl_series_uri,
                "future_covs_uri": etl_future_covariates_uri,
                "past_covs_uri": etl_past_covariates_uri,
                "year_range": year_range,
                "resolution": resolution,
                "darts_model": darts_model,
                "hyperparams_entrypoint": hyperparams_entrypoint,
                "cut_date_val": cut_date_val,
                "test_end_date": test_end_date,
                "cut_date_test": cut_date_test,
                "device": device,
                "forecast_horizon": forecast_horizon,
                "stride": stride,
                "retrain": retrain,
                "scale": scale,
                "scale_covs": scale_covs,
                "multiple": multiple,
                "eval_series": eval_series,
                "n_trials": n_trials,
                "num_workers": num_workers,
                "day_first": day_first,
                "eval_method": eval_method,
                "loss_function": loss_function,
                "evaluate_all_ts": evaluate_all_ts,
                "grid_search" : grid_search,
                "num_samples": num_samples,
            }
            train_opt_run = _get_or_run("optuna_search", optuna_params, git_commit, ignore_previous_runs)

        else:
            train_params = {
                "series_uri": etl_series_uri,
                "future_covs_uri": etl_future_covariates_uri,
                "past_covs_uri": etl_past_covariates_uri,
                "darts_model": darts_model,
                "hyperparams_entrypoint": hyperparams_entrypoint,
                "cut_date_val": cut_date_val,
                "cut_date_test": cut_date_test,
                "test_end_date": test_end_date,
                "device": device,
                "scale": scale,
                "scale_covs": scale_covs,
                "multiple": multiple,
                "training_dict": None,
                "num_workers": num_workers,
                "day_first": day_first,
                "resolution": resolution,
            }
            train_opt_run = _get_or_run("train", train_params, git_commit, ignore_previous_runs)

            # Log train params (mainly for logging hyperparams to father run)
        for param_name, param_value in train_opt_run.data.params.items():
            try:
                mlflow.log_param(param_name, param_value)
                print(param_name, param_value)
            except mlflow.exceptions.RestException:
                pass
            except mlflow.exceptions.MlflowException:
                pass

        train_opt_model_uri = train_opt_run.data.tags["model_uri"].replace("s3:/", S3_ENDPOINT_URL)
        train_opt_model_type = train_opt_run.data.tags["model_type"]
        train_opt_series_uri = train_opt_run.data.tags["series_uri"].replace("s3:/", S3_ENDPOINT_URL)
        train_opt_future_covariates_uri = train_opt_run.data.tags["future_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
        train_opt_past_covariates_uri = train_opt_run.data.tags["past_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
        train_opt_scaler_uri = train_opt_run.data.tags["scaler_uri"].replace("s3:/", S3_ENDPOINT_URL)
        train_opt_setup_uri = train_opt_run.data.tags["setup_uri"].replace("s3:/", S3_ENDPOINT_URL)

        # 4. Evaluation
        ## load setup file
        setup_file = download_online_file(
            train_opt_setup_uri, "setup.yml")
        setup = load_yaml_as_dict(setup_file)
        print(f"\nSplit info: {setup} \n")

        eval_params = {
                "series_uri": train_opt_series_uri,
                "future_covs_uri": train_opt_future_covariates_uri,
                "past_covs_uri": train_opt_past_covariates_uri,
                "scaler_uri": train_opt_scaler_uri,
                "cut_date_test": setup['test_start'],
                "test_end_date": setup['test_end'],
                "model_uri": train_opt_model_uri,
                "model_type": train_opt_model_type,
                "forecast_horizon": forecast_horizon,
                "stride": stride,
                "retrain": retrain,
                "shap_input_length" : shap_input_length,
                "shap_output_length" : forecast_horizon,
                "size" : shap_data_size,
                "analyze_with_shap" : analyze_with_shap,
                "multiple": multiple,
                "eval_series": eval_series,
                "day_first": day_first,
                "resolution": resolution,
                "eval_method": eval_method,
                "evaluate_all_ts": evaluate_all_ts,
                "m_mase": m_mase,
                "num_samples": num_samples,
            }

        if "input_chunk_length" in train_opt_run.data.params:
            eval_params["shap_input_length"] = train_opt_run.data.params["input_chunk_length"]
            
        # Naive models require retrain=True
        if "naive" in [train_opt_run.data.params["darts_model"].lower()]:
            eval_params["retrain"] = True
            print("\Warning: Switching retrain flag to True as Naive models require...\n")


        eval_run = _get_or_run("eval", eval_params, git_commit)

        # Log eval metrics to father run for consistency and clear results
        mlflow.log_metrics(eval_run.data.metrics)

if __name__ == "__main__":
    workflow()
