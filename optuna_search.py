import pretty_errors
from utils import none_checker, ConfigParser, download_online_file, load_local_csv_as_darts_timeseries, truth_checker, load_yaml_as_dict, load_model, load_scaler, multiple_dfs_to_ts_file
from preprocessing import scale_covariates, split_dataset, split_nans
from darts.utils.missing_values import extract_subseries

from functools import reduce
from darts.metrics import mape as mape_darts
from darts.metrics import mase as mase_darts
from darts.metrics import mae as mae_darts
from darts.metrics import rmse as rmse_darts
from darts.metrics import smape as smape_darts
from darts.models import (
    NaiveSeasonal,
)
# the following are used through eval(darts_model + 'Model')
from darts.models import RNNModel, BlockRNNModel, NBEATSModel, TFTModel, NaiveDrift, NaiveSeasonal, TCNModel, NHiTSModel, TransformerModel
# from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.gradient_boosted_model import LightGBMModel
from darts.models.forecasting.random_forest import RandomForest
from darts.utils.likelihood_models import ContinuousBernoulliLikelihood, GaussianLikelihood, DirichletLikelihood, ExponentialLikelihood, GammaLikelihood, GeometricLikelihood

import yaml
import mlflow
import click
import os
import torch
import logging
import pickle
import tempfile
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import shutil
import optuna
import pandas as pd
# Inference requirements to be stored with the darts flavor !!
from sys import version_info
import torch, cloudpickle, darts
import matplotlib.pyplot as plt
import pprint
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import random

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
mlflow_serve_conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
                'cloudpickle=={}'.format(cloudpickle.__version__),
                'darts=={}'.format(darts.__version__),
                'pretty_errors=={}'.format(pretty_errors.__version__),
                'torch=={}'.format(torch.__version__),
                'mlflow=={}'.format(mlflow.__version__)
            ],
        },
    ],
    'name': 'darts_infer_pl_env'
}

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
# a period of 5 epochs (`patience`)
def log_optuna(study, opt_tmpdir, hyperparams_entrypoint, mlrun, log_model=False, curr_loss=0, model=None, darts_model=None, scale="False", scalers_dir=None, features_dir=None, opt_all_results=None, past_covariates=None, future_covariates=None, evaluate_all_ts=False):
    scale = truth_checker(scale)
    if evaluate_all_ts: 
        mlflow.log_artifacts(opt_all_results, "optuna_val_results_all_timeseries")
    
    if log_model and (len(study.trials_dataframe()[study.trials_dataframe()["state"] == "COMPLETE"]) < 1 or study.best_trial.values[0] >= curr_loss):
        if darts_model in ['NHiTS', 'NBEATS', 'RNN', 'BlockRNN', 'TFT', 'TCN', 'Transformer']:
            logs_path = f"./darts_logs/{mlrun.info.run_id}"
            model_type = "pl"
        elif darts_model in ['LightGBM', 'RandomForest']:
            print('\nStoring the model as pkl to MLflow...')
            logging.info('\nStoring the model as pkl to MLflow...')
            forest_dir = tempfile.mkdtemp()

            pickle.dump(model, open(
                f"{forest_dir}/_model.pkl", "wb"))

            logs_path = forest_dir
            model_type = "pkl"

        if scale:
            source_dir = scalers_dir
            target_dir = logs_path
            file_names = os.listdir(source_dir)
            for file_name in file_names:
                shutil.move(os.path.join(source_dir, file_name),
                target_dir)
            
        ## Create and move model info in logs path
        model_info_dict = {
            "darts_forecasting_model":  model.__class__.__name__,
            "run_id": mlrun.info.run_id
        }
        with open('model_info.yml', mode='w') as outfile:
            yaml.dump(
                model_info_dict,
                outfile,
                default_flow_style=False)

        
        shutil.move('model_info.yml', logs_path)

        ## Rename logs path to get rid of run name
        if model_type == 'pkl':
            logs_path_new = logs_path.replace(
            forest_dir.split('/')[-1], mlrun.info.run_id)
            os.rename(logs_path, logs_path_new)
        elif model_type == 'pl':
            logs_path_new = logs_path
        
        mlflow_model_root_dir = "pyfunc_model"
            
        ## Log MLflow model and code
        mlflow.pyfunc.log_model(mlflow_model_root_dir,
                            loader_module="darts_flavor",
                            data_path=logs_path_new,
                            code_path=['../utils.py', '../inference.py', '../darts_flavor.py'],
                            conda_env=mlflow_serve_conda_env)
            
        shutil.rmtree(logs_path_new)

        print("\nArtifacts are being uploaded to MLflow...")
        logging.info("\nArtifacts are being uploaded to MLflow...")
        mlflow.log_artifacts(features_dir, "features")

        if scale:
            # mlflow.log_artifacts(scalers_dir, f"{mlflow_model_path}/scalers")
            mlflow.set_tag(
                'scaler_uri',
                f'{mlrun.info.artifact_uri}/{mlflow_model_root_dir}/data/{mlrun.info.run_id}/scaler_series.pkl')
        else:
            mlflow.set_tag('scaler_uri', 'mlflow_artifact_uri')



        mlflow.set_tag("run_id", mlrun.info.run_id)
        mlflow.set_tag("stage", "optuna_search")
        mlflow.set_tag("model_type", model_type)

        mlflow.set_tag(
            'setup_uri',
            f'{mlrun.info.artifact_uri}/features/split_info.yml')

        mlflow.set_tag("darts_forecasting_model",
            model.__class__.__name__)
        # model_uri
        mlflow.set_tag('model_uri', mlflow.get_artifact_uri(
            f"{mlflow_model_root_dir}/data/{mlrun.info.run_id}"))
        # inference_model_uri
        mlflow.set_tag('pyfunc_model_folder', mlflow.get_artifact_uri(
            f"{mlflow_model_root_dir}"))

        mlflow.set_tag('series_uri',
            f'{mlrun.info.artifact_uri}/features/series.csv')

        if future_covariates is not None:
            mlflow.set_tag(
                'future_covariates_uri',
                f'{mlrun.info.artifact_uri}/features/future_covariates_transformed.csv')
        else:
            mlflow.set_tag(
                'future_covariates_uri',
                'mlflow_artifact_uri')

        if past_covariates is not None:
            mlflow.set_tag(
                'past_covariates_uri',
                f'{mlrun.info.artifact_uri}/features/past_covariates_transformed.csv')
        else:
            mlflow.set_tag('past_covariates_uri',
                'mlflow_artifact_uri')

        print("\nArtifacts uploaded.")
        logging.info("\nArtifacts uploaded.")
    if not log_model:
        ######################
        # Log hyperparameters
        best_params = study.best_params
        if "scale" in best_params:
            best_params["scale_optuna"] = best_params["scale"]
            del best_params["scale"]
        mlflow.log_params(best_params)

        # Log log_metrics
        mlflow.log_metrics(study.best_trial.user_attrs)




    if len(study.trials_dataframe()[study.trials_dataframe()["state"] == "COMPLETE"]) <= 1: return

    plt.close()

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{opt_tmpdir}/plot_optimization_history.html")
    plt.close()

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{opt_tmpdir}/plot_param_importances.html")
    plt.close()

    fig = optuna.visualization.plot_slice(study)
    fig.write_html(f"{opt_tmpdir}/plot_slice.html")
    plt.close()

    study.trials_dataframe().to_csv(f"{opt_tmpdir}/{hyperparams_entrypoint}.csv")

    print("\nUploading optuna plots to MLflow server...")
    logging.info("\nUploading optuna plots to MLflow server...")

    mlflow.log_artifacts(opt_tmpdir, "optuna_results")

def append(x, y):
    return x.append(y)

def objective(series_csv, series_uri, future_covs_csv, future_covs_uri,
             past_covs_csv, past_covs_uri, year_range, resolution,
             darts_model, hyperparams_entrypoint, cut_date_val, test_end_date, 
             cut_date_test, device, forecast_horizon, stride, retrain, scale, 
             scale_covs, multiple, eval_series, mlrun, trial, study, opt_tmpdir, 
             num_workers, day_first, eval_method, loss_function, opt_all_results,
             evaluate_all_ts, num_samples):

                hyperparameters = ConfigParser('../config_opt.yml').read_hyperparameters(hyperparams_entrypoint)
                training_dict = {}
                for param, value in hyperparameters.items():
                    if type(value) == list and value and value[0] == "range":
                        if type(value[1]) == int:
                            training_dict[param] = trial.suggest_int(param, value[1], value[2], value[3])
                        else:
                            training_dict[param] = trial.suggest_float(param, value[1], value[2], value[3])
                    elif type(value) == list and value and value[0] == "list":
                        training_dict[param] = trial.suggest_categorical(param, value[1:])
                    elif type(value) == list and value and value[0] == "equal":
                        continue
                    else:
                        training_dict[param] = value
                for param, value in hyperparameters.items():
                    if type(value) == list and value and value[0] == "equal":
                        training_dict[param] = training_dict[value[1]]
                if 'scale' in training_dict:
                     scale = training_dict['scale']
                     del training_dict['scale']

                #TODO: Make it work with csvs also
                model, scaler, train_future_covariates, train_past_covariates, features_dir, scalers_dir = train(
                      series_uri=series_uri,
                      future_covs_uri=future_covs_uri,
                      past_covs_uri=past_covs_uri, # fix that in case REAL Temperatures come -> etl_temp_covs_uri. For forecasts, integrate them into future covariates!!
                      darts_model=darts_model,
                      hyperparams_entrypoint=hyperparams_entrypoint,
                      cut_date_val=cut_date_val,
                      cut_date_test=cut_date_test,
                      test_end_date=test_end_date,
                      device=device,
                      scale=scale,
                      scale_covs=scale_covs,
                      multiple=multiple,
                      training_dict=training_dict,
                      mlrun=mlrun,
                      num_workers=num_workers,
                      day_first=day_first,
                      resolution=resolution,
                      )
                try:
                    trial.set_user_attr("epochs_trained", model.epochs_trained)
                except:
                    pass
                metrics = validate(
                    series_uri=series_uri,
                    future_covariates=train_future_covariates,
                    past_covariates=train_past_covariates,
                    scaler=scaler,
                    cut_date_test=cut_date_test,
                    test_end_date=test_end_date,#check that again
                    model=model,
                    forecast_horizon=forecast_horizon,
                    stride=stride,
                    retrain=retrain,
                    multiple=multiple,
                    eval_series=eval_series,
                    cut_date_val=cut_date_val,
                    mlrun=mlrun,
                    resolution=resolution,
                    eval_method=eval_method,
                    opt_all_results=opt_all_results,
                    evaluate_all_ts=evaluate_all_ts,
                    study=study,
                    num_samples=num_samples,
                    )
                trial.set_user_attr("mape", float(metrics["mape"]))
                trial.set_user_attr("smape", float(metrics["smape"]))
                trial.set_user_attr("mase", float(metrics["mase"]))
                trial.set_user_attr("mae", float(metrics["mae"]))
                trial.set_user_attr("rmse", float(metrics["rmse"]))
                trial.set_user_attr("nrmse_max", float(metrics["nrmse_max"]))
                trial.set_user_attr("nrmse_mean", float(metrics["nrmse_mean"]))
                log_optuna(study, opt_tmpdir, hyperparams_entrypoint, mlrun, 
                    log_model=True, curr_loss=float(metrics[loss_function]), 
                    model=model, darts_model=darts_model, scale=scale, scalers_dir=scalers_dir, 
                    features_dir=features_dir, opt_all_results=opt_all_results, 
                    past_covariates=train_past_covariates, future_covariates=train_future_covariates, 
                    evaluate_all_ts=evaluate_all_ts)

                return metrics[loss_function]

def train(series_uri, future_covs_uri, past_covs_uri, darts_model,
          hyperparams_entrypoint, cut_date_val, cut_date_test,
          test_end_date, device, scale, scale_covs, multiple,
          training_dict, mlrun, num_workers, day_first, resolution):


    # Argument preprocessing

    ## test_end_date
    num_workers = int(num_workers)
    torch.set_num_threads(num_workers)

    my_stopper = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-6,
        mode='min',
    )

    test_end_date = none_checker(test_end_date)

    ## scale or not
    scale = truth_checker(scale)
    scale_covs = truth_checker(scale_covs)

    multiple = truth_checker(multiple)

    ## hyperparameters
    hyperparameters = training_dict

    ## device
    #print("param", hyperparameters)
    if device == 'gpu' and torch.cuda.is_available():
        device = 'gpu'
        print("\nGPU is available")
    else:
        device = 'cpu'
        print("\nGPU is available")
    ## series and covariates uri and csv
    series_uri = none_checker(series_uri)
    future_covs_uri = none_checker(future_covs_uri)
    past_covs_uri = none_checker(past_covs_uri)

    # redirect to local location of downloaded remote file
    if series_uri is not None:
        download_file_path = download_online_file(series_uri, dst_filename="load.csv")
        series_csv = download_file_path.replace('/', os.path.sep).replace("'", "")
    else:
        series_csv = None

    if  future_covs_uri is not None:
        download_file_path = download_online_file(future_covs_uri, dst_filename="future.csv")
        future_covs_csv = download_file_path.replace('/', os.path.sep).replace("'", "")
    else:
        future_covs_csv = None

    if  past_covs_uri is not None:
        download_file_path = download_online_file(past_covs_uri, dst_filename="past.csv")
        past_covs_csv = download_file_path.replace('/', os.path.sep).replace("'", "")
    else:
        past_covs_csv = None

    ## model
    # TODO: Take care of future covariates (RNN, ...) / past covariates (BlockRNN, NBEATS, ...)
    if darts_model in ["NBEATS", "BlockRNN", "TCN", "NHiTS", "Transformer"]:
        """They do not accept future covariates as they predict blocks all together.
        They won't use initial forecasted values to predict the rest of the block
        So they won't need to additionally feed future covariates during the recurrent process.
        """
        #TODO Concatenate future covs to past??
        #past_covs_csv = future_covs_csv
        future_covs_csv = None

    elif darts_model in ["RNN"]:
        """Does not accept past covariates as it needs to know future ones to provide chain forecasts
        its input needs to remain in the same feature space while recurring and with no future covariates
        this is not possible. The existence of past_covs is not permitted for the same reason. The
        feature space will change during inference. If for example I have current temperature and during
        the forecast chain I only have time covariates, as I won't know the real temp then a constant \
        architecture like LSTM cannot handle this"""
        past_covs_csv = None
        # TODO: when actual weather comes extend it, now the stage only accepts future covariates as argument.
    #elif: extend for other models!! (time_covariates are always future covariates, but some models can't handle them as so)

    future_covariates = none_checker(future_covs_csv)
    past_covariates = none_checker(past_covs_csv)


    ######################
    # Load series and covariates datasets
    time_col = "Datetime"
    series, id_l, ts_id_l = load_local_csv_as_darts_timeseries(
                local_path=series_csv,
                name='series',
                time_col=time_col,
                last_date=test_end_date,
                multiple=multiple,
                day_first=day_first,
                resolution=resolution)
    if future_covariates is not None:
        future_covariates, id_l_future_covs, ts_id_l_future_covs = load_local_csv_as_darts_timeseries(
                local_path=future_covs_csv,
                name='future covariates',
                time_col=time_col,
                last_date=test_end_date,
                multiple=True,
                day_first=day_first,
                resolution=resolution)
    else:
        future_covariates, id_l_future_covs, ts_id_l_future_covs = None, None, None
    if past_covariates is not None:
        past_covariates, id_l_past_covs, ts_id_l_past_covs = load_local_csv_as_darts_timeseries(
                local_path=past_covs_csv,
                name='past covariates',
                time_col=time_col,
                last_date=test_end_date,
                multiple=True,
                day_first=day_first,
                resolution=resolution)
    else:
        past_covariates, id_l_past_covs, ts_id_l_past_covs = None, None, None

    if scale:
        scalers_dir = tempfile.mkdtemp()
    else:
        scalers_dir = None
    features_dir = tempfile.mkdtemp()

    ######################
    # Train / Test split
    print(
        f"\nTrain / Test split: Validation set starts: {cut_date_val} - Test set starts: {cut_date_test} - Test set end: {test_end_date}")
    logging.info(
        f"\nTrain / Test split: Validation set starts: {cut_date_val} - Test set starts: {cut_date_test} - Test set end: {test_end_date}")

    ## series
    series_split = split_dataset(
            series,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            store_dir=features_dir,
            name='series',
            conf_file_name='split_info.yml',
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l)
        ## future covariates
    future_covariates_split = split_dataset(
            future_covariates,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            # store_dir=features_dir,
            name='future_covariates',
            multiple=True,
            id_l=id_l_future_covs,
            ts_id_l=ts_id_l_future_covs)
        ## past covariates
    past_covariates_split = split_dataset(
            past_covariates,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            # store_dir=features_dir,
            name='past_covariates',
            multiple=True,
            id_l=id_l_past_covs,
            ts_id_l=ts_id_l_past_covs)
    #################
    # Scaling
    print("\nScaling...")
    logging.info("\nScaling...")

    ## scale series
    series_transformed = scale_covariates(
            series_split,
            store_dir=features_dir,
            filename_suffix="series_transformed.csv",
            scale=scale,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l
            )
    if scale:
        pickle.dump(series_transformed["transformer"], open(f"{scalers_dir}/scaler_series.pkl", "wb"))
        ## scale future covariates
    future_covariates_transformed = scale_covariates(
            future_covariates_split,
            store_dir=features_dir,
            filename_suffix="future_covariates_transformed.csv",
            scale=scale_covs,
            multiple=True,
            id_l=id_l_future_covs,
            ts_id_l=ts_id_l_future_covs
            )
        ## scale past covariates
    past_covariates_transformed = scale_covariates(
            past_covariates_split,
            store_dir=features_dir,
            filename_suffix="past_covariates_transformed.csv",
            scale=scale_covs,
            multiple=True,
            id_l=id_l_past_covs,
            ts_id_l=ts_id_l_past_covs
            )

    ######################
    # Model training
    print("\nTraining model...")
    logging.info("\nTraining model...")
    pl_trainer_kwargs = {"callbacks": [my_stopper],
                         "accelerator": 'auto',
                        #  "gpus": 1,
                        #  "auto_select_gpus": True,
                         "log_every_n_steps": 10}

    print("\nTraining on series:\n")
    logging.info("\nTraining on series:\n")
    if multiple:
        for i, series in enumerate(series_transformed['train']):
            print(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
            logging.info(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
    else:
        print(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")
        logging.info(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")
    print("")

    #TODO maybe modify print to include split train based on nans
    #TODO make more efficient by also spliting covariates where the nans are split 
    print("TRAIN,,,,,", series_transformed['train'])
    series_transformed['train'], past_covariates_transformed['train'], future_covariates_transformed['train'] = \
            split_nans(series_transformed['train'], past_covariates_transformed['train'], future_covariates_transformed['train'])

    ## choose architecture
    if darts_model in ['NBEATS', 'RNN', 'BlockRNN', 'TFT', 'TCN', 'NHiTS', 'Transformer']:
        hparams_to_log = hyperparameters
        if 'learning_rate' in hyperparameters:
            hyperparameters['optimizer_kwargs'] = {'lr': hyperparameters['learning_rate']}
            del hyperparameters['learning_rate']

        if 'likelihood' in hyperparameters:
            hyperparameters['likelihood'] = eval(hyperparameters['likelihood']+"Likelihood"+"()")
        print(hyperparameters)
        model = eval(darts_model + 'Model')(
            force_reset=True,
            save_checkpoints=True,
            log_tensorboard=False,
            model_name=mlrun.info.run_id,
            pl_trainer_kwargs=pl_trainer_kwargs,
            **hyperparameters
        )
        ## fit model
        # try:
        # print(series_transformed['train'])
        # print(series_transformed['val'])
        #print("SERIES", series_transformed['train'])
        model.fit(series_transformed['train'],
            future_covariates=future_covariates_transformed['train'],
            past_covariates=past_covariates_transformed['train'],
            val_series=series_transformed['val'],
            val_future_covariates=future_covariates_transformed['val'],
            val_past_covariates=past_covariates_transformed['val'])

        # LightGBM and RandomForest
    elif darts_model in ['LightGBM', 'RandomForest']:

        if "lags_future_covariates" in hyperparameters:
            if truth_checker(str(hyperparameters["future_covs_as_tuple"])):
                hyperparameters["lags_future_covariates"] = tuple(
                    hyperparameters["lags_future_covariates"])
            hyperparameters.pop("future_covs_as_tuple")

        if future_covariates is None:
            hyperparameters["lags_future_covariates"] = None
        if past_covariates is None:
            hyperparameters["lags_past_covariates"] = None

        hparams_to_log = hyperparameters

        if darts_model == 'RandomForest':
            model = RandomForest(**hyperparameters)
        elif darts_model == 'LightGBM':
            model = LightGBMModel(**hyperparameters)

        print(f'\nTraining {darts_model}...')
        logging.info(f'\nTraining {darts_model}...')

        model.fit(
            series=series_transformed['train'],
            # val_series=series_transformed['val'],
            future_covariates=future_covariates_transformed['train'],
            past_covariates=past_covariates_transformed['train'],
            # val_future_covariates=future_covariates_transformed['val'],
            # val_past_covariates=past_covariates_transformed['val']
            )
    if scale:
        scaler = series_transformed["transformer"]
    else:
        scaler = None

    if future_covariates is not None:
        return_future_covariates = future_covariates_transformed['all']
    else:
        return_future_covariates = None

    if past_covariates is not None:
        return_past_covariates = past_covariates_transformed['all']
    else:
        return_past_covariates = None
    return model, scaler, return_future_covariates, return_past_covariates, features_dir, scalers_dir

def backtester(model,
               series_transformed,
               test_start_date,
               forecast_horizon,
               stride=None,
               series=None,
               transformer_ts=None,
               retrain=False,
               future_covariates=None,
               past_covariates=None,
               path_to_save_backtest=None,
               num_samples=1):
               #TODO Add mase
    """ Does the same job with advanced forecast but much more quickly using the darts
    bult-in historical_forecasts method. Use this for evaluation. The other only
    provides pure inference. Provide a unified timeseries test set point based
    on test_start_date. series_transformed does not need to be adjacent to
    training series. if transformer_ts=None then no inverse transform is applied
    to the model predictions.
    Parameters
    ----------
    Returns
    ----------
    """
    # produce the fewest forecasts possible.
    if stride is None:
        stride = forecast_horizon
    test_start_date = pd.Timestamp(test_start_date)

    #keep last non nan values
    #must be sufficient for historical_forecasts and mase calculation
    #TODO Add check for that in the beggining
    series = extract_subseries(series, min_gap_size=1)[-1]
    series_transformed = extract_subseries(series_transformed, min_gap_size=1)[-1]

    # produce list of forecasts
    #print("backtesting starting at", test_start_date, "series:", series_transformed)
    backtest_series_transformed = model.historical_forecasts(series_transformed,
                                                             future_covariates=future_covariates,
                                                             past_covariates=past_covariates,
                                                             start=test_start_date,
                                                             forecast_horizon=forecast_horizon,
                                                             stride=stride,
                                                             retrain=retrain,
                                                             last_points_only=False,
                                                             verbose=False,
                                                             num_samples=num_samples)

    # flatten lists of forecasts due to last_points_only=False
    if isinstance(backtest_series_transformed, list):
        backtest_series_transformed = reduce(
            append, backtest_series_transformed)

    # inverse scaling
    if transformer_ts is not None and series is not None:
        backtest_series = transformer_ts.inverse_transform(
            backtest_series_transformed)
    else:
        series = series_transformed
        backtest_series = backtest_series_transformed
        print("\nWarning: Scaler not provided. Ensure model provides normal scale predictions")
        logging.info(
            "\n Warning: Scaler not provided. Ensure model provides normal scale predictions")

    # Metrix
    test_series = series.drop_before(pd.Timestamp(test_start_date))
    #print("SERIES",test_series)
    metrics = {
        "smape": smape_darts(
            test_series,
            backtest_series),
        "mase": mase_darts(
            series.drop_before(pd.Timestamp(test_start_date)),
            backtest_series,
            insample=series.drop_after(pd.Timestamp(test_start_date))),
        "mae": mae_darts(
            series.drop_before(pd.Timestamp(test_start_date)),
            backtest_series),
        "rmse": rmse_darts(
            series.drop_before(pd.Timestamp(test_start_date)),
            backtest_series),
        "nrmse_max": rmse_darts(
            series.drop_before(pd.Timestamp(test_start_date)),
            backtest_series) / (
            series.drop_before(pd.Timestamp(test_start_date)).pd_dataframe().max()[0]- 
            series.drop_before(pd.Timestamp(test_start_date)).pd_dataframe().min()[0]),
        "nrmse_mean": rmse_darts(
            series.drop_before(pd.Timestamp(test_start_date)),
            backtest_series) / (
            series.drop_before(pd.Timestamp(test_start_date)).pd_dataframe().mean()[0])
    }
    if min(test_series.min(axis=1).values()) > 0 and min(backtest_series.min(axis=1).values()) > 0:
        metrics["mape"] = mape_darts(
            test_series,
            backtest_series)
    else:
        print("\nModel result or validation series not strictly positive. Setting mape to NaN...")
        logging.info("\nModel result or validation series not strictly positive. Setting mape to NaN...")
        metrics["mape"] = np.nan
    
    for key, value in metrics.items():
        print(key, ': ', value)

    return {"metrics": metrics, "backtest_series": backtest_series}


def validate(series_uri, future_covariates, past_covariates, scaler, cut_date_test, test_end_date,
             model, forecast_horizon, stride, retrain, multiple, eval_series, cut_date_val, mlrun, 
             resolution, eval_method, opt_all_results, evaluate_all_ts, study, num_samples, mode='remote'):
    # TODO: modify functions to support models with likelihood != None
    # TODO: Validate evaluation step for all models. It is mainly tailored for the RNNModel for now.

    # Argument processing
    stride = none_checker(stride)
    forecast_horizon = int(forecast_horizon)
    stride = int(forecast_horizon) if stride is None else int(stride)
    retrain = truth_checker(retrain)
    multiple = truth_checker(multiple)
    num_samples = int(num_samples)

    # Load model / datasets / scalers from Mlflow server

    ## load series from MLflow
    series_path = download_online_file(
        series_uri, "series.csv") if mode == 'remote' else series_uri
    series, id_l, ts_id_l = load_local_csv_as_darts_timeseries(
        local_path=series_path,
        last_date=test_end_date,
        multiple=multiple,
        resolution=resolution)

    if scaler is not None:
        if not multiple:
            series_transformed = scaler.transform(series)
        else:
            series_transformed = [scaler[i].transform(series[i]) for i in range(len(series))]
    else:
        series_transformed = series

    # Split in the same way as in training
    ## series
    series_split = split_dataset(
            series,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l)


    series_transformed_split = split_dataset(
            series_transformed,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l)
        
    if evaluate_all_ts and multiple:
        eval_results = {}
        ts_n = len(ts_id_l)
        for eval_i in range(ts_n):
            backtest_series = darts.timeseries.concatenate([series_split['train'][eval_i], series_split['val'][eval_i]]) if multiple else \
                            darts.timeseries.concatenate([series_split['train'], series_split['val']])
            backtest_series_transformed = darts.timeseries.concatenate([series_transformed_split['train'][eval_i], series_transformed_split['val'][eval_i]]) if multiple else \
                            darts.timeseries.concatenate([series_transformed_split['train'], series_transformed_split['val']])
            #print("testing on", eval_i, backtest_series_transformed)
            print(f"Validating timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")
            logging.info(f"Validating timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")
            print(f"Validating from {pd.Timestamp(cut_date_val)} to {backtest_series_transformed.time_index[-1]}...")
            logging.info(f"Validating from {pd.Timestamp(cut_date_val)} to {backtest_series_transformed.time_index[-1]}...")
            print("")
            validation_results = backtester(model=model,
                                            series_transformed=backtest_series_transformed,
                                            series=backtest_series,
                                            transformer_ts=scaler if (not multiple or (scaler == None)) else scaler[eval_i],
                                            test_start_date=cut_date_val,
                                            forecast_horizon=forecast_horizon,
                                            stride=stride,
                                            retrain=retrain,
                                            future_covariates=None if future_covariates == None else (future_covariates[0] if not multiple else future_covariates[eval_i]),
                                            past_covariates=None if past_covariates == None else (past_covariates[0] if not multiple else past_covariates[eval_i]),
                                            num_samples=num_samples,
                                            )
            eval_results[eval_i] = list(map(str, ts_id_l[eval_i])) + [validation_results["metrics"]["smape"],
                                                                      validation_results["metrics"]["mase"],
                                                                      validation_results["metrics"]["mae"],
                                                                      validation_results["metrics"]["rmse"],
                                                                      validation_results["metrics"]["mape"],
                                                                      validation_results["metrics"]["nrmse_max"],
                                                                      validation_results["metrics"]["nrmse_mean"]]

        eval_results = pd.DataFrame.from_dict(eval_results, orient='index', columns=["Timeseries ID", "smape", "mase", "mae", "rmse", "mape", "nrmse_max", "nrmse_mean"])
        trial_num = len(study.trials_dataframe()) - 1
        save_path = f"{opt_all_results}/trial_{trial_num}.csv"
        if os.path.exists(save_path):
            print(f"Path {opt_all_results}/trial_{trial_num}.csv already exists. Creating extra file for Trial {trial_num}...")
            logging.info(f"Path {opt_all_results}/trial_{trial_num}.csv already exists. Creating extra file for Trial {trial_num}...")
            all_letters = string.ascii_lowercase
            save_path = save_path.split(".")[0] + ''.join(random.choice(all_letters) for i in range(20)) + ".csv"
        eval_results.to_csv(save_path)
        print(eval_results.mean(axis=0, numeric_only=True).to_dict())
        return eval_results.mean(axis=0, numeric_only=True).to_dict()
    else:
        if multiple:
            eval_i = -1
            if eval_method == "ts_ID":
                for i, comps in enumerate(ts_id_l):
                    for comp in comps:
                        if eval_series == str(comp):
                            eval_i = i
            else:
                for i, comps in enumerate(id_l):
                    for comp in comps:
                        if eval_series == str(comp):
                            eval_i = i
        else:
            eval_i = 0
        # Evaluate Model

        backtest_series = darts.timeseries.concatenate([series_split['train'][eval_i], series_split['val'][eval_i]]) if multiple else \
                        darts.timeseries.concatenate([series_split['train'], series_split['val']])
        backtest_series_transformed = darts.timeseries.concatenate([series_transformed_split['train'][eval_i], series_transformed_split['val'][eval_i]]) if multiple else \
                        darts.timeseries.concatenate([series_transformed_split['train'], series_transformed_split['val']])
        #print("testing on", eval_i, backtest_series_transformed)
        if multiple:
            print(f"Validating timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")
            logging.info(f"Validating timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")

        print(f"Validating from {pd.Timestamp(cut_date_val)} to {backtest_series_transformed.time_index[-1]}...")
        logging.info(f"Validating from {pd.Timestamp(cut_date_val)} to {backtest_series_transformed.time_index[-1]}...")
        print("")

        validation_results = backtester(model=model,
                                        series_transformed=backtest_series_transformed,
                                        series=backtest_series,
                                        transformer_ts=scaler if (not multiple or (scaler == None)) else scaler[eval_i],
                                        test_start_date=cut_date_val,
                                        forecast_horizon=forecast_horizon,
                                        stride=stride,
                                        retrain=retrain,
                                        future_covariates=None if future_covariates == None else (future_covariates[0] if not multiple else future_covariates[eval_i]),
                                        past_covariates=None if past_covariates == None else (past_covariates[0] if not multiple else past_covariates[eval_i]),
                                        num_samples=num_samples,
                                        )

        return validation_results["metrics"]

@click.command()
# load arguments
@click.option("--series-csv",
              type=str,
              default="None",
              help="Local timeseries csv. If set, it overwrites the local value."
              )
@click.option("--series-uri",
              type=str,
              default='mlflow_artifact_uri',
              help="Remote timeseries csv file. If set, it overwrites the local value."
              )
@click.option("--future-covs-csv",
              type=str,
              default='None'
              )
@click.option("--future-covs-uri",
              type=str,
              default='mlflow_artifact_uri'
              )
@click.option("--past-covs-csv",
              type=str,
              default='None'
              )
@click.option("--past-covs-uri",
              type=str,
              default='mlflow_artifact_uri'
              )
@click.option('--year-range',
    default="None",
    type=str,
    help='Choose year range to include in the dataset.'
)

@click.option("--resolution",
    default="None",
    type=str,
    help="Change the resolution of the dataset (minutes)."
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
@click.option("--scale",
              type=str,
              default="true",
              help="Whether to scale the target series")
@click.option("--scale-covs",
              type=str,
              default="true",
              help="Whether to scale the covariates")
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
@click.option("--num-workers",
        type=str,
        default="4",
        help="Number of threads that will be used by pytorch")
@click.option("--day-first",
    type=str,
    default="true",
    help="Whether the date has the day before the month")

@click.option("--eval-method",
    type=click.Choice(
        ['ts_ID',
         'ID']),
    default="ts_ID",
    help="what ID type is speciffied in eval_series: if ts_ID is speciffied, then we look at Timeseries ID column. Else, we look at ID column ")

@click.option("--eval-method",
    type=click.Choice(
        ['ts_ID',
         'ID']),
    default="ts_ID",
    help="what ID type is speciffied in eval_series: if ts_ID is speciffied, then we look at Timeseries ID column. Else, we look at ID column ")

@click.option("--loss-function",
    type=click.Choice(
        ["mape",
         "smape",
         "mase", 
         "mae",
         "rmse",
         "nrmse_max",
         "nrmse_mean"]),
    default="mape",
    help="Loss function to use for optuna")

@click.option("--evaluate-all-ts",
    type=str,
    default="false",
    help="Whether to validate the models for all timeseries, and return the mean of their metrics")

@click.option("--grid-search",
    type=str,
    default="false",
    help="Whether to run a grid search or use tpe in optuna")

@click.option("--num-samples",
    type=str,
    default="1",
    help="Number of samples to use for evaluating/validating a probabilistic model's output")


def optuna_search(series_csv, series_uri, future_covs_csv, future_covs_uri,
          past_covs_csv, past_covs_uri, year_range, resolution, darts_model, hyperparams_entrypoint,
           cut_date_val, cut_date_test, test_end_date, device, forecast_horizon, stride, retrain,
           scale, scale_covs, multiple, eval_series, n_trials, num_workers, day_first, eval_method, loss_function, evaluate_all_ts, grid_search, num_samples):

        n_trials = none_checker(n_trials)
        n_trials = int(n_trials)
        evaluate_all_ts = truth_checker(evaluate_all_ts)
        with mlflow.start_run(run_name=f'optuna_test_{darts_model}', nested=True) as mlrun:
            if grid_search:
                hyperparameters = ConfigParser('../config_opt.yml').read_hyperparameters(hyperparams_entrypoint)
                training_dict = {}
                for param, value in hyperparameters.items():
                    if type(value) == list and value and value[0] == "range":
                        if type(value[1]) == int:
                            training_dict[param] = list(range(value[1], value[2], value[3]))
                        else:
                            training_dict[param] = list(range(value[1], value[2], value[3]))
                    elif type(value) == list and value and value[0] == "list":
                        training_dict[param] = value[1:]
                study = optuna.create_study(storage="sqlite:///memory.db", study_name=hyperparams_entrypoint, load_if_exists=True, sampler=optuna.samplers.GridSampler(training_dict))
            else:
                study = optuna.create_study(storage="sqlite:///memory.db", study_name=hyperparams_entrypoint, load_if_exists=True)

            opt_tmpdir = tempfile.mkdtemp()

            if evaluate_all_ts:
                opt_all_results = tempfile.mkdtemp()
            else:
                opt_all_results = None
            study.optimize(lambda trial: objective(series_csv, series_uri, future_covs_csv, future_covs_uri, past_covs_csv, past_covs_uri, year_range, resolution,
                       darts_model, hyperparams_entrypoint, cut_date_val, test_end_date, cut_date_test, device,
                       forecast_horizon, stride, retrain, scale, scale_covs,
                       multiple, eval_series, mlrun, trial, study, opt_tmpdir, num_workers, day_first, eval_method, loss_function, opt_all_results, evaluate_all_ts, num_samples),
                       n_trials=n_trials, n_jobs = 1)

            log_optuna(study, opt_tmpdir, hyperparams_entrypoint, mlrun, opt_all_results=opt_all_results, evaluate_all_ts=evaluate_all_ts)

            return

if __name__ =='__main__':
    print("\n=========== OPTUNA SEARCH =============")
    logging.info("\n=========== OPTUNA SEARCH =============")
    mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    logging.info("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    optuna_search()
