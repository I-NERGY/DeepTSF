import pretty_errors
from utils import none_checker, truth_checker, download_online_file, load_local_csv_as_darts_timeseries, load_model, load_scaler

from functools import reduce
from darts.metrics import mape as mape_darts
from darts.metrics import mase as mase_darts
from darts.metrics import mae as mae_darts
from darts.metrics import rmse as rmse_darts
from darts.metrics import smape as smape_darts
from darts.models import (
    NaiveSeasonal,
)
from darts import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import logging
import click
import mlflow
import shutil
from preprocessing import split_dataset
import tempfile


# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# DARTS

# def eval_model(model, train, val, n_steps, future_covariates=None, past_covariates=None):
#     pred = model.predict(n=n_steps,95

#                          future_covariates=future_covariates,
#                          past_covariates=past_covariates)
#     series = train.append(val)
#     plt.figure(figsize=(20, 10))
#     series.drop_before(pd.Timestamp(
#         pred.time_index[0] - datetime.timedelta(days=7))) \
#         .drop_after(pred.time_index[-1]).plot(label='actual')
#     pred.plot(label='forecast')
#     plt.legend()
#     mape_error = darts.metrics.mape(val, pred)
#     print('MAPE = {:.2f}%'.format(mape_error))
#     return mape_error

def append(x, y):
    return x.append(y)

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
               path_to_save_backtest=None):
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

    # produce list of forecasts
    backtest_series_transformed = model.historical_forecasts(series_transformed,
                                                             future_covariates=future_covariates,
                                                             past_covariates=past_covariates,
                                                             start=test_start_date,
                                                             forecast_horizon=forecast_horizon,
                                                             stride=stride,
                                                             retrain=retrain,
                                                             last_points_only=False,
                                                             verbose=False)

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

    # plot all test
    fig1 = plt.figure(figsize=(15, 8))
    ax1 = fig1.add_subplot(111)
    backtest_series.plot(label='forecast')
    series \
        .drop_before(pd.Timestamp(pd.Timestamp(test_start_date) - datetime.timedelta(days=7))) \
        .drop_after(backtest_series.time_index[-1]) \
        .plot(label='actual')
    ax1.legend()
    ax1.set_title(
        f'Backtest, starting {test_start_date}, {forecast_horizon}-steps horizon')
    # plt.show()

    # plot one week (better visibility)
    forecast_start_date = pd.Timestamp(
        test_start_date + datetime.timedelta(days=7))
        
    fig2 = plt.figure(figsize=(15, 8))
    ax2 = fig2.add_subplot(111)
    backtest_series \
        .drop_before(pd.Timestamp(forecast_start_date)) \
        .drop_after(forecast_start_date + datetime.timedelta(days=7)) \
        .plot(label='Forecast')
    series \
        .drop_before(pd.Timestamp(forecast_start_date)) \
        .drop_after(forecast_start_date + datetime.timedelta(days=7)) \
        .plot(label='Actual')
    ax2.legend()
    ax2.set_title(
        f'Weekly forecast, Start date: {forecast_start_date}, Forecast horizon (timesteps): {forecast_horizon}, Forecast extended with backtesting...')

    # Metrix
    test_series = series.drop_before(pd.Timestamp(test_start_date))
    metrics = {
        "mape": mape_darts(
            test_series, 
            backtest_series),
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
            backtest_series)
    }
    for key, value in metrics.items():
        print(key, ': ', value)
    

    # save plot
    if path_to_save_backtest is not None:
        os.makedirs(path_to_save_backtest, exist_ok=True)
        mape = metrics['mape']
        fig1.savefig(os.path.join(path_to_save_backtest,
            f'test_start_date_{test_start_date.date()}_forecast_horizon_{forecast_horizon}_mape_{mape:.2f}.png'))
        fig2.savefig(os.path.join(path_to_save_backtest,
            f' week2_forecast_start_date_{test_start_date.date()}_forecast_horizon_{forecast_horizon}.png'))
        backtest_series.drop_before(pd.Timestamp(test_start_date)) \
            .to_csv(os.path.join(path_to_save_backtest, 'predictions.csv'))

        backtest_series_transformed.drop_before(pd.Timestamp(test_start_date)) \
            .to_csv(os.path.join(path_to_save_backtest, 'predictions_transformed.csv'))
        series_transformed.drop_before(pd.Timestamp(test_start_date)) \
            .to_csv(os.path.join(path_to_save_backtest, 'test_transformed.csv'))

    return {"metrics": metrics, "eval_plot": plt, "backtest_series": backtest_series}

@click.command()
@click.option("--mode",
              type=str,
              default='remote',
              help='Whether to look for files locally or remotely'
              )
@click.option("--series-uri",
              type=str,
              default='mlflow_artifact_uri',
              help='Provide the uri of the series file'
              )
@click.option("--future-covs-uri",
              type=str,
              default='mlflow_artifact_uri',
              help='Provide the uri of the future covariates file'
              )
@click.option("--past-covs-uri",
              type=str,
              default='mlflow_artifact_uri',
              help='Provide the uri of the past covariates file'
              )
@click.option("--scaler-uri",
              type=str,
              default='mlflow_artifact_uri',
              help='Provide the uri of the future covariates file'
              )
@click.option("--cut-date-test",
              type=str,
              default='20210101',
              help="Test set start date [str: 'YYYYMMDD']",
              )
@click.option("--test-end-date",
              type=str,
              default='20213112',
              help="Test end start date [str: 'YYYYMMDD']",
              )
@click.option("--model-uri",
              type=str,
              default='mlflow_artifact_uri',
              help='Remote URI of the model to be evaluated'
              )
@click.option("--model-type",
              default='pl',
              type=click.Choice(
                  ['pl',
                   'pkl']),
              help='Type of Model'
              )
@click.option("--forecast-horizon",
              type=str,
              default="96")
@click.option("--stride",
              type=str,
              default="None")
@click.option("--retrain",
              type=str,
              default="false")
def evaluate(mode, series_uri, future_covs_uri, past_covs_uri, scaler_uri, cut_date_test, test_end_date, model_uri, model_type, forecast_horizon, stride, retrain):
    # TODO: modify functions to support models with likelihood != None
    # TODO: Validate evaluation step for all models. It is mainly tailored for the RNNModel for now.

    # Argument processing
    stride = none_checker(stride)
    forecast_horizon = int(forecast_horizon)
    stride = int(forecast_horizon) if stride is None else int(stride)
    retrain = truth_checker(retrain)

    future_covariates_uri = none_checker(future_covs_uri)
    past_covariates_uri = none_checker(past_covs_uri)

    # Load model / datasets / scalers from Mlflow server

    ## load series from MLflow
    series_path = download_online_file(
        series_uri, "series.csv") if mode == 'remote' else series_uri
    series = load_local_csv_as_darts_timeseries(
        local_path=series_path,  
        last_date=test_end_date)

    if future_covariates_uri is not None:
        future_covs_path = download_online_file(
            future_covariates_uri, "future_covariates.csv") if mode == 'remote' else future_covariates_uri
        future_covariates = load_local_csv_as_darts_timeseries(
            local_path=future_covs_path,  
            last_date=test_end_date)
    else:
        future_covariates = None

    if past_covariates_uri is not None:
        past_covs_path = download_online_file(
            past_covariates_uri, "past_covariates.csv") if mode == 'remote' else past_covariates_uri
        past_covariates = load_local_csv_as_darts_timeseries(
            local_path=past_covs_path,  
            last_date=test_end_date)
    else:
        past_covariates = None

    # TODO: Also implement for local files -> Done?
    ## load model from MLflow
    model = load_model(model_uri, mode, model_type)
    scaler = load_scaler(scaler_uri=none_checker(scaler_uri), mode=mode)

    if scaler is not None:
        series_transformed = scaler.transform(series)

    # Split in the same way as in training
    ## series
    series_split = split_dataset(
        series, 
        val_start_date_str=cut_date_test, 
        test_start_date_str=cut_date_test,
        test_end_date=test_end_date)
    
    series_transformed_split = split_dataset(
        series_transformed, 
        val_start_date_str=cut_date_test, 
        test_start_date_str=cut_date_test,
        test_end_date=test_end_date)

    # Evaluate Model
    evaltmpdir = tempfile.mkdtemp()
    with mlflow.start_run(run_name='eval', nested=True) as mlrun: 
        mlflow.set_tag("run_id", mlrun.info.run_id)
        mlflow.set_tag("stage", "evaluation")
        evaluation_results = backtester(model=model,
                                series_transformed=series_transformed_split['all'],
                                series=series_split['all'],
                                transformer_ts=scaler,
                                test_start_date=cut_date_test,
                                forecast_horizon=forecast_horizon,
                                stride=stride,
                                retrain=retrain,
                                future_covariates=future_covariates,
                                past_covariates=past_covariates,
                                path_to_save_backtest=evaltmpdir)
                                
        series_split['test'].to_csv(
            os.path.join(evaltmpdir, "test.csv"))

        print("\nUploading evaluation results to MLflow server...")
        logging.info("\nUploading evaluation results to MLflow server...")

        mlflow.log_metrics(evaluation_results["metrics"])
        mlflow.log_artifacts(evaltmpdir, "eval_results")
        
        print("\nArtifacts uploaded. Deleting local copies...")
        logging.info("\nArtifacts uploaded. Deleting local copies...")

        print("\nEvaluation succesful.\n")
        logging.info("\nEvaluation succesful.\n")

        # Set tags
        mlflow.set_tag("run_id", mlrun.info.run_id)

        return

if __name__ == '__main__':
    print("\n=========== EVALUATION =============")
    logging.info("\n=========== EVALUATION =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    evaluate()
