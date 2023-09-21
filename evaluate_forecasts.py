import pretty_errors
from utils import none_checker, truth_checker, download_online_file, load_local_csv_as_darts_timeseries, load_model, load_scaler, multiple_dfs_to_ts_file
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
import random
import shap
from typing import Union
from typing import List
import darts
import json
import statistics
# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)


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
               path_to_save_backtest=None,
               m_mase=1,
               num_samples=1):
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
    #print("EVALUATIING ON SERIES:", series_transformed)
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
    #print(backtest_series, series)

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
        "smape": smape_darts(
            test_series,
            backtest_series),
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
        print("\nModel result or testing series not strictly positive. Setting mape to NaN...")
        logging.info("\nModel result or testing series not strictly positive. Setting mape to NaN...")
        metrics["mape"] = np.nan
    try:
        metrics["mase"] = mase_darts(
            series.drop_before(pd.Timestamp(test_start_date)),
            backtest_series,
            insample=series.drop_after(pd.Timestamp(test_start_date)),
            m = m_mase)
        print("BACKTEST", series)
    except:
        print("\nSeries is periodical. Setting mase to NaN...")
        logging.info("\nModel result or testing series not strictly positive. Setting mape to NaN...")
        metrics["mase"] = np.nan


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
        try:
            backtest_series.drop_before(pd.Timestamp(test_start_date)) \
            .to_csv(os.path.join(path_to_save_backtest, 'predictions.csv'))
        except:
            backtest_series.drop_before(pd.Timestamp(test_start_date)).quantile_df() \
            .to_csv(os.path.join(path_to_save_backtest, 'predictions.csv'))

        try:
            backtest_series_transformed.drop_before(pd.Timestamp(test_start_date)) \
            .to_csv(os.path.join(path_to_save_backtest, 'predictions_transformed.csv'))
        except:
            backtest_series_transformed.drop_before(pd.Timestamp(test_start_date)).quantile_df() \
            .to_csv(os.path.join(path_to_save_backtest, 'predictions_transformed.csv'))

        series_transformed.drop_before(pd.Timestamp(test_start_date)) \
        .to_csv(os.path.join(path_to_save_backtest, 'test_transformed.csv'))

    return {"metrics": metrics, "eval_plot": plt, "backtest_series": backtest_series}

def build_shap_dataset(size: Union[int, float],
                       train: darts.TimeSeries,
                       test: darts.TimeSeries,
                       shap_input_length: int,
                       shap_output_length: int,
                       past_covs: darts.TimeSeries = None,
                       future_covs: darts.TimeSeries = None):
    """
    Produces the dataset to be fed to SHAP's explainer. It chooses a subset of
    the validation dataset and it returns a dataframe of these samples along
    with their corresponding covariates if needed by the model. Naive model not supported
    Parameters
    ----------
    size
        The number of samples to be produced. If it is a float, it represents
        the proportion of possible samples of the validation dataset to be
        chosen. If it is an int, it represents the absolute numbe of samples to
        be produced.
    train
        The training dataset of the model. It is needed to set the background samples
        of the explainer.
    test
        The validation dataset of the model.
    shap_input_length
        The length of each sample of the dataset. Also, the input_chunk_length of the model if 
        it exists as a parameter. It can also be set by the user
    shap_output_length
        The length of each sample of the result. Also, the forecast horizon used for the model
    past_covs
        Whether the model has been trained with past covariates
    future_covs
        Whether the model has been trained with future covariates
    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        -First position of tuple:
            A dataframe consisting of the samples of the validation dataset that
            were chosen, along with their corresponding covariates. Its exact form
            is as follows:
            0 timestep  1 timestep  ... <shap_input_length - 1> timestep \
            Step 0 of past covariate 0 ... Step <shap_input_length - 1> of past covariate 0 \
            Step 0 of past covariate 1 ... Step <shap_input_length - 1> of past covariate <past_covs.n_components> \
            Step 0 of future covariate 0 ... Step <shap_input_length + shap_output_length - 1> of future covariate <future_covs.n_components>
        -Second position of tuple:
            A dataframe containing the sample providing the values that replace the data's values that are simulated to be
            missing. Each feature's value is the median of the TimeSeries it originates from. So, if it's a covariate feature,
            its value will be the median of this covariate, and if it is a feature of the dataset, its value will be the median
            of the training dataset.
    """
    #data: The dataset to be given to SHAP
    data = []
    #background: Dataframe containing the sample providing the values that replace the data's values that are simulated to be missing
    background = []
    #columns: The name of the columns of the dataframes
    columns = []
    #Whether it is the first time the for loop is run
    first_iter = True
    samples = set()

    #Choosing the samples of val we will use randomly
    if(type(size) == float):
        size = int(size*(len(test) - shap_input_length + 1))
    if size == len(test) - shap_input_length + 1:
        samples = set(range(size))
    else:
        for i in range(size):
            while(True):
                r = random.randint(0, len(test) - shap_input_length + 1)
                if r not in samples:
                    break
            samples.add(r)

    for i in samples:
        curr = test[i:i + shap_input_length]
        curr_date = int(curr.time_index[0].timestamp())
        curr_values = curr.random_component_values(copy=False)
        #if np.isnan(curr_values).any():
            #print(curr_date, "has NaN values")
            #continue
        data.append(curr_values.flatten())
#    print(data[-1].flatten())
        if first_iter:
            columns.extend([str(i) + " timestep" for i in range(shap_input_length)])
            median_of_train = statistics.median(map(lambda x : x.median(axis=0).values()[0,0], train))
#            print("median of train", median_of_train)
            background.extend([median_of_train for _ in range(shap_input_length)])
        if past_covs != None:
            for ii in range(past_covs.n_components):
                data[-1] = np.concatenate([data[-1], past_covs.univariate_component(ii)[i:i + shap_input_length].random_component_values(copy=False).flatten()])
                if first_iter:
                    columns.extend(["Step " + str(iii) + " of past covariate " + str(ii) for iii in range(shap_input_length)])
                    background.extend([past_covs.univariate_component(ii).median(axis=0).values()[0,0] for _ in range(shap_input_length)])
        if future_covs != None:
            for ii in range(future_covs.n_components):
                data[-1] = np.concatenate([data[-1], future_covs.univariate_component(ii)[i:i + shap_input_length + shap_output_length].random_component_values(copy=False).flatten()])
                if first_iter:
                    columns.extend(["Step " + str(iii) + " of future covariate " + str(ii) for iii in range(shap_input_length + shap_output_length)])
                    background.extend([future_covs.univariate_component(ii).median(axis=0).values()[0,0] for _ in range(shap_input_length + shap_output_length)])
        data[-1] = np.concatenate([data[-1], [curr_date]])
        if first_iter:
            columns.extend(["Datetime"])
            background.extend([curr_date])
        first_iter = False

    data = pd.DataFrame(data, columns=columns)
    background = pd.DataFrame([background], columns=columns)
    return data, background

def predict(x: darts.TimeSeries,
            n_past_covs: int,
            n_future_covs: int,
            shap_input_length: int,
            shap_output_length: int,
            model,
            scaler_list: List[darts.dataprocessing.transformers.Scaler],
            scale: bool = True,
            num_samples: int = 1):
    """
    The function to be given to KernelExplainer, in order for it to produce predictions
    for all samples of x. These samples have the format described in the above function. Also,
    this function scales the data if necessary and is called multiple times by the SHAP explainer
    Parameters
    ----------
    x
        The dataset of samples to be predicted.
    n_past_covs
        The number of past covariates the model was trained on. This is needed to produce the
        timeseries to be given to the models predict method.
    n_future_covs
        The number of future covariates the model was trained on.
    shap_input_length
        The length of each sample of the dataset. Also, the input_chunk_length of the model if 
        it exists as a parameter. It can also be set by the user
    shap_output_length
        The length of each sample of the result. Also, the forecast horizon used for the model.
    model
        The model to be explained.
    scaler_list
        The list of scalers that were used. First, the training data scaler is listed. Then, all
        the covariate scalers are listed in the order in which they appear in x.
    scale
        Whether to scale the data and the covariates
    Returns
    -------
    numpy.array
        The numpy array of the model's results. Its line number is equal to the number of samples
        provided, and its column number is equal to shap_output_length.
    """

    series_list = []
    past_covs_list = []
    future_covs_list = []
    for sample in x:
    #    print(sample)
        index = [datetime.datetime.utcfromtimestamp(sample[-1]) + pd.offsets.DateOffset(hours=i) for i in range(shap_input_length)]
        index_future = [datetime.datetime.utcfromtimestamp(sample[-1]) + pd.offsets.DateOffset(hours=i) for i in range(shap_input_length + shap_output_length)]
        sample = np.array(sample, dtype=np.float32)
        data = sample[:shap_input_length]
        ts = TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index, columns=["Load"]))
    #    print(ts.dtype)
        if scale:
            ts = scaler_list[0].transform(ts)
    #    print(ts)
        past_covs = None
        future_covs = None
        for i in range(shap_input_length, shap_input_length*(n_past_covs+1), shap_input_length):
#            print(i, "p")
            data = sample[i:i+shap_input_length]
            if i == shap_input_length:
                past_covs = TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index, columns=["Covariate"]))
                if scale:
                    past_covs = scaler_list[i//shap_input_length].transform(past_covs)
            else:
                temp = TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index, columns=["Covariate"]))
                if scale:
                    past_covs = past_covs.stack(scaler_list[i//shap_input_length].transform(temp))
                else:
                    past_covs = past_covs.stack(temp)
        if past_covs == None: 
            past_covs_list = None 
        else:
            past_covs_list.append(past_covs)
        for i in range(shap_input_length*(n_past_covs+1), shap_input_length*(n_past_covs+1) + (shap_input_length + shap_output_length)*n_future_covs, shap_input_length + shap_output_length):
#            print(i, "f")
            data = sample[i:i+shap_input_length+shap_output_length]
            scale_index = 1 + n_past_covs + (i - shap_input_length*(n_past_covs+1))//(shap_input_length+shap_output_length)
            if i == shap_input_length*(n_past_covs+1):
                future_covs = TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index_future, columns=["Covariate"]))
                if scale:
                    future_covs = scaler_list[scale_index].transform(future_covs)
            else:
                temp = TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index_future, columns=["Covariate"]))
                if scale:
                    future_covs = future_covs.stack(scaler_list[scale_index].transform(temp))
                else:
                    future_covs = future_covs.stack(temp)
        if future_covs == None:
            future_covs_list = None
        else:
            future_covs_list.append(future_covs)
        series_list.append(ts)
    #    print("asdssd", past_covs, future_covs)
    try:
        res = model.predict(shap_output_length, series_list, past_covariates=past_covs_list, future_covariates=future_covs_list, num_samples=num_samples)
    except:
        res = model.predict(shap_output_length, series_list, num_samples=num_samples)
    if scale:
        res = list(map(lambda elem : scaler_list[0].inverse_transform(elem).univariate_values(), res))
    else:
        res = list(map(lambda elem : elem.univariate_values(), res))
   #     print("max", np.array(res).max(), np.array(res).min())
#    print("result 1", np.array(res)[0])
#    print("sample 1", x[0])
    return np.array(res)
#lambda x: model_rnn.predict(TimeSeries.from_dataframe(pd.DataFrame(index=(x[-1] + pd.offsets.DateOffset(hours=i) for i in range(96)), data=x[:-1])))

def bar_plot_store_json(shap_values, data, filename):
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(20, len(feature_order)):]
    feature_inds = feature_order[:20]
    feature_inds = reversed(feature_inds)
    feature_names = data.columns
    global_shap_values = np.abs(shap_values).mean(0)
    bar_plot_dict = {}
    for i in feature_inds:
        bar_plot_dict[feature_names[i]] = global_shap_values[i]
    with open(filename, "w") as out:
        json.dump(bar_plot_dict, out)

def call_shap(n_past_covs: int,
              n_future_covs: int,
              shap_input_length: int,
              shap_output_length: int,
              model,
              scaler_list: List[darts.dataprocessing.transformers.Scaler],
              background: darts.TimeSeries,
              data: darts.TimeSeries,
              scale: bool = True,
              num_samples: int = 1):
    """
    The function that calls KernelExplainer, and stores to the MLflow server
    some plots explaining the output of the model. More specifficaly, ...
    Parameters
    ----------
    n_past_covs
        The number of past covariates the model was trained on. This is needed to produce the
        timeseries to be given to the models predict method.
    n_future_covs
        The number of future covariates the model was trained on.
    shap_input_length
        The length of each sample of the dataset. Also, the input_chunk_length of the model if 
        it exists as a parameter. It can also be set by the user
    shap_output_length
        The length of each sample of the result. Also, the forecast horizon used for the model. 
    model
        The model to be explained.
    scaler_list
        The list of scalers that were used. First, the training data scaler is listed. Then, all
        the covariate scalers are listed in the order in which they appear in x.
    scale
        Whether to scale the data and the covariates
    background
        The sample that provides the values that replace the data's values that are simulated to be
        missing
    data
        The samples to be tested
    """

    shap.initjs()
#    print("background", background)
    explainer = shap.KernelExplainer(lambda x : predict(x, n_past_covs, n_future_covs, shap_input_length, shap_output_length, model, scaler_list, scale), background, num_samples=num_samples)
#    print("all samples", data)
    shap_values = explainer.shap_values(data, nsamples="auto", normalize=False)
#    print("values for 0 output", shap_values[0])
    plt.close()
    interprtmpdir = tempfile.mkdtemp()
    sample = random.randint(0, len(data) - 1)
    for out in [0, shap_output_length//2, shap_output_length-1]:
#        print(len(shap_values))
#        print(out)
        shap.summary_plot(shap_values[out], data, show=False)
        plt.savefig(f"{interprtmpdir}/summary_plot_all_samples_out_{out}.png")
        plt.close()
        os.remove(f"{interprtmpdir}/summary_plot_all_samples_out_{out}.png")
        shap.summary_plot(shap_values[out], data, show=False)
        plt.savefig(f"{interprtmpdir}/summary_plot_all_samples_out_{out}.png")
        plt.close()
        shap.summary_plot(shap_values[out], data, plot_type='bar', show=False)
        plt.savefig(f"{interprtmpdir}/summary_plot_bar_graph_out_{out}.png")
        plt.close()
        bar_plot_store_json(shap_values[out], data, f"{interprtmpdir}/summary_plot_bar_data_out_{out}.json")
        shap.force_plot(explainer.expected_value[out],shap_values[out][sample,:], data.iloc[sample,:],  matplotlib = True, show = False)
        plt.savefig(f"{interprtmpdir}/force_plot_of_{sample}_sample_starting_at_{datetime.datetime.utcfromtimestamp(data.iloc[sample][-1])}_{out}_output.png")
        plt.close()

        print("\nUploading SHAP interpretation results to MLflow server...")
        logging.info("\nUploading SHAP interpretation results to MLflow server...")

        mlflow.log_artifacts(interprtmpdir, "interpretation")



@click.command()
@click.option("--mode",
              type=str,
              default='remote',
              help='Whether to look for files locally or remotely'
              )
@click.option("--series-uri",
              type=str,
              default='None',
              help='Provide the uri of the series file'
              )
@click.option("--future-covs-uri",
              type=str,
              default='None',
              help='Provide the uri of the future covariates file'
              )
@click.option("--past-covs-uri",
              type=str,
              default='None',
              help='Provide the uri of the past covariates file'
              )
@click.option("--scaler-uri",
              type=str,
              default='None',
              help='Provide the uri of the future covariates file'
              )
@click.option("--cut-date-test",
              type=str,
              default='None',
              help="Test set start date [str: 'YYYYMMDD']",
              )
@click.option("--test-end-date",
              type=str,
              default='20213112',
              help="Test end start date [str: 'YYYYMMDD']",
              )
@click.option("--model-uri",
              type=str,
              default='None',
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
              default="None")
@click.option("--stride",
              type=str,
              default="None")
@click.option("--retrain",
              type=str,
              default="false",
              help="Whether to retrain model during backtesting")

@click.option("--shap-input-length",
             type=str,
             default="None",
             help="The length of each sample of the dataset to be used during analysis with SHAP.")

@click.option("--shap-output-length",
             type=str,
             default="None",
             help="The length of each sample of the result. Also, the forecast horizon used for the model")

@click.option("--size",
             type=str,
             default="100",
             help="Size of shap dataset in samples")

@click.option("--analyze-with-shap",
             type=str,
             default="False",
             help="Whether to do SHAP analysis on the model. Naive is not supported")
@click.option("--multiple",
    type=str,
    default="false",
    help="Whether to train on multiple timeseries")

@click.option("--eval-series",
    type=str,
    default="None",
    help="On which timeseries to run the backtesting. Only for multiple timeseries")

@click.option("--cut-date-val",
              type=str,
              default='None',
              help="Val set start date [str: 'YYYYMMDD']",
              )
@click.option("--day-first",
    type=str,
    default="true",
    help="Whether the date has the day before the month")

@click.option("--resolution",
    default="None",
    type=str,
    help="The resolution of the dataset in minutes."
)
@click.option("--eval-method",
    type=click.Choice(
        ['ts_ID',
         'ID']),
    default="ts_ID",
    help="what ID type is speciffied in eval_series: if ts_ID is speciffied, then we look at Timeseries ID column. Else, we look at ID column")

@click.option("--evaluate-all-ts",
    type=str,
    default="false",
    help="Whether to validate the models for all timeseries, and return the mean of their metrics")

@click.option("--m-mase",
    type=str,
    default="1",
    help="m to use for mase metric")

@click.option("--num-samples",
    type=str,
    default="1",
    help="Number of samples to use for evaluating/validating a probabilistic model's output")


def evaluate(mode, series_uri, future_covs_uri, past_covs_uri, scaler_uri, cut_date_test, test_end_date, model_uri, model_type, forecast_horizon, stride, retrain, shap_input_length, shap_output_length, size, analyze_with_shap, multiple, eval_series, cut_date_val, day_first, resolution, eval_method, evaluate_all_ts, m_mase, num_samples):
    # TODO: modify functions to support models with likelihood != None
    # TODO: Validate evaluation step for all models. It is mainly tailored for the RNNModel for now.

    # Argument processing
    stride = none_checker(stride)
    forecast_horizon = int(forecast_horizon)
    m_mase = int(m_mase)
    num_samples = int(num_samples)
    stride = int(forecast_horizon) if stride is None else int(stride)
    retrain = truth_checker(retrain)
    analyze_with_shap = truth_checker(analyze_with_shap)
    multiple = truth_checker(multiple)
    future_covariates_uri = none_checker(future_covs_uri)
    past_covariates_uri = none_checker(past_covs_uri)
    evaluate_all_ts = truth_checker(evaluate_all_ts)
    try:
        size = int(size)
    except:
        size = float(size)
    try:
        shap_input_length = int(shap_input_length)
    except:
        pass
    shap_output_length = int(shap_output_length)

    # Load model / datasets / scalers from Mlflow server

    ## load series from MLflow
    series_path = download_online_file(
        series_uri, "series.csv") if mode == 'remote' else series_uri
    series, id_l, ts_id_l = load_local_csv_as_darts_timeseries(
        local_path=series_path,
        last_date=test_end_date,
        multiple=multiple,
        day_first=day_first,
        resolution=resolution)


    if future_covariates_uri is not None:
        future_covs_path = download_online_file(
            future_covariates_uri, "future_covariates.csv") if mode == 'remote' else future_covariates_uri
        future_covariates, id_l_future_covs, ts_id_l_future_covs = load_local_csv_as_darts_timeseries(
            local_path=future_covs_path,
            last_date=test_end_date,
            multiple=True,
            day_first=day_first,
            resolution=resolution)
    else:
        future_covariates = None

    if past_covariates_uri is not None:
        past_covs_path = download_online_file(
            past_covariates_uri, "past_covariates.csv") if mode == 'remote' else past_covariates_uri
        past_covariates, id_l_past_covs, ts_id_l_past_covs = load_local_csv_as_darts_timeseries(
            local_path=past_covs_path,
            last_date=test_end_date,
            multiple=True,
            day_first=day_first,
            resolution=resolution)
    else:
        past_covariates = None

    # TODO: Also implement for local files -> Done?
    ## load model from MLflow
    model = load_model(model_uri, mode)
    scaler = load_scaler(scaler_uri=none_checker(scaler_uri), mode=mode)

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
            val_start_date_str=cut_date_test,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l)

    series_transformed_split = split_dataset(
            series_transformed,
            val_start_date_str=cut_date_test,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l)


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
    evaltmpdir = tempfile.mkdtemp()
    with mlflow.start_run(run_name='eval', nested=True) as mlrun:
        mlflow.set_tag("run_id", mlrun.info.run_id)
        mlflow.set_tag("stage", "evaluation")
        #print("TESTING ON", eval_i, series_transformed_split['all'][eval_i])
        if evaluate_all_ts and multiple:
            eval_results = {}
            ts_n = len(ts_id_l)
            for eval_i in range(ts_n):
                backtest_series_transformed = series_transformed_split['all'] if not multiple else series_transformed_split['all'][eval_i]
                #print("testing on", eval_i, backtest_series_transformed)
                print(f"Testing timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")
                logging.info(f"Validating timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")
                print(f"Testing from {pd.Timestamp(cut_date_test)} to {backtest_series_transformed.time_index[-1]}...")
                logging.info(f"Testing from {pd.Timestamp(cut_date_test)} to {backtest_series_transformed.time_index[-1]}...")
                print("")
                evaluation_results = backtester(model=model,
                                            series_transformed=backtest_series_transformed,
                                            series=series_split['all'] if not multiple else series_split['all'][eval_i],
                                            transformer_ts=scaler if (not multiple or (scaler == None)) else scaler[eval_i],
                                            test_start_date=cut_date_test,
                                            forecast_horizon=forecast_horizon,
                                            stride=stride,
                                            retrain=retrain,
                                            future_covariates=None if future_covariates == None else (future_covariates[0] if not multiple else future_covariates[eval_i]),
                                            past_covariates=None if past_covariates == None else (past_covariates[0] if not multiple else past_covariates[eval_i]),
                                            path_to_save_backtest=f"{evaltmpdir}/{ts_id_l[eval_i][0]}",
                                            m_mase=m_mase,
                                            num_samples=num_samples)
                eval_results[eval_i] = list(map(str, ts_id_l[eval_i][:1])) + [evaluation_results["metrics"]["smape"],
                                                                              evaluation_results["metrics"]["mase"],
                                                                              evaluation_results["metrics"]["mae"],
                                                                              evaluation_results["metrics"]["rmse"],
                                                                              evaluation_results["metrics"]["mape"],
                                                                              evaluation_results["metrics"]["nrmse_max"],
                                                                              evaluation_results["metrics"]["nrmse_mean"]]

            eval_results = pd.DataFrame.from_dict(eval_results, orient='index', columns=["Timeseries ID", "smape", "mase", "mae", "rmse", "mape", "nrmse_max", "nrmse_mean"])
            save_path = f"{evaltmpdir}/evaluation_results_all_ts.csv"
            eval_results.to_csv(save_path)
            evaluation_results["metrics"] = eval_results.mean(axis=0, numeric_only=True).to_dict()
        else:
            if multiple:
                print(f"Testing timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}")
                logging.info(f"Testing timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}")

            backtest_series_transformed = series_transformed_split['all'] if not multiple else series_transformed_split['all'][eval_i]
            print(f"Testing from {pd.Timestamp(cut_date_test)} to {backtest_series_transformed.time_index[-1]}...")
            logging.info(f"Testing from {pd.Timestamp(cut_date_test)} to {backtest_series_transformed.time_index[-1]}...")

            print("")

            evaluation_results = backtester(model=model,
                                            series_transformed=backtest_series_transformed,
                                            series=series_split['all'] if not multiple else series_split['all'][eval_i],
                                            transformer_ts=scaler if (not multiple or (scaler == None)) else scaler[eval_i],
                                            test_start_date=cut_date_test,
                                            forecast_horizon=forecast_horizon,
                                            stride=stride,
                                            retrain=retrain,
                                            future_covariates=None if future_covariates == None else (future_covariates[0] if not multiple else future_covariates[eval_i]),
                                            past_covariates=None if past_covariates == None else (past_covariates[0] if not multiple else past_covariates[eval_i]),
                                            path_to_save_backtest=evaltmpdir,
                                            m_mase=m_mase,
                                            num_samples=num_samples)
            if analyze_with_shap:
                data, background = build_shap_dataset(size=size,
                                                train=series_split['train'],
                                                test=series_split['test']\
                                                     if not multiple else series_split['test'][eval_i],
                                                shap_input_length=shap_input_length,
                                                shap_output_length=shap_output_length,
                                                future_covs=None if future_covariates == None else (future_covariates[0] if not multiple else future_covariates[eval_i]),
                                                past_covs=None if past_covariates == None else (past_covariates[0] if not multiple else past_covariates[eval_i]))

                #print(data, background)
                #TODO check SHAP with covariates
                call_shap(n_past_covs=0 if past_covariates == None else past_covariates.n_components,
                    n_future_covs=0 if future_covariates == None else future_covariates.n_components,
                    shap_input_length=shap_input_length,
                    shap_output_length=shap_output_length,
                    model=model,
                    scaler_list=[scaler if (not multiple or (scaler == None)) else scaler[eval_i],],
                    background=background,
                    data=data,
                    scale=(scaler != None),
                    num_samples=num_samples)
        if not multiple:
            series_split['test'].to_csv(
                    os.path.join(evaltmpdir, "test.csv"))
        else:
            multiple_dfs_to_ts_file(series_split['test'], id_l, ts_id_l, os.path.join(evaltmpdir, "test.csv"))

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
