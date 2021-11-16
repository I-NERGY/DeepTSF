from functools import reduce
from darts.datasets import AirPassengersDataset
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.metrics import mape as mape_darts
from darts.metrics import mase as mase_darts
from darts.models import (
    NaiveSeasonal,
)
from darts import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
import pprint
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import matplotlib.pyplot as plt
import darts
import datetime
from tqdm import tqdm
import os
import logging

pp = pprint.PrettyPrinter(indent=4)


def simple_n_day_ahead_forecast(model, days_ahead, steps, train, test, scaler, exog_test=None, scaler_exog=None, naive_remebers_k_timesteps=1):

    print(f"Simple {days_ahead} day ahead forecast:\n")

    if exog_test is not None:
        if scaler_exog is not None:
            exog_test = pd.Series(scaler_exog.transform(
                exog_test.values), index=test.index)
        exog_test = exog_test[:steps * days_ahead]

    if scaler is not None:

        forecast_scaled = model.forecast(steps * days_ahead, exog=exog_test) \
            if 'SARIMAX' in str(model) \
            else model.forecast(steps * days_ahead)

        if isinstance(forecast_scaled, np.ndarray):
            predictions = scaler.inverse_transform(
                forecast_scaled.reshape(-1, 1)).reshape(-1)
        else:
            predictions = scaler.inverse_transform(
                forecast_scaled.values.reshape(-1, 1)).reshape(-1)
    else:

        predictions = \
            model.forecast(steps * days_ahead, exog=exog_test) \
            if 'SARIMAX' in str(model) \
            else model.forecast(steps * days_ahead)
        # if exog_test \
        # else model.forecast(steps * days_ahead)

    # adjust test set based on days_ahead given
    test = test[:len(predictions)]
    predictions = pd.Series(predictions, index=test.index)

    metrics = {
        "MAPE": mape(test, predictions),
        "MSE": mse(test, predictions),
        "RMSE": np.sqrt(mse(test, predictions))
    }

    if train is not None:

        ground_truth_line = \
            pd.DataFrame(index=pd.concat([train[-7*24:], test]).index)
        ground_truth_line['Train'] = train[-7*24:]
        ground_truth_line['Test'] = test

        series = TimeSeries.from_series(train)
        naive_model = NaiveSeasonal(K=naive_remebers_k_timesteps)
        naive_model.fit(series)
        naive_pred = naive_model.predict(steps * days_ahead)
        naive_pred = TimeSeries.pd_series(naive_pred)

        # naive_pred=[train.tolist()[-1]] + test.tolist()[:-1]
        metrics = {
            "MAPE naive": mape(test, naive_pred),
            "MAPE": mape(test, predictions),
            "MSE": mse(test, predictions),
            "RMSE": np.sqrt(mse(test, predictions))
        }

        plt.figure()
        plot = ground_truth_line.plot(figsize=(15, 7),
                                      label='Data',
                                      legend=True,
                                      title=f"{days_ahead} day ahead forecast")
        predictions.plot(label='Forecast', legend=True)
        naive_pred.plot(
            label=f'Naive method (#memory_steps={naive_remebers_k_timesteps})', legend=True)
        plot.grid()
        plt.show()

        return predictions, metrics


def advanced_n_day_ahead_forecast(model, days_ahead, steps, train, test, scaler):

    print(f"Advanced {days_ahead} day ahead forecast:\n")

    predictions = pd.Series(dtype='float64')

    if scaler:

        for n_day in range(days_ahead):
            cur_timestep = n_day * steps
            next_timestep = cur_timestep + steps

            day_preds_scaled = model.forecast(steps)
            if isinstance(day_preds_scaled, np.ndarray):
                day_preds = pd.Series(
                    scaler.inverse_transform(
                        day_preds_scaled.reshape(-1, 1)).reshape(-1),
                    index=test[cur_timestep:next_timestep].index)
            else:
                day_preds = pd.Series(
                    scaler.inverse_transform(
                        day_preds_scaled.values.reshape(-1, 1)).reshape(-1),
                    index=test[cur_timestep:next_timestep].index)
            predictions = pd.concat([predictions, pd.Series(day_preds)])
            y_news = test.iloc[cur_timestep: next_timestep].values
            # y_news = y_news.asfreq('H')
            model = model.append(y_news)

    else:

        for n_day in range(days_ahead):
            cur_timestep = n_day * steps
            next_timestep = cur_timestep + steps
            day_preds = model.forecast(steps)
            predictions = pd.concat([predictions, day_preds])
            y_news = test.iloc[cur_timestep: next_timestep]
            y_news = y_news.asfreq('H')
            model = model.append(y_news)

    # adjust test set based on days_ahead given
    test = test[:len(predictions)]

    predictions.name = test.name
    predictions = predictions.rename_axis('datetime')

    metrics = {
        "MAPE": mape(test, predictions),
        "MSE": mse(test, predictions),
        "RMSE": np.sqrt(mse(test, predictions))
    }

    if train is not None:

        ground_truth_line = pd.DataFrame(
            index=pd.concat([train[-7*24:], test]).index)
        ground_truth_line['Train'] = train[-7*24:]
        ground_truth_line['Test'] = test
        ground_truth_line

        series = TimeSeries.from_series(train)
        naive_model = NaiveSeasonal(K=naive_remebers_k_timesteps)
        naive_model.fit(series)
        naive_pred = naive_model.predict(steps * days_ahead)
        naive_pred = TimeSeries.pd_series(naive_pred)

        # naive_pred= [train.tolist()[-1]] + test.tolist()[:-1]

        metrics = {
            "MAPE naive": mape(test, naive_pred),
            "MAPE": mape(test, predictions),
            "MSE": mse(test, predictions),
            "RMSE": np.sqrt(mse(test, predictions))
        }

        plt.figure()
        plot = ground_truth_line.plot(figsize=(
            15, 3), label='Data', legend=True, title=f"{days_ahead} day ahead forecast")
        predictions.plot(label='Forecast', legend=True)
        plot.grid()
        plt.show()

    return predictions, metrics

# DARTS
def eval_model(model, train, val, n_steps, future_covariates=None, past_covariates=None):
    pred = model.predict(n=n_steps,
                         future_covariates=future_covariates,
                         past_covariates=past_covariates)
    series = train.append(val)
    plt.figure(figsize=(20, 10))
    series.drop_before(pd.Timestamp(
        pred.time_index[0] - datetime.timedelta(days=7))) \
        .drop_after(pred.time_index[-1]).plot(label='actual')
    pred.plot(label='forecast')
    plt.legend()
    mape_error = darts.metrics.mape(val, pred)
    print('MAPE = {:.2f}%'.format(mape_error))
    return mape_error


def append(x, y):
    return x.append(y)


def backtester(model,
               series_transformed,
               backtest_start_date,
               forecast_horizon,
               stride=None,
               transformer_ts=None,
               retrain=False,
               future_covariates=None,
               past_covariates=None,
               path_to_save_backtest=None):
    """ Does the same job with advanced forecast but much more quickly using the darts
    bult-in historical_forecasts method. Use this for evaluation. The other only 
    provides pure inference. Provide a unified timeseries set test point based 
    on backtest_start_date. series_transformed does not need to be adjacent to 
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

    # produce list of forecasts
    backtest_series_transformed = model.historical_forecasts(series_transformed,
                                                             future_covariates=future_covariates,
                                                             past_covariates=past_covariates,
                                                             start=pd.Timestamp(
                                                                 backtest_start_date),
                                                             forecast_horizon=forecast_horizon,
                                                             stride=stride,
                                                             retrain=retrain,
                                                             last_points_only=False,
                                                             verbose=True)

    # flatten lists of forecasts due to last_points_only=False
    if isinstance(backtest_series_transformed, list):
        backtest_series_transformed = reduce(
            append, backtest_series_transformed)

    # inverse scaling
    if transformer_ts is not None:
        series = transformer_ts.inverse_transform(series_transformed)
        backtest_series = transformer_ts.inverse_transform(
            backtest_series_transformed)
    else:
        series = series_transformed
        backtest_series = backtest_series_transformed

    # plot
    plt.figure(figsize=(15, 8))
    backtest_series.plot(label='backtest')
    series \
        .drop_before(pd.Timestamp(pd.Timestamp(backtest_start_date) - datetime.timedelta(days=7))) \
        .drop_after(backtest_series.time_index[-1]) \
        .plot(label='actual')
    plt.legend()
    plt.title(
        f'Backtest, starting {backtest_start_date}, {forecast_horizon}-steps horizon')

    # mape
    mape_error = mape_darts(series, backtest_series)
    print('MAPE: {:.2f}%'.format(mape_error))

    # save plot
    if path_to_save_backtest is not None:
        plt.savefig(os.path.join(path_to_save_backtest,
                    f'backtest_start_date_{backtest_start_date.date()}_forecast_horizon_{forecast_horizon}_mape_{mape_error:.2f}.png'))

    return mape_error, backtest_series
