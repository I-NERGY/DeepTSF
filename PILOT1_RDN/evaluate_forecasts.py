
from darts.datasets import AirPassengersDataset
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.metrics import mape, mase
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

pp = pprint.PrettyPrinter(indent=4)

def simple_n_day_ahead_forecast(model, days_ahead, steps, train, test, scaler, exog_test=None, scaler_exog=None, naive_remebers_k_timesteps=1):

    print(f"Simple {days_ahead} day ahead forecast:\n")

    if exog_test is not None:
        if scaler_exog is not None:
            exog_test = pd.Series(scaler_exog.transform(exog_test.values), index=test.index)
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
            pd.DataFrame(index = pd.concat([train[-7*24:], test]).index)
        ground_truth_line['Train'] = train[-7*24:]
        ground_truth_line['Test'] = test

        series = TimeSeries.from_series(train)
        naive_model = NaiveSeasonal(K=naive_remebers_k_timesteps)
        naive_model.fit(series);
        naive_pred = naive_model.predict(steps * days_ahead)
        naive_pred = TimeSeries.pd_series(naive_pred)

        # naive_pred=[train.tolist()[-1]] + test.tolist()[:-1]
        metrics = {
            "MAPE naive": mape(test, naive_pred),
            "MAPE": mape(test, predictions),
            "MSE": mse(test, predictions),
            "RMSE": np.sqrt(mse(test, predictions))
            }

        plt.figure();
        plot = ground_truth_line.plot(figsize=(15, 7), 
                                      label='Data', 
                                      legend=True, 
                                      title=f"{days_ahead} day ahead forecast")
        predictions.plot(label='Forecast', legend=True);
        naive_pred.plot(label=f'Naive method (#memory_steps={naive_remebers_k_timesteps})', legend=True);
        plot.grid()
        plt.show()

        return predictions, metrics

def advanced_n_day_ahead_forecast(model, days_ahead, steps, train, test, scaler):

    print(f"Advanced {days_ahead} day ahead forecast:\n")

    predictions=pd.Series(dtype = 'float64')

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
    test=test[:len(predictions)]

    predictions.name=test.name
    predictions=predictions.rename_axis('datetime')

    metrics={
               "MAPE": mape(test, predictions),
               "MSE": mse(test, predictions),
               "RMSE": np.sqrt(mse(test, predictions))
              }

    if train is not None:

        ground_truth_line=pd.DataFrame(
            index =pd.concat([train[-7*24:], test]).index)
        ground_truth_line['Train']= train[-7*24:]
        ground_truth_line['Test']= test
        ground_truth_line

        series = TimeSeries.from_series(train)
        naive_model = NaiveSeasonal(K=naive_remebers_k_timesteps)
        naive_model.fit(series)
        naive_pred = naive_model.predict(steps * days_ahead)
        naive_pred = TimeSeries.pd_series(naive_pred)

        # naive_pred= [train.tolist()[-1]] + test.tolist()[:-1]

        metrics={
                "MAPE naive": mape(test, naive_pred),
                "MAPE": mape(test, predictions),
                "MSE": mse(test, predictions),
                "RMSE": np.sqrt(mse(test, predictions))
                }
        
        plt.figure()
        plot = ground_truth_line.plot(figsize=(15, 3), label='Data', legend=True, title=f"{days_ahead} day ahead forecast")
        predictions.plot(label='Forecast', legend=True)
        plot.grid()
        plt.show()
    
    return predictions, metrics


def darts_block_forecast(model, block_n_steps, series, future_covariates, past_covariates):

    # if future_covariates is not None and past_covariates is not None:
    pred = model.predict(n=block_n_steps,
                         future_covariates=future_covariates,
                         past_covariates=past_covariates,
                         series=series)

    # elif future_covariates is not None:
    #     pred = model.predict(n=block_n_steps,
    #                          future_covariates=future_covariates,
    #                          series=series)

    # elif past_covariates is not None:
    #     pred = model.predict(n=block_n_steps,
    #                          past_covariates=past_covariates,
    #                          series=series)

    # else:
    #     pred = model.predict(n=block_n_steps,
    #                          series=series)
    return pred


def darts_block_n_step_ahead_forecast(model, train, val, block_n_steps=24, n_blocks=31, future_covariates=None, past_covariates=None):

    series = train.append(val)
    history = train

    # calculate predictions in blocks, updating the history after each block
    for i in range(n_blocks):
        pred_i = darts_single_forecast(
            model, block_n_steps, history, future_covariates, past_covariates)
        pred = pred_i if i == 0 else pred.append(pred_i)
        history = history.append(val[block_n_steps*(i):block_n_steps*(i+1)])

    # evaluate
    plt.figure(figsize=(20, 10))
    series.drop_before(pd.Timestamp(pred.time_index[0] - datetime.timedelta(
        days=5))).drop_after(pred.time_index[-1]).plot(label='actual')
    pred.plot(label='forecast')
    plt.legend()
    mape_error = mape(val, pred)
    print('MAPE = {:.2f}%'.format(mape_error))

    return mape_error
