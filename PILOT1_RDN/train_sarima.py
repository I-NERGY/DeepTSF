#!/usr/bin/env python
# coding: utf-8

# # Training and forecast example of an automated ARIMA model (TBATS).

from rdn_train_test_split import rdn_train_test_split_old
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels
import pickle
from datetime import timedelta
# from tbats import TBATS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import math
# from evaluate_forecasts import simple_n_day_ahead_forecast
import os


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



# ## Load dataset


ts60 = pd.read_csv(
    '../../RDN/Load_Data/2009-2019-global-load.csv', index_col=0, parse_dates=True)
load60 = ts60['Load'].dropna()
load60.head()

steps = 96
days_ahead = 1
last_train_year_id = 2018
last_train_month_id = 12
last_train_day_id = 31
last_train_day = datetime(last_train_year_id, last_train_month_id, last_train_day_id)  # a week finishes there
model_id_date = f'{last_train_year_id}_{last_train_month_id}_{last_train_day_id}'

# choose type of scaler
scaler = StandardScaler()

# model storing directory
# model_dir = '../../RDN/Load Data(2018-2019)/models/'
model_dir = '.'

# ## Train / test split
train, test, train_std, test_std, scalert = rdn_train_test_split_old(load60,
                                                                     last_train_day,
                                                                     days_ahead,
                                                                     steps,
                                                                     freq='15min',
                                                                     scaler_class=scaler)

# ## SARIMA

# ### Parsimonious SARIMA
# print("Training SARIMA parsimonious...\n")
# sarima_pars = sm.tsa.statespace.SARIMAX(endog=train_std, order=(2, 1, 1),
#                                         seasonal_order=(2, 1, 2, 24)).fit(max_iter=200, method='powell')
# print(sarima_pars.summary())

# # #### Simple n-day ahead forecast


# _, _, scaler, _ = simple_n_day_ahead_forecast(sarima_pars, days_ahead,
#                                               steps, train, test, scaler)


# # #### Save model locally

# # workaround to make pickle work for SARIMAX
# def __getnewargs__(self):
#     return (tuple(i for i in self.params_complete))


# statsmodels.api.tsa.statespace.SARIMAX.__getnewargs__ = __getnewargs__

# # storing
# fname = os.path.join(model_dir, f"rdn_sarima_pars_{model_id_date}.pkl")
# with open(fname, 'wb') as file:
#     pickle.dump(sarima_pars, file)

# ### Full SARIMA model from r gridsearch
# A sarima is fitted as selected in the gridsearch process of R_tal_time_series_analysis.ipynb (further analysis can be sought there). However we have limited the seasonal autoregressive terms to 1 for speed and simplicity as longs as the model could not be saved otherwise. AIC, residuals, ACF, PACFs are presented again for completeness.
print("Training SARIMA full...\n")
sarima = sm.tsa.statespace.SARIMAX(endog=train_std, order=(2, 1, 1),
                                   seasonal_order=(4, 1, 2, 96)).fit(max_iter=200, method='powell')
print(sarima.summary())

res = sarima.resid
fig, ax = plt.subplots(2, 1, figsize=(20, 20))
fig = sm.graphics.tsa.plot_acf(res, lags=100, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=100, ax=ax[1])
plt.show()

# #### Simple n-day ahead forecast

simple_n_day_ahead_forecast(sarima, days_ahead, steps, train, test, scaler)

# #### Save model locally


def __getnewargs__(self):
    return (tuple(i for i in self.params_complete))


statsmodels.api.tsa.statespace.SARIMAX.__getnewargs__ = __getnewargs__

# storing
fname = os.path.join(model_dir, f"rdn_sarima_{model_id_date}.pkl")
file = open(fname, 'wb')
pickle.dump(sarima, file)
file.close()
