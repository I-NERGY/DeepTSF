#!/usr/bin/env python
# coding: utf-8

# # Training and forecast example of an automated ARIMA model (TBATS).

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
from tbats import TBATS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import math
from evaluate_forecasts import simple_n_day_ahead_forecast
import os


# ## Load dataset


ts60 = pd.read_csv('../../RDN/Load Data (2018-2019)/artifacts/load_60min.csv', index_col=0, parse_dates=True)
load60 = ts60['Load'].dropna()
load60.head()

steps = 24
days_ahead = 7
last_train_year_id = 2019
last_train_month_id = 12
last_train_day_id = 1
last_train_day = datetime(last_train_year_id, last_train_month_id, last_train_day_id) # a week finishes there
model_id_date = f'{last_train_year_id}_{last_train_month_id}_{last_train_day_id}'

scaler_class = StandardScaler()

# model storing directory
# model_dir = '../../RDN/Load Data(2018-2019)/models/'
model_dir = '.'

# ## Train / test split


from rdn_train_test_split import rdn_train_test_split
train, test = rdn_train_test_split(load60,
                                   last_train_day,
                                   days_ahead,
                                   steps,
                                   freq='H')

scaler = scaler_class.fit(train.values.reshape(-1, 1))
train_scaled = pd.Series(scaler.transform(
    train.values.reshape(-1, 1)).reshape(-1), index=train.index)
test_scaled = pd.Series(scaler.transform(
    test.values.reshape(-1, 1)).reshape(-1), index=test.index)

# ## SARIMAX
# Gridsearch has run in R code. External variables are added to better capture complex seasonal relationships. Time variables are expected to help capture here more complex seasonalities and time relations. For that reason the parsimonious sarima will be used. Model is stored to disk. Weather is not yet available for RDN timeseries


# weather = pd.read_csv('../../RDN/Load Data (2018-2019)/artifacts/weather_curated_60min.csv',
#                       index_col=0, parse_dates=True)[datetime(2020, 11, 6):]
# weather.head()


time = pd.read_csv(
    '../../RDN/Load Data (2018-2019)/artifacts/time_60min.csv', index_col=1, parse_dates=True)

time_useful = time[['month_sin', 'month_cos', 
                    'weekday_sin', 'weekday_cos',
                    'hour_sin', 'hour_cos', 
                    'holiday', 
                    'monthday_sin',
                    'monthday_cos', 
                    'yearday_cos', 
                    'yearday_sin']]
time_useful['holiday'] = time_useful['holiday'].astype(int)
print(time_useful.head())


# exog_training = pd.concat((weather.loc[train.index], time_useful.loc[train.index]), axis=1)
# exog_testing = pd.concat((weather.loc[test.index], time_useful.loc[test.index]), axis=1)
# exog_training.head()
exog_training = time_useful.loc[train.index]
exog_testing = time_useful.loc[test.index]

# # ### Parsimonious SARIMAX
# print("Training SARIMAX parsimonious...\n")
# sarimax_pars = sm.tsa.statespace.SARIMAX(endog=train_scaled,
#                                          order=(2, 1, 1),
#                                          exog=exog_training.values,
#                                          seasonal_order=(2, 1, 2, 24)) \
#                     .fit(max_iter=200, method='powell')
# print(sarimax_pars.summary())

# # #### Simple n-day ahead forecast
# # No need for scaler in exogenous variables. There are already scaled from preprocessing stage.


# simple_n_day_ahead_forecast(sarimax_pars,
#                             days_ahead,
#                             steps,
#                             train,
#                             test,
#                             scaler,
#                             exog_training)

# # #### Save model to disk

# def __getnewargs__(self):
#     return (tuple(i for i in self.params_complete))


# statsmodels.api.tsa.statespace.SARIMAX.__getnewargs__ = __getnewargs__

# # storing
# fname = os.path.join(model_dir, f"rdn_sarimax_pars_{model_id_date}.pkl")
# f = open(fname, 'wb')
# pickle.dump(sarimax_pars, f)
# f.close()

# ### Full model
# Same process here for the full model. Unable to store this model...
print("Training SARIMAX full...\n")
sarimax = sm.tsa.statespace.SARIMAX(endog=train_scaled,
                                    order=(2, 1, 1),
                                    exog=exog_training.values,
                                    seasonal_order=(4, 1, 2, 24)) \
    .fit(max_iter=50, method='powell')
print(sarimax.summary())


predictions, metrics = \
    simple_n_day_ahead_forecast(sarimax,
                            days_ahead,
                            steps,
                            train,
                            test,
                            scaler,
                            exog_training)

print(metrics)

# store
def __getnewargs__(self):
    return (tuple(i for i in self.params_complete))


statsmodels.api.tsa.statespace.SARIMAX.__getnewargs__ = __getnewargs__

fname = os.path.join(model_dir, f"rdn_sarimax_{model_id_date}.pkl")
f = open(fname, 'wb')
pickle.dump(sarimax, f)
f.close()
