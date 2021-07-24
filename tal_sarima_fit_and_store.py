
from tbats import TBATS
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse 
from sklearn.metrics import mean_absolute_percentage_error as mape
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels
import pickle
from datetime import timedelta

# Load dataset

ts60 = pd.read_csv('../VEOLIA/artifacts/timeseries_60min.csv', index_col=0, parse_dates=True)
load60 = ts60['Diff Load Activa Total (60 minuto)'].dropna()
load60.head()

# Train / test split
last_train_day = datetime(2021, 6, 20)
train = load60[:(last_train_day - timedelta(hours=1))]  
train = train.asfreq('H')
test = load60[last_train_day:]
test = test.asfreq('H')

# Configuration
# Steps for day ahead forecasts are preset to 24 (1 day)
steps = 24

# SARIMA model from r gridsearch
sarima = sm.tsa.statespace.SARIMAX(endog=train, order=(4, 1, 1),
                                   seasonal_order=(2, 1, 1, 24)).fit(max_iter=50, method='powell')
print(sarima.summary())

# Save model locally
# workaround to make pickle work for SARIMAX
def __getnewargs__(self):
    return (tuple(i for i in self.params_complete))

statsmodels.api.tsa.statespace.SARIMAX.__getnewargs__ = __getnewargs__

# Store model locally
fname = "../VEOLIA/models/tal_sarima2.pkl"  
file = open(fname, 'wb')
pickle.dump(sarima, file)
file.close()