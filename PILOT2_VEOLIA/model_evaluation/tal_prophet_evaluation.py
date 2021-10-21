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

## Load dataset
ts60 = pd.read_csv('../../VEOLIA/artifacts/timeseries_60min.csv', index_col=0, parse_dates=True)
load60 = ts60['Diff Load Activa Total (60 minuto)'].dropna()
load60.head()

# Train / Test
last_train_day = datetime(2021, 6, 20)
train = load60[:(last_train_day - timedelta(hours=1))]  
train = train.asfreq('H')
test = load60[last_train_day:]
test = test.asfreq('H')
test.name = train.name

# Steps for day ahead forecasts are preset to 24 (1 day)
steps = 24
days_ahead = 11
#####################################################################
# Load model
file = open('../../VEOLIA/models/tal_prophet.pkl', 'rb')
# dump information to that file
model_prophet = pickle.load(file)
# close the file
file.close()

# evaluation
future = pd.DataFrame(test.index[:steps]).rename(columns={'datetime':'ds'})
forecast_prophet = model_prophet.predict(future)
predictions = forecast_prophet['yhat'].values
ground_truth = test[:steps]

print("MAPE:", mape(ground_truth, predictions))
print("MSE:", mse(ground_truth, predictions))
print("RMSE:", np.sqrt(mse(ground_truth, predictions)))

# plot forecast
comparison_prophet = pd.DataFrame(ground_truth)
comparison_prophet['Forecast'] = predictions
training = pd.DataFrame(train, columns=comparison_prophet.filter(like='Diff').columns.tolist())
comparison_prophet = pd.concat([training, comparison_prophet])
plot = comparison_prophet[datetime(2021, 6, 19):].plot(figsize=(12,7))
plot.grid()

#####################################################################
# Simple 11-day ahead forecast with error propagation
steps= test.shape[0]
future = pd.DataFrame(test.index[:steps]).rename(columns={'datetime':'ds'})
forecast_prophet = model_prophet.predict(future)
predictions = pd.Series(forecast_prophet['yhat'].values)
predictions.index = test.index
ground_truth = test[:steps]
print(f"Simple {days_ahead} day ahead forecast:")
ground_truth_line = pd.concat([train[- int(days_ahead / 2) * 24 :], test])

print("MAPE:", mape(test, predictions))
print("MSE:", mse(test, predictions))
print("RMSE:", np.sqrt(mse(test, predictions)))

plt.figure()
plot = ground_truth_line.plot(figsize=(15, 3), label='Data', legend=True, title=f"{days_ahead} day ahead forecast")
predictions.plot(label='Forecast', legend=True)
plot.grid()
plt.show()