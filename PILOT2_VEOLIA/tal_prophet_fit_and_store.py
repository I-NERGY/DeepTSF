import pandas as pd
from datetime import datetime
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta

# Load dataset
ts60 = pd.read_csv('../../VEOLIA/artifacts/timeseries_60min.csv', index_col=0, parse_dates=True)
load60 = ts60['Diff Load Activa Total (60 minuto)'].dropna()
load60.head()

# Load calendar
cal60 = pd.read_csv('../../VEOLIA/artifacts/time_60min.csv', index_col=0, parse_dates=True)

holidays = pd.DataFrame({
  'holiday': 'Valladolid',
  'ds': np.unique(pd.to_datetime(cal60[cal60['holiday']==1]['datetime'],
        format="%Y-%m-%d %H:%M:%S").apply(lambda x:x.date()).values).tolist(),
})

## Train / test split
last_train_day = datetime(2021, 6, 20)
train = load60[:(last_train_day - timedelta(hours=1))]
train = train.asfreq('H')
test = load60[last_train_day:]
test = test.asfreq('H')

# Training prophet
# add datetime as column for prophet
train_prophet = pd.DataFrame()
train_prophet['ds'] = train.index
train_prophet['y'] = train.values

# include holidays

# define the model
model_prophet = Prophet(holidays = holidays)
# fit the model
model_prophet.fit(train_prophet)

## Save model locally
fname = "../../VEOLIA/models/tal_prophet.pkl"  
with open(fname, 'wb') as file:
    pickle.dump(model_prophet, file)

# quick evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
steps=24
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
train = pd.DataFrame(train, columns=comparison_prophet.filter(like='Diff').columns.tolist())
comparison_prophet = pd.concat([train, comparison_prophet])
plt = comparison_prophet[datetime(2021, 6, 19):].plot(figsize=(12,7))
plt.grid()
