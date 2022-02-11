import pandas as pd
import statsmodels
import statsmodels.api as sm
import numpy as np
from darts.models import RNNModel
from darts import TimeSeries
from inference import darts_block_n_step_ahead_forecast
from etl import get_time_covariates
import os
import torch
import re
import pickle
from darts.models import RNNModel
from darts.dataprocessing.transformers import MissingValuesFiller

# def predict_load(input_dict, model_path='/models/tal_sarima.pkl'):
#     sarima = statsmodels.tsa.statespace.sarimax.SARIMAXResults.load('models/tal_sarima.pkl')
#     news = input_dict['news']
#     sarima = sarima.append(news)
#     predictions = sarima.forecast(input_dict['days_ahead'] * input_dict['daily_steps'])
#     print(predictions)
#     return predictions


def predict_load(input_dict, model_name, models_path, freq=None):

    # load model
    model_folder_file_list = os.listdir(os.path.join(models_path, model_name))
    model_filename = [fname for fname in model_folder_file_list if re.search('pth.tar$', fname)][0]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    lstm = torch.load(os.path.join(models_path, model_name, model_filename), map_location=device)
    lstm.device = device
    # lstm = RNNModel.load_from_checkpoint(model_name=model_name, best=True)

    # get inputs
    news = input_dict['news']
    dates = input_dict['datetime']
    forecast_horizon = input_dict["forecast_horizon"]

    # build history darts Series
    history = pd.Series(data=news, index=pd.to_datetime(dates, infer_datetime_format=True))
    if freq is None:
        freq = pd.infer_freq(history.index)
    history = TimeSeries.from_series(history, freq=freq)
    
    # scale history
    scaler = pickle.load(open(os.path.join(models_path, model_name, 'scaler.pkl'), "rb"))
    history_scaled = scaler.transform(history)

    # need here preprocessing step to create covariates for the given dates!!!!
    scaler_cov = pickle.load(open(os.path.join(models_path, model_name, 'scaler_cov.pkl'), "rb"))

    # series = TimeSeries.from_series(pd.Series(data=np.append(history.data_array.values, np.zeros(forecast_horizon)
    series_scaled = history_scaled.append_values(np.zeros(forecast_horizon)) # add extra dates here to get future covariates below
    future_covariates_scaled = scaler_cov.transform(get_time_covariates(series_scaled))


    pred_scaled = darts_block_n_step_ahead_forecast(model=lstm,
                                      history=history_scaled,
                                      test=None,
                                      block_n_steps=forecast_horizon,
                                      n_blocks=1,
                                      future_covariates=future_covariates_scaled,
                                      past_covariates=None,
                                      path_to_save_eval=None)

    # inverse scale pred
    pred = scaler.inverse_transform(pred_scaled)
    # as pandas series (to avoid importing darts also in the predict_load_server.py)
    return pred.pd_series()

# # example usage
# if __name__ == '__main__':
#     model_path = 'models/tal_sarima.pkl'

#     # parameters to be provided by the model client
#     days_ahead = 6
#     days_to_append = 1
#     daily_steps = 24

#     # time series to be provided by the client
#     # create random input vector (known history / news)
#     np.random.seed(0)
#     y_news = np.random.normal(2300, 30, days_to_append * daily_steps)
    
#     # the final parameter dict to be provided by the client
#     input_dict = {'days_to_append': days_to_append, 
#                   'days_ahead': days_ahead, 
#                   'daily_steps': daily_steps,
#                   'news': y_news 
#                  }
#     print(predict_load(input_dict, model_path))
