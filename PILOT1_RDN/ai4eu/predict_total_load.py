import pandas as pd
import statsmodels
import statsmodels.api as sm
import numpy as np
from darts.models import RNNModel
from darts import TimeSeries
from evaluate_forecasts import darts_block_n_step_ahead_forecast

# def predict_load(input_dict, model_path='/models/tal_sarima.pkl'):
#     sarima = statsmodels.tsa.statespace.sarimax.SARIMAXResults.load('models/tal_sarima.pkl')
#     news = input_dict['news']
#     sarima = sarima.append(news)
#     predictions = sarima.forecast(input_dict['days_ahead'] * input_dict['daily_steps'])
#     print(predictions)
#     return predictions


def predict_load(input_dict, model_name='LSTM_120', freq=None):

    # load model
    lstm = RNNModel.load_from_checkpoint(work_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),
        model_name=model_name, best=True)

    # get inputs
    news = input_dict['news']
    dates = input_dict['dates']
    forecast_horizon = input_dict["forecast_horizon"]

    # build history darts Series
    history = pd.Series(data=news, index=pd.to_datetime(dates, infer_datetime_format=True))
    history = TimeSeries.from_series(history, freq=freq)
    
    # scale history
    scaler = pickle.load(open(os.path.join(models, 'scaler.pkl'), "rb"))
    history = scaler.fit_transform(history)

    # need here preprocessign step to create covariates for the given dates!!!!
#######################
##################
    pred = darts_block_n_step_ahead_forecast(model=lstm,
                                      history=history,
                                      test=None,
                                      block_n_steps=forecast_horizon,
                                      n_blocks=1,
                                      future_covariates=future_covariates,
                                      past_covariates=None,
                                      path_to_save_eval=None)

    # inverse scale pred
    pred = scaler.inverse_transform(pred)
    print(pred)
    return pred # as pandas series (to avoid importing darts also in the predict_load_server.py)

def produce_time_covariates():


# example usage
if __name__ == '__main__':
    model_path = 'models/tal_sarima.pkl'

    # parameters to be provided by the model client
    days_ahead = 6
    days_to_append = 1
    daily_steps = 24

    # time series to be provided by the client
    # create random input vector (known history / news)
    np.random.seed(0)
    y_news = np.random.normal(2300, 30, days_to_append * daily_steps)
    
    # the final parameter dict to be provided by the client
    input_dict = {'days_to_append': days_to_append, 
                  'days_ahead': days_ahead, 
                  'daily_steps': daily_steps,
                  'news': y_news 
                 }
    print(predict_load(input_dict, model_path))
