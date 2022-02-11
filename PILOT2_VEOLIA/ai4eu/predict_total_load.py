import pandas as pd
import statsmodels
import statsmodels.api as sm
import numpy as np

def predict_load(input_dict, model_path='/models/tal_sarima.pkl'):
    sarima = statsmodels.tsa.statespace.sarimax.SARIMAXResults.load('models/tal_sarima.pkl')
    news = input_dict['news']
    sarima = sarima.append(news)
    predictions = sarima.forecast(input_dict['days_ahead'] * input_dict['daily_steps'])
    print(predictions)
    return predictions


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
