from datetime import timedelta
import pandas as pd

# def rdn_train_test_split(timeseries, last_train_day, days_ahead, steps, freq='H', scaler_class=None):
    
#     train = timeseries[:(last_train_day - timedelta(hours=1))].asfreq(freq)
#     test = timeseries[last_train_day:last_train_day + timedelta(hours=days_ahead * steps - 1)].asfreq(freq)
    
#     # Scaling
#     if scaler_class:
       
#         scaler_class = scaler_class.fit(train.values.reshape(-1, 1))

#         train_scaled = pd.Series(scaler_class.transform(train.values.reshape(-1, 1)).reshape(-1), index=train.index)
#         test_scaled = pd.Series(scaler_class.transform(test.values.reshape(-1, 1)).reshape(-1), index=test.index)
        
#         return train, test, train_scaled, test_scaled, scaler_class
    
#     else:
        
#         return train, test


def rdn_train_test_split(timeseries, last_train_day, days_ahead, steps, freq='H'):

    train = timeseries[:(last_train_day - timedelta(hours=1))].asfreq(freq)
    test = timeseries[last_train_day:last_train_day +
                      timedelta(hours=days_ahead * steps - 1)].asfreq(freq)

    return train, test
    
