from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.timeseries_generation import holidays_timeseries
from pandas import DatetimeIndex
import darts
import pandas as pd

def get_time_covariates(series):
    if isinstance(series, pd.Series):
        series = darts.TimeSeries.from_series(series)
    year_60 = datetime_attribute_timeseries(
        time_index=series, attribute='year')
    month_60 = datetime_attribute_timeseries(
        time_index=series, attribute='month', cyclic=True)
    day_60 = datetime_attribute_timeseries(
        time_index=series, attribute='day', cyclic=True)
    hour_60 = datetime_attribute_timeseries(
        time_index=series, attribute='hour', cyclic=True)
    dayofweek_60 = datetime_attribute_timeseries(
        time_index=series, attribute='dayofweek')
    dayofyear_60 = datetime_attribute_timeseries(
        time_index=series, attribute='dayofyear')
    holidays_60 = holidays_timeseries(
        time_index=series.time_index, country_code='PT')
    weekofyear_60 = darts.TimeSeries.from_series(series.time_index.isocalendar().week)

    covariates = year_60.stack(month_60) \
        .stack(day_60) \
        .stack(hour_60) \
        .stack(dayofweek_60) \
        .stack(dayofyear_60) \
        .stack(weekofyear_60) \
        .stack(holidays_60)

    return covariates
