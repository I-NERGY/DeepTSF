from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.timeseries_generation import holidays_timeseries

def get_time_covariates(series):
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
    weekofyear_60 = datetime_attribute_timeseries(
        time_index=series, attribute='weekofyear')
    holidays_60 = holidays_timeseries(
        time_index=series.time_index, country_code='PT')

    covariates = year_60.stack(month_60) \
        .stack(day_60) \
        .stack(hour_60) \
        .stack(dayofweek_60) \
        .stack(dayofyear_60) \
        .stack(weekofyear_60) \
        .stack(holidays_60)
    
    return covariates

