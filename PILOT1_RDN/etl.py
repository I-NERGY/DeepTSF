from os import times
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import darts
from pandas import DatetimeIndex
from darts.utils.timeseries_generation import holidays_timeseries
import datetime
import pandas as pd
import math
from datetime import timezone
import matplotlib.pyplot as plt
import mlflow 
import click
from distutils import util
import os
from utils import read_config
import shutil
import logging

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
os.environ["MLFLOW_TRACKING_URI"] = read_config(
    'config.yml', 'mlflow_settings')['mlflow_tracking_uri']
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# DATE HANDLERS
def isholiday(x, holiday_list):
    if x in holiday_list:
        return True
    return False


def isweekend(x):
    if x == 6 or x == 0:
        return True
    return False


def create_calendar(timeseries, timestep_minutes, holiday_list, local_timezone):

    calendar = pd.DataFrame(
        timeseries.index.tolist(),
        columns=['datetime']
    )

    calendar['year'] = calendar['datetime'].apply(lambda x: x.year)
    calendar['month'] = calendar['datetime'].apply(lambda x: x.month)
    calendar['yearweek'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%V")) - 1)
    calendar['day'] = calendar['datetime'].apply(lambda x: x.day)
    calendar['hour'] = calendar['datetime'].apply(lambda x: x.hour)
    calendar['minute'] = calendar['datetime'].apply(lambda x: x.minute)
    calendar['second'] = calendar['datetime'].apply(lambda x: x.second)
    calendar['weekday'] = calendar['datetime'].apply(lambda x: x.weekday())
    calendar['monthday'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%d")) - 1)
    calendar['weekend'] = calendar['weekday'].apply(lambda x: isweekend(x))
    calendar['yearday'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%j")) - 1)

    # first convert to utc and then to timestamp
    calendar['timestamp'] = calendar['datetime'].apply(lambda x: local_timezone.localize(
        x).replace(tzinfo=timezone.utc).timestamp()).astype(int)

    # national_holidays = Province(name="valladolid").national_holidays()
    # regional_holidays = Province(name="valladolid").regional_holidays()
    # local_holidays = Province(name="valladolid").local_holidays()
    # holiday_list = national_holidays + regional_holidays + local_holidays

    calendar['holiday'] = calendar['datetime'].apply(
        lambda x: isholiday(x.date(), holiday_list))
    return calendar


def add_cyclical_time_features(calendar):
    """ 
    The function below is useful to create sinusoidal transformations of time features. 
    This article explains why 2 transformations are necessary: 
    https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
    """

    calendar['month_sin'] = calendar['month'].apply(
        lambda x: math.sin(2*math.pi/12*(x-1)))
    calendar['weekday_sin'] = calendar['weekday'].apply(
        lambda x: math.sin(2*math.pi/7*(x)))
    calendar['monthday_sin'] = calendar['monthday'].apply(
        lambda x: math.sin(2*math.pi/30*(x)))
    calendar['yearday_sin'] = calendar['yearday'].apply(
        lambda x: math.sin(2*math.pi/365*(x)))
    calendar['hour_sin'] = calendar['hour'].apply(
        lambda x: math.sin(2*math.pi/24*(x)))
    calendar['yearday_sin'] = calendar['yearday'].apply(
        lambda x: math.sin(2*math.pi/52.1428*(x)))

    calendar['month_cos'] = calendar['month'].apply(
        lambda x: math.cos(2*math.pi/12*(x-1)))
    calendar['weekday_cos'] = calendar['weekday'].apply(
        lambda x: math.cos(2*math.pi/7*(x)))
    calendar['monthday_cos'] = calendar['monthday'].apply(
        lambda x: math.cos(2*math.pi/30*(x)))
    calendar['yearday_cos'] = calendar['yearday'].apply(
        lambda x: math.cos(2*math.pi/365*(x)))
    calendar['hour_cos'] = calendar['hour'].apply(
        lambda x: math.cos(2*math.pi/24*(x)))
    calendar['yearday_cos'] = calendar['yearday'].apply(
        lambda x: math.cos(2*math.pi/52.1428*(x)))

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 5, 1)
    calendar['month_sin'][:50000].plot()
    plt.subplot(2, 5, 2)
    calendar['weekday_sin'][:1000].plot()
    plt.subplot(2, 5, 3)
    calendar['monthday_sin'][:1000].plot()
    plt.subplot(2, 5, 4)
    calendar['yearday_sin'][:1000000].plot()
    plt.subplot(2, 5, 5)
    calendar['hour_sin'][:96].plot()

    plt.subplot(2, 5, 6)
    calendar['month_cos'][:50000].plot()
    plt.subplot(2, 5, 7)
    calendar['weekday_cos'][:1000].plot()
    plt.subplot(2, 5, 8)
    calendar['monthday_cos'][:1000].plot()
    plt.subplot(2, 5, 9)
    calendar['yearday_cos'][:1000000].plot()
    plt.subplot(2, 5, 10)
    calendar['hour_cos'][:96].plot()

    return calendar


def get_time_covariates(series, country_code='PT'):
    """ Do it the darts way"""
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
        time_index=series.time_index, country_code=country_code)
    weekofyear_60 = darts.TimeSeries.from_series(
        series.time_index.isocalendar().week)

    covariates = year_60.stack(month_60) \
        .stack(day_60) \
        .stack(hour_60) \
        .stack(dayofweek_60) \
        .stack(dayofyear_60) \
        .stack(weekofyear_60) \
        .stack(holidays_60)

    return covariates

@click.command(
    help="Given a timeseries CSV file (see load_raw_data), resamples it, converts it to darts " \
         "and optionally creates the time covariates. The new timeseries is logged as artifact to MLflow"
)
@click.option("--series-csv", 
    type=str, 
    default="../../RDN/Load_Data/load_2018_2021.csv"
    # default='artifact_uri'
)
@click.option('--year-range',  
    default="2018-2019",
    type=str,
    help='Choose year range to include in the dataset.'
)
@click.option("--resolution", 
    default="15", 
    type=str,
    help="Change the resolution of the dataset (minutes)."
)
@click.option(
    "--time-covs", 
    default=" ", 
    type=str,
    help="Add time covariates to the timeseries"
)
def etl(series_csv, year_range, resolution, time_covs):

    ## Process parameters from click and MLProject
    series_csv = series_csv.replace('/', os.path.sep).replace("'", "")

    # Convert some str parameters to boolean (MLproject does not permit boolean)
    # time_covs = util.strtobool(time_covs)

    # Year range handling
    if "-" in year_range:
        year_range = year_range.split("-")
    if isinstance(year_range, list):
        year_range = [int(i) for i in year_range]
        year_min = min(year_range)
        year_max = max(year_range)
    else:
        year = int(year_range)
        year_max = year
        year_min = year


    resolution = int(resolution)

    with mlflow.start_run() as mlrun:
        ts = pd.read_csv(series_csv, 
                        delimiter=',', 
                        header=0, 
                        index_col=0, 
                        parse_dates=True)

        # temporal filtering
        ts = ts[ts.index >= pd.Timestamp(str(year_min) + '0101 00:00:00')]
        ts = ts[ts.index <= pd.Timestamp(str(year_max) + '1231 23:59:59')]
        print(ts)
        
        # _res_darts.drop_before(pd.Timestamp(year_min))
        # ts_res_darts = ts_res_darts.drop_after(pd.Timestamp(year_max))

        if resolution != "15":
            ts_res = ts.resample(f'{str(resolution)}T').sum()
        else:
            ts_res = ts

        # darts dataset creation
        ts_res_darts = darts.TimeSeries.from_dataframe(ts_res, freq=f'{str(resolution)}min')

        # time variables creation
        if time_covs != " ":
            time_covariates = get_time_covariates(ts_res_darts, time_covs)
        
        # Darts future_covariates creation (darts) - a priori known covariates for the timestep of the forecasts
        future_covariates = time_covariates

        # ## TODO: Darts past_covariates creation 
        past_covariates = None

        # store locally as csv in folder
        os.makedirs("features", exist_ok=True)
        ts_res_darts.to_csv('features/series.csv')
        if future_covariates is not None:
            future_covariates.to_csv('features/future_covariates.csv')
        if past_covariates is not None: 
            past_covariates.to_csv('features/past_covariates.csv')

        print("Uploading features to MLflow server...")
        mlflow.log_artifacts("features", "features")
        print("Artifacts uploaded. Deleting local copies...")
        shutil.rmtree("features")
        print("Local copies deleted.")


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    logging.info("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    etl()