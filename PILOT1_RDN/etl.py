import pretty_errors
from utils import none_checker
import os
from os import times
from utils import download_online_file
from utils import truth_checker
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import darts
from darts.utils.timeseries_generation import holidays_timeseries
import pandas as pd
import math
from datetime import timezone
import matplotlib.pyplot as plt
import mlflow
import click
import shutil
import logging
from darts.dataprocessing.transformers import MissingValuesFiller
import tempfile
from exceptions import CountryDoesNotExist
import holidays
from calendar import isleap
from pytz import timezone
import pytz
import numpy as np

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
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
        x).replace(tzinfo=pytz.utc).timestamp()).astype(int)

    # national_holidays = Province(name="valladolid").national_holidays()
    # regional_holidays = Province(name="valladolid").regional_holidays()
    # local_holidays = Province(name="valladolid").local_holidays()
    # holiday_list = national_holidays + regional_holidays + local_holidays

    calendar['holiday'] = calendar['datetime'].apply(
        lambda x: isholiday(x.date(), holiday_list))
    WNweekday = calendar['datetime'].apply(
        lambda x: x.weekday() if not isholiday(x.date(), holiday_list) else 5 if x.weekday() == 4 else 6)
    calendar['WN'] = WNweekday + calendar['hour']/24 + calendar['minute']/(24*60)
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

    year = datetime_attribute_timeseries(
        time_index=series, attribute='year')

    month = datetime_attribute_timeseries(
        time_index=series, attribute='month', cyclic=True)

    dayofyear = datetime_attribute_timeseries(
        time_index=series, attribute='dayofyear', cyclic=True)

    hour = datetime_attribute_timeseries(
        time_index=series, attribute='hour', cyclic=True)

    # minute = datetime_attribute_timeseries(
    #     time_index=series, attribute='minute', cyclic=True)

    dayofweek = datetime_attribute_timeseries(
        time_index=series, attribute='dayofweek', cyclic=True)

    weekofyear = datetime_attribute_timeseries(
        time_index=series, attribute='weekofyear', cyclic=True)

    # dayofyear = datetime_attribute_timeseries(
    #     time_index=series, attribute='dayofyear')

    holidays = holidays_timeseries(
        time_index=series.time_index, country_code=country_code)

    # weekofyear = darts.TimeSeries.from_series(
    #     series.time_index.isocalendar().week)

    covariates = year.stack(month) \
        .stack(dayofyear) \
        .stack(hour) \
        .stack(dayofweek) \
        .stack(weekofyear) \
        .stack(holidays)
    # .stack(minute)

    return covariates

def remove_outliers(ts: pd.DataFrame,
                    name: str = "Portugal",
                    std_dev: float = 4.5,
                    resolution: str = "15",
                    print_removed: bool = False):
    """
    Reads the input dataframe and replaces its outliers with NaNs by removing
    values that are more than std_dev standard deviations away from their 1 month
    mean or both. This function works with datasets that have NaN values.

    Parameters
    ----------
    ts
        The pandas.DataFrame to be processed
    name
        The name of the country to be displayed on the plots
    std_dev
        The number to be multiplied with the standard deviation of
        each 1 month  period of the dataframe. The result is then used as
        a cut-off value as described above
    resolution
        The resolution of the dataset
    print_removed
        If true it will print the removed values

    Returns
    -------
    pandas.DataFrame
        The original dataframe with its outliers values replaced with NaNs
    """

    #Dates with NaN values are removed from the dataframe
    ts = ts.dropna()
    #Removing all non postive values
    a = ts.loc[ts["Load"] <= 0]
    #Calculating monthly mean and standard deviation and removing values
    #that are more than std_dev standard deviations away from the mean
    mean_per_month = ts.groupby(lambda x: x.month).mean().to_numpy()
    mean = ts.index.to_series().apply(lambda x: mean_per_month[x.month - 1])
    std_per_month = ts.groupby(lambda x: x.month).std().to_numpy()
    std = ts.index.to_series().apply(lambda x: std_per_month[x.month - 1])
    a = pd.concat([a, ts.loc[-std_dev * std + mean > ts['Load']]])
    a = pd.concat([a, ts.loc[ts['Load'] > std_dev * std + mean]])

    #Plotting Removed values and new series
    a = a.sort_values(by='Date')
    a = a[~a.index.duplicated(keep='first')]
    if print_removed: print(f"Removed from {name}", a)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ts.index, ts['Load'], color='black', label = f'Load of {name}')
    ax.scatter(a.index, a['Load'], color='blue', label = 'Removed Outliers')
    plt.legend()
    mlflow.log_figure(fig, 'removed_outliers.png')

    res = ts.drop(index=a.index)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(res.index, res['Load'], color='black', label = f'Load of {name}')
    plt.legend()
    mlflow.log_figure(fig, 'new_series.png')

    return res.asfreq(resolution+'min'), a

def impute(ts: pd.DataFrame,
           holidays,
           max_thr: int = 48,
           a: float = 0.3,
           wncutoff: float = 0.000694,
           ycutoff: int = 3,
           ydcutoff: int = 30,
           resolution: str = "15",
           debug: bool = False):
    """
    Reads the input dataframe and imputes the timeseries using a weighted average of historical data
    and simple interpolation. The weights of each method are exponentially dependent on the distance
    to the nearest non NaN value. More specifficaly, with increasing distance, the weight of
    simple interpolation decreases, and the weight of the historical data increases. If there is
    a consecutive subseries of NaNs longer than max_thr, then it is not imputed and returned with NaN
    values.

    Parameters
    ----------
    ts
        The pandas.DataFrame to be processed
    holidays
        The holidays of the country this timeseries belongs to
    max_thr
        If there is a consecutive subseries of NaNs longer than max_thr,
        then it is not imputed and returned with NaN values
    a
        The weight that shows how quickly simple interpolation's weight decreases as
        the distacne to the nearest non NaN value increases
    wncutoff
        Historical data will only take into account dates that have at most wncutoff distance
        from the current null value's WN(Week Number)
    ycutoff
        Historical data will only take into account dates that have at most ycutoff distance
        from the current null value's year
    ydcutoff
        Historical data will only take into account dates that have at most ydcutoff distance
        from the current null value's yearday
    resolution
        The resolution of the dataset
    debug
        If true it will print helpfull intermediate results

    Returns
    -------
    pandas.DataFrame
        The imputed dataframe
    """

    #Returning calendar of the country ts belongs to
    calendar = create_calendar(ts, int(resolution), holidays, timezone("UTC"))
    calendar.index = calendar["datetime"]

    imputed_values = ts[ts["Load"].isnull()]

    #null_dates: Series with all null dates to be imputed
    null_dates = imputed_values.index


    if debug:
        for date in null_dates:
            print(date)

    #isnull: An array which stores whether each value is null or not
    isnull = ts["Load"].isnull().values

    #d: List with distances to the nearest non null value
    d = [len(null_dates) for _ in range(len(null_dates))]

    #leave_nan: List with all the values to be left NaN because there are
    #more that max_thr consecutive ones
    leave_nan = [False for _ in range(len(null_dates))]

    #Calculating the distances to the nearest non null value that is earlier in the series
    count = 1
    for i in range(len(null_dates)):
        d[i] = min(d[i], count)
        if i < len(null_dates) - 1:
            if null_dates[i+1] == null_dates[i] + pd.offsets.DateOffset(minutes=int(resolution)):
                count += 1
            else:
                count = 1

    #Calculating the distances to the nearest non null value that is later in the series
    count = 1
    for i in range(len(null_dates)-1, -1, -1):
        d[i] = min(d[i], count)
        if i > 0:
            if null_dates[i-1] == null_dates[i] - pd.offsets.DateOffset(minutes=int(resolution)):
                count += 1
            else:
                count = 1

    #If d[i] >= max_thr // 2, that means we have a consecutive subseries of NaNs longer than max_thr.
    #We mark this subseries so that it does not get imputed
    for i in range(len(null_dates)):
        if d[i] == max_thr // 2:
            for ii in range(max(0, i - max_thr // 2 + 1), min(i + max_thr // 2, len(null_dates))):
                leave_nan[ii] = True
        elif d[i] > max_thr // 2:
            leave_nan[i] = True

    #This is the interpolated version of the time series
    ts_interpolatied = ts.interpolate(inplace=False)

    #We copy the time series so that we don't change it while iterating
    res = ts.copy()

    for i, null_date in enumerate(null_dates):
        if leave_nan[i]: continue

        print(null_date)

        #WN: Day of the week + hour/24 + minute/(24*60). Holidays are handled as
        #either Saturdays(if the real day is a Friday) or Sundays(in every other case)
        currWN = calendar.loc[null_date]['WN']
        currYN = calendar.loc[null_date]['yearday']
        currY = calendar.loc[null_date]['year']

        #weight of interpolated series, decreases as distance to nearest known value increases
        w = np.e ** (-a * d[i])

        #Historical value is calculated as the mean of values that have at most wncutoff distance to the current null value's
        #WN, ycutoff distance to its year, and ydcutoff distance to its yearday
        historical = ts[(~isnull) & ((calendar['WN'] - currWN < wncutoff) & (calendar['WN'] - currWN > -wncutoff) &\
                                    (calendar['year'] - currY < ycutoff) & (calendar['year'] - currY > -ycutoff) &\
                                    (((calendar['yearday'] - currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < ydcutoff) |\
                                    ((-calendar['yearday'] + currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < ydcutoff)))]["Load"]

        if debug: print("~~~~~~Date~~~~~~~",null_date, "~~~~~~~Dates summed~~~~~~~~~~",historical,sep="\n")

        historical = historical.mean()

        #imputed value is calculated as a wheighted average of the histrorical value and the value from intrepolation
        res.loc[null_date] = w * ts_interpolatied.loc[null_date] + (1 - w) * historical

        imputed_values.loc[null_date] = res.loc[null_date]

        if debug:
            print(res.loc[null_date])

    return res, imputed_values



@click.command(
    help="Given a timeseries CSV file (see load_raw_data), resamples it, "
         "drops duplicates converts it to darts and optionally creates "
         "the time covariates. The new timeseries is logged as artifact "
         "to MLflow"
)
@click.option("--series-csv",
    type=str,
    default="../../RDN/Load_Data/2009-2019-global-load.csv",
    help="Local timeseries csv. It gets overwritten by uri if given."
)
@click.option("--series-uri",
    type=str,
    default="mlflow_artifact_uri",
    help="Remote timeseries csv file.  If set, it overwrites the local value."
)
@click.option('--year-range',
    default="None",
    type=str,
    help='The year range to include in the dataset.'
)
@click.option("--resolution",
    default="15",
    type=str,
    help="The resolution of the dataset in minutes."
)
@click.option("--time-covs",
    default="PT",
    type=click.Choice(["None", "PT"]),
    help="Optionally add time covariates to the timeseries. [Options: None or Country Code based on the Python 'holidays' package]"
)
@click.option("--day-first",
    type=str,
    default="true",
    help="Whether the date has the day before the month")

@click.option("--country",
    type=str,
    default="Portugal",
    help="The country this dataset belongs to")

@click.option("--std-dev",
    type=str,
    default="4.5",
    help="The number to be multiplied with the standard deviation of \
          each 1 month  period of the dataframe. The result is then used as \
          a cut-off value as described above")

@click.option("--max-thr",
    type=str,
    default="48",
    help="If there is a consecutive subseries of NaNs longer than max_thr, \
          then it is not imputed and returned with NaN values")

@click.option("--a",
    type=str,
    default="0.3",
    help="The weight that shows how quickly simple interpolation's weight decreases as \
          the distacne to the nearest non NaN value increases")

@click.option("--wncutoff",
    type=str,
    default="0.000694",
    help="Historical data will only take into account dates that have at most wncutoff distance \
          from the current null value's WN(Week Number)")

@click.option("--ycutoff",
    type=str,
    default="3",
    help="Historical data will only take into account dates that have at most ycutoff distance \
          from the current null value's year")

@click.option("--ydcutoff",
    type=str,
    default="30",
    help="Historical data will only take into account dates that have at most ydcutoff distance \
          from the current null value's yearday")

def etl(series_csv, series_uri, year_range, resolution, time_covs, day_first, country, std_dev, max_thr, a, wncutoff, ycutoff, ydcutoff):
    # TODO: play with get_time_covariates and create sinusoidal
    # transformations for all features (e.g dayofyear)
    # Also check if current transformations are ok

    if series_uri != "mlflow_artifact_uri":
        download_file_path = download_online_file(series_uri, dst_filename="load.csv")
        series_csv = download_file_path

    ## Process parameters from click and MLProject
    series_csv = series_csv.replace("'", "")
    std_dev = float(std_dev)
    max_thr = int(max_thr)
    a = float(a)
    wncutoff = float(wncutoff)
    ycutoff = int(ycutoff)
    ydcutoff = int(ydcutoff)

    # Time col check
    time_covs = none_checker(time_covs)

    # Day first check
    day_first = truth_checker(day_first)

    print("\nLoading source dataset..")
    logging.info("\nLoading source dataset..")

    ts = pd.read_csv(series_csv,
                     delimiter=',',
                     header=0,
                     index_col=0,
                     parse_dates=True,
                     dayfirst=day_first)

    # Year range handling
    if none_checker(year_range) is None:
        year_range = f"{ts.index[0].year}-{ts.index[-1].year}"
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

    # Tempdir for artifacts
    tmpdir = tempfile.mkdtemp()

    with mlflow.start_run(run_name='etl', nested=True) as mlrun:

        # temporal filtering
        print("\nTemporal filtering...")
        logging.info("\nTemporal filtering...")
        ts = ts[ts.index >= pd.Timestamp(str(year_min) + '0101 00:00:00')]
        ts = ts[ts.index <= pd.Timestamp(str(year_max) + '1231 23:59:59')]

        # ts.to_csv(f'{tmpdir}/1_filtered.csv')

        if resolution != "15":
            print("\nResampling as given frequency is different than 15 minutes")
            logging.info("\nResampling as given frequency is different than 15 minutes")
            ts_res = ts.resample(resolution+'min').sum()
        else:
            ts_res = ts

        # ts_res.to_csv(f'{tmpdir}/2_resampled.csv')

        # drop duplicate index entries, keeping the first
        print("\nDropping duplicate time index entries, keeping first one...")
        logging.info("\nDropping duplicate time index entries, keeping first one...")
        # TODO: fix!!! It is not working correctly! Remember IISA 2022...
        ts_res = ts_res[~ts_res.index.duplicated(keep='first')]

        # ts_res.to_csv(f'{tmpdir}/3_dropped_duplicates.csv')

        print("\nPerfrorming Outlier Detection...")
        logging.info("\nPerfrorming Outlier Detection...")

        ts_res, removed = remove_outliers(ts=ts_res,
                                          name=country,
                                          std_dev=std_dev,
                                          resolution=resolution)
        #holidays_: The holidays of country
        try:
            code = compile(f"holidays.{country}()", "<string>", "eval")
            country_holidays = eval(code)
        except:
            raise CountryDoesNotExist()

        print("\nPerfrorming Imputation of the Dataset...")
        logging.info("\nPerfrorming Imputation of the Dataset...")

        ts_res, imputed_values = impute(ts=ts_res,
                                        holidays=country_holidays,
                                        max_thr=max_thr,
                                        a=a,
                                        wncutoff=wncutoff,
                                        ycutoff=ycutoff,
                                        ydcutoff=ydcutoff,
                                        resolution=resolution)

        print("\nCreating darts data frame...")
        logging.info("\nCreating darts data frame...")

        # explicitly redefine frequency
        ts_res = ts_res.asfreq(resolution+'min')

        # ts_res.to_csv(f'{tmpdir}/4_asfreq.csv')

        # darts dataset creation
        ts_res_darts = darts.TimeSeries.from_dataframe(ts_res)

        # ts_res_darts.to_csv(f'{tmpdir}/4_read_as_darts.csv')


        # ts_res_darts.to_csv(f'{tmpdir}/5_filled_na.csv')

        # time variables creation

        #TODO: Make training happen withot outlier detection
        if time_covs is not None:
            print("\nCreating time covariates dataset...")
            logging.info("\nCreating time covariates dataset...")
            time_covariates = get_time_covariates(ts_res_darts, time_covs)
        else:
            print("\nSkipping the creation of time covariates")
            logging.info("\nSkipping the creation of time covariates")
            time_covariates = None

        # store locally as csv in folder
        print("\nCreating local folder to store the datasets as csv...")
        logging.info("\nCreating local folder to store the datasets as csv...")
        ts_res_darts.to_csv(f'{tmpdir}/series.csv')

        print("\nStoring removed values from outlier detection as csv locally...")
        logging.info("\nStoring removed values from outlier detection as csv...")
        removed.to_csv(f'{tmpdir}/removed.csv')

        print("\nStoring imputed dates and their values as csv locally...")
        logging.info("\nStoring imputed dates and their values as csv locally...")
        imputed_values.to_csv(f'{tmpdir}/imputed_values.csv')

        print("\nStoring datasets locally...")
        logging.info("\nStoring datasets...")
        if time_covariates is not None:
            time_covariates.to_csv(f'{tmpdir}/time_covariates.csv')

        print("\nUploading features to MLflow server...")
        logging.info("\nUploading features to MLflow server...")
        mlflow.log_artifacts(tmpdir, "features")

        print("\nArtifacts uploaded. Deleting local copies...")
        logging.info("\nArtifacts uploaded. Deleting local copies...")
        shutil.rmtree(tmpdir)

        print("\nETL succesful.")
        logging.info("\nETL succesful.")

        # mlflow tags
        mlflow.set_tag("run_id", mlrun.info.run_id)
        mlflow.set_tag("stage", "etl")
        mlflow.set_tag('series_uri', f'{mlrun.info.artifact_uri}/features/series.csv')
        if time_covariates is not None:
            mlflow.set_tag('time_covariates_uri', f'{mlrun.info.artifact_uri}/features/time_covariates.csv')
        else: # default naming for non available time covariates uri
            mlflow.set_tag('time_covariates_uri', 'mlflow_artifact_uri')

        return


if __name__ == "__main__":
    print("\n=========== ETL =============")
    logging.info("\n=========== ETL =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
    logging.info("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
    etl()
