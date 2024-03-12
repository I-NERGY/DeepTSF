import sys
sys.path.append('..')
import pretty_errors
from utils import none_checker
import os
from os import times
from utils import download_online_file, truth_checker, multiple_ts_file_to_dfs, multiple_dfs_to_ts_file
from utils import plot_imputation, plot_removed, get_weather_covariates
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
from exceptions import CountryDoesNotExist, NoUpsamplingException, TsUsedIdDoesNotExcist
import holidays
from calendar import isleap
from pytz import timezone
import pytz
import numpy as np
from tqdm import tqdm
from pytz import country_timezones
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

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

def get_time_covariates(series, country_code='PT', id_name='0'):
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

    ts_list_covariates =  year.stack(month). \
                               stack(dayofyear). \
                               stack(hour). \
                               stack(dayofweek). \
                               stack(weekofyear). \
                               stack(holidays)
    
    ts_list_covariates = [ts_list_covariates.univariate_component(i).pd_dataframe() for i in range(ts_list_covariates.n_components)]

    id_l_covariates = ["year", 
                            "month_sin",
                            "month_cos", 
                            "dayofyear_sin",
                            "dayofyear_cos",
                            "hour_sin", 
                            "hour_cos",
                            "dayofweek_sin", 
                            "dayofweek_cos",
                            "weekofyear_sin",
                            "weekofyear_cos",
                            "holidays"]
    ts_id_l_covariates = [id_name for _ in range(12)]

    return ts_list_covariates, id_l_covariates, ts_id_l_covariates

def cut_extra_samples(ts_list):
    print("\nMaking all components of each ts start and end on the same timestep...")
    logging.info("\nnMaking all components of each ts start and end on the same timestep...")
    ts_list_cut = []

    for i, ts in enumerate(ts_list):
        earliest_end = min(comp.index[-1] for comp in ts)
        latest_beginning = max(comp.index[0] for comp in ts)

        print(f"\nMaking series {i} \ {len(ts_list) - 1} start on {latest_beginning} and end on {earliest_end}...")
        logging.info(f"\nMaking series {i} \ {len(ts_list) - 1} start on {latest_beginning} and end on {earliest_end}...")
        ts_list_cut.append(list(comp[(comp.index <= earliest_end) & (comp.index >= latest_beginning)] for comp in ts))
    return ts_list_cut

def add_weather_covariates(res_, res_future, id_l_future_covs, ts_id_l_future_covs, ts_id_l, fields=["shortwave_radiation"]):
    other_covs = (res_future == [])
    for i, ts in tqdm(list(enumerate(res_))):
        end = ts[0].index[-1]
        start = ts[0].index[0]
        covs_nans = get_weather_covariates(start, end, fields)
        covs = []
        for cov in covs_nans:
            covs.append(cov.asfreq('60min').interpolate(inplace=False, method='time'))
        for elem in covs:
            assert elem.isnull().sum().sum() == 0
        if other_covs:
            res_future.append(covs) 
            id_l_future_covs.append(list(map(lambda elem : elem + "_" + ts_id_l[i][0], fields)))
            ts_id_l_future_covs.append([ts_id_l[i][0] for _ in range(len(covs))])
            #TODO check multivariate and multiple
            #TODO check past and future covs more exactly
        else:
            res_future[i].extend(covs) 
            id_l_future_covs[i].extend(list(map(lambda elem : elem + "_" + ts_id_l[i][0], fields)))
            ts_id_l_future_covs[i].extend([ts_id_l[i][0] for _ in range(len(covs))])
    return res_, res_future, id_l_future_covs, ts_id_l_future_covs


def remove_outliers(ts: pd.DataFrame,
                    name: str = "Portugal",
                    std_dev: float = 4.5,
                    resolution: str = "15",
                    print_removed: bool = False,
                    min_non_nan_interval: int = -1,
                    outlier_dir: str = ""):
    """
    Reads the input dataframe and replaces its outliers with NaNs by removing
    values that are more than std_dev standard deviations away from their 1 month
    mean. This function works with datasets that have NaN values.

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
    Tuple[pandas.DataFrame, pandas.DataFrame]
        The original dataframe with its outlier values replaced with NaNs, along
        with a dataframe consisting of the removed values.
    """
    null_dates = ts[ts["Value"].isnull()].index
    #Datetimes with NaN values are removed from the dataframe
    ts = ts.dropna()
    #Keeping 0s for UC7
    a = pd.DataFrame(columns=ts.columns)
    a.index.name = ts.index.name

    #Calculating monthly mean and standard deviation and removing values
    #that are more than std_dev standard deviations away from the mean
    mean_per_month = ts.groupby(lambda x: str(x.month) + "-" + str(x.year)).mean().to_dict()["Value"]

    resolution = int(resolution)
    nan_sum = 0
    prev = null_dates[0]
    cut_off = ts.quantile(.99)[0]
    for null_date in null_dates[1:]:
        nan_sum += 1
        if (null_date - prev)!= pd.Timedelta(resolution, "min"):
            # if (nan_sum - 1) * mean_per_month[str(prev.month) + "-" + str(prev.year)]*0.5 <= ts.loc[prev + pd.Timedelta(resolution, "min")][0] and (nan_sum - 1 > 2) or ts.loc[prev + pd.Timedelta(resolution, "min")][0] == 0:
            #     a.loc[prev + pd.Timedelta(resolution, "min")] = ts.loc[prev + pd.Timedelta(resolution, "min")]
            if cut_off <= ts.loc[prev + pd.Timedelta(resolution, "min")][0] or ts.loc[prev + pd.Timedelta(resolution, "min")][0] == 0:
                a.loc[prev + pd.Timedelta(resolution, "min")] = ts.loc[prev + pd.Timedelta(resolution, "min")]
            nan_sum = 1
        prev = null_date
    try:
        if cut_off <= ts.loc[prev + pd.Timedelta(resolution, "min")][0] or ts.loc[prev + pd.Timedelta(resolution, "min")][0] == 0:
            a.loc[prev + pd.Timedelta(resolution, "min")] = ts.loc[prev + pd.Timedelta(resolution, "min")]
    except:
        pass

    if min_non_nan_interval != -1:
        #UC7 Do that for l_interpolation also
        #If after imputation there exist continuous intervals of non nan values in the train set that are smaller 
        #than min_non_nan_interval time steps, these intervals are all replaced by nan values
        not_nan_values = ts[(~ts["Value"].isnull())]
        not_nan_dates = not_nan_values.index
        prev = not_nan_dates[0]
        start = prev

        for not_nan_day in not_nan_dates[1:]:
            if (not_nan_day - prev)!= pd.Timedelta(resolution, "min"):
                if prev - start < pd.Timedelta(resolution * min_non_nan_interval, "min"):
                    for date in pd.date_range(start=start, end=prev, freq=str(resolution) + "min"):
                        a.loc[date] = ts.loc[date].values[0]

                start = not_nan_day
            prev = not_nan_day
        if prev - start < pd.Timedelta(resolution * min_non_nan_interval, "min"):
            for date in pd.date_range(start=start, end=prev, freq=str(resolution) + "min"):
                a.loc[date] = ts.loc[date].values[0]


    #Plotting Removed values and new series
    a = a.sort_values(by='Datetime')
    a = a[~a.index.duplicated(keep='first')]

    res = ts.drop(index=a.index)
    if print_removed: print(f"Removed from {name}", a)

    removed = a.copy()
    #maybe they are removed also
    #fix
    removed.loc[res.index[0]] = pd.NA
    removed.loc[res.index[-1]] = pd.NA
    removed = removed.sort_values(by='Datetime')
    removed = removed[~removed.index.duplicated(keep='first')]

    if not removed.empty:
        full_removed = removed.asfreq(str(resolution)+'min')
    else:
        full_removed = removed.copy()

    if not ts.empty:
        full_ts = ts.asfreq(str(resolution)+'min')
    else:
        full_ts = full_ts.copy()

    plot_removed(full_removed, full_ts, name, outlier_dir)

    return res.asfreq(str(resolution)+'min'), a

def impute(ts: pd.DataFrame,
           holidays,
           max_thr: int = -1,
           a: float = 0.3,
           wncutoff: float = 0.000694,
           ycutoff: int = 3,
           ydcutoff: int = 30,
           #TODO Add to params
           dcutoff: int = 6,
           resolution: str = "15",
           debug: bool = False,
           name: str = "PT",
           l_interpolation: bool = False,
           cut_date_val: str = "20221208",
           min_non_nan_interval: int = 24,
           impute_dir=""):
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
        then it is not imputed and returned with NaN values. If -1, every 
        value will be imputed regardless of how long the consecutive 
        subseries of NaNs it belongs to is
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
    l_interpolation
        Whether to only use linear interpolation 
    cut_date_val
        All dates before cut_date_val that have nan values are imputed using historical data
        from dates which are also before cut_date_val. Datetimes after cut_date_val are not affected
        by this
    min_non_nan_interval
        If after imputation there exist continuous intervals of non nan values that are smaller than min_non_nan_interval
        hours, these intervals are all replaced  by nan values

    Returns
    -------
    pandas.DataFrame
        The imputed dataframe
    """
    if max_thr == -1: max_thr = len(ts)
    if l_interpolation:
        imputed_values = ts[ts["Value"].isnull()]

        #null_dates: Series with all null dates to be imputed
        null_dates = imputed_values.index

        if debug:
            for date in null_dates:
                print(date)

        #isnull: An array which stores whether each value is null or not
        isnull = ts["Value"].isnull().values


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

        #using max_thr for linear interp. for UC7
        res = ts.interpolate(inplace=False)

        null_zip = [(i, null_date) for (i, null_date) in enumerate(null_dates) if leave_nan[i]]

        for i, null_date in tqdm(null_zip):
            res.loc[null_date] = np.NaN
        imputed_values = res[ts["Value"].isnull()].copy()

    else:
        #Returning calendar of the country ts belongs to
        calendar = create_calendar(ts, int(resolution), holidays, timezone("UTC"))
        calendar.index = calendar["datetime"]
        imputed_values = ts[ts["Value"].isnull()].copy()

        #null_dates: Series with all null dates to be imputed
        null_dates = imputed_values.index

        if debug:
            for date in null_dates:
                print(date)

        #isnull: An array which stores whether each value is null or not
        isnull = ts["Value"].isnull().values

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

        null_zip = [(i, null_date) for (i, null_date) in enumerate(null_dates) if not leave_nan[i]]

        for i, null_date in tqdm(null_zip):

            #WN: Day of the week + hour/24 + minute/(24*60). Holidays are handled as
            #either Saturdays(if the real day is a Friday) or Sundays(in every other case)
            currWN = calendar.loc[null_date]['WN']
            currYN = calendar.loc[null_date]['yearday']
            currY = calendar.loc[null_date]['year']
            currDN = calendar.loc[null_date]['DN']

            #weight of interpolated series, decreases as distance to nearest known value increases
            w = np.e ** (-a * d[i])

            #Historical value is calculated as the mean of values that have at most wncutoff distance to the current null value's
            #WN, ycutoff distance to its year, and ydcutoff distance to its yearday
            #All dates before cut_date_val that have nan values are imputed using historical data
            #from dates which are also before cut_date_val
            if null_date < pd.Timestamp(cut_date_val):
                historical = ts[(~isnull) & (((calendar['WN'] - currWN < wncutoff) & (calendar['WN'] - currWN > -wncutoff) &\
                                        (ts["Value"].index < pd.Timestamp(cut_date_val)) &\
                                        (calendar['year'] - currY < ycutoff) & (calendar['year'] - currY > -ycutoff) &\
                                        (((calendar['yearday'] - currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < ydcutoff) |\
                                        ((-calendar['yearday'] + currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < ydcutoff))) |\
                                        ((((calendar['yearday'] - currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < dcutoff) |\
                                        ((-calendar['yearday'] + currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < dcutoff)) &\
                                        ((calendar['DN'] - currDN < int(resolution)/120) & (- calendar['DN'] + currDN < int(resolution)/120))))]["Value"]

                # historical = ts[(~isnull) &\
                #                         (((calendar['yearday'] - currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < dcutoff) |\
                #                         ((-calendar['yearday'] + currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < dcutoff)) &\
                #                         (calendar['hour'] == currH)]["Value"]

                
            #Dates after cut_date_val are not affected by cut_date_val
            else:
                historical = ts[(~isnull) & (((calendar['WN'] - currWN < wncutoff) & (calendar['WN'] - currWN > -wncutoff) &\
                                        (calendar['year'] - currY < ycutoff) & (calendar['year'] - currY > -ycutoff) &\
                                        (((calendar['yearday'] - currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < ydcutoff) |\
                                        ((-calendar['yearday'] + currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < ydcutoff))) |\
                                        ((((calendar['yearday'] - currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < dcutoff) |\
                                        ((-calendar['yearday'] + currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < dcutoff)) &\
                                        ((calendar['DN'] - currDN < int(resolution)/120) & (- calendar['DN'] + currDN < int(resolution)/120))))]["Value"]

            if debug: print("~~~~~~Date~~~~~~~",null_date, "~~~~~~~Dates summed~~~~~~~~~~",historical,sep="\n")

            historical = historical.mean()

            #imputed value is calculated as a wheighted average of the histrorical value and the value from intrepolation
            res.loc[null_date] = w * ts_interpolatied.loc[null_date] + (1 - w) * historical

            imputed_values.loc[null_date] = res.loc[null_date]

            if debug:
                print(res.loc[null_date])
    non_nan_intervals_to_nan = {}
    if min_non_nan_interval != -1:
        #UC7 Do that for l_interpolation also
        #If after imputation there exist continuous intervals of non nan values in the train set that are smaller 
        #than min_non_nan_interval time steps, these intervals are all replaced by nan values
        not_nan_values = res[(~res["Value"].isnull())]
        not_nan_dates = not_nan_values.index
        prev = not_nan_dates[0]
        start = prev

        for not_nan_day in not_nan_dates[1:]:
            if (not_nan_day - prev)!= pd.Timedelta(int(resolution), "min"):
                if prev - start < pd.Timedelta(int(resolution) * min_non_nan_interval, "min"):
                    print(f"Non Nan interval from {start} to {prev} is smaller than {min_non_nan_interval} time steps. Making this also Nan")
                    for date in pd.date_range(start=start, end=prev, freq=resolution + "min"):
                        non_nan_intervals_to_nan[date] = res.loc[date].values[0]
                        res.loc[date] = pd.NA
                        imputed_values.loc[date] = pd.NA

                start = not_nan_day
            prev = not_nan_day
        if prev - start < pd.Timedelta(int(resolution) * min_non_nan_interval, "min"):
            for date in pd.date_range(start=start, end=prev, freq=resolution + "min"):
                non_nan_intervals_to_nan[date] = res.loc[date].values[0]
                res.loc[date] = pd.NA
                imputed_values.loc[date] = pd.NA
    imputed_values = imputed_values[(~imputed_values["Value"].isnull())]
    non_nan_intervals_to_nan = pd.DataFrame.from_dict(non_nan_intervals_to_nan, columns=["Value"], orient='index')
    non_nan_intervals_to_nan.index.name = "Datetime"
    if not res.empty:
        full_res = res.asfreq(str(resolution)+'min')
    else:
        full_res = res.copy()

    if not imputed_values.empty:
        full_imputed_values = imputed_values.asfreq(str(resolution)+'min')
    else:
        full_imputed_values = imputed_values.copy()
    
    if not non_nan_intervals_to_nan.empty:
        full_non_nan_intervals_to_nan = non_nan_intervals_to_nan.asfreq(str(resolution)+'min')
    else:
        full_non_nan_intervals_to_nan = non_nan_intervals_to_nan.copy()

    plot_imputation(full_res, full_imputed_values, full_non_nan_intervals_to_nan, name, impute_dir)
    return res, imputed_values


def utc_to_local(df, country_code):
    # Get dictionary of countries and their timezones
    timezone_countries = {country: timezone 
                            for country, timezones in country_timezones.items()
                            for timezone in timezones}

    print(timezone_countries)
    local_timezone = timezone_countries[country_code]

    print(f"\nUsing timezone {local_timezone}...")
    logging.info(f"\nUsing timezone {local_timezone}...")


    # convert dates to given timezone, get timezone info
    #print(df.index.to_series().tz_localize("UTC"))
    df['Local Datetime'] = df.index.to_series().dt.tz_localize("UTC").dt.tz_convert(local_timezone)

    # remove timezone information-naive, because next localize() recquires it 
    # but keep dates to local timezone
    df['Local Datetime'] = df['Local Datetime'].dt.tz_localize(None)

    # localize based on timezone, ignore daylight saving time, shift forward if ambiguous datetimes
    df['Local Datetime'] = df['Local Datetime'].dt.tz_localize(local_timezone,
                                                        ambiguous=np.array([False] * df.shape[0]),
                                                        nonexistent='shift_forward')

    df['Local Datetime'] = df['Local Datetime'].dt.tz_localize(None)


    #set index to local time
    df.set_index('Local Datetime', inplace=True)

    df.index.name = "Datetime"

    #print(df)

def save_consecutive_nans(ts, resolution, tmpdir, name):
    """
    Function that saves the time ranges that are left in the time series with 
    consecutive nans. A file is saved containing this information named 
    cons_nans_left_<name}>.txt. All time ranges are inclusive in both sides.

    Parameters
    ----------
    ts
        The pandas.DataFrame to be processed
    resolution
        The resolution of the time series
    tmpdir
        The folder where to save the file
    name
        The name of the time series that is being processed. Also included in the 
        name of the file to be saved.

    Returns
    -------
    str
        The contents of the saved file in string format.
    """
    resolution = int(resolution)
    output = "Consecutive nans left in df:\n"
    null_dates = ts[ts["Value"].isnull()].index
    prev = null_dates[0]
    output = output + str(prev) + " - "
    null_date = ""
    for null_date in null_dates[1:]:
        if (null_date - prev)!= pd.Timedelta(resolution, "min"):
            output = output + str(prev) + "\n" + str(null_date) + " - "
        prev = null_date
    output = output + str(null_date)
    
    f = open(f'{tmpdir}/cons_nans_left_{name}.txt', "w")
    f.write(output)
    f.close()
    return output

def sum_wo_nans(arraylike):
    #function used to ingore nans in
    #case of summation, and if other
    #samples exist 
    if np.isnan(arraylike).all():
        return np.nan
    else:
        return np.sum(arraylike)

def resample(series, new_resolution, method):
    """
    Undersamples a time series to the desired new resolution using the desired method.

    Parameters
    ----------
    series
        The pandas.DataFrame to be processed
    new_resolution
        The resolution of the resampled final time series
    method
        The method of the resampling. The possibilities include averaging, summation, 
        and downsampling. More specifically:
            - averaging: An new dataframe will be produced with datetimes that are
            separated by new_resolution minutes. The first datetime of the new dataframe
            will be the same as that of the old one. Also, all other datetimes (lets call
            them datetime) will be calculated using the average of the datetimes of the
            old dataset that belong to the date range [datetime, datetime + new_resolution).
            NaNs are ignored if non NaN values exist in [datetime, datetime + new_resolution), 
            otherwise datetime remains NaN in the new dataset.
            - summation: The same applies here as averaging. The differences are that all other 
            datetimes (lets call them datetime) will be calculated using the sum of the datetimes of the
            old dataset that belong to the date range [datetime, datetime + new_resolution).
            NaNs are considered 0 if non NaN values exist in [datetime, datetime + new_resolution), 
            otherwise datetime remains NaN in the new dataset.
            - downsampling: The same applies here as averaging. The differences are that all other 
            datetimes (lets call them datetime) will be the first available sample of the
            old dataset that belongs to the date range [datetime, datetime + new_resolution). So,
            if datetime existed in the old dataset, it will have the same value in the new one.

    Returns
    -------
    pandas.DataFrame
        The resampled dataframe
    """
    if method == "averaging":
        return pd.DataFrame(series["Value"].resample(new_resolution+'min').mean(), columns=["Value"])
    elif method == "summation":
        return pd.DataFrame(series["Value"].resample(new_resolution+'min').apply(sum_wo_nans), columns=["Value"])
    else:
        return pd.DataFrame(series["Value"].resample(new_resolution+'min').first(), columns=["Value"])

def preprocess_covariates(ts_list, id_list, cov_id, infered_resolution, resolution, type, multiple, year_min, year_max, resampling_agg_method):
    """
    Basic preprocessing of a covariate time series. This function is called for one series of
    the past or future covariates at a time. The steps are repeated for all componets of this
    series. More methods / details may be added according to the particular use case / covariate 
    being processed.

    Parameters
    ----------
    ts_list
        A list of dataframes each of which is a component of the timeseries being processed
    id_list
        The list of ids of the components being processed
    cov_id
        The id of the covariate time series being processed
    infered_resolution
        The current resolution of the covariate time series
    resolution
        The intended resolution of the covariate time series (same as the intended resolution 
        of the main time series)
    type
        Wether it's past or future covariates
    multiple
        Wether the main time series comes from a multiple file format or not
    year_min
        The first year that is kept from the main time series, and also from the covariates   
    year_max
        The last year that is kept from the main time series, and also from the covariates
    resampling_agg_method
        Method to use for resampling. Choice between averaging, summation and downsampling
    Returns
    -------
    List[pandas.DataFrame]
        A list containing all preprocessed components of the chosen covariate time series
    """
    #One can add more preprocessing here according to the exact caracteristics of the covariates 
    result = []
    for ts, ts_id in zip(ts_list, id_list):
        
        #Temporal fitting
        ts_res = ts[ts.index >= pd.Timestamp(str(year_min) + '0101 00:00:00')]
        ts_res = ts_res[ts_res.index <= pd.Timestamp(str(year_max) + '1231 23:59:59')]

        #Linear interpolation
        ts_res = ts_res.interpolate(inplace=False)

        #resampling
        if int(resolution) < int(infered_resolution):
            raise NoUpsamplingException()
                    
        if int(resolution) != int(infered_resolution):
            print(f"\nResampling {ts_id} component of {cov_id} timeseries of {type} covariates as given frequency different than infered resolution")
            logging.info(f"\nResampling {ts_id} component of {cov_id} timeseries of {type} covariates as given frequency different than infered resolution")
            ts_res = resample(ts_res, resolution, resampling_agg_method)
        
        result.append(ts_res)
    return result

@click.command(
    help="Given a timeseries CSV file (see load_raw_data), resamples it, "
         "drops duplicates converts it to darts and optionally creates "
         "the time covariates. The new timeseries is logged as artifact "
         "to MLflow"
)
@click.option("--series-csv",
    type=str,
    default="None",
    help="Local timeseries csv. It gets overwritten by uri if given."
)
@click.option("--series-uri",
    type=str,
    default="None",
    help="Remote timeseries csv file.  If set, it overwrites the local value."
)
@click.option('--year-range',
    default="None",
    type=str,
    help='The year range to include in the dataset.'
)
@click.option("--resolution",
    default="None",
    type=str,
    help="The resolution of the dataset in minutes."
)
@click.option("--time-covs",
    default="false",
    type=str,
    help="Whether to add time covariates to the timeseries."
)
@click.option("--day-first",
    type=str,
    default="true",
    help="Whether the date has the day before the month")

@click.option("--country",
    type=str,
    default="PT",
    help="The country code this dataset belongs to")

@click.option("--std-dev",
    type=str,
    default="4.5",
    help="The number to be multiplied with the standard deviation of \
          each 1 month  period of the dataframe. The result is then used as \
          a cut-off value as described above")

@click.option("--max-thr",
              type=str,
              default="-1",
              help="If there is a consecutive subseries of NaNs longer than max_thr, \
                    then it is not imputed and returned with NaN values. If -1, every value will\
                    be imputed regardless of how long the consecutive subseries of NaNs it belongs \
                    to is")
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

@click.option("--multiple",
    type=str,
    default="false",
    help="Whether to train on multiple timeseries")
     
@click.option("--l-interpolation",
    type=str,
    default="false",
    help="Whether to only use linear interpolation")

@click.option("--rmv-outliers",
    type=str,
    default="true",
    help="Whether to remove outliers")

@click.option("--convert-to-local-tz",
    type=str,
    default="true",
    help="Whether to convert time")


@click.option("--ts-used-id",
    type=str,
    default="None",
    help="If not None, only ts with this id will be used for training and evaluation. Applicable only on multiple ts files")

@click.option("--infered-resolution-series",
    type=str,
    default="15",
    help="infered resolution of the dataset from load_raw_data")

@click.option("--min-non-nan-interval",
    type=str,
    default="24",
    help="If after imputation there exist continuous intervals of non nan values that are smaller than min_non_nan_interval \
        time steps, these intervals are all replaced by nan values")

@click.option("--cut-date-val",
              type=str,
              default='None',
              help="Validation set start date [str: 'YYYYMMDD'] \
                  All dates before cut_date_val that have nan values are imputed using historical data \
                  from dates which are also before cut_date_val. Datetimes after cut_date_val are not affected \
                  by this")

@click.option("--infered-resolution-past",
    type=str,
    default="15",
    help="infered resolution of the past covariates from load_raw_data")

@click.option("--past-covs-csv",
    type=str,
    default="None",
    help="Local past covaraites csv file"
    )
@click.option("--past-covs-uri",
    default="None",
    help="Remote past covariates csv file. If set, it overwrites the local value."
    )
@click.option("--infered-resolution-future",
    type=str,
    default="15",
    help="infered resolution of the future covariates from load_raw_data")

@click.option("--future-covs-csv",
    type=str,
    default="None",
    help="Local future covaraites csv file"
    )
@click.option("--future-covs-uri",
    default="None",
    help="Remote future covariates csv file. If set, it overwrites the local value."
    )

@click.option("--resampling-agg-method",
    default="averaging",
    type=click.Choice(['averaging',
                       'summation',
                       'downsampling']),
    multiple=False,
    help="Method to use for resampling."
    )

def etl(series_csv, series_uri, year_range, resolution, time_covs, day_first, 
        country, std_dev, max_thr, a, wncutoff, ycutoff, ydcutoff, multiple, 
        l_interpolation, rmv_outliers, convert_to_local_tz, ts_used_id,
        infered_resolution_series, min_non_nan_interval, cut_date_val,
        infered_resolution_past, past_covs_csv, past_covs_uri, infered_resolution_future,
        future_covs_csv, future_covs_uri, resampling_agg_method):

    # TODO: play with get_time_covariates and create sinusoidal
    # transformations for all features (e.g dayofyear)
    # Also check if current transformations are ok

    disable_warnings(InsecureRequestWarning)


    # If series_uri is given, series_csv will be downloaded from there
    if none_checker(series_uri) != None:
        download_file_path = download_online_file(series_uri, dst_filename="load.csv")
        series_csv = download_file_path

    past_covs_csv = none_checker(past_covs_csv)
    past_covs_uri = none_checker(past_covs_uri)
    future_covs_csv = none_checker(future_covs_csv)
    future_covs_uri = none_checker(future_covs_uri)

    # If uri is given, covariates will be downloaded from there
    if past_covs_uri != None:
        download_file_path = download_online_file(past_covs_uri, dst_filename="past_covs.csv")
        past_covs_csv = download_file_path

    if future_covs_uri != None:
        download_file_path = download_online_file(future_covs_uri, dst_filename="future_covs.csv")
        future_covs_csv = download_file_path

    ## Process parameters from click and MLProject
    series_csv = series_csv.replace("'", "")
    std_dev = float(std_dev)
    max_thr = int(max_thr)
    a = float(a)
    wncutoff = float(wncutoff)
    ycutoff = int(ycutoff)
    ydcutoff = int(ydcutoff)
    min_non_nan_interval = int(min_non_nan_interval)
    l_interpolation = truth_checker(l_interpolation)
    rmv_outliers = truth_checker(rmv_outliers)
    convert_to_local_tz = truth_checker(convert_to_local_tz)
    time_covs = truth_checker(time_covs)
    day_first = truth_checker(day_first)
    multiple = truth_checker(multiple)

    ts_used_id = none_checker(ts_used_id)

    print("\nLoading source dataset..")
    logging.info("\nLoading source dataset..")

    #Read past / futute covariates
    if past_covs_csv != None:
        ts_list_past_covs, id_l_past_covs, ts_id_l_past_covs = \
                multiple_ts_file_to_dfs(past_covs_csv, day_first, infered_resolution_past)
    else:
        ts_list_past_covs, id_l_past_covs, ts_id_l_past_covs = [], [], []
    if future_covs_csv != None:
        ts_list_future_covs, id_l_future_covs, ts_id_l_future_covs = \
                multiple_ts_file_to_dfs(future_covs_csv, day_first, infered_resolution_future)
    else:
        ts_list_future_covs, id_l_future_covs, ts_id_l_future_covs = [], [], []

    if multiple:
        ts_list, id_l, ts_id_l = multiple_ts_file_to_dfs(series_csv, day_first, infered_resolution_series)
        # selecting only ts_used_id from multiple ts if the user wants to
        if ts_used_id != None:
            try:
                ts_used_id = int(ts_used_id)
            except:
                pass
            try:
                index = ts_id_l.index([ts_used_id for _ in range(len(ts_id_l[0]))])
            except:
                raise TsUsedIdDoesNotExcist()
            ts_list = [ts_list[index]]
            id_l = [id_l[index]]
            ts_id_l = [ts_id_l[index]]
            if past_covs_csv != None:
                ts_list_past_covs = [ts_list_past_covs[index]]
                id_l_past_covs = [id_l_past_covs[index]]
                ts_id_l_past_covs = [ts_id_l_past_covs[index]]
            if future_covs_csv != None:
                ts_list_future_covs = [ts_list_future_covs[index]]
                id_l_future_covs = [id_l_future_covs[index]]
                ts_id_l_future_covs = [ts_id_l_future_covs[index]]


    else:
        ts_list = [[pd.read_csv(series_csv,
                         delimiter=',',
                         header=0,
                         index_col=0,
                         parse_dates=True,
                         dayfirst=day_first)]]
        id_l, ts_id_l = [["Timeseries"]], [["Timeseries"]]

    # Year range handling
    if none_checker(year_range) is None:
        year_range = f"{ts_list[0][0].index[0].year}-{ts_list[0][0].index[-1].year}"
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
    outlier_dir = tempfile.mkdtemp()
    impute_dir = tempfile.mkdtemp()

    #TODO Chenck if series number == covariate series number

    with mlflow.start_run(run_name='etl', nested=True) as mlrun:
        #these are the final dataframe lists to be returned from etl
        res_ = []
        res_past = []
        res_future = []
        for ts_num, ts in enumerate(ts_list):
            res_.append([])
            for comp_num, comp in enumerate(ts):
                #All preprocessing is done on each component separately 
                print(f"\n---> Starting etl of ts {ts_num+1} / {len(ts_list)}, component {comp_num+1} / {len(ts)}, id {id_l[ts_num][comp_num]}...")
                logging.info(f"\n---> Starting etl of ts {ts_num+1} / {len(ts_list)}, component {comp_num+1} / {len(ts)}, id {id_l[ts_num][comp_num]}...")
                if convert_to_local_tz:
                    print(f"\nConverting to local Timezone...")
                    logging.info(f"\nConverting to local Timezone...")
                    try:
                        utc_to_local(comp, id_l[ts_num][comp_num])
                    except:
                        try:
                            print(f"\ID not a country code, trying country argument...")
                            logging.info(f"\ID not a country code, trying country argument...")
                            utc_to_local(comp, country)
                        except:
                            print(f"\nError occured, keeping time provided by the file...")
                            logging.info(f"\nError occured, keeping time provided by the file...")

                # temporal filtering
                print(f"\nTemporal filtering...")
                logging.info(f"\nTemporal filtering...")
                comp = comp[comp.index >= pd.Timestamp(str(year_min) + '0101 00:00:00')]
                comp = comp[comp.index <= pd.Timestamp(str(year_max) + '1231 23:59:59')]

                comp_res = comp
                # drop duplicate index entries, keeping the first
                print("\nDropping duplicate time index entries, keeping first one...")
                logging.info("\nDropping duplicate time index entries, keeping first one...")

                comp_res = comp_res[~comp_res.index.duplicated(keep='first')]

                if rmv_outliers:
                    print("\nPerfrorming Outlier Detection...")
                    logging.info("\nPerfrorming Outlier Detection...")
                    comp_res, removed = remove_outliers(ts=comp_res,
                                                        name=id_l[ts_num][comp_num],
                                                        resolution=infered_resolution_series,
                                                        std_dev=std_dev,
                                                        outlier_dir=outlier_dir)

                #holidays_: The holidays of country
                if l_interpolation:
                    country_holidays = None
                else:
                    try:
                        code = compile(f"holidays.{id_l[ts_num][comp_num]}()", "<string>", "eval")
                        country_holidays = eval(code)
                    except:
                        country_holidays = []
                        print("\nID column does not specify valid country. Using country argument instead")
                        logging.info("\nID column does not specify valid country. Using country argument instead")
                        try:
                            code = compile(f"holidays.{country}()", "<string>", "eval")
                            country_holidays = eval(code)
                        except:
                            raise CountryDoesNotExist()
                print("\nPerfrorming Imputation of the Dataset...")
                logging.info("\nPerfrorming Imputation of the Dataset...")
                #print(comp_res)
                comp_res, imputed_values = impute(ts=comp_res,
                                                  holidays=country_holidays,
                                                  max_thr=max_thr,
                                                  a=a,
                                                  wncutoff=wncutoff,
                                                  ycutoff=ycutoff,
                                                  ydcutoff=ydcutoff,
                                                  resolution=infered_resolution_series,
                                                  name=id_l[ts_num][comp_num],
                                                  l_interpolation=l_interpolation,
                                                  cut_date_val=cut_date_val,
                                                  min_non_nan_interval=min_non_nan_interval,
                                                  impute_dir=impute_dir)
                
                if int(resolution) < int(infered_resolution_series):
                    raise NoUpsamplingException()
                if int(resolution) != int(infered_resolution_series):
                    print(f"\nResampling as given frequency different than infered resolution")
                    logging.info(f"\nResampling as given frequency is different than infered resolution")
                    comp_res = resample(comp_res, resolution, resampling_agg_method)
                
                #if there are still nans in the component, we save their date ranges in a file
                if comp_res.isnull().sum().sum() > 0:
                    save_consecutive_nans(comp_res, resolution, impute_dir, id_l[ts_num][comp_num])
                    
                print("\nCreating darts data frame...")
                logging.info("\nCreating darts data frame...")

                # explicitly redefine frequency
                comp_res = comp_res.asfreq(resolution+'min')

                if 'W6 positive_active' in id_l[ts_num][comp_num]:
                    comp_res = -comp_res


                # ts_res.to_csv(f'{tmpdir}/4_asfreq.csv')

                # darts dataset creation
                comp_res_darts = darts.TimeSeries.from_dataframe(comp_res)

                # if id_l[ts_num][comp_num] in ["W6 positive_active"]:
                #    comp_res = -comp_res
                #print("NULL VALUES", comp_res_darts.pd_dataframe().isnull().sum().sum())

                # ts_res_darts.to_csv(f'{tmpdir}/4_read_as_darts.csv')

                #comp_res_darts.to_csv("/new_vol_300/opt/energy-forecasting-theo/model_registry/lgbm_uc6_w6_pos_ac_serving/sample.csv")
                # ts_res_darts.to_csv(f'{tmpdir}/5_filled_na.csv')

                # time variables creation

                if comp_num == len(ts) - 1:
                    #We add covariates when the preprocessing of the final component of the series has finished
                    if past_covs_csv != None:
                        res_past.append(preprocess_covariates(ts_list_past_covs[ts_num], id_l_past_covs[ts_num], ts_id_l_past_covs[ts_num][0], infered_resolution_past, resolution, "past", multiple, year_min, year_max, resampling_agg_method))
                    if future_covs_csv != None:
                        res_future.append(preprocess_covariates(ts_list_future_covs[ts_num], id_l_future_covs[ts_num], ts_id_l_future_covs[ts_num][0], infered_resolution_future, resolution, "future", multiple, year_min, year_max, resampling_agg_method))
                    
                    if time_covs:
                        print("\nCreating time covariates dataset...")
                        logging.info("\nCreating time covariates dataset...")
                        try:
                            ts_list_covariates, id_l_covariates, ts_id_l_covariates = get_time_covariates(comp_res_darts, id_l[ts_num][comp_num], ts_id_l[ts_num][0])
                        except:
                            print("\nID column does not specify valid country. Using country argument instead")
                            logging.info("\nID column does not specify valid country. Using country argument instead")
                            try:
                                ts_list_covariates, id_l_covariates, ts_id_l_covariates = get_time_covariates(comp_res_darts, country,  ts_id_l[ts_num][0])
                            except:
                                raise CountryDoesNotExist()

                        #Adding time covariates to the rest of the future covariates
                        if future_covs_csv == None:
                            res_future.append(ts_list_covariates) 
                            id_l_future_covs.append(id_l_covariates)
                            ts_id_l_future_covs.append(ts_id_l_covariates)
                            #TODO check multivariate and multiple
                        else:
                            res_future[-1].extend(ts_list_covariates) 
                            id_l_future_covs[-1].extend(id_l_covariates)
                            ts_id_l_future_covs[-1].extend(ts_id_l_covariates)
                    
                    else:
                        print("\nSkipping the creation of time covariates")
                        logging.info("\nSkipping the creation of time covariates")
                        ts_list_covariates, id_l_covariates, ts_id_l_covariates = None, None, None
                if rmv_outliers:
                    print("\nStoring removed values from outlier detection as csv locally...")
                    logging.info("\nStoring removed values from outlier detection as csv...")
                    removed.to_csv(f'{outlier_dir}/removed_{id_l[ts_num][comp_num]}.csv')

                print("\nStoring imputed dates and their values as csv locally...")
                logging.info("\nStoring imputed dates and their values as csv locally...")
                imputed_values.to_csv(f'{impute_dir}/imputed_values_{id_l[ts_num][comp_num]}.csv')


                res_[-1].append(comp_res)
        if multiple:
            res_ = cut_extra_samples(res_)
        
        # res_, res_future, id_l_future_covs, ts_id_l_future_covs = add_weather_covariates(res_,
        #                                                                                  res_future, 
        #                                                                                  id_l_future_covs,
        #                                                                                  ts_id_l_future_covs,
        #                                                                                  ts_id_l,
        #                                                                                  fields=["shortwave_radiation","direct_radiation","diffuse_radiation"])
        #                                                                                  #fields=["temperature","shortwave_radiation","direct_radiation","diffuse_radiation"])
        print("\nCreating local folder to store the datasets as csv...")
        logging.info("\nCreating local folder to store the datasets as csv...")
        if not multiple:
            # store locally as csv in folder
            comp_res_darts.to_csv(f'{tmpdir}/series.csv')

            print("\nStoring datasets locally...")
            logging.info("\nStoring datasets...")
        else:
            multiple_dfs_to_ts_file(res_, id_l, ts_id_l, f'{tmpdir}/series.csv')
        if res_past != []:
            multiple_dfs_to_ts_file(res_past, id_l_past_covs, ts_id_l_past_covs, f'{tmpdir}/past_covs.csv')
        if res_future != []:
            print(res_future)
            multiple_dfs_to_ts_file(res_future, id_l_future_covs, ts_id_l_future_covs, f'{tmpdir}/future_covs.csv')
        print("\nUploading features to MLflow server...")
        logging.info("\nUploading features to MLflow server...")
        mlflow.log_artifacts(tmpdir, "features")
        mlflow.log_artifacts(outlier_dir, "outlier_detection_results")
        mlflow.log_artifacts(impute_dir, "imputation_results")

        print("\nArtifacts uploaded. Deleting local copies...")
        logging.info("\nArtifacts uploaded. Deleting local copies...")
        shutil.rmtree(tmpdir)
        shutil.rmtree(outlier_dir)
        shutil.rmtree(impute_dir)

        print("\nETL succesful.")
        logging.info("\nETL succesful.")

        # mlflow tags
        #TODO tags for outliers and imputation?
        mlflow.set_tag("run_id", mlrun.info.run_id)
        mlflow.set_tag("stage", "etl")
        mlflow.set_tag('series_uri', f'{mlrun.info.artifact_uri}/features/series.csv')
        if res_past != []:
            mlflow.set_tag('past_covs_uri', f'{mlrun.info.artifact_uri}/features/past_covs.csv')
        else: # default naming for non available time covariates uri
            mlflow.set_tag('past_covs_uri', 'None')
        if res_future != []:
            mlflow.set_tag('future_covs_uri', f'{mlrun.info.artifact_uri}/features/future_covs.csv')
        else: # default naming for non available time covariates uri
            mlflow.set_tag('future_covs_uri', 'None')
        return


if __name__ == "__main__":
    print("\n=========== ETL =============")
    logging.info("\n=========== ETL =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
    logging.info("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
    etl()
