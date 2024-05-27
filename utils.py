
from dotenv import load_dotenv
import tempfile
import pretty_errors
import os
import mlflow
import pandas as pd
import yaml
import darts
from pandas.tseries.frequencies import to_offset
from math import ceil
cur_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np
load_dotenv()
from tqdm import tqdm
import logging
from exceptions import MandatoryArgNotSet, NotValidConfig, EmptySeries, DifferentFrequenciesMultipleTS
import json
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)
import requests
from datetime import date
import pvlib
import pandas as pd
import matplotlib.pyplot as plt
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pi
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.timeseries_generation import holidays_timeseries
import math
from datetime import timezone
from darts.dataprocessing.transformers import MissingValuesFiller
import tempfile
import holidays
from pytz import timezone
import pytz
from datetime import datetime

class ConfigParser:
    def __init__(self, config_file=f'{cur_dir}/config.yml', config_string=None):
        import yaml
        try:
            with open(config_file, "r") as ymlfile:
                self.config = yaml.safe_load(ymlfile)
                if config_string != None:
                    assert config_string in self.config['hyperparameters']
        except:
            try:
                assert "{" in config_string
                self.config = yaml.safe_load(config_string)
            except:
                raise NotValidConfig()

    def read_hyperparameters(self, hyperparams_entrypoint=""):
            try:
                return self.config['hyperparameters'][hyperparams_entrypoint]
            except:
                return self.config

    def read_entrypoints(self):
        return self.config['hyperparameters']


def download_file_from_s3_bucket(object_name, dst_filename, dst_dir=None, bucketName='mlflow-bucket'):
    import boto3
    import tempfile
    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    s3_resource = boto3.resource(
        's3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    bucket = s3_resource.Bucket(bucketName)
    local_path = os.path.join(dst_dir, dst_filename)
    bucket.download_file(object_name, local_path)
    return local_path

def get_pv_forecast(ts_id, start=None, end=None, inference=False, kW=185, use_saved=False, format="long"):
    # start = ts[0].index[0] - pd.Timedelta("1d"),    
    # end = ts[0].index[-1] + pd.Timedelta("1d"),
    #TODO Add parameters to mlflow
    #TODO Add visualization
    if use_saved:
        covs_list, cov_id_l, cov_ts_id_l = multiple_ts_file_to_dfs("/new_vol_300/opt/energy-forecasting-theo/uc7-data-ops/pvlib/UC6_covs_final.csv", True, "60", format=format)
        covs_list = covs_list[0]
        covs_list_final = [covs_list]
        #[['diffuse_radiation_W4 positive_active', 'direct_normal_irradiance_W4 positive_active', 'shortwave_radiation_W4 positive_active', 'temperature_W4 positive_active', 'windspeed_10m_W4 positive_active']]
        #temp_air, wind_speed, ghi, dhi, dni

        covs_list_final = []
        covs_list_final.append(covs_list[3])
        covs_list_final.append(covs_list[4])
        covs_list_final.append(covs_list[2])
        covs_list_final.append(covs_list[0])
        covs_list_final.append(covs_list[1])
        covs_list_final = [covs_list_final]
        print(f"Using covs of {cov_id_l[0][0]}")
        return darts.TimeSeries.from_dataframe(pvlib_forecast(covs_weather=covs_list_final[0], start=start, end=end, kW=kW))
    else:
        covs_list, cov_id_l, cov_ts_id_l = add_weather_covariates(start-pd.Timedelta("1d"),
                                                              end+pd.Timedelta("1d"),
                                                              [], 
                                                              [],
                                                              [], 
                                                              [ts_id],
                                                              fields=["temperature","windspeed_10m","shortwave_radiation","diffuse_radiation","direct_normal_irradiance"],
                                                              inference=inference)
        

        covs_list_final = []
        for ts, id_t in zip(covs_list, cov_id_l):
            new = []
            for comp, id in zip(ts, id_t):
                temp = impute(comp,
                        holidays.IT(),
                        max_thr = 2000,
                        a = 0.3,
                        wncutoff = 0.000694,
                        ycutoff = 3,
                        ydcutoff = 30,
                        resolution = "60min",
                        debug = False,
                        name = id,
                        l_interpolation = False,
                        cut_date_val = "20231022",
                        min_non_nan_interval = -1)
                new.append(temp)
                covs_list_final.append(new)
        print(f"Using covs of {ts_id[0]}")
        return darts.TimeSeries.from_dataframe(pvlib_forecast(covs_weather=covs_list_final[0], start=start, end=end, kW=kW))

def pvlib_forecast(covs_weather=[], start=None, end=None, kW=185):
    #init params
    latitude=42.567055
    longitude=12.607027
    surface_tilt=0 
    surface_azimuth=180
    modules_per_string=25
    strings_per_inverter=215
    altitude=0
    location=Location(latitude, longitude, altitude=altitude)

    #make weather
    weather = covs_weather[3].copy()
    weather.columns = ["dhi"]
    weather["dni"] = covs_weather[4]
    weather["ghi"] = covs_weather[2]
    weather["temp_air"] = covs_weather[0]
    weather["wind_speed"] = covs_weather[1]

    if start:
        weather = weather.loc[(weather.index >= start)]

    if end:
        weather = weather.loc[(weather.index <= end)] 

    #initialize system
    module_name = 'Canadian_Solar_CS5P_220M___2009_'
    inverter_name = 'ABB__ULTRA_1100_TL_OUTD_2_US_690_x_y_z__690V_' #'Power_Electronics__FS3000CU15__690V_' #'ABB__PVS980_58_2000kVA_K__660V_' #'ABB__ULTRA_1100_TL_OUTD_2_US_690_x_y_z__690V_'

    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod') 
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    module = sandia_modules[module_name]
    inverter = sapm_inverters[inverter_name]
    temperature_model_parameters = {'a': -2.4, 'b': -0.0455, 'deltaT': 8}

    system=PVSystem(surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
                module_parameters=module, inverter_parameters=inverter,
                temperature_model_parameters=temperature_model_parameters,
                modules_per_string=modules_per_string, strings_per_inverter=strings_per_inverter
                )
    

    modelchain=ModelChain(system, location)

    modelchain.run_model(weather)
    solar_data=modelchain.results.ac
    solar_data=pd.DataFrame(solar_data, columns=(['Value']))

    # Convert to our kW
    solar_data = solar_data * kW / 1000000.0

    return solar_data

def load_yaml_as_dict(filepath):
    import yaml
    with open(filepath, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            return parsed_yaml
        except yaml.YAMLError as exc:
            print(exc)
            return

def save_dict_as_yaml(filepath, data):
    with open(filepath, 'w') as savefile:
        yaml.dump(data, savefile, default_flow_style=False)

def load_local_pkl_as_object(local_path):
    import pickle
    pkl_object = pickle.load(open(local_path, "rb"))
    return pkl_object


def download_online_file(client, url, dst_filename=None, dst_dir=None):
    import sys
    import tempfile
    import requests
    print("Donwloading_online_file")
    print(url)
    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    if dst_filename is None:
        dst_filename = url.split('/')[-1]
    filepath = os.path.join(dst_dir, dst_filename)
    url = url.split('mlflow-bucket')[-1]
    client.fget_object("mlflow-bucket", url, filepath)
    # print(req)
    # if req.status_code != 200:
    #     raise Exception(f"\nResponse is not 200\nProblem downloading: {url}")
    #     sys.exit()
    # url_content = req.content
    # filepath = os.path.join(dst_dir, dst_filename)
    # file = open(filepath, 'wb')
    # file.write(url_content)
    # file.close()
    return filepath


def download_mlflow_file(client, url, dst_dir=None):
    S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    if url.startswith('s3://mlflow-bucket/'):
        url = url.replace("s3:/", S3_ENDPOINT_URL)
        local_path = download_online_file(
            client, url, dst_dir=dst_dir)
    elif url.startswith('runs:/'):
        mlflow_client = mlflow.tracking.MlflowClient()
        run_id = url.split('/')[1]
        mlflow_path = '/'.join(url.split('/')[3:])
        local_path = mlflow_client.download_artifacts(run_id, mlflow_path, dst_dir)
    elif url.startswith('http://'):
        local_path = download_online_file(
            client, url, dst_dir=dst_dir)
    return local_path


def load_pkl_model_from_server(client, model_uri):
    print("\nLoading remote PKL model...")
    model_path = download_mlflow_file(client, f'{model_uri}/_model.pkl')
    print(model_path)
    best_model = load_local_pkl_as_object(model_path)
    return best_model


def load_local_pl_model(model_root_dir):

    from darts.models.forecasting.gradient_boosted_model import LightGBMModel
    from darts.models.forecasting.random_forest import RandomForest
    from darts.models import RNNModel, BlockRNNModel, NBEATSModel, TFTModel, NaiveDrift, NaiveSeasonal, TCNModel, NHiTSModel, TransformerModel
    print("\nLoading local PL model...")
    model_info_dict = load_yaml_as_dict(
        os.path.join(model_root_dir, 'model_info.yml'))

    darts_forecasting_model = model_info_dict[
        "darts_forecasting_model"]

    model = eval(darts_forecasting_model)

    # model_root_dir = model_root_dir.replace('/', os.path.sep)

    print(f"Loading model from local directory:{model_root_dir}")

    best_model = model.load_from_checkpoint(model_root_dir, best=True)

    return best_model


def load_pl_model_from_server(model_root_dir):

    import tempfile

    print("\nLoading remote PL model...")
    mlflow_client = mlflow.tracking.MlflowClient()
    print(model_root_dir)
    model_run_id = model_root_dir.split("/")[5]
    mlflow_relative_model_root_dir = model_root_dir.split("/artifacts/")[1]

    local_dir = tempfile.mkdtemp()
    mlflow_client.download_artifacts(
        run_id=model_run_id, path=mlflow_relative_model_root_dir, dst_path=local_dir)
    best_model = load_local_pl_model(os.path.join(
        local_dir, mlflow_relative_model_root_dir))
    return best_model


def load_model(client, model_root_dir, mode="remote"):

    # Get model type as tag of model's run
    import mlflow
    print(model_root_dir)

    if mode == 'remote':
        mlflow_client = mlflow.tracking.MlflowClient()
        run_id = model_root_dir.split('/')[-1]
        model_run = mlflow_client.get_run(run_id)
        model_type = model_run.data.tags.get('model_type')
    else:
        if "_model.pth.tar" in os.listdir(model_root_dir):
            model_type = 'pl'
        else:
            model_type = "pkl"

    # Load accordingly
    if mode == "remote" and model_type == "pl":
        model = load_pl_model_from_server(model_root_dir=model_root_dir)
    #TODO Check if working with pl models
    elif mode == "remote" and model_type == "pkl":
        model = load_pkl_model_from_server(client, model_root_dir)
    elif mode == "local" and model_type == 'pl':
        model = load_local_pl_model(model_root_dir=model_root_dir)
    else:
        print("\nLoading local PKL model...")
        # pkl loads the exact model file so:
        model_uri = os.path.join(model_root_dir, '_model.pkl')
        model = load_local_pkl_as_object(model_uri)
    return model


def load_scaler(scaler_uri=None, mode="remote"):

    import tempfile

    if scaler_uri is None:
        print("\nNo scaler loaded.")
        return None

    if mode == "remote":
        run_id = scaler_uri.split("/")[-2]
        mlflow_filepath = scaler_uri.split("/artifacts/")[1]

        mlflow_client = mlflow.tracking.MlflowClient()
        local_dir = tempfile.mkdtemp()
        print("Scaler: ", scaler_uri)
        print("Run: ", run_id)
        mlflow_client.download_artifacts(
            run_id=run_id,
            path=mlflow_filepath,
            dst_path=local_dir
        )
        # scaler_path = download_online_file(
        #     scaler_uri, "scaler.pkl") if mode == 'remote' else scaler_uri
        scaler = load_local_pkl_as_object(
            os.path.join(local_dir, mlflow_filepath))
    else:
        scaler = load_local_pkl_as_object(scaler_uri)
    return scaler

def load_ts_id(load_ts_id_uri=None, mode="remote"):

    import tempfile

    if load_ts_id_uri is None:
        print("\nNo ts id list loaded.")
        return None

    if mode == "remote":
        run_id = load_ts_id_uri.split("/")[-2]
        mlflow_filepath = load_ts_id_uri.split("/artifacts/")[1]

        mlflow_client = mlflow.tracking.MlflowClient()
        local_dir = tempfile.mkdtemp()
        print("ts id list: ", load_ts_id_uri)
        print("Run: ", run_id)
        mlflow_client.download_artifacts(
            run_id=run_id,
            path=mlflow_filepath,
            dst_path=local_dir
        )
        ts_id_l = load_local_pkl_as_object(
            os.path.join(local_dir, mlflow_filepath))
    else:
        ts_id_l = load_local_pkl_as_object(load_ts_id_uri)
    return ts_id_l


def truth_checker(argument):
    """ Returns True if string has specific truth values else False"""
    return argument.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'on']


def none_checker(argument):
    """ Returns True if string has specific truth values else False"""
    if argument is None:
        return None
    if argument.lower() in ['none', 'nope', 'nan', 'na', 'null', 'nope', 'n/a', 'mlflow_artifact_uri']:
        return None
    else:
        return argument


def load_artifacts(run_id, src_path, dst_path=None):
    import tempfile
    import os
    import mlflow
    if dst_path is None:
        dst_dir = tempfile.mkdtemp()
    else:
        dst_dir = os.path.sep.join(dst_path.split("/")[-1])
        os.makedirs(dst_dir, exist_ok=True)
    fname = src_path.split("/")[-1]
    return mlflow.artifacts.download_artifacts(artifact_path=src_path, dst_path="/".join([dst_dir, fname]), run_id=run_id)


def load_local_model_as_torch(local_path):
    import torch
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(local_path, map_location=device)
    model.device = device
    return model

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
    calendar['dayofyear'] = calendar['datetime'].apply(lambda x: x.day_of_year)
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
    calendar['DN'] = calendar['hour'] + calendar['minute']/(60)
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

def impute(ts: pd.DataFrame,
           holidays,
           max_thr: int = -1,
           a: float = 0.3,
           wncutoff: float = 0.000694,
           ycutoff: int = 3,
           ydcutoff: int = 30,
           resolution: str = "15min",
           debug: bool = False,
           name: str = "PT",
           l_interpolation: bool = False,
           cut_date_val: str = "20221208",
           min_non_nan_interval: int = 24):
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
        imputed_values = ts[ts[ts.columns[0]].isnull()]

        #null_dates: Series with all null dates to be imputed
        null_dates = imputed_values.index

        if debug:
            for date in null_dates:
                print(date)

        #isnull: An array which stores whether each value is null or not
        isnull = ts[ts.columns[0]].isnull().values


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
                if null_dates[i+1] == null_dates[i] + pd.offsets.DateOffset(seconds=to_seconds(resolution)):
                    count += 1
                else:
                    count = 1

        #Calculating the distances to the nearest non null value that is later in the series
        count = 1
        for i in range(len(null_dates)-1, -1, -1):
            d[i] = min(d[i], count)
            if i > 0:
                if null_dates[i-1] == null_dates[i] - pd.offsets.DateOffset(seconds=to_seconds(resolution)):
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
        imputed_values = res[ts[ts.columns[0]].isnull()].copy()

    else:
        #Returning calendar of the country ts belongs to
        calendar = create_calendar(ts, resolution, holidays, timezone("UTC"))
        calendar.index = calendar["datetime"]
        imputed_values = ts[ts[ts.columns[0]].isnull()].copy()

        #null_dates: Series with all null dates to be imputed
        null_dates = imputed_values.index

        if debug:
            for date in null_dates:
                print(date)

        #isnull: An array which stores whether each value is null or not
        isnull = ts[ts.columns[0]].isnull().values

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
                if null_dates[i+1] == null_dates[i] + pd.offsets.DateOffset(seconds=to_seconds(resolution)):
                    count += 1
                else:
                    count = 1

        #Calculating the distances to the nearest non null value that is later in the series
        count = 1
        for i in range(len(null_dates)-1, -1, -1):
            d[i] = min(d[i], count)
            if i > 0:
                if null_dates[i-1] == null_dates[i] - pd.offsets.DateOffset(seconds=to_seconds(resolution)):
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
            currDayOfYear = calendar.loc[null_date]['dayofyear']
            currH = calendar.loc[null_date]['hour']
            currDN = calendar.loc[null_date]['DN']

            #weight of interpolated series, decreases as distance to nearest known value increases
            w = np.e ** (-a * d[i])

            #Historical value is calculated as the mean of values that have at most wncutoff distance to the current null value's
            #WN, ycutoff distance to its year, and ydcutoff distance to its yearday
            #All dates before cut_date_val that have nan values are imputed using historical data
            #from dates which are also before cut_date_val
            dcutoff = 6
            while True:   
                if null_date < pd.Timestamp(cut_date_val):
                    historical = ts[(~isnull) & (ts[ts.columns[0]].index < pd.Timestamp(cut_date_val)) &\
                                        ((((calendar['yearday'] - currYN) < 0) &\
                                        ((-calendar['yearday'] + currYN) < dcutoff)) &\
                                        ((calendar['DN'] - currDN < to_seconds(resolution)/(120*60)) & (- calendar['DN'] + currDN < to_seconds(resolution)/(120*60))))][ts.columns[0]]

                
                #Dates after cut_date_val are not affected by cut_date_val
                else:
                    historical = ts[(~isnull) & ((((calendar['yearday'] - currYN) < 0) &\
                                        ((-calendar['yearday'] + currYN) < dcutoff)) &\
                                        ((calendar['DN'] - currDN < to_seconds(resolution)/(120*60)) & (- calendar['DN'] + currDN < to_seconds(resolution)/(120*60))))][ts.columns[0]]
            
                if historical.empty:
                    dcutoff += 1
                    if dcutoff>20: 
                        break
                    continue


                historical = historical.mean()

                #imputed value is calculated as a wheighted average of the histrorical value and the value from intrepolation
                res.loc[null_date] = w * ts_interpolatied.loc[null_date] + (1 - w) * historical

                imputed_values.loc[null_date] = res.loc[null_date]

                if debug:
                    print(res.loc[null_date])
                break
    non_nan_intervals_to_nan = {}
    if min_non_nan_interval != -1:
        #UC7 Do that for l_interpolation also
        #If after imputation there exist continuous intervals of non nan values in the train set that are smaller 
        #than min_non_nan_interval time steps, these intervals are all replaced by nan values
        not_nan_values = res[(~res[res.columns[0]].isnull())]
        not_nan_dates = not_nan_values.index
        prev = not_nan_dates[0]
        start = prev

        for not_nan_day in not_nan_dates[1:]:
            if (not_nan_day - prev)!= pd.Timedelta(resolution):
                if prev - start < pd.Timedelta(to_seconds(resolution) * min_non_nan_interval, "sec"):
                    print(f"Non Nan interval from {start} to {prev} is smaller than {min_non_nan_interval} time steps. Making this also Nan")
                    for date in pd.date_range(start=start, end=prev, freq=resolution):
                        non_nan_intervals_to_nan[date] = res.loc[date].values[0]
                        res.loc[date] = pd.NA
                        imputed_values.loc[date] = pd.NA

                start = not_nan_day
            prev = not_nan_day
        if prev - start < pd.Timedelta(to_seconds(resolution) * min_non_nan_interval, "sec"):
            for date in pd.date_range(start=start, end=prev, freq=resolution):
                non_nan_intervals_to_nan[date] = res.loc[date].values[0]
                res.loc[date] = pd.NA
                imputed_values.loc[date] = pd.NA
    imputed_values = imputed_values[(~imputed_values[imputed_values.columns[0]].isnull())]
    imputed_values.to_csv("inputed.csv")
    non_nan_intervals_to_nan = pd.DataFrame.from_dict(non_nan_intervals_to_nan, columns=[res.columns[0]], orient='index')
    non_nan_intervals_to_nan.index.name = "Datetime"
    non_nan_intervals_to_nan.to_csv("non_nan_intervals_to_nan.csv")
    res.to_csv("res.csv")
    if not res.empty:
        full_res = res.asfreq(resolution)
    else:
        full_res = res.copy()

    if not imputed_values.empty:
        full_imputed_values = imputed_values.asfreq(resolution)
    else:
        full_imputed_values = imputed_values.copy()
    
    if not non_nan_intervals_to_nan.empty:
        full_non_nan_intervals_to_nan = non_nan_intervals_to_nan.asfreq(resolution)
    else:
        full_non_nan_intervals_to_nan = non_nan_intervals_to_nan.copy()

    #plot(full_res, full_imputed_values, full_non_nan_intervals_to_nan, name)
    return res#, imputed_values


def get_weather_covariates(start, end, fields=["shortwave_radiation"], name="W6 positive_active", inference=False):
    if type(fields) == str:
        fields = [fields]
    if inference:
        req_fields = ",".join(fields)
        result = requests.get(
            f"https://api.open-meteo.com/v1/gfs?latitude=42.564&longitude=12.643&hourly={req_fields}&forecast_days=10&timezone=auto"
        ).text
        result = json.loads(result)
        df_list = []
        for field in fields:
            data = result['hourly'][field]
            index = pd.to_datetime(result['hourly']['time'])
            assert len(data) == len(index)
            df = pd.DataFrame(data=data, index=index, columns=[field]).asfreq("60min")
            df.index.name = "Datetime"
            df_list.append(df)
        if name in ["W6 positive_active", "W6 positive_reactive"]:
            result_db = requests.get(
                    f"http://38.242.137.200:8000/api/v1/query?source=gfs_forecast&coordinates=(42.569,%2012.608)&start_date={start}&end_date={end}&fields={req_fields}"
                ).text
        elif name in ["W4 positive_active", "W4 positive_reactive"]:
            result_db = requests.get(
                    f"http://38.242.137.200:8000/api/v1/query?source=gfs_forecast&coordinates=(42.567,%2012.607)&start_date={start}&end_date={end}&fields={req_fields}"
                ).text
        else:
            result_db = requests.get(
                f"http://38.242.137.200:8000/api/v1/query?source=gfs_forecastt&coordinates=(42.564,%2012.643)&start_date={start}&end_date={end}&fields={req_fields}"
            ).text
        result_db = json.loads(result_db)
        df_list_db = []
        for field in fields:
            data = result_db['results'][field]['value']
            index = pd.to_datetime(result_db['time'])
            assert len(data) == len(index)
            df = pd.DataFrame(data=data, index=index, columns=[field]).asfreq("60min")
            df.index.name = "Datetime"
            end_date = '2023-10-28'

            # Specify the shift time    
            shift_time = 0

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[:end_date] = subset_df_shifted

            start_date = '2023-10-06'
            end_date = '2023-10-13'

            # Specify the shift time    
            shift_time = 3

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            start_date = '2023-10-14'
            end_date = '2023-10-18'

            # Specify the shift time    
            shift_time = 2

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            start_date = '2023-10-29'
            end_date = '2023-10-29'

            # Specify the shift time    
            shift_time = -4

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            
            start_date = '2023-10-29'

            # Specify the shift time    
            shift_time = -1

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:] = subset_df_shifted


            # shift_time = -2

            # df = df.shift(periods=shift_time)
            df_list_db.append(df)
        result = []
        for ts, ts_db in zip(df_list, df_list_db):
            ts = ts[ts.index > ts_db.index[-1]]
            result_df = pd.concat([ts_db, ts])
            result_df.to_csv("test_.csv")
            result.append(result_df)

        return result
    else:
        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")
        req_fields = ",".join(fields)
        if name in ["W6 positive_active", "W6 positive_reactive"]:
            result = requests.get(
                f"http://38.242.137.200:8000/api/v1/query?source=gfs_forecast&coordinates=(42.569,%2012.608)&start_date={start}&end_date={end}&fields={req_fields}"
            ).text
        elif name in ["W4 positive_active", "W4 positive_reactive"]:
            result = requests.get(
                f"http://38.242.137.200:8000/api/v1/query?source=gfs_forecast&coordinates=(42.567,%2012.607)&start_date={start}&end_date={end}&fields={req_fields}"
            ).text
        else:
            print("Error, no weather available for this time series")
        result = json.loads(result)
        df_list = []
        for field in fields:
            data = result['results'][field]['value']
            index = pd.to_datetime(result['time'])
            assert len(data) == len(index)
            df = pd.DataFrame(data=data, index=index, columns=[field]).asfreq("60min")
            df.index.name = "Datetime"

            end_date = '2023-10-28'

            # Specify the shift time    
            shift_time = 0

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[:end_date] = subset_df_shifted

            start_date = '2023-10-06'
            end_date = '2023-10-13'

            # Specify the shift time    
            shift_time = 3

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            start_date = '2023-10-14'
            end_date = '2023-10-18'

            # Specify the shift time    
            shift_time = 2

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            start_date = '2023-10-29'
            end_date = '2023-10-29'

            # Specify the shift time    
            shift_time = -4

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            
            start_date = '2023-10-29'

            # Specify the shift time    
            shift_time = -1

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:] = subset_df_shifted


            # shift_time = -2

            # df = df.shift(periods=shift_time)

            df_list.append(df)
        return df_list

def add_weather_covariates(start, end, res_future, id_l_future_covs, ts_id_l_future_covs, ts_id_l, fields=["shortwave_radiation"], inference=False):
    other_covs = (res_future == [])
    for i, ts in tqdm(list(enumerate(ts_id_l))):
        if not inference:
            covs_nans = get_weather_covariates(start, end, fields, ts_id_l[i][0], inference)
        else:
            covs_nans = get_weather_covariates(start, 
                                               pd.Timestamp(date.today()).ceil(freq='D') + pd.Timedelta("10D"), 
                                               fields,
                                               ts_id_l[i][0],
                                               inference)
        covs = []
        for cov in covs_nans:
            covs.append(cov)
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
    return res_future, id_l_future_covs, ts_id_l_future_covs

def load_local_csv_as_darts_timeseries(local_path, name='Time Series', time_col='Datetime', last_date=None, multiple = False, day_first=True, resolution="15min", format="long"):

    import logging
    import darts
    import numpy as np
    import pandas as pd

    try:
        if multiple:
            #TODO Fix this too (
            #file_name, format)
            ts_list, id_l, ts_id_l = multiple_ts_file_to_dfs(series_csv=local_path, day_first=day_first, resolution=resolution, format=format)
            covariate_l  = []

            print("\nTurning dataframes to timeseries...")
            logging.info("\nTurning dataframes to timeseries...")
            for comps in tqdm(ts_list):
                first = True
                for df in comps:
                    covariates = darts.TimeSeries.from_dataframe(
                                df,
                                fill_missing_dates=True,
                                freq=None)
                    covariates = covariates.astype(np.float32)
                    if last_date is not None:
                        try:
                            covariates.drop_after(pd.Timestamp(last_date))
                        except:
                            pass
                    if first:
                        first = False
                        covariate_l.append(covariates)
                    else:
                        covariate_l[-1] = covariate_l[-1].stack(covariates)
            covariates = covariate_l
        else:
            id_l, ts_id_l = [[]], [[]]
            covariates = darts.TimeSeries.from_csv(
                local_path, time_col=time_col,
                fill_missing_dates=True,
                freq=None)
            covariates = covariates.astype(np.float32)
            if last_date is not None:
                try:
                    covariates.drop_after(pd.Timestamp(last_date))
                except:
                    pass
    except (FileNotFoundError, PermissionError) as e:
        print(
            f"\nBad {name} file.  The model won't include {name}...")
        logging.info(
            f"\nBad {name} file. The model won't include {name}...")
        covariates = None
    return covariates, id_l, ts_id_l


def parse_uri_prediction_input(client, model_input: dict, model, ts_id_l) -> dict:

    series_uri = model_input['series_uri']

    # str to int
    batch_size = int(model_input["batch_size"])
    roll_size = int(model_input["roll_size"])
    forecast_horizon = int(model_input["n"])

    multiple = truth_checker(model_input["multiple"])
    weather_covariates = model_input["weather_covariates"]
    resolution = model_input["resolution"]
    ts_id_pred = model_input["ts_id_pred"]
    #multiple, resolution and weather_covariates now compulsory

    ## Horizon
    n = int(
        model_input["n"]) if model_input["n"] is not None else model.output_chunk_length
    roll_size = int(
        model_input["roll_size"]) if model_input["roll_size"] is not None else model.output_chunk_length

    ## TODO: future and past covariates (load and transform to darts)
    past_covariates_uri = model_input["past_covariates_uri"]
    future_covariates_uri = model_input["future_covariates_uri"]

    batch_size = int(model_input["batch_size"]
                     ) if model_input["batch_size"] is not None else 1

    if 'runs:/' in series_uri or 's3://mlflow-bucket/' in series_uri or series_uri.startswith('http://'):
        print('\nDownloading remote file of recent time series history...')
        series_uri = download_mlflow_file(client, series_uri)

    if multiple:
        predict_series_idx = [elem[0] for elem in ts_id_l].index(ts_id_pred)
    else: 
        predict_series_idx = 0

    if "history" not in model_input:
        history, id_l, ts_id_l_series = load_local_csv_as_darts_timeseries(
            local_path=series_uri,
            name='series',
            time_col='Datetime',
            last_date=None,
            multiple=multiple,
            resolution=model_input["resolution"],
            format="long" )

        
        if multiple:
             idx = [elem[0] for elem in ts_id_l_series].index(ts_id_pred)
             history = [history[idx]]
        else:
            #TODO Fix multivariate
            history = [history]
    else:
            history = darts.TimeSeries.from_dataframe(
                model_input["history"],
                fill_missing_dates=True,
                freq=None)
            history = [history.astype(np.float32)]

    if none_checker(future_covariates_uri) is not None:
        pass
    #TODO
    else:
        future_covariates = None

    if none_checker(past_covariates_uri) is not None:
        pass
    else:
        past_covariates = None

    if weather_covariates:
        #TODO Fix that
        covs_nans = get_weather_covariates(history[0].pd_dataframe().index[0], 
                                           pd.Timestamp(date.today()).ceil(freq='D') + pd.Timedelta("10D"), 
                                           weather_covariates,
                                           inference=True)
        covs = []
        for cov in covs_nans:
            covs.append(cov.asfreq('60min').interpolate(inplace=False, method='time').ffill())
        future_covariates = darts.TimeSeries.from_dataframe(
            covs[0])
        
        for cov in covs[1:]:
            future_covariates = future_covariates.stack(darts.TimeSeries.from_dataframe(
                cov))
        future_covariates = [future_covariates]

    return {
        "n": n,
        "history": history,
        "roll_size": roll_size,
        "future_covariates": future_covariates,
        "past_covariates": past_covariates,
        "batch_size": batch_size,
        "predict_series_idx": predict_series_idx,
    }

def multiple_ts_file_to_dfs(series_csv: str = "../../RDN/Load_Data/2009-2019-global-load.csv",
                            day_first: bool = True,
                            resolution: str = "15min",
                            value_name="Value",
                            format="long"):
    """
    Reads the input multiple ts file, and returns a tuple containing a list of the time series it consists
    of, along with their ids and timeseries ids. 

    Parameters
    ----------
    series_csv
        The file name of the csv to be read. It must be in the multiple ts form described in the documentation
    day_first
        Wether the day appears before the month in dates
    resolution
        The resolution of the dataset
    value_name
        The name of the value column of the returned dataframes

    Returns
    -------
    Tuple[List[List[pandas.DataFrame]], List[List[str]], List[List[str]]]
        A tuple with the list of lists of dataframes to be returned, the ids 
        of their components, and the timeseries ids. For example, if the function
        reads a file with 2 time series (with ids ts_1 and ts_2), and each one 
        consists of 3 components (with ids ts_1_1, ts_1_2, ts_1_3, ts_2_1, ts_2_2, ts_2_3),
        then the function will return:
        (res, id_l, ts_id_l), where:
        res = [[ts_1_comp_1, ts_1_comp_2, ts_1_comp_3], [ts_2_comp_1, ts_2_comp_2, ts_2_comp_3]]
        id_l = [[ts_1_1, ts_1_2, ts_1_3], [ts_2_1, ts_2_2, ts_2_3]]
        ts_id_l = [[ts_1, ts_1, ts_1], [ts_2, ts_2, ts_2]]
        All of the above lists of lists have the same number of lists and each sublist the same
        amount of elements as the sublist of any other list of lists in the corresponding location.
        This is true because each sublist corresponds to a times eries, and each element of this
        sublist corresponds to a component of this time series.
    """

    ts = pd.read_csv(series_csv,
                     sep=None,
                     header=0,
                     index_col=0,
                     parse_dates=True,
                     dayfirst=day_first,
                     engine='python')
    res = []
    id_l = []
    ts_id_l = []
    ts_ids = list(np.unique(ts["Timeseries ID"]))
    first = True
    print("\nTurning multiple ts file to dataframe list...")
    logging.info("\nTurning multiple ts file to dataframe list...")
    for ts_id in tqdm(ts_ids):
        curr_ts = ts[ts["Timeseries ID"] == ts_id]
        ids = list(np.unique(curr_ts["ID"]))
        res.append([])
        id_l.append([])
        ts_id_l.append([])
        for id in ids:
            curr_comp = curr_ts[curr_ts["ID"] == id]
            if format == 'short':
                curr_comp = pd.melt(curr_comp, id_vars=['Date', 'ID', 'Timeseries ID'], var_name='Time', value_name=value_name)
                curr_comp["Datetime"] = pd.to_datetime(curr_comp['Date'] + curr_comp['Time'], format='%Y-%m-%d%H:%M:%S')
            else:
                curr_comp["Datetime"] = pd.to_datetime(curr_comp["Datetime"])
            curr_comp = curr_comp.set_index("Datetime")
            series = curr_comp[value_name].sort_index().dropna()
            if resolution!=None:
                series = series.asfreq(resolution)
            elif first:
                infered_resolution = to_standard_form(pd.to_timedelta(np.diff(series.index).min()))
                series = series.asfreq(infered_resolution)
                first = False
            else:
                temp = to_standard_form(pd.to_timedelta(np.diff(series.index).min()))
                if temp != infered_resolution:
                    raise DifferentFrequenciesMultipleTS(temp, infered_resolution, id)
                else:
                    series = series.asfreq(temp)
                    infered_resolution = temp

            res[-1].append(pd.DataFrame({value_name : series}))
            id_l[-1].append(id)
            ts_id_l[-1].append(ts_id)
    if resolution != None:
        return res, id_l, ts_id_l
    else:
        return res, id_l, ts_id_l, infered_resolution

def multiple_dfs_to_ts_file(res_l, id_l, ts_id_l, save_dir, save=True, format="long", value_name="Value"):
    ts_list = []
    print("\nTurning dataframe list to multiple ts file...")
    logging.info("\nTurning dataframe list to multiple ts file...")
    for ts_num, (ts, id_ts, ts_id_ts) in tqdm(list(enumerate(zip(res_l, id_l, ts_id_l)))):
        if type(ts) == darts.timeseries.TimeSeries:
            ts = [ts.univariate_component(i).pd_dataframe() for i in range(ts.n_components)]
        for comp_num, (comp, id, ts_id) in enumerate(zip(ts, id_ts, ts_id_ts)):
            comp.columns.values[0] = value_name
            if format == "short":
                comp["Date"] = comp.index.date
                comp["Time"] = comp.index.time
                comp = pd.pivot_table(comp, index=["Date"], columns=["Time"])
                comp = comp[value_name]
            comp["ID"] = id
            comp["Timeseries ID"] = ts_id
            ts_list.append(comp)
    if format == "long":
        res = pd.concat(ts_list).reset_index()
        res.rename(columns={'index': 'Datetime'}, inplace=True)
    else:
        res = pd.concat(ts_list).sort_values(by=["Date", "Timeseries ID", "ID"])
        res = res.reindex(columns=sorted(res.columns, key=lambda x : 0 if isinstance(x, str) else int(datetime.combine(datetime.today().date(), x).timestamp())))
        res = res.reset_index()
    if save:
        res.to_csv(save_dir)
    return res


def check_mandatory(argument, argument_name, mandatory_prerequisites):
    if none_checker(argument) is None:
        raise MandatoryArgNotSet(argument_name, mandatory_prerequisites)
    

def plot_imputation(df, imputed_datetimes, non_nan_intervals_to_nan, name, impute_dir):

    layout = go.Layout(
        title=f'Imputation of time series {name}',
        template='plotly_dark',
    )

    # Create the figure
    fig = go.Figure(layout=layout)

    # Plot the time series, ignoring NaN values
    if not df.empty:
        fig.add_trace(go.Scatter(x=df.index, y=df['Value'], fillcolor='blue', mode='lines', name=f'Original time series'))
    if not imputed_datetimes.empty:
        fig.add_trace(go.Scatter(x=imputed_datetimes.index, y=imputed_datetimes['Value'], fillcolor='green', mode='lines', name=f'Imputed values'))
    if not non_nan_intervals_to_nan.empty:
        fig.add_trace(go.Scatter(x=non_nan_intervals_to_nan.index, y=non_nan_intervals_to_nan['Value'], fillcolor='red', mode='lines', name=f'Values set to Nan'))


    # Update layout for better visualization
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Value')

    # Save the plot
    fig.write_html(f'{impute_dir}/imputed_series_{name}.html')

def plot_removed(removed, res, name, outlier_dir):
    #TODO Fix size issue 

    layout = go.Layout(
        title=f'Outlier Detection of time series {name}',
        template='plotly_dark',
    )

    # Create the figure
    fig = go.Figure(layout=layout)

    # Plot the time series, ignoring NaN values
    if not removed.empty:
        fig.add_trace(go.Scatter(x=res.index, y=res['Value'], fillcolor='blue', mode='lines', name=f'Original time series'))
    if not res.empty:
        fig.add_trace(go.Scatter(x=removed.index, y=removed['Value'], fillcolor='green', mode='lines+markers', marker= dict(size = 8, color= 'red'), name=f'Removed values'))


    # Update layout for better visualization
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Value')

    # Show the plot
    fig.write_html(f'{outlier_dir}/removed_outliers_{name}.html')

def plot_series(df_list, ts_name_list, save_dir, id_list=None):
    if df_list == []: return
    if ts_name_list == []:return
    name = ", ".join(ts_name_list)

    layout = go.Layout(
        title=f'Plot of {name}',
        template='plotly_dark',
    )

    # Create the figure
    fig = go.Figure(layout=layout)

    if type(df_list[0]) == darts.timeseries.TimeSeries:
        temp_df_list = []
        temp_ts_name_list = []
        for ts_idx, ts in enumerate(df_list):
            temp_df_list.extend([ts.univariate_component(i).pd_dataframe() for i in range(ts.n_components)])
            if ts.n_components == 1:
                temp_ts_name_list.extend([ts_name_list[ts_idx] for _ in range(ts.n_components)])
            elif id_list:
                temp_ts_name_list.extend([ts_name_list[ts_idx] + " - component " + id_list[i] for i in range(ts.n_components)])
            else:
                temp_ts_name_list.extend([ts_name_list[ts_idx] + " - component " + str(i) for i in range(ts.n_components)])
        for df in temp_df_list: 
            df.columns.values[0] = "Value"
        df_list = temp_df_list
        ts_name_list = temp_ts_name_list

    # Plot the time series, ignoring NaN values
    for df, ts_name in zip(df_list, ts_name_list):
        if not df.empty:
            fig.add_trace(go.Scatter(x=df.index, y=df[df.columns[0]], fillcolor='blue', mode='lines', name=f'Time series {ts_name}'))

    # Update layout for better visualization
    fig.update_layout(xaxis_title='Date',
                      yaxis_title=df.columns[0])

    # Show the plot
    fig.write_html(save_dir)


def allow_empty_series_fun(ts_list, id_l, ts_id_l, allow_empty_series=False):
    # TODO: that works only for multiple, extend for multivariate
    ts_list_ret, id_l_ret, ts_id_l_ret = [], [], []
    for ts, id, ts_id in zip(ts_list, id_l, ts_id_l):
        if not ts[0].empty:
            ts_list_ret.append(ts)
            id_l_ret.append(id)
            ts_id_l_ret.append(ts_id)
        elif not allow_empty_series: 
            raise EmptySeries()
    return ts_list_ret, id_l_ret, ts_id_l_ret

def to_seconds(resolution):
    return ceil(pd.to_timedelta(to_offset(resolution)).total_seconds())

def to_standard_form(freq):

    total_seconds = int(freq.total_seconds())

    if total_seconds % 86400 == 0:
        if total_seconds // 86400 == 1:
            return '1d'  # Daily frequency
        else:
            return f'{total_seconds // 86400}d'
    elif total_seconds % 3600 == 0:
        if total_seconds // 3600 == 1:
            return '1h'  # Hourly frequency
        else:
            return f'{total_seconds // 3600}h'
    elif total_seconds % 60 == 0:
        if total_seconds // 60 == 1:
            return '1min'  # Minutely frequency
        else:
            return f'{total_seconds // 60}min'
    else:
        return f'{total_seconds}s'  # Secondly frequency


def change_form(freq, change_format_to="pandas_form"):
    import re

    # Dictionary to map time units from short to long forms and vice versa
    time_units = {
        "s": "second",
        "min": "minute",
        "h": "hour",
        "d": "day"
    }
    
    # Identify the number and unit from the frequency
    match = re.match(r"(\d+)?(\w+)", freq)
    if not match:
        raise ValueError("Invalid frequency format.")
    
    number, unit = match.groups()

    if not number:
      number = 1
    
    # Convert to the desired format
    if change_format_to == "print_form":
        # From pandas form (e.g., '1h') to human-readable form (e.g., '1 hour')
        full_unit = time_units.get(unit, "unknown")  # Default to 'unknown' if unit not found
        if int(number) > 1:
            full_unit += 's'  # Make plural if more than one
        return f"{number} {full_unit}"
    elif change_format_to == "pandas_form":
        # From human-readable form (e.g., '1 hour') to pandas form (e.g., '1h')
        for short, long in time_units.items():
            if long in freq:
                # Check if the unit matches and convert it
                if ' ' in freq:
                    num, _ = freq.split()
                return f"{num}{short}"
    else:
        raise ValueError("Invalid change_format_to value. Use 'pandas_form' or 'print_form'.")

def make_time_list(resolution):
    import re

    # List of all supported resolutions in increasing order
    all_resolutions = ["1s", "2s", "5s", "15s", "30s", "1min", "2min", "5min", "15min", "30min", "1h", "2h", "6h", "1d", "2d", "5d", "10d"]

    input_seconds = to_seconds(resolution)

    # Filter and convert resolutions
    resolutions = [{"value": change_form(resolution, change_format_to="print_form"), "default": True}]
    for res in all_resolutions:
        if to_seconds(res) > input_seconds:
            resolutions.append({"value": change_form(res, "print_form"), "default": False})
    return resolutions

#epestrepse kai IDs
#prwta psakse ID meta SC
#TODO: Fix it. It does not get any progress any more
# def get_training_progress_by_tag(fn, tag):
#     # assert(os.path.isdir(output_dir))

#     image_str = tf.compat.v1.placeholder(tf.string)

#     im_tf = tf.image.decode_image(image_str)

#     sess = tf.compat.v1.InteractiveSession()
#     with sess.as_default():
#         count = 0
#         values = []
#         for e in summary_iterator(fn):
#             for v in e.summary.value:
#                 if tag == v.tag:
#                     values.append(v.simple_value)
#     sess.close()
#     return {"Epoch": range(1, len(values) + 1), "Value": values}

# def log_curves(tensorboard_event_folder, output_dir='training_curves'):
    # TODO: fix deprecation warnings

    # tf.compat.v1.disable_eager_execution()

    # # locate tensorboard event file
    # event_file_names = os.listdir(tensorboard_event_folder)
    # if len(event_file_names) > 1:
    #     logging.info(
    #         "Searching for term 'events.out.tfevents.' in logs folder to extract tensorboard file...\n")
    #     print(
    #         "Searching for term 'events.out.tfevents.' in logs folder to extract tensorboard file...\n")
    # tensorboard_folder_list = os.listdir(tensorboard_event_folder)
    # event_file_name = [fname for fname in tensorboard_folder_list if
    #     "events.out.tfevents." in fname][0]
    # tensorboard_event_file = os.path.join(tensorboard_event_folder, event_file_name)

    # # test for get_training_progress_by_tag
    # print(tensorboard_event_file)

    # # local folder
    # print("Creating local folder to store the datasets as csv...")
    # logging.info("Creating local folder to store the datasets as csv...")
    # os.makedirs(output_dir, exist_ok=True)

    # # get metrix and store locally
    # training_loss = pd.DataFrame(get_training_progress_by_tag(tensorboard_event_file, 'training/loss_total'))
    # training_loss.to_csv(os.path.join(output_dir, 'training_loss.csv'))

    # # testget_training_progress_by_tag
    # print(training_loss)

    # ## consider here nr_epoch_val_period
    # validation_loss = pd.DataFrame(get_training_progress_by_tag(tensorboard_event_file, 'validation/loss_total'))

    # # test for get_training_progress_by_tag
    # print(validation_loss)
    # print(validation_loss.__dict__)

    # validation_loss["Epoch"] = (validation_loss["Epoch"] * int(len(training_loss) / len(validation_loss)) + 1).astype(int)
    # validation_loss.to_csv(os.path.join(output_dir, 'validation_loss.csv'))

    # learning_rate = pd.DataFrame(get_training_progress_by_tag(tensorboard_event_file, 'training/learning_rate'))
    # learning_rate.to_csv(os.path.join(output_dir, 'learning_rate.csv'))

    # sns.lineplot(x="Epoch", y="Value", data=training_loss, label="Training")
    # sns.lineplot(x="Epoch", y="Value", data=validation_loss, label="Validation", marker='o')
    # plt.grid()
    # plt.legend()
    # plt.title("Loss")
    # plt.savefig(os.path.join(output_dir, 'loss.png'))

    # plt.figure()
    # sns.lineplot(x="Epoch", y="Value", data=learning_rate)
    # plt.grid()
    # plt.title("Learning rate")
    # plt.savefig(os.path.join(output_dir, 'learning_rate.png'))

    # print("Uploading training curves to MLflow server...")
    # logging.info("Uploading training curves to MLflow server...")
    # mlflow.log_artifacts(output_dir, output_dir)

    # print("Artifacts uploaded. Deleting local copies...")
    # logging.info("Artifacts uploaded. Deleting local copies...")
    # shutil.rmtree(output_dir)
