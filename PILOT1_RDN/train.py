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

# get environment variables
from dotenv import load_dotenv
load_dotenv()

# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
os.environ["MLFLOW_TRACKING_URI"] = read_config(
    'config.yml', 'mlflow_settings')['mlflow_tracking_uri']
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

def train