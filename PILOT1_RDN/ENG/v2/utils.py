from tabnanny import check
from threading import local
from dotenv import load_dotenv
import yaml
import os
import scipy.misc
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import pandas as pd
import mlflow
import shutil
import torch
import darts
import pickle
import requests
import tempfile
import yaml
import boto3
import sys

cur_dir = os.path.dirname(os.path.realpath(__file__))

class ConfigParser:
    def __init__(self, config_file_path=f'{cur_dir}/config.yml'):
        with open(config_file_path, "r") as ymlfile:
            self.config = yaml.safe_load(ymlfile)
            self.mlflow_tracking_uri = self.config['mlflow_settings']['mlflow_tracking_uri']

    def read_hyperparameters(self, hyperparams_entrypoint):
        return self.config['hyperparameters'][hyperparams_entrypoint]

# get environment variables
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"] 
MLFLOW_S3_ENDPOINT_URL = os.environ["MLFLOW_S3_ENDPOINT_URL"] 

def download_file_from_s3_bucket(object_name, dst_filename, dst_dir=None, bucketName='mlflow-bucket'):
    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    s3_resource = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    bucket = s3_resource.Bucket(bucketName)
    local_path = os.path.join(dst_dir, dst_filename)
    bucket.download_file(object_name, local_path)
    return local_path

def download_online_file(url, dst_filename, dst_dir=None):
    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    req = requests.get(url)
    if req.status_code != 200:
        raise Exception(f"\nResponse is not 200\nProblem downloading: {url}")
        sys.exit()
    url_content = req.content
    filepath = os.path.join(dst_dir, dst_filename)
    file = open(filepath, 'wb')
    file.write(url_content)
    file.close()
    return filepath

def load_yaml_as_dict(filepath):
    with open(filepath, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            return parsed_yaml
        except yaml.YAMLError as exc:
            print(exc)
            return 

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

def load_artifacts(run_id, src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    client = mlflow.tracking.MlflowClient()
    client.download_artifacts(run_id=run_id, path=src_path, dst_path=dst_path)

def load_local_model_as_torch(local_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(local_path, map_location=device)
    model.device = device
    return model

def load_local_csv_as_darts_timeseries(local_path, name='Time Series', time_col='Date', last_date=None):
    try:
        covariates = darts.TimeSeries.from_csv(
            local_path, time_col=time_col, 
            fill_missing_dates=True, 
            freq=None)
        if last_date is not None:
            covariates.drop_after(pd.Timestamp(last_date))
    except (FileNotFoundError, PermissionError) as e:
        print(
            f"\nBad {name} file.  The model won't include {name}...")
        logging.info(
            f"\nBad {name} file. The model won't include {name}...")
        covariates = None
    return covariates

def load_local_pkl_as_object(local_path):
    pkl_object = pickle.load(open(local_path, "rb"))
    return pkl_object


class MlflowArtifactDownloader():
    def __init__(self, run_id=None, uri=None):
        self.run_id = run_id

    def load_model_artifact_as_torch(self, checkpoints_dir="checkpoints", 
        keyword_to_search_in_best_model="model_best"):

        client = mlflow.tracking.MlflowClient()
        model_dir_list = client.list_artifacts(run_id=self.run_id, path=checkpoints_dir)
        src_path = [fileinfo.path for fileinfo in model_dir_list if 
            keyword_to_search_in_best_model in fileinfo.path][0]

        tmpdir = tempfile.mkdtemp()
        load_artifacts(self.run_id, src_path=src_path, dst_path=tmpdir)
        local_path = os.path.join(tmpdir, src_path.replace('/', os.path.sep))
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = torch.load(local_path, map_location=device)
        model.device = device
        return model

    def load_csv_artifact_as_darts(self, time_col="Date", 
        prefix="features", suffix="test.csv"):

        # ideally needs tmpfiles but does not work in venv
        
        src_path = "/".join([prefix, suffix])
        tmpdir = tempfile.mkdtemp()
        load_artifacts(self.run_id, src_path=src_path, dst_path=tmpdir)
        local_path = os.path.join(tmpdir, src_path.replace('/', os.path.sep))

        darts_ts = darts.TimeSeries.from_csv(local_path, time_col=time_col)

        return darts_ts

    def load_pkl_artifact_as_object(self, prefix="scalers", suffix="scaler_series.pkl"):

        # ideally needs tmpfiles but does not work in venv
        
        src_path = "/".join([prefix, suffix])
        tmpdir = tempfile.mkdtemp()
        load_artifacts(self.run_id, src_path=src_path, dst_path=tmpdir)
        local_path = os.path.join(tmpdir, src_path)

        pkl_object = pickle.load(open(local_path, "rb"))
        return pkl_object

def get_training_progress_by_tag(fn, tag):
    # assert(os.path.isdir(output_dir))

    image_str = tf.compat.v1.placeholder(tf.string)
    
    im_tf = tf.image.decode_image(image_str)

    sess = tf.compat.v1.InteractiveSession()
    with sess.as_default():
        count = 0
        values = []
        for e in summary_iterator(fn):
            for v in e.summary.value:
                if tag == v.tag:
                    values.append(v.simple_value)
    sess.close()
    return {"Epoch": range(1, len(values) + 1), "Value": values}

def log_curves(tensorboard_event_folder, output_dir='training_curves'):
    # TODO: fix deprecation warnings

    tf.compat.v1.disable_eager_execution()

    # locate tensorboard event file
    event_file_names = os.listdir(tensorboard_event_folder)
    if len(event_file_names) > 1:
        os.logging("MORE TENSORBOARD FILES HAVE BEEN DETECTED! ONLY FIRST USED!")
        print("MORE TENSORBOARD FILES HAVE BEEN DETECTED! ONLY FIRST USED!")
    tensorboard_event_file = os.path.join(tensorboard_event_folder, event_file_names[0])

    # local folder
    print("Creating local folder to store the datasets as csv...")
    logging.info("Creating local folder to store the datasets as csv...")
    os.makedirs(output_dir, exist_ok=True)

    # get metrix and store locally
    training_loss = pd.DataFrame(get_training_progress_by_tag(tensorboard_event_file, 'training/loss_total'))
    training_loss.to_csv(os.path.join(output_dir, 'training_loss.csv'))
    
    ## consider here nr_epoch_val_period
    validation_loss = pd.DataFrame(get_training_progress_by_tag(tensorboard_event_file, 'validation/loss_total'))
    validation_loss["Epoch"] = (validation_loss["Epoch"] * int(len(training_loss) / len(validation_loss)) + 1).astype(int)
    validation_loss.to_csv(os.path.join(output_dir, 'validation_loss.csv'))

    learning_rate = pd.DataFrame(get_training_progress_by_tag(tensorboard_event_file, 'training/learning_rate'))    
    learning_rate.to_csv(os.path.join(output_dir, 'learning_rate.csv'))

    sns.lineplot(x="Epoch", y="Value", data=training_loss, label="Training")
    sns.lineplot(x="Epoch", y="Value", data=validation_loss, label="Validation", marker='o')
    plt.grid()
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(output_dir, 'loss.png'))

    plt.figure()
    sns.lineplot(x="Epoch", y="Value", data=learning_rate)
    plt.grid()
    plt.title("Learning rate")
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'))

    print("Uploading training curves to MLflow server...")
    logging.info("Uploading training curves to MLflow server...")
    mlflow.log_artifacts(output_dir, output_dir)

    print("Artifacts uploaded. Deleting local copies...")
    logging.info("Artifacts uploaded. Deleting local copies...")
    shutil.rmtree(output_dir)

    return
    



