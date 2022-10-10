
import pretty_errors
import os
# import tensorflow as tf
# from tensorflow.python.summary.summary_iterator import summary_iterator
# import seaborn as sns
import mlflow
import pandas as pd
cur_dir = os.path.dirname(os.path.realpath(__file__))

class ConfigParser:
    def __init__(self, config_file_path=f'{cur_dir}/config.yml'):
        import yaml
        with open(config_file_path, "r") as ymlfile:
            self.config = yaml.safe_load(ymlfile)
            # self.mlflow_tracking_uri = self.config['mlflow_settings']['mlflow_tracking_uri']

    def read_hyperparameters(self, hyperparams_entrypoint):
        return self.config['hyperparameters'][hyperparams_entrypoint]

    def read_entrypoints(self):
        return self.config['hyperparameters']

def download_file_from_s3_bucket(object_name, dst_filename, dst_dir=None, bucketName='mlflow-bucket'):
    import boto3, tempfile
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

def download_online_file(url, dst_filename=None, dst_dir=None):
    import sys, tempfile, requests

    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    req = requests.get(url)
    if req.status_code != 200:
        raise Exception(f"\nResponse is not 200\nProblem downloading: {url}")
        sys.exit()
    url_content = req.content
    if dst_filename is None:
        dst_filename = url.split('/')[-1]
    filepath = os.path.join(dst_dir, dst_filename)
    file = open(filepath, 'wb')
    file.write(url_content)
    file.close()
    return filepath


def download_mlflow_file(url, dst_dir=None):
    import tempfile
    from dotenv import load_dotenv
    load_dotenv()
    S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    if url.startswith('s3://mlflow-bucket/'):
        url = url.replace("s3:/", S3_ENDPOINT_URL)
        local_path = download_online_file(
            url, dst_dir=dst_dir)
    elif url.startswith('runs:/'):
        client = mlflow.tracking.MlflowClient()
        run_id = url.split('/')[1]
        mlflow_path = '/'.join(url.split('/')[3:])
        local_path = client.download_artifacts(run_id, mlflow_path, dst_dir)
    elif url.startswith('http://'):
        local_path = download_online_file(
            url, dst_dir=dst_dir)
    return local_path

def load_pkl_model_from_server(model_uri):
    print("\nLoading remote PKL model...")
    model_path = download_online_file(f'{model_uri}/_model.pkl', '_model.pkl')
    print(model_path)
    best_model = load_local_pkl_as_object(model_path)
    return best_model

def load_local_pl_model(model_root_dir):

    from darts.models.forecasting.gradient_boosted_model import LightGBMModel
    from darts.models.forecasting.random_forest import RandomForest
    from darts.models import RNNModel, BlockRNNModel, NBEATSModel, TFTModel, NaiveDrift, NaiveSeasonal, TCNModel

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
    client = mlflow.tracking.MlflowClient()
    print(model_root_dir)
    model_run_id = model_root_dir.split("/")[5]
    mlflow_relative_model_root_dir = model_root_dir.split("/artifacts/")[1]

    local_dir = tempfile.mkdtemp()
    client.download_artifacts(
        run_id=model_run_id, path=mlflow_relative_model_root_dir, dst_path=local_dir)
    best_model = load_local_pl_model(os.path.join(local_dir, mlflow_relative_model_root_dir))
    return best_model


def load_model(model_root_dir, mode="remote"):

    # Get model type as tag of model's run
    import mlflow
    print(model_root_dir)

    if mode == 'remote':
        client = mlflow.tracking.MlflowClient()
        run_id = model_root_dir.split('/')[-1]
        model_run = client.get_run(run_id)
        model_type = model_run.data.tags.get('model_type')
    else:
        if "_model.pth.tar" in os.listdir(model_root_dir):
            model_type = 'pl'
        else:
            model_type = "pkl"

    # Load accordingly
    if mode == "remote" and model_type == "pl":
        model = load_pl_model_from_server(model_root_dir=model_root_dir)
    elif mode == "remote" and model_type == "pkl":
        model = load_pkl_model_from_server(model_root_dir)
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
        run_id = scaler_uri.split("/")[5]
        mlflow_filepath = scaler_uri.split("/artifacts/")[1]

        client = mlflow.tracking.MlflowClient()
        local_dir = tempfile.mkdtemp()

        client.download_artifacts(
            run_id=run_id,
            path=mlflow_filepath,
            dst_path=local_dir
            )
        # scaler_path = download_online_file(
        #     scaler_uri, "scaler.pkl") if mode == 'remote' else scaler_uri
        scaler = load_local_pkl_as_object(os.path.join(local_dir, mlflow_filepath))
    else:
        scaler = load_local_pkl_as_object(scaler_uri)
    return scaler


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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(local_path, map_location=device)
    model.device = device
    return model

def load_local_csv_as_darts_timeseries(local_path, name='Time Series', time_col='Date', last_date=None, multiple = False):

    import logging, darts
    import numpy as np
    import pandas as pd

    try:
        if multiple:
            #TODO Fix this too
            ts_list, country_l, country_code_l = multiple_ts_file_to_dfs(series_csv=local_path, day_first=True)
            covariate_l  = []
            print("liiiist", local_path)
            for df in ts_list:
                covariates = darts.TimeSeries.from_dataframe(
                             df, time_col=time_col,
                             fill_missing_dates=True,
                             freq=None)
                covariates = covariates.astype(np.float32)
                if last_date is not None:
                    covariates.drop_after(pd.Timestamp(last_date))
                covariate_l.append(covariates)
            covariates = covariate_l
        else:
            country_code_l, country_l = [], []
            covariates = darts.TimeSeries.from_csv(
                local_path, time_col=time_col,
                fill_missing_dates=True,
                freq=None)
            covariates = covariates.astype(np.float32)
            if last_date is not None:
                covariates.drop_after(pd.Timestamp(last_date))
    except (FileNotFoundError, PermissionError) as e:
        print(
            f"\nBad {name} file.  The model won't include {name}...")
        logging.info(
            f"\nBad {name} file. The model won't include {name}...")
        covariates = None
    return covariates, country_l, country_code_l

def parse_uri_prediction_input(model_input: dict, model) -> dict:

    series_uri = model_input['series_uri']

    # str to int
    batch_size = int(model_input["batch_size"])
    roll_size = int(model_input["roll_size"])
    forecast_horizon = int(model_input["n"])

    ## Horizon
    n = int(model_input["n"]) if model_input["n"] is not None else model.output_chunk_length
    roll_size = int(model_input["roll_size"]) if model_input["roll_size"] is not None else model.output_chunk_length

    ## TODO: future and past covariates (load and transform to darts)
    past_covariates_uri = model_input["past_covariates_uri"]
    future_covariates_uri = model_input["future_covariates_uri"]

    batch_size = int(model_input["batch_size"]) if model_input["batch_size"] is not None else 1

    if 'runs:/' in series_uri or 's3://mlflow-bucket/' in series_uri or series_uri.startswith('http://'):
        print('\nDownloading remote file of recent time series history...')
        series_uri = download_mlflow_file(series_uri)

    history = load_local_csv_as_darts_timeseries(
        local_path=series_uri,
        name='series',
        time_col='Date',
        last_date=None)

    if none_checker(future_covariates_uri) is not None:
        future_covariates = darts.TimeSeries.from_csv(
            future_covariates_uri, time_col='Date')
    else:
        future_covariates = None

    if none_checker(past_covariates_uri) is not None:
        past_covariates = darts.TimeSeries.from_csv(
            past_covariates_uri, time_col='Date')
    else:
        past_covariates = None

    return {
        "n": n,
        "history": history,
        "roll_size": roll_size,
        "future_covariates": future_covariates,
        "past_covariates": past_covariates,
        "batch_size": batch_size,
    }

def multiple_ts_file_to_dfs(series_csv: str = "../../RDN/Load_Data/2009-2019-global-load.csv",
                            day_first: str = "true"):
    ts = pd.read_csv(series_csv,
                     sep=None,
                     header=0,
                     index_col=0,
                     parse_dates=True,
                     dayfirst=day_first,
                     engine='python')
    print("tssss", ts)
    res = []
    country = []
    country_code = []
    for i in range(max(ts["ID"])):
        curr = ts[ts["ID"] == i]
        curr = pd.melt(curr, id_vars=['Day', 'ID', 'Country', 'Country Code'], var_name='Time', value_name='Load')
        curr["Date"] = pd.to_datetime(curr['Day'] + curr['Time'], format='%Y-%m-%d%H:%M:%S')
        curr = curr.set_index("Date")
        series = curr["Load"].sort_index().dropna().asfreq('60'+'min')
        res.append(pd.DataFrame({"Load" : series}))
        country.append(curr["Country"].values[0])
        country_code.append(curr["Country Code"].values[0])
    return res, country, country_code

def multiple_dfs_to_ts_file(res_l, country_l, country_code_l, save_dir):
    ts_list = []
    for i, (ts, country, country_code) in enumerate(zip(res_l, country_l, country_code_l)):
        ts["Day"] = ts.index.date
        ts["Time"] = ts.index.time
        ts = pd.pivot_table(ts, index=["Day"], columns=["Time"])
        ts = ts["Load"]
        ts["ID"] = i
        ts["Country"] = country
        ts["Country Code"] = country_code
        ts_list.append(ts)
    res = pd.concat(ts_list).sort_values(by=["Day", "ID"])
    columns = list(res.columns)
    columns.remove("ID")
    columns = ["ID"] + columns
    res = res[columns].reset_index()
    res.to_csv(save_dir)

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

    # return
