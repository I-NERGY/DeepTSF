import pretty_errors
from utils import none_checker, ConfigParser, download_online_file, load_local_csv_as_darts_timeseries, truth_checker, load_yaml_as_dict, get_pv_forecast #, log_curves
from preprocessing import scale_covariates, split_dataset, split_nans

# the following are used through eval(darts_model + 'Model')
from darts.models import RNNModel, BlockRNNModel, NBEATSModel, TFTModel, NaiveDrift, NaiveSeasonal, TCNModel, NHiTSModel, TransformerModel
# from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.gradient_boosted_model import LightGBMModel
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.arima import ARIMA
from darts.utils.likelihood_models import ContinuousBernoulliLikelihood, GaussianLikelihood, DirichletLikelihood, ExponentialLikelihood, GammaLikelihood, GeometricLikelihood

import yaml
import mlflow
import click
import os
import torch
import logging
import pickle
import tempfile
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import shutil
import pandas as pd

# Inference requirements to be stored with the darts flavor !!
from sys import version_info
import torch, cloudpickle, darts
PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
mlflow_serve_conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
                'cloudpickle=={}'.format(cloudpickle.__version__),
                'darts=={}'.format(darts.__version__),
                'pretty_errors=={}'.format(pretty_errors.__version__),
                'torch=={}'.format(torch.__version__),
                'mlflow=={}'.format(mlflow.__version__)
            ],
        },
    ],
    'name': 'darts_infer_pl_env'
}

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)

# stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
# a period of 5 epochs (`patience`)
my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=1e-6,
    mode='min',
)

@click.command()
@click.option("--series-csv",
              type=str,
              default="None",
              help="Local timeseries csv. If set, it overwrites the local value."
              )
@click.option("--series-uri",
              type=str,
              default='None',
              help="Remote timeseries csv file. If set, it overwrites the local value."
              )
@click.option("--future-covs-csv",
              type=str,
              default='None'
              )
@click.option("--future-covs-uri",
              type=str,
              default='None'
              )
@click.option("--past-covs-csv",
              type=str,
              default='None'
              )
@click.option("--past-covs-uri",
              type=str,
              default='None'
              )
@click.option("--darts-model",
              type=click.Choice(
                  ['NBEATS',
                   'Transformer',
                   'NHiTS',
                   'TCN',
                   'RNN',
                   'BlockRNN',
                   'TFT',
                   'ARIMA',
                   'LightGBM',
                   'RandomForest',
                   'Naive']),
              multiple=False,
              default='None',
              help="The base architecture of the model to be trained"
              )
@click.option("--hyperparams-entrypoint", "-h",
              type=str,
              default='None',
              help=""" The entry point of config.yml under the 'hyperparams'
              one containing the desired hyperparameters for the selected model"""
              )
@click.option("--cut-date-val",
              type=str,
              default='None',
              help="Validation set start date [str: 'YYYYMMDD']"
              )
@click.option("--cut-date-test",
              type=str,
              default='None',
              help="Test set start date [str: 'YYYYMMDD']",
              )
@click.option("--test-end-date",
              type=str,
              default='None',
              help="Test set ending date [str: 'YYYYMMDD']",
              )
@click.option("--device",
              type=click.Choice(
                  ['gpu',
                   'cpu']),
              multiple=False,
              default='gpu',
              )
@click.option("--scale",
              type=str,
              default="true",
              help="Whether to scale the target series")
@click.option("--scale-covs",
              type=str,
              default="true",
              help="Whether to scale the covariates")
@click.option("--multiple",
    type=str,
    default="false",
    help="Whether to train on multiple timeseries")

@click.option("--training-dict",
        type=str,
        default="None",
        help="In case of an optuna run, the yaml with the dictionary with the current model's hyperparameters")

@click.option("--num-workers",
        type=str,
        default="4",
        help="Number of threads that will be used by pytorch")
        
@click.option("--day-first",
    type=str,
    default="true",
    help="Whether the date has the day before the month")
@click.option("--resolution",
    default="None",
    type=str,
    help="The resolution of the dataset in minutes."
)
@click.option("--pv-ensemble",
    default="False",
    type=str,
    help="Wether to subtract the pv production forecasts from the training series and add it again during testing or not."
    )

def train(series_csv, series_uri, future_covs_csv, future_covs_uri,
          past_covs_csv, past_covs_uri, darts_model,
          hyperparams_entrypoint, cut_date_val, cut_date_test,
          test_end_date, device, scale, scale_covs, multiple,
          training_dict, num_workers, day_first, resolution,
          pv_ensemble):

    num_workers = int(num_workers)
    torch.set_num_threads(num_workers)

    # Argument preprocessing

    ## test_end_date
    test_end_date = none_checker(test_end_date)

    ## scale or not
    scale = truth_checker(scale)
    scale_covs = truth_checker(scale_covs)

    multiple = truth_checker(multiple)
    pv_ensemble = truth_checker(pv_ensemble)



    ## hyperparameters
    hyperparameters = ConfigParser(config_string=hyperparams_entrypoint).read_hyperparameters(hyperparams_entrypoint)

    ## device
    if device == 'gpu' and torch.cuda.is_available():
        device = 'gpu'
        print("\nGPU is available")
    else:
        device = 'cpu'
        print("\nGPU is available")
    ## series and covariates uri and csv
    series_uri = none_checker(series_uri)
    future_covs_uri = none_checker(future_covs_uri)
    past_covs_uri = none_checker(past_covs_uri)

    # redirect to local location of downloaded remote file
    if series_uri is not None:
        download_file_path = download_online_file(series_uri, dst_filename="load.csv")
        series_csv = download_file_path
    if  future_covs_uri is not None:
        download_file_path = download_online_file(future_covs_uri, dst_filename="future.csv")
        future_covs_csv = download_file_path
    if  past_covs_uri is not None:
        download_file_path = download_online_file(past_covs_uri, dst_filename="past.csv")
        past_covs_csv = download_file_path

    series_csv = series_csv.replace('/', os.path.sep).replace("'", "")
    future_covs_csv = future_covs_csv.replace('/', os.path.sep).replace("'", "")
    past_covs_csv = past_covs_csv.replace('/', os.path.sep).replace("'", "")

    ## model
    # TODO: Take care of future covariates (RNN, ...) / past covariates (BlockRNN, NBEATS, ...)
    if darts_model in ["NBEATS", "BlockRNN", "TCN", "NHiTS", "Transformer"]:
        """They do not accept future covariates as they predict blocks all together.
        They won't use initial forecasted values to predict the rest of the block
        So they won't need to additionally feed future covariates during the recurrent process.
        """
        #past_covs_csv = future_covs_csv
        #TODO Make tool add past_covs to future_covs in this case
        future_covs_csv = None
        # TODO: when actual weather comes extend it, now the stage only accepts future covariates as argument.

    elif darts_model in ["RNN", "ARIMA"]:
        """Does not accept past covariates as it needs to know future ones to provide chain forecasts
        its input needs to remain in the same feature space while recurring and with no future covariates
        this is not possible. The existence of past_covs is not permitted for the same reason. The
        feature space will change during inference. If for example I have current temperature and during
        the forecast chain I only have time covariates, as I won't know the real temp then a constant \
        architecture like LSTM cannot handle this"""
        past_covs_csv = None
        # TODO: when actual weather comes extend it, now the stage only accepts future covariates as argument.
    #elif: extend for other models!! (time_covariates are always future covariates, but some models can't handle them as so)
    
    elif darts_model=='Naive':
        past_covs_csv = None
        future_covs_csv = None
        scale = False
    
    future_covariates = none_checker(future_covs_csv)
    past_covariates = none_checker(past_covs_csv)

    with mlflow.start_run(run_name=f'train_{darts_model}', nested=True) as mlrun:

        mlflow_model_root_dir = "pyfunc_model"

        ######################
        # Load series and covariates datasets
        time_col = "Datetime"
        series, id_l, ts_id_l = load_local_csv_as_darts_timeseries(
                local_path=series_csv,
                name='series',
                time_col=time_col,
                last_date=test_end_date,
                multiple=multiple,
                day_first=day_first,
                resolution=resolution)
        if future_covariates is not None:
            future_covariates, id_l_future_covs, ts_id_l_future_covs = load_local_csv_as_darts_timeseries(
                local_path=future_covs_csv,
                name='future covariates',
                time_col=time_col,
                last_date=test_end_date,
                multiple=True,
                day_first=day_first,
                resolution=resolution)
        else:
            future_covariates, id_l_future_covs, ts_id_l_future_covs = None, None, None
        if past_covariates is not None:
            past_covariates, id_l_past_covs, ts_id_l_past_covs = load_local_csv_as_darts_timeseries(
                local_path=past_covs_csv,
                name='past covariates',
                time_col=time_col,
                last_date=test_end_date,
                multiple=True,
                day_first=day_first,
                resolution=resolution)
        else:
            past_covariates, id_l_past_covs, ts_id_l_past_covs = None, None, None

        if (len(id_l) != 1 or len(id_l[0]) > 1) and darts_model=='ARIMA':
            raise Exception("ARIMA does not support multiple time series") 

        print("\nCreating local folders...")
        logging.info("\nCreating local folders...")

        scalers_dir = tempfile.mkdtemp()
        features_dir = tempfile.mkdtemp()

        ######################
        # Train / Test split
        print(
            f"\nTrain / Test split: Validation set starts: {cut_date_val} - Test set starts: {cut_date_test} - Test set end: {test_end_date}")
        logging.info(
             f"\nTrain / Test split: Validation set starts: {cut_date_val} - Test set starts: {cut_date_test} - Test set end: {test_end_date}")
             
        ## series
        series_split = split_dataset(
            series,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            store_dir=features_dir,
            name='series',
            conf_file_name='split_info.yml',
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l)
        ## future covariates
        future_covariates_split = split_dataset(
            future_covariates,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            # store_dir=features_dir,
            name='future_covariates',
            multiple=True,
            id_l=id_l_future_covs,
            ts_id_l=ts_id_l_future_covs)
        ## past covariates
        past_covariates_split = split_dataset(
            past_covariates,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            # store_dir=features_dir,
            name='past_covariates',
            multiple=True,
            id_l=id_l_past_covs,
            ts_id_l=ts_id_l_past_covs)
        

        if pv_ensemble:
            print("\nSubtracting pv forecast from train and val series")
            logging.info("\nSubtracting pv forecast from train and val series")
            for i in range(len(series_split['train'])):

                series_split['train'][i] = series_split['train'][i] + get_pv_forecast(ts_id_l[i], 
                                                                                  start=series_split['train'][i].pd_dataframe().index[0], 
                                                                                  end=series_split['train'][i].pd_dataframe().index[-1], 
                                                                                  inference=False, 
                                                                                  kW=60, 
                                                                                  use_saved=True)
                series_split['val'][i] = series_split['val'][i] + get_pv_forecast(ts_id_l[i], 
                                                                                  start=series_split['val'][i].pd_dataframe().index[0], 
                                                                                  end=series_split['val'][i].pd_dataframe().index[-1], 
                                                                                  inference=False, 
                                                                                  kW=60, 
                                                                                  use_saved=True)


        #################
        # Scaling
        print("\nScaling...")
        logging.info("\nScaling...")

        ## scale series
        series_transformed = scale_covariates(
            series_split,
            store_dir=features_dir,
            filename_suffix="series_transformed.csv",
            scale=scale,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l
            )
        if scale:
            pickle.dump(series_transformed["transformer"], open(f"{scalers_dir}/scaler_series.pkl", "wb"))
        pickle.dump(ts_id_l, open(f"{scalers_dir}/ts_id_l.pkl", "wb"))
        ## scale future covariates
        future_covariates_transformed = scale_covariates(
            future_covariates_split,
            store_dir=features_dir,
            filename_suffix="future_covariates_transformed.csv",
            scale=scale_covs,
            multiple=True,
            id_l=id_l_future_covs,
            ts_id_l=ts_id_l_future_covs
            )
        ## scale past covariates
        past_covariates_transformed = scale_covariates(
            past_covariates_split,
            store_dir=features_dir,
            filename_suffix="past_covariates_transformed.csv",
            scale=scale_covs,
            multiple=True,
            id_l=id_l_past_covs,
            ts_id_l=ts_id_l_past_covs
            )

        ######################
        # Model training
        print("\nTraining model...")
        logging.info("\nTraining model...")
        pl_trainer_kwargs = {"callbacks": [my_stopper],
                             "accelerator": 'auto',
                            #  "gpus": 1,
                            #  "auto_select_gpus": True,
                             "log_every_n_steps": 10}
        print("\nTraining on series:\n")
        logging.info("\nTraining on series:\n")
        if multiple:
            for i, series in enumerate(series_transformed['train']):
                print(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
                logging.info(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
        else:
            print(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")
            logging.info(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")
        print("")

        print("\Validating on series:\n")
        logging.info("\Validating on series:\n")
        if multiple:
            for i, series in enumerate(series_transformed['val']):
                print(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
                logging.info(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
        else:
            print(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")
            logging.info(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")

        #TODO maybe modify print to include split train based on nans
        #TODO make more efficient by also spliting covariates where the nans are split
            
        series_transformed['train'], past_covariates_transformed['train'], future_covariates_transformed['train'] = \
            split_nans(series_transformed['train'], past_covariates_transformed['train'], future_covariates_transformed['train'])
        ## choose architecture
        
        if darts_model in ['NHiTS', 'NBEATS', 'RNN', 'BlockRNN', 'TFT', 'TCN', 'Transformer']:
            darts_model = darts_model+"Model"
            
            print(f'\nTrained Model: {darts_model}')
            hparams_to_log = hyperparameters
            if 'learning_rate' in hyperparameters:
                hyperparameters['optimizer_kwargs'] = {'lr': hyperparameters['learning_rate']}
                del hyperparameters['learning_rate']

            if 'likelihood' in hyperparameters:
                hyperparameters['likelihood'] = eval(hyperparameters['likelihood']+"Likelihood"+"()")
            model = eval(darts_model)(
                save_checkpoints=True,
                log_tensorboard=False,
                model_name=mlrun.info.run_id,
                pl_trainer_kwargs=pl_trainer_kwargs,
                **hyperparameters
            )

            # for i, series in enumerate(series_transformed['val']):
            #     series.pd_dataframe().to_csv(f"{i}_val_partial_lgbm_ens")

            model.fit(series_transformed['train'],
                future_covariates=future_covariates_transformed['train'],
                past_covariates=past_covariates_transformed['train'],
                val_series=series_transformed['val'],
                val_future_covariates=future_covariates_transformed['val'],
                val_past_covariates=past_covariates_transformed['val'])

            logs_path = f"./darts_logs/{mlrun.info.run_id}"
            model_type = "pl"
            # TODO: Implement this step without tensorboard (fix utils.py: get_training_progress_by_tag)
            # log_curves(tensorboard_event_folder=f"./darts_logs/{mlrun.info.run_id}/logs", output_dir='training_curves')
        
        # Naive Models    
        elif darts_model == 'Naive':
            # Identify resolution
            daily_timesteps = int(24 * 60 // (pd.to_timedelta(series_transformed['train'][0].time_index[1]-series_transformed['train'][0].time_index[0]).seconds//60))
            seasonality_timesteps = daily_timesteps * int(hyperparameters['days_seasonality'])
            print(f'\nTrained Model: NaiveSeasonal, with seasonality (in timesteps): {seasonality_timesteps}') 

            hparams_to_log = hyperparameters
            # for ts in 

            model = NaiveSeasonal(K = seasonality_timesteps)
            model.fit(series_transformed['train'][-1])
            model_type = 'pkl'

        # LightGBM and RandomForest
        elif darts_model in ['LightGBM', 'RandomForest']:
            print(f'\nTrained Model: {darts_model}') 

            try:
                if "lags_future_covariates" in hyperparameters:
                    if truth_checker(str(hyperparameters["future_covs_as_tuple"])):
                        hyperparameters["lags_future_covariates"] = tuple(
                            hyperparameters["lags_future_covariates"])
                    hyperparameters.pop("future_covs_as_tuple")
            except:
                pass

            if future_covariates is None:
                hyperparameters["lags_future_covariates"] = None

            if past_covariates is None:
                hyperparameters["lags_past_covariates"] = None

            hparams_to_log = hyperparameters

            if darts_model == 'RandomForest':
                model = RandomForest(**hyperparameters)
            elif darts_model == 'LightGBM':
                model = LightGBMModel(**hyperparameters)

            print(f'\nTraining {darts_model}...')
            logging.info(f'\nTraining {darts_model}...')

            # for i, series in enumerate(series_transformed['train']):
            #     series.pd_dataframe().to_csv(f"{i}_series_partial_lgbm_ens")

            model.fit(
                series=series_transformed['train'],
                # val_series=series_transformed['val'],
                future_covariates=future_covariates_transformed['train'],
                past_covariates=past_covariates_transformed['train'],
                # val_future_covariates=future_covariates_transformed['val'],
                # val_past_covariates=past_covariates_transformed['val']
                )
            model_type = "pkl"

        elif darts_model == 'ARIMA':
            print(f'\nTrained Model: {darts_model}') 

            hparams_to_log = hyperparameters
            model = ARIMA(**hyperparameters)

            print(f'\nTraining {darts_model} on only the last non-nan subseries as it only supports single ts forecasting...')
            logging.info(f'\nTraining {darts_model} on only the last non-nan subseries as it only supports single ts forecasting...')

            model.fit(
                series=series_transformed['train'][-1],
                future_covariates=future_covariates_transformed['train'],
                )
            model_type = "pkl"
        
        if model_type == 'pkl':
            model_dir = tempfile.mkdtemp()
            pickle.dump(model, open(
                f"{model_dir}/_model.pkl", "wb"))
            logs_path = model_dir
            

        ######################
        # Log hyperparameters
        mlflow.log_params(hparams_to_log)

        ######################
        # Log artifacts
        target_dir = logs_path
        
        ## Move scaler in logs path
        if scale:
            source_dir = scalers_dir
            file_names = os.listdir(source_dir)
            for file_name in file_names:
                shutil.move(os.path.join(source_dir, file_name),
                target_dir)

        ## Create and move model info in logs path
        model_info_dict = {
            "darts_forecasting_model":  model.__class__.__name__,
            "run_id": mlrun.info.run_id
            }
        with open('model_info.yml', mode='w') as outfile:
            yaml.dump(
                model_info_dict,
                outfile,
                default_flow_style=False)

        shutil.move('model_info.yml', logs_path)

        ## Rename logs path to get rid of run name
        if model_type == 'pkl':
            logs_path_new = logs_path.replace(
            model_dir.split('/')[-1], mlrun.info.run_id)
            os.rename(logs_path, logs_path_new)
        elif model_type == 'pl':
            logs_path_new = logs_path

        ## Log MLflow model and code
        # if model_type == 'pl':
        mlflow.pyfunc.log_model(mlflow_model_root_dir,
                                loader_module="darts_flavor",
                                data_path=logs_path_new,
                                code_path=['../exceptions.py', '../utils.py', '../inference.py', '../darts_flavor.py'],
                                conda_env=mlflow_serve_conda_env)
        # elif model_type == 'pkl':
        #     mlflow.pyfunc.log_model(mlflow_model_root_dir,
        #                             loader_module="loader_module_pkl",
        #                             data_path=logs_path_new,
        #                             code_path=['utils.py', 'inference.py', 'loader_module_pkl.py'])

        ## Clean logs_path: Now it is necessary to avoid conflicts
        shutil.rmtree(logs_path_new)

        ######################
        # Set tags
        print("\nArtifacts are being uploaded to MLflow...")
        logging.info("\nArtifacts are being uploaded to MLflow...")
        mlflow.log_artifacts(features_dir, "features")

        if scale:
            # mlflow.log_artifacts(scalers_dir, f"{mlflow_model_path}/scalers")
            mlflow.set_tag(
                'scaler_uri',
                f'{mlrun.info.artifact_uri}/{mlflow_model_root_dir}/data/{mlrun.info.run_id}/scaler_series.pkl')
        else:
            mlflow.set_tag('scaler_uri', 'None')
        mlflow.set_tag(
            'ts_id_l_uri',
            f'{mlrun.info.artifact_uri}/{mlflow_model_root_dir}/data/{mlrun.info.run_id}/ts_id_l.pkl')
        mlflow.set_tag("run_id", mlrun.info.run_id)
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("model_type", model_type)

        mlflow.set_tag("darts_forecasting_model",
            model.__class__.__name__)

        mlflow.set_tag('series_uri',
            f'{mlrun.info.artifact_uri}/features/series.csv')

        if future_covariates is not None:
            mlflow.set_tag(
                'future_covariates_uri',
                f'{mlrun.info.artifact_uri}/features/future_covariates_transformed.csv')
        else:
            mlflow.set_tag(
                'future_covariates_uri',
                'None')

        if past_covariates is not None:
            mlflow.set_tag(
                'past_covariates_uri',
                f'{mlrun.info.artifact_uri}/features/past_covariates_transformed.csv')
        else:
            mlflow.set_tag('past_covariates_uri',
                'None')

        mlflow.set_tag(
            'setup_uri',
            f'{mlrun.info.artifact_uri}/features/split_info.yml')

        # model_uri
        mlflow.set_tag('model_uri', mlflow.get_artifact_uri(
            f"{mlflow_model_root_dir}/data/{mlrun.info.run_id}"))
        # inference_model_uri
        mlflow.set_tag('pyfunc_model_folder', mlflow.get_artifact_uri(
            f"{mlflow_model_root_dir}"))

        print("\nArtifacts uploaded.")
        logging.info("\nArtifacts uploaded.")
        return

if __name__ =='__main__':
    print("\n=========== TRAINING =============")
    logging.info("\n=========== TRAINING =============")
    mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    logging.info("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    train()
