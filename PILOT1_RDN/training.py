from utils import none_checker, ConfigParser, download_online_file, load_local_csv_as_darts_timeseries, log_curves, truth_checker
from preprocessing import scale_covariates, split_dataset

# the following are used through eval(darts_model + 'Model')
import darts
from darts.models import RNNModel, BlockRNNModel, NBEATSModel, TFTModel, NaiveDrift, NaiveSeasonal, TCNModel
from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.gradient_boosted_model import LightGBMModel
from darts.models.forecasting.random_forest import RandomForest
from darts.utils.likelihood_models import ContinuousBernoulliLikelihood, GaussianLikelihood, DirichletLikelihood, ExponentialLikelihood, GammaLikelihood, GeometricLikelihood
from darts.metrics import mape as mape_darts

import mlflow
import click
import os
import shutil
import torch
import logging
import pickle
import tempfile
import pretty_errors
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
# a period of 5 epochs (`patience`)
my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=0.00,
    mode='min',
)

pl_trainer_kwargs = {"callbacks": [my_stopper],
                     "accelerator": "gpu", 
                     "gpus": -1, 
                     "auto_select_gpus": True,
                     "log_every_n_steps": 20}

@click.command()
@click.option("--series-csv",
              type=str,
              default="../../RDN/Load_Data/series.csv",
              help="Local timeseries csv. If set, it overwrites the local value."
              )
@click.option("--series-uri",
              type=str,
              default='mlflow_artifact_uri',
              help="Remote timeseries csv file. If set, it overwrites the local value."
              )
@click.option("--future-covs-csv",
              type=str,
              default='None'
              )
@click.option("--future-covs-uri",
              type=str,
              default='mlflow_artifact_uri'
              )
@click.option("--past-covs-csv",
              type=str,
              default='None'
              )
@click.option("--past-covs-uri",
              type=str,
              default='mlflow_artifact_uri'
              )
@click.option("--darts-model",
              type=click.Choice(
                  ['NBEATS',
                   'TCN',
                   'RNN',
                   'BlockRNN',
                   'TFT',
                   'LightGBM',
                   'RandomForest',
                   'Naive',
                   'AutoARIMA']),
              multiple=False,
              default='RNN',
              help="The base architecture of the model to be trained"
              )
@click.option("--hyperparams-entrypoint", "-h",
              type=str,
              default='LSTM1',
              help=""" The entry point of config.yml under the 'hyperparams'
              one containing the desired hyperparameters for the selected model"""
              )
@click.option("--cut-date-val",
              type=str,
              default='20200101',
              help="Validation set start date [str: 'YYYYMMDD']"
              )
@click.option("--cut-date-test",
              type=str,
              default='20210101',
              help="Test set start date [str: 'YYYYMMDD']",
              )
@click.option("--test-end-date",
              type=str,
              default='None',
              help="Test set ending date [str: 'YYYYMMDD']",
              )
@click.option("--device",
              type=click.Choice(
                  ['cuda', 
                   'cpu']),
              multiple=False,
              default='cuda',
              )
def train(series_csv, series_uri, future_covs_csv, future_covs_uri, 
          past_covs_csv, past_covs_uri, darts_model, 
          hyperparams_entrypoint, cut_date_val, cut_date_test,
          test_end_date, device):
    
    # Argument preprocessing

    ## test_end_date
    test_end_date = none_checker(test_end_date) 

    ## hyperparameters   
    hyperparameters = ConfigParser().read_hyperparameters(hyperparams_entrypoint)

    ## device
    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
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
    if darts_model in ["NBEATS", "BlockRNN", "TCN"]:
        """They do not accept future covariates as they predict blocks all together.
        They won't use initial forecasted values to predict the rest of the block 
        So they won't need to additionally feed future covariates during the recurrent process.
        """
        past_covs_csv = future_covs_csv
        future_covs_csv = None

    elif darts_model in ["RNN"]:
        """Does not accept past covariates as it needs to know future ones to provide chain forecasts
        its input needs to remain in the same feature space while recurring and with no future covariates
        this is not possible. The existence of past_covs is not permitted for the same reason. The 
        feature space will change during inference. If for example I have current temperature and during 
        the forecast chain I only have time covariates, as I won't know the real temp then a constant \
        architecture like LSTM cannot handle this"""
        future_covs_csv = past_covs_csv
        past_covs_csv = None
    #elif: extend for other models!! (time_covariates are always future covariates, but some models can't handle them as so)

    future_covariates = none_checker(future_covs_csv)
    past_covariates = none_checker(past_covs_csv)

    with mlflow.start_run(run_name=f'train_{darts_model}', nested=True) as mlrun:
        ######################
        # Load series and covariates datasets
        series = load_local_csv_as_darts_timeseries(
                local_path=series_csv, 
                name='series', 
                time_col='Date', 
                last_date=test_end_date)
        # print(series.time_index)
        # a = pd.date_range(start='20180101 00:00:00',
        #               end='20211212 23:00:00').difference(series.time_index)
        # print(f'\n\n {a}')
        if future_covariates is not None:
            future_covariates = load_local_csv_as_darts_timeseries(
                local_path=future_covs_csv, 
                name='future covariates', 
                time_col='Date', 
                last_date=test_end_date)
        if past_covariates is not None:
            past_covariates = load_local_csv_as_darts_timeseries(
                local_path=past_covs_csv, 
                name='past covariates', 
                time_col='Date', 
                last_date=test_end_date)

        print("\nCreating local folder to store the scaler as pkl...")
        logging.info("\nCreating local folder to store the scaler as pkl...")
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
            store_dir=features_dir, 
            name='series',
            conf_file_name='split_info.yml')
        ## future covariates
        future_covariates_split = split_dataset(
            future_covariates, 
            val_start_date_str=cut_date_val, 
            test_start_date_str=cut_date_test, 
            # store_dir=features_dir,
            name='future_covariates')
        ## past covariates
        past_covariates_split = split_dataset(
            past_covariates, 
            val_start_date_str=cut_date_val, 
            test_start_date_str=cut_date_test,
            # store_dir=features_dir, 
            name='past_covariates')

        ######################
        # Scaling
        print("\nScaling...")
        logging.info("\nScaling...")

        ## scale series
        scaled_series = scale_covariates(
            series_split, 
            store_dir=features_dir, 
            filename_suffix="series_transformed.csv")
        pickle.dump(scaled_series["transformer"], open(f"{scalers_dir}/scaler_series.pkl", "wb"))
        ## scale future covariates
        scaled_future_covariates = scale_covariates(
            future_covariates_split, 
            store_dir=features_dir, 
            filename_suffix="future_covariates_transformed.csv")
        ## scale past covariates
        scaled_past_covariates = scale_covariates(
            past_covariates_split,
            store_dir=features_dir, 
            filename_suffix="past_covariates_transformed.csv")

        ######################
        # Save scaled features and scalers locally and then to mlflow server
        print("\nDatasets and scalers are being uploaded to MLflow...")
        logging.info("\nDatasets and scalers are being uploaded to MLflow...")
        mlflow.log_artifacts(scalers_dir, "scalers")
        mlflow.log_artifacts(features_dir, "features")
        print("\nDatasets uploaded. ...")
        logging.info("\nDatasets uploaded. ...")

        ######################
        # Model training
        print("\nTraining model...")
        logging.info("\nTraining model...")

        ## choose architecture
        if darts_model in ['NBEATS', 'RNN', 'BlockRNN', 'TFT', 'TCN']:
            hparams_to_log = hyperparameters
            if 'learning_rate' in hyperparameters:
                hyperparameters['optimizer_kwargs'] = {'lr': hyperparameters['learning_rate']}
                del hyperparameters['learning_rate']

            if 'likelihood' in hyperparameters:
                hyperparameters['likelihood'] = eval(hyperparameters['likelihood']+"Likelihood"+"()")

            model = eval(darts_model + 'Model')(
                save_checkpoints=True,
                log_tensorboard=False,
                model_name=mlrun.info.run_id,
                pl_trainer_kwargs=pl_trainer_kwargs,
                **hyperparameters
            )                
            ## fit model
            # try:
            # print(scaled_series['train'])
            # print(scaled_series['val'])
            model.fit(scaled_series['train'],
                future_covariates=scaled_future_covariates['train'],
                past_covariates=scaled_past_covariates['train'],
                val_series=scaled_series['val'],
                val_future_covariates=scaled_future_covariates['val'],
                val_past_covariates=scaled_past_covariates['val'])

            # TODO: Package Models as python functions for MLflow (see RiskML and https://mlflow.org/docs/0.5.0/models.html#python-function-python-function)
            print('\nStoring torch model to MLflow...')
            logging.info('\nStoring torch model to MLflow...')
            # model_dir_list = os.listdir(f"./.darts/checkpoints/{mlrun.info.run_id}")
            # best_model_name = [fname for fname in model_dir_list if "model_best" in fname][0]
            # best_model_path = f"./darts_logs/{mlrun.info.run_id}/_model.pth.tar"
            logs_path = f"./darts_logs/{mlrun.info.run_id}/"
            mlflow.log_artifacts(logs_path)
            # mlflow.log_artifact(best_model_path)
            
            # TODO: Implement this step without tensorboard (fix utils.py: get_training_progress_by_tag)
            # log_curves(tensorboard_event_folder=f"./darts_logs/{mlrun.info.run_id}/logs", output_dir='training_curves')

            # TODO: Implement early stopping without keyboard interupt ?? (consider tags as well)
            # except KeyboardInterrupt:
            #     # TODO: Package Models as python functions for MLflow (see RiskML and https://mlflow.org/docs/0.5.0/models.html#python-function-python-function)
            #     model_dir_list = os.listdir(f"./.darts/checkpoints/{mlrun.info.run_id}")
            #     best_model_name = [fname for fname in model_dir_list if "model_best" in fname][0]
            #     best_model_path = f"./.darts/checkpoints/{mlrun.info.run_id}/{best_model_name}"
            #     mlflow.log_artifact(best_model_path, f"checkpoints")
            #     log_curves(tensorboard_event_folder=f"./.darts/runs/{mlrun.info.run_id}", 
            #     output_dir='training_curves')
            #     mlflow.log_param("status", "forced_stop")                
        
        # LightGBM and RandomForest
        elif darts_model in ['LightGBM', 'RandomForest']:

            if "lags_future_covariates" in hyperparameters:
                if truth_checker(str(hyperparameters["future_covs_as_tuple"])):
                    hyperparameters["lags_future_covariates"] = tuple(
                        hyperparameters["lags_future_covariates"])
                hyperparameters.pop("future_covs_as_tuple")
            
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

            model.fit(
                series=scaled_series['train'],
                # val_series=scaled_series['val'],
                future_covariates=scaled_future_covariates['train'],
                past_covariates=scaled_past_covariates['train'], 
                # val_future_covariates=scaled_future_covariates['val'],
                # val_past_covariates=scaled_past_covariates['val']
                )

            print('\nStoring the model as pkl to MLflow...')
            logging.info('\nStoring the model as pkl to MLflow...')
            forest_dir = tempfile.mkdtemp()

            pickle.dump(model, open(
                f"{forest_dir}/model_best_{darts_model}.pkl", "wb"))
            mlflow.log_artifacts(forest_dir, "checkpoints")
        
        # ARIMA
        elif darts_model == 'AutoARIMA':
            hparams_to_log={}
            print("\nEmploying AutoARIMA - Ignoring hyperparameters")
            logging.info("\nEmploying AutoARIMA - Ignoring hyperparameters")
            model = AutoARIMA(random_state=0, information_criterion='bic')
            model.fit(scaled_series['train'].append(scaled_series['val']), future_covariates=future_covariates)
            # hparams = model.get_params()
            # print(model.summary())

            print("\nStoring the model as pkl...")
            logging.info("\nStoring the model as pkl...")
            arima_dir = tempfile.mkdtemp()

            pickle.dump(model, open(
                f"{arima_dir}/model_best_arima.pkl", "wb"))
            mlflow.log_artifacts(arima_dir, "checkpoints")


            # hyperparameters = {k: eval(v) for (k, v) in hyperparameters.items()}
            # model = ARIMA().gridsearch(
            #     parameters=hyperparameters, 
            #     series=series_split['train'].append(series_split['val']),
            #     past_covariates=past_covariates,
            #     future_covariates=future_covariates,
            #     forecast_horizon=hyperparameters[forecast_horizon],
            #     stride=hyperparameters[forecast_horizon],
            #     start= cut_date_val,
            #     last_points_only=False,
            #     metric=mape_darts,
            #     n_jobs=-1,
            #     verbose=True
            #     )

        # Log hyperparameters
        mlflow.log_params(hparams_to_log)

        # Set tags
        mlflow.set_tag("run_id", mlrun.info.run_id)
        mlflow.set_tag("stage", "training")

        client = mlflow.tracking.MlflowClient()
        model_dir_list = client.list_artifacts(run_id=mlrun.info.run_id)
        src_path = [fileinfo.path for fileinfo in model_dir_list if '_model' in fileinfo.path][0]
        mlflow.set_tag('model_uri', mlflow.get_artifact_uri(src_path))

        mlflow.set_tag('series_uri', f'{mlrun.info.artifact_uri}/features/series.csv')

        # check if future_covariates exist
        if future_covariates is not None:
            mlflow.set_tag('future_covariates_uri', f'{mlrun.info.artifact_uri}/features/future_covariates_transformed.csv')
        else:
            mlflow.set_tag('future_covariates_uri', 'mlflow_artifact_uri')

        # check if past_covariates exist
        if past_covariates is not None:
            mlflow.set_tag('past_covariates_uri', f'{mlrun.info.artifact_uri}/features/past_covariates_transformed.csv')
        else:
            mlflow.set_tag('past_covariates_uri', 'mlflow_artifact_uri')

        mlflow.set_tag('scaler_uri', f'{mlrun.info.artifact_uri}/scalers/scaler_series.pkl')
        mlflow.set_tag('setup_uri', f'{mlrun.info.artifact_uri}/features/split_info.yml')
        
        return

if __name__ =='__main__':
    print("\n=========== TRAINING =============")
    logging.info("\n=========== TRAINING =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    logging.info("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    train()
