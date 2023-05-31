import logging
import pretty_errors
from utils import ConfigParser, multiple_dfs_to_ts_file
import os
import pandas as pd
import yaml
from darts.dataprocessing.transformers import Scaler
import datetime
from darts.utils.missing_values import extract_subseries
# get environment variables
from dotenv import load_dotenv
load_dotenv()
from exceptions import NanInSet
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

def split_nans(covariates, past_covs, future_covs):
    result = []
    past_covs_return = [] if past_covs != None else None
    future_covs_return = [] if future_covs != None else None
    for i, covariate in enumerate(covariates):
        if covariate.pd_dataframe().isnull().sum().sum() > 0:
            covariate = extract_subseries(covariate, min_gap_size=1)#, mode='any') TODO update darts!!

            print(f"Spliting train into {len(covariate)} consecutive series\n")
            logging.info(f"Spliting train into {len(covariate)} consecutive series\n")
            result.extend(covariate)
            if past_covs != None:
                past_covs_return.extend([past_covs[i] for _ in range(len(covariate))])
            if future_covs != None:
                future_covs_return.extend([future_covs[i] for _ in range(len(covariate))])
        else:
            result.append(covariate)
            if past_covs != None:
                past_covs_return.append(past_covs[i])
            if future_covs != None:
                future_covs_return.append(future_covs[i])
    return result, past_covs_return, future_covs_return

def split_dataset(covariates, val_start_date_str, test_start_date_str,
        test_end_date=None, store_dir=None, name='series',
        conf_file_name='split_info.yml', multiple=False,
        source_l=[], source_code_l=[], id_l=[], ts_id_l=[]):
    if covariates is not None:
        if not multiple:
            covariates = [covariates]
        covariates_train = []
        covariates_val = []
        covariates_test = []
        covariates_return = []
        for covariate in covariates:                
            if test_end_date is not None and covariate.time_index[-1].strftime('%Y%m%d') > test_end_date:
                covariate = covariate.drop_after(
                    pd.Timestamp(test_end_date) + datetime.timedelta(days=1))

            covariate_train, covariate_val = covariate.split_before(
                pd.Timestamp(val_start_date_str))

            if val_start_date_str == test_start_date_str:
                covariate_test = covariate_val
            else:
                covariate_val, covariate_test = covariate_val.split_before(
                    pd.Timestamp(test_start_date_str))
            
            print(covariate_val)
            if covariate_val.pd_dataframe().isnull().sum().sum() > 0:
                print(f"Validation set can not have any nan values\n")
                logging.info(f"Validation set can not have any nan values\n")
                raise NanInSet()
            if covariate_test.pd_dataframe().isnull().sum().sum() > 0:
                print(f"Test set can not have any nan values\n")
                logging.info(f"Test set can not have any nan values\n")
                raise NanInSet()
            covariates_train.append(covariate_train)
            covariates_test.append(covariate_test)
            covariates_val.append(covariate_val)
            covariates_return.append(covariate)

        if store_dir is not None:
            #print("covariates", covariates)
            split_info = {
                "val_start": val_start_date_str,
                "test_start": test_start_date_str,
                "test_end": covariates_test[0].time_index[-1].strftime('%Y%m%d')
            }
            #print(split_info)
            with open(f'{store_dir}/{conf_file_name}', 'w') as outfile:
                yaml.dump(split_info, outfile, default_flow_style=False)
            if not multiple:
                covariates_return[0].to_csv(f"{store_dir}/{name}.csv")
            else:
                multiple_dfs_to_ts_file(covariates_return, source_l, source_code_l, id_l, ts_id_l, f"{store_dir}/{name}.csv")
        if not multiple:
            covariates_train = covariates_train[0]
            covariates_val = covariates_val[0]
            covariates_test = covariates_test[0]
            covariates_return = covariates_return[0]
    else:
        covariates_train = None
        covariates_val = None
        covariates_test = None
        covariates_return = None


    return {"train": covariates_train,
            "val": covariates_val,
            "test": covariates_test,
            "all": covariates_return
           }

def scale_covariates(covariates_split, store_dir=None, filename_suffix='', scale=True, multiple=False, source_l=[], source_code_l=[], id_l=[], ts_id_l=[]):
    covariates_train = covariates_split['train']
    covariates_val = covariates_split['val']
    covariates_test = covariates_split['test']
    covariates = covariates_split['all']
    #print("SCALEEEEEEE",covariates)
    if covariates is not None:
        if scale:
            if not multiple:
                # scale them between 0 and 1:
                transformer = Scaler()
                # TODO: future covariates are a priori known!
                # i can fit on all dataset, but I won't do it as this function works for all covariates!
                # this is a problem only if not a full year is contained in the training set
                covariates_train_transformed = \
                    transformer.fit_transform(covariates_train, n_jobs=-1)
                covariates_val_transformed = \
                    transformer.transform(covariates_val, n_jobs=-1)
                covariates_test_transformed = \
                    transformer.transform(covariates_test, n_jobs=-1)
                covariates_transformed = \
                    transformer.transform(covariates, n_jobs=-1)
                transformers = transformer
            else:
                transformers = []
                covariates_train_transformed = []
                covariates_val_transformed = []
                covariates_test_transformed = []
                covariates_transformed = []
                for covariate_train, covariate_val, covariate_test, covariate in \
                zip(covariates_train, covariates_val, covariates_test, covariates):
                    transformer = Scaler()
                    # TODO: future covariates are a priori known!
                    # i can fit on all dataset, but I won't do it as this function works for all covariates!
                    # this is a problem only if not a full year is contained in the training set
                    covariates_train_transformed.append(\
                        transformer.fit_transform(covariate_train, n_jobs=-1))
                    covariates_val_transformed.append(\
                        transformer.transform(covariate_val, n_jobs=-1))
                    covariates_test_transformed.append(\
                        transformer.transform(covariate_test, n_jobs=-1))
                    covariates_transformed.append(\
                        transformer.transform(covariate, n_jobs=-1))
                    transformers.append(transformer)

        else:
            # To avoid scaling
            covariates_train_transformed = covariates_train
            covariates_val_transformed = covariates_val
            covariates_test_transformed = covariates_test
            covariates_transformed = covariates
            transformers = None

        if store_dir is not None:
            if not multiple:
                #print(covariates_transformed)
                covariates_transformed.to_csv(
                    f"{store_dir}/{filename_suffix}")
            else:
                multiple_dfs_to_ts_file(covariates_transformed, source_l, source_code_l, id_l, ts_id_l, f"{store_dir}/{filename_suffix}")


        return {"train": covariates_train_transformed,
                "val": covariates_val_transformed,
                "test": covariates_test_transformed,
                "all": covariates_transformed,
                "transformer": transformers
                }
    else:
        return {"train":None,
                "val": None,
                "test": None,
                "all": None,
                "transformer": None
                }