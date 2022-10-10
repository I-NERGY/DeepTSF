import pretty_errors
from utils import ConfigParser, multiple_dfs_to_ts_file
import os
import pandas as pd
import yaml
from darts.dataprocessing.transformers import Scaler
import datetime

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

def split_dataset(covariates, val_start_date_str, test_start_date_str,
        test2_start_date_str=None, test_end_date=None, store_dir=None, name='series',
        conf_file_name='split_info.yml', multiple=False,
        country_l=[], country_code_l=[], opt_test=False):
    if covariates is not None:
        if not multiple:
            covariates = [covariates]
        covariates_train = []
        covariates_val = []
        covariates_test = []
        if opt_test: covariates_test2 = []
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
            if opt_test:
                if test_start_date_str == test2_start_date_str:
                    covariate_test2 = covariate_test
                else:
                    covariate_test, covariate_test2 = covariate_test.split_before(
                        pd.Timestamp(test2_start_date_str))
                covariates_test2.append(covariate_test2)
            covariates_train.append(covariate_train)
            covariates_test.append(covariate_test)
            covariates_val.append(covariate_val)

        if store_dir is not None:
            print("covariates", covariates)
            if opt_test:
                split_info = {
                    "val_start": val_start_date_str,
                    "test_start": test_start_date_str,
                    "test2_start": test2_start_date_str,
                    "test_end": covariates_test[0].time_index[-1].strftime('%Y%m%d')
                }
            else:
                split_info = {
                    "val_start": val_start_date_str,
                    "test_start": test_start_date_str,
                    "test_end": covariates_test[0].time_index[-1].strftime('%Y%m%d')
                }
            print(split_info)
            with open(f'{store_dir}/{conf_file_name}', 'w') as outfile:
                yaml.dump(split_info, outfile, default_flow_style=False)
            if not multiple:
                covariates[0].to_csv(f"{store_dir}/{name}.csv")
            else:
                multiple_dfs_to_ts_file(covariates, country_l, country_code_l, f"{store_dir}/{name}.csv")
        if not multiple:
            covariates_train = covariates_train[0]
            covariates_val = covariates_val[0]
            covariates_test = covariates_test[0]
            if opt_test: covariates_test2 = covariates_test2[0]
            covariates = covariates[0]
    else:
        covariates_train = None
        covariates_val = None
        covariates_test = None
        if opt_test: covariates_test2 =  None


    res = {"train": covariates_train,
            "val": covariates_val,
            "test": covariates_test,
            "all": covariates
           }
    if opt_test: res["test2"] = covariates_test2
    return res

def scale_covariates(covariates_split, store_dir=None, filename_suffix='', scale=True, multiple=False, country_l=[], country_code_l=[],  opt_test=False):
    covariates_train = covariates_split['train']
    covariates_val = covariates_split['val']
    covariates_test = covariates_split['test']
    covariates = covariates_split['all']
    if opt_test: covariates_test2 = covariates_split['test2']
    print("SCALEEEEEEE",covariates)
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
                if opt_test:
                    covariates_test2_transformed = \
                        transformer.transform(covariates_test2, n_jobs=-1)
                covariates_transformed = \
                    transformer.transform(covariates, n_jobs=-1)
                transformers = transformer
            else:
                print("baaad")
                transformers = []
                covariates_train_transformed = []
                covariates_val_transformed = []
                covariates_test_transformed = []
                if opt_test:
                    covariates_test2_transformed = []
                else:
                    covariates_test2 = covariates_test
                covariates_transformed = []
                for covariate_train, covariate_val, covariate_test, covariate_test2, covariate in \
                zip(covariates_train, covariates_val, covariates_test, covariates_test2, covariates):
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
                    if opt_test:
                        covariates_test2_transformed.append(\
                            transformer.transform(covariate_test2, n_jobs=-1))
                    covariates_transformed.append(\
                        transformer.transform(covariate, n_jobs=-1))
                    transformers.append(transformer)

        else:
            # To avoid scaling
            covariates_train_transformed = covariates_train
            covariates_val_transformed = covariates_val
            covariates_test_transformed = covariates_test
            if opt_test: covariates_test2_transformed = covariates_test2
            covariates_transformed = covariates
            transformer = None

        if store_dir is not None:
            if not multiple:
                print(covariates_transformed)
                covariates_transformed.to_csv(
                    f"{store_dir}/{filename_suffix}")
            else:
                multiple_dfs_to_ts_file(covariates_transformed, country_l, country_code_l, f"{store_dir}/{filename_suffix}")


        res = {"train": covariates_train_transformed,
                "val": covariates_val_transformed,
                "test": covariates_test_transformed,
                "all": covariates_transformed,
                "transformer": transformers
                }
        if opt_test: res["test2"] = covariates_test2_transformed
        return res
    else:
        res = {"train":None,
                "val": None,
                "test": None,
                "all": None,
                "transformer": None
                }
        if opt_test: res["test2"] = None
        return res
