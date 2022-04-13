import pretty_errors
from utils import ConfigParser
import os
import pandas as pd
import yaml
from darts.dataprocessing.transformers import Scaler

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

def split_dataset(covariates, val_start_date_str, test_start_date_str, 
    store_dir=None, name='series', conf_file_name='split_info.yml'):
    if covariates is not None:
        covariates_train, covariates_val = covariates.split_before(
            pd.Timestamp(val_start_date_str))
        covariates_val, covariates_test = covariates_val.split_before(
            pd.Timestamp(test_start_date_str))
        if store_dir is not None:
            split_info = {
                "val_start": val_start_date_str,
                "test_start": test_start_date_str,
                "test_end": covariates.time_index[-1].strftime('%Y%m%d')
            }
            with open(f'{store_dir}/{conf_file_name}', 'w') as outfile:
                yaml.dump(split_info, outfile, default_flow_style=False)
            covariates.to_csv(f"{store_dir}/{name}.csv")
    else:
        covariates_train = None
        covariates_val = None
        covariates_test = None
        
    return {"train":covariates_train,
            "val": covariates_val,
            "test": covariates_test,
            "all": covariates 
           }

def scale_covariates(covariates_split, store_dir=None, filename_suffix=''):
    covariates_train = covariates_split['train'] 
    covariates_val = covariates_split['val']
    covariates_test = covariates_split['test'] 
    covariates = covariates_split['all'] 
    if covariates is not None:
        # scale them between 0 and 1:
        transformer = Scaler()
        # TODO: future covariates are a priori known! 
        # i can fit on all dataset, but I won't do it as this function works for all covariates! 
        # this is a problem only if not a full year is contained in the training set
        covariates_train_transformed = \
            transformer.fit_transform(covariates_train) 
        covariates_val_transformed = \
            transformer.transform(covariates_val)
        covariates_test_transformed = \
            transformer.transform(covariates_test)
        covariates_transformed = \
            transformer.transform(covariates)
        if store_dir is not None:
            covariates_transformed.to_csv(
                f"{store_dir}/{filename_suffix}")

        return {"train":covariates_train_transformed,
                "val": covariates_val_transformed,
                "test": covariates_test_transformed,
                "all": covariates_transformed, 
                "transformer": transformer
                }
    else:
        return {"train":None,
                "val": None,
                "test": None,
                "all": None,
                "transformer": None 
                }
