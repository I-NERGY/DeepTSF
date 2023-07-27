# DeepTSF

Repository for DeepTSF timeseries forecasting tool.  

## To create conda environment file

```conda env export conda.yaml```

If it is packaged from windows to linux do the following instead:

```conda env export --no-builds > conda.yaml```

And remember to remove any windows related dependencies such as:
- vc
- wincertstore
- vs2015_runtime
- win_inet_pton
- pywin32

## Reproduce the conda environment manually
```conda env create -f conda.yaml```

## Reproduce the conda environment automatically
Run any 'mlflow run' command *without* the option `--env-manager=local`

## Inference example (replace uris accordingly)
```mlflow run --experiment-name trash --entry-point inference . --env-manager=local -P series_uri=runs:/bd0c727f76a849b48824daa7147f4b82/artifacts/features/series.csv -P pyfunc_model_folder=runs:/bd0c727f76a849b48824daa7147f4b82/pyfunc_model```  

or  

```python inference.py --pyfunc-model-folder runs:/bd0c727f76a849b48824daa7147f4b82/pyfunc_model --series-uri runs:/bd0c727f76a849b48824daa7147f4b82/artifacts/features/series.csv```  

or  

```python inference.py --pyfunc-model-folder s3://mlflow-bucket/2/bd0c727f76a849b48824daa7147f4b82/artifacts/pyfunc_model --series-uri s3://mlflow-bucket/2/bd0c727f76a849b48824daa7147f4b82/artifacts/features/series.csv```

## Full pipeline example
```mlflow run --experiment-name 2009-2019 --entry-point exp_pipeline . -P hyperparams_entrypoint=nbeats0_10 -P darts_model=NBEATS -P ignore_previous_runs=t -P cut_date_val=20180101 -P cut_date_test=20190101 -P year_range=2009-2019 -P time_covs=None --env-manager=local```

## Multiple timeseries suport
This pipeline supports training with multiple timeseries, which should be provided in a csv file using the following file format (along with example values):

    Index | Day         | ID | Country | Country Code | 00:00:00 | 00:00:00 + resolution | ... | 24:00:00 - resolution
    0     | 2015-04-09  | 0  | Portugal| PT           | 5248     | 5109                  | ... | 5345
    1     | 2015-04-09  | 1  | Spain   | ES           | 25497    | 23492                 | ... | 25487
    .
    .

Columns can be in any order and ID must alwaws be convertible to an int, and consequtive. Also, all the above
column names must be present in the file, and the hour columns must be consequtive and separated by resolution 
minutes. The lines can be at any order as long as the Day column is increasing for each country.

The parameters of MLflow that the user can set are the following:

* ```series_csv``` (default series.csv), the path to the local time series file to use. If series_uri has a non-default value, or if from_mongo is true, then series_csv has no effect.

* ```series_uri``` (default online_artifact), the uri of the online time series file to use. If series_uri is not online_artifact, and from_mongo is false, then this is the time series DeepTSF will use. 

* ```past_covs_csv``` (default None), the path to the local time series file to use as past covariates. If past_covs_uri is not None, then this has no effect.

* ```past_covs_uri``` (default None), the uri of the online time series file to use as past covariates. If past_covs_uri is not None, then this is the file DeepTSF will use as past covariates.

* ```future_covs_csv``` (default None), the path to the local time series file to use as future covariates. If future_covs_uri is not None, then this has no effect.

* ```future_covs_uri``` (default None), the uri of the online time series file to use as future covariates. If future_covs_uri is not None, then this is the file DeepTSF will use as future covariates.

TODO: Check that
* ```resolution``` (default 15), the resolution that all datasets will use. If this is not the resolution of a time series, then it is resampled to use that resolution. In case of single timeseries, all prepprocessing is done in this resolution. In other words resampling is done before prosocessing. In case of multiple timeseries however, the resolution is infered from load_raw_data. All preprosessing is done using the infered resolution and this afterwards resampling is performed. 

TODO: this should be for covariates also
* ```year_range``` (default 2009-2019), the years to use from the datasets (inclusive). All values outside of those dates will be dropped.

* ```time_covs``` (default false), whether to add time covariates to the time series. If true, then the following time covariates will be added as future covariates:
    * The month
    * The day of the year
    * The hour
    * The day of the week
    * The week of the year
    * Whether its a holiday or not

* ```darts_model``` (default RNN), the base architecture of the model to be trained. The possible options are:
    * NBEATS
    * NHiTS
    * Transformer
    * RNN
    * TCN
    * BlockRNN
    * TFT
    * LightGBM
    * RandomForest
    * Naive
    TODO ???
    * AutoARIMA

* ```hyperparams_entrypoint``` (default LSTM1), the entry point containing the desired hyperparameters for the selected model. The file that will be searched for the entrypoint will be config.yml if opt_test is false, and config_opt.yml otherwise. More info for the required file format below
TODO Where?

* ```cut_date_val``` (default 20190101), the validation set start date. All values before that will be the training series. Format: str, 'YYYYMMDD'

TODO Test inclusive or not

* ```cut_date_test``` (default 20200101), the test set start date. Values between that and cut_date_val will be the validation series. If cut_date_test = cut_date_test, then the test and validation sets will be the same (from cut_date_test to test_end_date). Format: str, 'YYYYMMDD'

* ```test_end_date``` (default None), the test set ending date. Values between that and cut_date_test will be the testing series. All values after that will be ignored. If None, all the timeseries from cut_date_test will be the test set. Format: str, 'YYYYMMDD'

* ```device``` (default gpu), whether to run the pipeline on the gpu, or just use the cpu. 

* ```forecast_horizon``` (default 96), the number of timesteps that the model being evaluated is going to predict in each step of backtesting.

* ```stride``` (default None), the number of time steps between two consecutive steps of backtesting. If it is None, then stride = forecast_horizon

* ```retrain``` (default false), whether to retrain model during backtesting

TODO: ??? more info on that

* ```ignore_previous_runs``` (default true), whether to ignore previous step runs while running the pipeline

* ```scale``` (default true), whether to scale the target series

* ```scale_covs``` (default true), whether to scale the covariates

* ```day_first``` (default true), whether the date has the day before the month in timeseries file.

TODO: Check oti ontws einai country code kai oxi country
* ```country``` (default PT), the country code this dataset belongs to

* ```std_dev``` (default 4.5), argument of the outlier detection method. It is the number to be multiplied with the standard deviation of each 1 month period of the dataframe. The result is then used as a cut-off value.

* ```max_thr``` (default -1), argument of the imputation method. If there is a consecutive subseries of NaNs longer than max_thr, then it is not imputed and returned with NaN values. If it is -1, every value will be imputed regardless of how long the consecutive subseries of NaNs it belongs to is.

TODO More about methods here
* ```a``` (default 0.3), argument of the imputation method.
It is the weight that shows how quickly simple interpolation's weight decreases as the distacne to the nearest non NaN value increases.

TODO Why 0.000694?
* ```wncutoff``` (default 0.000694), argument of the imputation method. Historical data will only take into account dates that have at most wncutoff distance from the current null value's WN (Week Number). 

* ```ycutoff``` (default 3), argument of the imputation method. Historical data will only take into account dates that have at most ycutoff distance from the current null value's year.

* ```ydcutoff``` (default 30), argument of the imputation method. Historical data will only take into account dates that have at most ydcutoff distance from the current null value's yearday .

* ```shap_data_size``` (default 10), The size of shap dataset in samples. The SHAP coefficients are going to be computed for this number of random samples of the test dataset. If it is a float, it represents the proportion of samples of the test dataset that will be chosen. If it is an int, it represents the absolute number of samples to be produced.

TODO All models supported, fix documentation

* ```shap_data_size``` (default false), whether to do SHAP analysis on the model.

* ```multiple``` (default false), whether to train on multiple timeseries. This applies to the main time series. Covariates can be multivariate, but the number of time series must be the same as the main time series. The only exception to this is if we have multiple time series and a single past or future covariate. In this case, we consider this series to be the covariate to all the main time series.

TODO change to PT
* ```eval_series``` (default PT), on which country to run the backtesting.Only for multiple timeseries

* ```n_trials``` (default 100), how many trials optuna will run. If we run a simple grid search, this might be bigger than the possible number of parameter combinations to be tested. In this case, optuna will only run the maximum possible number of combinations.

* ```opt_test``` (default false), whether we are running optuna or not. Also, DeepTSF will check config_opt.yml for the model parameters if this is true, and config.yml otherwise.

* ```from_mongo``` (default false), whether to read the dataset from mongodb, or from other sources. If this is true, it overrides all other options (series_csv, series_uri)

* ```mongo_name``` (default rdn_load_data), which mongo file to read

* ```num_workers``` (default 4), number of threads that will be used by pytorch

-- eval method

* ```l_interpolation``` (default false), whether to only use linear interpolation, or use the imputation method described above

* ```rmv_outliers``` (default true), whether to remove outliers or not

* ```loss_function``` (default mape), loss function to use as objective function for optuna. Possible options are 

TODO Write files and what is saved
* ```evaluate_all_ts``` (default false), whether to validate the models for all timeseries, and return the mean of their metrics . Only applicable for multiple time series.

* ```convert_to_local_tz``` (default false), whether to convert to local time. ??????If we have a multiple time series file, ID column is considered as the country to transform each time series' time to. If this is not a country code, then country argument is used.

* ```grid_search``` (default false), whether to run an exhaustive grid search or use tpe method in optuna.

???--input-chunk-length

* ```grid_search``` (default None), if not None, only ts with this id will be used for training and evaluation. Applicable only on multiple ts files

* ```m_mase``` (default 1), the forecast horizon of the naive method used in MASE

* ```min_non_nan_interval``` (default 24), if after imputation there exist continuous intervals of non nan values that are smaller than min_non_nan_interval hours, these intervals are all replaced by nan values

* ```num_samples``` (default 1), number of samples to use for evaluating/validating a probabilistic model's output