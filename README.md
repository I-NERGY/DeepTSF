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

The stages of the pipeline, along with the MLflow parameters that are related to each one are presented below:

## Data loading

### Description of this step
Firstly, the dataset is loaded from local or online sources. Currently Deep-TSF supports csv files of the schema that is discussed in section File format. In this context, the connectors that enable data ingestion vary depending on the use case and the schema of the respective data source and shall be engineered by the DeepTSF user. We provide an example of this connector, which works with MongoDB (function load_data_to_csv in load_raw_data.py for each use case). After that, validation is performed to ensure that the files provided by the user respect the required schema. The files are saved on the MLflow tracking server so that they are available for the data pre-processing stage.

### Input file format
The format of the csv files DeepTSF can accept depends on the nature of the problem it is trying to solve. More specifically, in case of a single time series file, its format is:

|Datetime | Value|
|- | -| 
|2015-04-09 00:00:00 | 7893 |
|2015-04-09 01:00:00 | 8023 |
|2015-04-09 02:00:00 | 8572 |
|... |... | 


In this table, the Datetime column simply stores the dates and times of each observation, and the Value column stores the value that has been observed.


If we are solving a multiple and / or multivariate time series problem, then the file format (along with example values) is:

|Index | Date | ID | Timeseries ID | 00:00:00 | ... |
|-|-|-|-|-|-|
0  | 2015-04-09  | PT  | PT | 5248 | ... |
1  | 2015-04-09  | ES  | ES | 25497 | ...|
... | ... | ...  | ... | ... | ...

The columns that can be present in the csv have the following meaning
- Index: Simply a monotonic integer range
- Date: The Date each row is referring to
- ID: Each ID corresponds to a component of a time series in the file. This ID must be unique for each time series component in the file. If referring to country loads it can be the country code. In this case, this will be used to obtain the country holidays for the imputation function as well as the time covariates.
- Timeseries ID (Optional): Timeseries ID column is not compulsory, and shows the time series to which each component belongs. If Timeseries ID is not present, it is assumed that each component represents one separate series (the column is set to ID).
- Time columns: Columns that store the Value of each component. They must be consecutive and separated by resolution minutes. They should start at 00:00:00, and end at 24:00:00 - resolution


The checks that are performed when valifating a file are the following:

For all time series:
- The dataframe can not be empty
- All the dates must be sorted

For non-multiple time series:
- Column Datetime must be used as an index
- If the time series is the main dataset, Load must be the only other column in the dataframe
- If the time series is a covariates time series, there must be only one column in the dataframe named arbitrarily

For multiple timeseries:
- Columns Date, ID, and the time columns exist in any order
- Only the permitted column names exist in the dataframe (see Multiple timeseries file format bellow)
- All timeseries in the dataframe have the same number of components

### Parameters of the pipeline

* ```series_csv``` (default series.csv), the path to the local time series file to use. If series_uri has a non-default value, or if from_mongo is true, then series_csv has no effect.

* ```series_uri``` (default online_artifact), the uri of the online time series file to use. If series_uri is not online_artifact, and from_mongo is false, then this is the time series DeepTSF will use. 

* ```past_covs_csv``` (default None), the path to the local time series file to use as past covariates. If past_covs_uri is not None, then this has no effect.

* ```past_covs_uri``` (default None), the uri of the online time series file to use as past covariates. If past_covs_uri is not None, then this is the file DeepTSF will use as past covariates.

* ```future_covs_csv``` (default None), the path to the local time series file to use as future covariates. If future_covs_uri is not None, then this has no effect.

* ```future_covs_uri``` (default None), the uri of the online time series file to use as future covariates. If future_covs_uri is not None, then this is the file DeepTSF will use as future covariates.

* ```day_first``` (default true), whether the date has the day before the month in timeseries file.

* ```multiple``` (default false), whether to train on multiple timeseries. This applies to the main time series. Covariates can be multivariate, but the number of time series must be the same as the main time series. The only exception to this is if we have multiple time series and a single past or future covariate. In this case, we consider this series to be the covariate to all the main time series.

* ```from_mongo``` (default false), whether to read the dataset from mongodb, or from other sources. If this is true, it overrides all other options (series_csv, series_uri)

* ```mongo_name``` (default rdn_load_data), which mongo file to read

## Data pre-processing

### Description of this step

For each component of each time series, outlier detection is optionally conducted by removing values that differ more than an arbitrary number (defined by the user) of standard deviations from their monthly average, or that are zero in the case of a non-negative time series. Outliers are replaced by missing values. Subsequently, missing data may be imputed by using a weighted average of historical data and simple interpolation. This imputation method is analyzed below in more detail.

### Imputation method
This method imputes the timeseries using a weighted average of historical data
and simple interpolation. The weights of each method are exponentially dependent on the distance to the nearest non NaN value. More specifficaly, with increasing distance, the weight of simple interpolation decreases, and the weight of the historical data increases. The imputation result is calculated based on the following formulas:  

$w = e^{a d_i}$

 $result = w L + (1 - w) H$


 where $L$ is the simple interpolation, $H$ the historical data and $d_i$ the distance. $a$ is a constant that determines how quickly simple interpolation will lose its significance to the result. 
 
 The historical data that imputes a particular datetime (we will refer to it as $current$) is calculated using the average values of datetimes that fulfil all of the following conditions:

 * They have at most wncutoff distance to the current null value's
    week number (WN), where $WN = week + hour/24 + minute/(24\cdot60)$. $week$ is the day of the week (0-6) of $current$, $hour$ is the hour (0-23), and minute is the minute (0-59). wncutoff's default value (0.000694) is a little less than a minute. This means that we take into account the datetimes that have the same week, hour and minute as $current$. Holidays are handled as
    either Saturdays (if the real day is a Friday) or Sundays (in every other case).

 * They have at most ycutoff distance (in years) from the $current$'s year. 

 * They have at most ydcutoff distance (in days) from the $current$'s yearday, where yearday is the number of days that have passed since January 1st of $current$'s year. We also use mod to include the datetimes of the previous or next year that are still at most ydcutoff days away from current's yearday. The exact formula that is used is presented below:

    $(yearday - currentYearday) \mod (daysInYear) < ydcutoff$ 

    or

    $(yearday + currentYearday) \mod (daysInYear) < ydcutoff$

    where $yearday$ is the yearday of the datetime to be included in the historical forecast, $daysInYear$ are the days in the year of that datetime, and $currentYearday$ is $current$'s yearday  

* If $current$ is before cut_date_val, it is imputed using historical data
from dates which are also before cut_date_val.


 The parameters of the pipeline associated with this method are presented below, along with all parameters of data pre-processing:

### Parameters of the pipeline
TODO: Check that leave it be
* ```resolution``` (default 15), the resolution that all datasets will use. If this is not the resolution of a time series, then it is resampled to use that resolution. In case of single timeseries, all prepprocessing is done in this resolution. In other words resampling is done before prosocessing. In case of multiple timeseries however, the resolution is infered from load_raw_data. All preprosessing is done using the infered resolution and then afterwards resampling is performed. 

TODO: this should be for covariates also
* ```year_range``` (default 2009-2019), the years to use from the datasets (inclusive). All values outside of those dates will be dropped.

* ```time_covs``` (default false), whether to add time covariates to the time series. If true, then the following time covariates will be added as future covariates:
    * The month
    * The day of the year
    * The hour
    * The day of the week
    * The week of the year
    * Whether its a holiday or not

* ```convert_to_local_tz``` (default false), whether to convert to local time. If we have a multiple time series file, ID column is considered as the country to transform each time series' time to. If this is not a country code, then country argument is used.

* ```min_non_nan_interval``` (default 24), if after imputation there exist continuous intervals of non nan values that are smaller than min_non_nan_interval hours, these intervals are all replaced by nan values

TODO write only for countries

* ```country``` (default PT), the country code this dataset belongs to. 

* ```std_dev``` (default 4.5), argument of the outlier detection method. It is the number to be multiplied with the standard deviation of each 1 month period of the dataframe. The result is then used as a cut-off value.

* ```max_thr``` (default -1), argument of the imputation method. If there is a consecutive subseries of NaNs longer than max_thr, then it is not imputed and returned with NaN values. If it is -1, every value will be imputed regardless of how long the consecutive subseries of NaNs it belongs to is.

* ```a``` (default 0.3), argument of the imputation method.
It is the weight that shows how quickly simple interpolation's weight decreases as the distacne to the nearest non NaN value increases. For more information see the section about our imputation method above.

* ```wncutoff``` (default 0.000694), argument of the imputation method. Historical data will only take into account dates that have at most wncutoff distance from the current null value's WN (Week Number). 

* ```ycutoff``` (default 3), argument of the imputation method. Historical data will only take into account dates that have at most ycutoff distance from the current null value's year.

* ```ydcutoff``` (default 30), argument of the imputation method. Historical data will only take into account dates that have at most ydcutoff distance from the current null value's yearday.

* ```l_interpolation``` (default false), whether to only use linear interpolation, or use the imputation method described above

* ```rmv_outliers``` (default true), whether to remove outliers or not

## Training and validation

### Description of this step

After the pre-processing stage, the data is scaled using min-max scaling, and is split into training, validation, and testing data sets. Then, the training of the model begins using only the training data set. The currently supported models are N-BEATS , Transformer, NHiTS, temporal convolutional networks, (block) recurrent neural networks, temporal fusion transformers, LightGBM, random forest, and seasonal naive. The latter can serve as an effective baseline depending on the seasonality of the time series.

 Hyperparameter optimization can be also triggered using the Optuna library. DeepTSF supports both exhaustive and Tree-Structured Parzen Estimator-based (TPE-based) hyperparameter search. The first method tests all possible combinations of the tested hyperparameters, while the second one uses probabilistic methods to explore the combinations that result to optimal values of the user-defined loss function. Ultimately, a method based on functional analysis of variance (fANOVA) and random forests, is used to calculate the importance of each hyperparameter during optimization.

 ### Providing the model's parameters to DeepTSF

The user can provide hyperparameters for the model they want to train using the YAML files config.yml if hyperparameter optimization is not performed, and config_opt.yml otherwise. More specifically, the entrypoint DeepTSF will try to find is set as a parameter of the pipeline (hyperparams_entrypoint), and that needs to also exist in the corresponding file. 

In config.yml's case, the entry point will look like this:
    
    hyperparams_entrypoint:
        parameter1: value1
        parameter2: value2
        ...

Where parameter and value are each model's hyperparameter, and its desired value respectivelly. All model parameters not specified by the user take their default values according to darts.

In config_opt.yml's case, the parameters of the model that the user doesn't want to test can be given as in config.yml. The parameters that have to be tested must have their values in a list format as follows:
* Format ["range", start, end, step]: a list of hyperparameter values are considered ranging from value "start" till "end" with the step being defined by the last value of the list. 
* Format ["list", value\_1, ..., value\_n]: All the listed parameters (\{value\_1, ..., value\_n\}) are considered in the grid. 

```
hyperparams_entrypoint:
    parameter_not_to_be_tested: value1
    parameter_range:  ["range", start, end, step]
    parameter_list: ["list", value\_1, ..., value\_n]
    ...
 ```
Finally, if the user wants, they can test whether to scale the data or not just by including the hyperparameter
```
scale: ["list", "True", "False"]
``` 


### Parameters of the pipeline

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

* ```hyperparams_entrypoint``` (default LSTM1), the entry point containing the desired hyperparameters for the selected model. The file that will be searched for the entrypoint will be config.yml if opt_test is false, and config_opt.yml otherwise. More info for the required file format above

* ```cut_date_val``` (default 20190101), the validation set start date. All values before that will be the training series. Format: str, 'YYYYMMDD'

TODO Test inclusive or not

* ```cut_date_test``` (default 20200101), the test set start date. Values between that and cut_date_val will be the validation series. If cut_date_test = cut_date_test, then the test and validation sets will be the same (from cut_date_test to test_end_date). Format: str, 'YYYYMMDD'

* ```test_end_date``` (default None), the test set ending date. Values between that and cut_date_test will be the testing series. All values after that will be ignored. If None, all the timeseries from cut_date_test will be the test set. Format: str, 'YYYYMMDD'

* ```device``` (default gpu), whether to run the pipeline on the gpu, or just use the cpu. 

* ```retrain``` (default false), whether to retrain model during backtesting

TODO: ??? more info on that

* ```ignore_previous_runs``` (default true), whether to ignore previous step runs while running the pipeline

* ```scale``` (default true), whether to scale the target series

* ```scale_covs``` (default true), whether to scale the covariates

* ```n_trials``` (default 100), how many trials hyperparameter optimization will run. If we run a simple grid search, this might be bigger than the possible number of parameter combinations to be tested. In this case, DeepTSF will only run the maximum possible number of combinations.

* ```opt_test``` (default false), whether we are running hyperparameter optimization or not. Also, DeepTSF will check config_opt.yml for the model parameters if this is true, and config.yml otherwise.

* ```num_workers``` (default 4), number of threads that will be used by pytorch

* ```loss_function``` (default mape), loss function to use as objective function for optuna. Possible options are:
    - mape
    - smape
    - mase 
    - mae
    - rmse
    - nrmse_max
    - nrmse_mean

* ```grid_search``` (default false), whether to run an exhaustive grid search (if true) or use the tpe method in optuna.


## Evaluation and explanation

### Description of this step

Εvaluation is performed through backtesting on the testing data set. Specifically, for each time series given to the function, it consecutively forecasts time series blocks of length equal to the forecast horizon of the model from the beginning until the end of the test set. This operation takes place by default with a stride equal to forecast horizon but can be changed by the user. 

Then, evaluation metrics are calculated using the resulting forecasted time series. The evaluation metrics that are supported are: mean absolute error (MAE), root mean squared error (RMSE), min-max and mean normalized mean squared error (NRMSE), mean absolute percentage error (MAPE), standardized mean absolute percentage error (sMAPE), and mean absolute scaled error (MASE). In the case of multiple time series, it is possible for all evaluation sub-series to be tested leading to an average value for each one of the metrics. In this case, DeepTSF stores the results for all time series. 

Additionally, it is possible to analyze the output of DL and DL models using SHapley Additive exPlanations. Each SHAP coefficient indicates how much the output of the model changes, given the current value of the corresponding feature. In DeepTSF's implementation, the lags after the start of each sample are considered as the features of each model. Following that, a beeswarm plot is produced. In addition, a minimal bar graph is produced showing the average of the absolute value of the SHAP coefficients for each attribute. Finally, three force plot charts are produced, showing the exact value of its SHAP coefficients for a random sample. The above mentioned artifacts are accessible through the MLflow tracking UI.

### Parameters of the pipeline

* ```forecast_horizon``` (default 96), the number of timesteps that the model being evaluated is going to predict in each step of backtesting.

* ```stride``` (default None), the number of time steps between two consecutive steps of backtesting. If it is None, then stride = forecast_horizon

* ```shap_data_size``` (default 10), The size of shap dataset in samples. The SHAP coefficients are going to be computed for this number of random samples of the test dataset. If it is a float, it represents the proportion of samples of the test dataset that will be chosen. If it is an int, it represents the absolute number of samples to be produced.

TODO All models supported, fix documentation

* ```shap_data_size``` (default false), whether to do SHAP analysis on the model.

TODO change to PT
* ```eval_series``` (default PT), on which timeseries to run the backtesting. Only for multiple timeseries. 


TODO change from code to id
* ```eval_method``` (default ts_ID), what ID type is speciffied in eval_series: if ts_ID is speciffied, then we look at Timeseries ID column. Else, we look at ID column. In this case, all components of the timeseries that has the component with eval_series ID are used in the evaluation step. 

* ```evaluate_all_ts``` (default false), whether to validate the models for all timeseries, and return the mean of their metrics as a result. Only applicable for multiple time series. In this case, a file is produced (evaluation_results_all_ts) showing detailed results for all metrics and timeseries.  

TODO get rid of input-chunk-length

* ```ts_used_id``` (default None), if not None, only time series with this id will be used for training and evaluation. Applicable only on multiple time series files

* ```m_mase``` (default 1), the forecast horizon of the naive method used in MASE metric

* ```num_samples``` (default 1), number of samples to use for evaluating/validating a probabilistic model's output