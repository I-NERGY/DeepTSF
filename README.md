# Energy-forecasting

Repository for energy demand forecasting 

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
