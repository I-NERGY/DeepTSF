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
```mlflow run --experiment-name <inference> --entry-point inference . --env-manager=local -P pyfunc_model_folder=runs:/946a0a2aef704408ace3c5a424402e3a/pyfunc_model -P series_uri='s3://mlflow-bucket/2/9d49c464031746c29353e590072094c2/artifacts/features/series.csv'```

## Full pipeline example
```mlflow run --experiment-name 2009-2019 --entry-point exp_pipeline . -P hyperparams_entrypoint=nbeats0_10 -P darts_model=NBEATS -P ignore_previous_runs=t -P cut_date_val=20180101 -P cut_date_test=20190101 -P year_range=2009-2019 -P time_covs=None --env-manager=local```