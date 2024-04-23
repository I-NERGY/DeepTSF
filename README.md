[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://github.com/I-NERGY/DeepTSF/blob//License.txt)
# DeepTSF

This is the repository for DeepTSF timeseries forecasting tool. The whitepaper for this project can be found in [1]. For the extensive DeepTSF documentation please navigate to our [Wiki](https://github.com/I-NERGY/DeepTSF/wiki/DeepTSF-documentation). 

## Installation

To set up DeepTSF on your local system, you need clone the main branch of this repository:

```git clone https://github.com/I-NERGY/DeepTSF.git```

Alternatively you can use the dedicated Github release instead of cloning the main branch.

After that you need to navigate to the root directory of DeepTSF:

```cd /path/to/repo/of/DeepTSF```

Το enable the communication of the client with the logging servers (MLflow, Minio, Postgres), a .env file is needed. 
An example (.env.example) is provided, with default environment variables.

After that, you can set up DeepTSF either using conda (CLI for data scientists) or Docker (full deployment).

### Set up locally using Docker (Recommended)

To set up locally using docker first go to DeepTSF's root directory and rename .env.example to .env. Then run the following command in DeepTSF's root directory:

```docker-compose up```

DeepTSF is up and running. Navigate to [http://localhost:3000](http://localhost:3000) and start your experiments!

- Optional step for advanced users: 

In a new terminal window, you can copy the timeseries file you desire to run into the container, 
and then run the following to gain access to the container's file system:

```docker cp <path_to_file> deeptsf-backend:/app```

```docker exec -it deeptsf_backend bash```

Now you are running bash in the main DeepTSF container! Choose the directory which best corresponds to your 
problem, and switch to that: <br>
- uc2 for general problems. The app will execute a national load forecasting
  use case if from_database is set to true. So preferrably set from_database=False unless you create your own database connection.<br>
- uc6 and uc7 are related to other use cases and are still under development.

So run the following to set up your working environment:
```cd uc2```

```conda activate DeepTSF_env```

```export MLFLOW_TRACKING_URI=https://localhost:5000```

```export GIT_PYTHON_REFRESH=quiet```


Then, you can execute any experiment you want. An example working command (also demonstrated in the whitepaper [1]), is shown below:

```mlflow run --experiment-name example --entry-point exp_pipeline . -P series_csv=user_datasets/Italy.csv -P convert_to_local_tz=false -P day_first=false -P from_database=false -P multiple=false -P imputation_method=peppanen -P resolution=1h -P rmv_outliers=true -P country=IT -P year_range=2015-2022 -P cut_date_val=20200101 -P cut_date_test=20210101 -P test_end_date=20211231 -P scale=true -P darts_model=NBEATS -P hyperparams_entrypoint=NBEATS_example -P loss_function=mape -P opt_test=true -P grid_search=false -P n_trials=100 -P device=gpu -P ignore_previous_runs=t -P forecast_horizon=24 -P m_mase=24 -P analyze_with_shap=False --env-manager=local```

Don't forget to change series_csv argument to match the file's location in the container 
(if you followed the previous instructions it must be located in the parent directory).

### Bare metal installation 

This installation is only recommended for advanced users that require advanced
pipeline parameterization and functionalities such as hyperparameter tuning.

#### Set up mlflow tracking server

To run DeepTSF on your system you first have to install the mlflow tracking and minio server.

```git clone https://github.com/I-NERGY/mlflow-tracking-server.git```

```cd mlflow-server```

After that, you need to get the server to run

```docker-compose up```

The MLflow server and client may run on different computers. In this case, remember to change
the addresses on the .env file.

#### Set up the DeepTSF backend (CLI functionality) locally using conda.

You can use conda.yaml to reproduce the conda environment manually. Simply 
execute the following command which creates a new conda enviroment called
DeepTSF_env:

```cd /path/to/repo/of/DeepTSF```

```conda env create -f conda.yaml```

Then activate the new environment:

```conda activate DeepTSF_env```

Alternatively to those 2 commands, you can reproduce the conda environment automatically,
by running any 'mlflow run' command *without* the option `--env-manager=local`. 
This option however is not encouraged for every day use as it rebuilds the conda environment from scratch every time.

Then, set the MLFLOW_TRACKING_URI to the uri of the mlflow tracking server (by default http://localhost:5000). 
Please do not omit this step as this environment variable will not get inherited from the .env file. 

```export MLFLOW_TRACKING_URI=https://localhost:5000```

For the extensive DeepTSF documentation please navigate to our [Wiki](https://github.com/I-NERGY/DeepTSF/wiki/DeepTSF-documentation). 

## References
[1]: Pelekis, S., Karakolis, E., Pountridis, T., Kormpakis, G., Lampropoulos, G., Mouzakits, S., & Askounis, D. (2023). DeepTSF: Codeless machine learning operations for time series forecasting. ArXiv https://arxiv.org/abs/2308.00709  <br>