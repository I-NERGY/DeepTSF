import os

import mlflow
from dagster import op, get_dagster_logger, graph, Definitions, ScheduleDefinition
from dagster import Config
from dagster_shell.ops import shell_op


class CliConfig(Config):
    experiment_name: str = 'uc2_example'
    from_database: str = 'false'
    series_csv: str = 'user_datasets/Italy.csv'
    convert_to_local_tz: str = 'false'
    day_first: str = 'false'
    multiple: str = 'false'
    imputation_method: str = 'pad'
    resolution: str = '60min'
    rmv_outliers: str = 'true'
    country: str = 'IT'
    year_range: str = '2015-2022'
    cut_date_val: str = '20200101'
    cut_date_test: str = '20210101'
    test_end_date: str = '20211231'
    scale: str = 'true'
    darts_model: str = 'LightGBM'
    hyperparams_entrypoint: str = '"{lags: 120}"'
    loss_function: str = 'mape'
    opt_test: str = 'false'
    grid_search: str = 'false'
    n_trials: str = '100'
    device: str = 'gpu'
    ignore_previous_runs: str = 't'
    forecast_horizon: str = '24'
    m_mase: str = '24'
    analyze_with_shap: str = 'false'
    evaluate_all_ts: str = 'false'
    ts_used_id: str = None




@op
def cli_command(config: CliConfig):
    log = get_dagster_logger()
    log.info('Current directory {}'.format(os.getcwd()))
    # pipeline_run = mlflow.projects.run(
    #     uri="./uc2/",
    #     experiment_name=config.experiment_name,
    #     entry_point="exp_pipeline",
    #     parameters=params,
    #     env_manager="local"
    #     )
    return "cd /app/uc2 && mlflow run " \
           f"--experiment-name {config.experiment_name} " \
           "--entry-point exp_pipeline . " \
           f"-P from_database={config.from_database} " \
           f"-P series_csv={config.series_csv} " \
           f"-P convert_to_local_tz={config.convert_to_local_tz} " \
           f"-P day_first={config.day_first}  " \
           f"-P multiple={config.multiple} " \
           f"-P imputation_method={config.imputation_method} " \
           f"-P resolution={config.resolution} " \
           f"-P rmv_outliers={config.rmv_outliers} " \
           f"-P country={config.country} " \
           f"-P year_range={config.year_range} " \
           f"-P cut_date_val={config.cut_date_val} " \
           f"-P cut_date_test={config.cut_date_test} " \
           f"-P test_end_date={config.test_end_date} " \
           f"-P scale={config.scale} " \
           f"-P darts_model={config.darts_model} " \
           f"-P hyperparams_entrypoint={config.hyperparams_entrypoint} " \
           f"-P loss_function={config.loss_function} " \
           f"-P opt_test={config.opt_test} " \
           f"-P grid_search={config.grid_search} " \
           f"-P n_trials={config.n_trials} " \
           f"-P device={config.device} " \
           f"-P ignore_previous_runs={config.ignore_previous_runs} " \
           f"-P forecast_horizon={config.forecast_horizon} " \
           f"-P m_mase={config.m_mase} " \
           f"-P analyze_with_shap={config.analyze_with_shap} " \
           f"-P evaluate_all_ts={config.evaluate_all_ts} " \
           f"-P  ts_used_id={config.ts_used_id} " \
           f"-P  eval_series={config.ts_used_id} " \
           "--env-manager=local"


@op
def experiment_steps_status(context, prev):
    log = get_dagster_logger()
    experiment = mlflow.get_experiment_by_name(context.run_config['ops']['cli_command']['config']['experiment_name'])
    runs = mlflow.search_runs([experiment.experiment_id])
    if len(runs) >= 5:
        last_experiment_runs = runs.head(5)
        expected_values = ['evaluation', 'training', 'etl', 'load_raw_data', 'main']
        actual_values = last_experiment_runs['tags.stage']
        for i in range(len(expected_values)):
            expected = expected_values[i]
            actual = actual_values[i]
            if expected != actual:
                raise Exception('Expected steps missing')
        all_steps_finished = (last_experiment_runs['status'] == 'FINISHED').all()
        if not all_steps_finished:
            raise Exception('Steps exist that did not finish successfully')
        return


@graph
def mlflow_cli_graph():
    experiment_steps_status(shell_op(cli_command()))


uc2_mlflow_cli_job = mlflow_cli_graph.to_job(
    name="mlflow_cli_uc2"
)

defs = Definitions(jobs=[uc2_mlflow_cli_job])

basic_schedule = ScheduleDefinition(job=uc2_mlflow_cli_job, cron_schedule="0 0 * * *")
