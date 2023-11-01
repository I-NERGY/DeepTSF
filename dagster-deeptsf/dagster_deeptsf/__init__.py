from dagster import Definitions, load_assets_from_modules

from . import assets
from dagster_deeptsf.job_uc2 import uc2_mlflow_cli_job, basic_schedule

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    jobs=[uc2_mlflow_cli_job],
    schedules=[basic_schedule],
)
