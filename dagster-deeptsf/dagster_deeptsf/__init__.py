from dagster import Definitions, load_assets_from_modules

from . import assets
from dagster_deeptsf.job_uc2 import uc2_mlflow_cli_job, basic_schedule

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    jobs=[uc2_mlflow_cli_job],
    schedules=[basic_schedule],
)


# """Definitions that provide Dagster code locations."""
# from dagster import Definitions
#
# from dagster_example.assets.cereal import cereals, highest_calorie_cereal, highest_protein_cereal
# from dagster_example.jobs import complex_job, hello_cereal_job
# from dagster_example.schedules import every_weekday_9am
#
# defs = Definitions(
#     assets=[cereals, highest_calorie_cereal, highest_protein_cereal],
#     jobs=[complex_job, hello_cereal_job],
#     schedules=[every_weekday_9am],
# )
