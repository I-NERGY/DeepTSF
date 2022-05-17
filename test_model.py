import mlflow
from dotenv import load_dotenv
import click
load_dotenv()
mlflow.set_tracking_uri('http://131.154.97.48:5000/')

@click.command()
@click.option("--run",
              type=str,
              default='bd68a7755c614730a75c3fd61a4765de',
              help='run'
              )
def tester(run):
	model = mlflow.pyfunc.load_model(f"runs:/{run}/pyfunc_model")
	print(model)

if __name__ =='__main__':
    tester()
