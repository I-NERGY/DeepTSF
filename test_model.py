import mlflow
from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri('http://131.154.97.48:5000/')
model = mlflow.pyfunc.load_model("runs:/326914ae41ea4fe6ae37805963eaaa3e/pyfunc_model")
print(model)