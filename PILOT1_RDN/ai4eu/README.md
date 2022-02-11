# [I-NERGY](https://www.i-nergy.eu/) load forecasting

This is a forecasting service for predicting the aggregated hourly net electrical load of the Portuguese transmission system operator (REN). The core of the service is a totally recurrent LSTM deep neural network. The model has been trained on the REN load time series for the years 2018 and 2019 (except December 2019). The service is served as a docker container and a client script is also provided to help the user form their inference requests. The model is totally configurable in terms of:

* *Provided ground truth data points:* The client can update the existing model with the desired length of new data points that have been observed. The provided input should follow the format of the csv file history_sample.csv.
* *Forecast horizons:* The client can request a forecast horizon of their preference. It should be noted that large forecast horizons lead to worse results due to the error propagation caused by the LSTM recurrence. 

## Download and installation

* Use *Deploy to local* option.
* Extract the downloaded solution.
* Create a Kubernetes namespace. (Kubernetes Cluster shall be available)  
`
kubectl create ns <namespace_name>
`
* Inside solution folder locate and run set up script.  
`
python .\kubernetes-client-script.py -n <namespace_name>
`
* Verify the status of the created pods.  
`
kubectl get pods –n <namespace_name>
`

## Usage

In case this service is not combined with another service and therefore it is not executed by an orchestrator as a pipeline solution, a gRPC client shall be implemented. To facilitate this process a simple example client's implementation is provided (see client.py in Documents section) along with relevant guidelines.

```python
import grpc
import yaml
import logging
import load_prediction_pb2
import load_prediction_pb2_grpc
import numpy as np
import pandas as pd
import os

# get config variables
logging.basicConfig(level=logging.DEBUG)
with open("config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)
data_file_name = config['client']['sample_data']

data_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), *data_file_name.split("/"))
print(f"Looking for example data in: \n {data_file}. \nIf this is not the case modify config.yml file accordingly. ")
sample_history = pd.read_csv(data_file, parse_dates=True, index_col=0)

SAMPLE_DATA = {
    "forecast_horizon": 24,
    # must be longer than input chunk
    "news": sample_history.values.reshape(-1).tolist(),
    # must be longer than input chunk
    "datetime": sample_history.index.strftime('%Y%m%d %H:%M:%S').values.tolist()
}

def get_load_prediction(stub):
    return stub.GetLoadPrediction(
        load_prediction_pb2.Input(
            forecast_horizon=SAMPLE_DATA['forecast_horizon'],
            news=SAMPLE_DATA["news"],
            datetime=SAMPLE_DATA["datetime"]
        )
    )

def run():
    with grpc.insecure_channel(
        ("{}:{}").format(config["client"]["host"], config["client"]["port"])
    ) as channel:
        stub = load_prediction_pb2_grpc.PredictLoadStub(channel)
        try:
            response = get_load_prediction(stub)
        except grpc.RpcError as e:
            print(f"grpc error occured:{e.details()} , {e.code().name}")
        except Exception as e:
            print(f"error occured: {e}")
        else:
            df = pd.DataFrame(
                {"Datetime": pd.to_datetime(
                    (list(response.datetime))), "Forecasted Load": list(response.load)},
            )
            print(df)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    # with open("config.yml", "r") as ymlfile:
    #     config = yaml.safe_load(ymlfile)
    run()

```

Users are able to configure the following fields of the request's payload:

* *forecast_horizon* : provide value equal or greater than 1, referring to the number of hours of the desired forecasting horizon.
* *news* : replace this value with the historical data that has been observed after model training. A python list containing the values of the appended histrory. Can be also altered in the provided csv history_sample.csv.
* *dates* : a python list containing the dates of the appended history. Can be also altered in the provided csv history_sample.csv.

In order to locally execute client:

* install required dependencies (i.e., `grpcio`, `pandas`, `pyyaml`).
* generate the imported classes. (classes are also available inside container and therefore can be copied, if users have access to container.)
  * install `grpcio-tools`.
  * locate file load_prediction.proto inside folder *microservice*
  * create classes: `python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. load_prediction.proto`
* configure request's payload as described above.
* replace values of host & port to the ones of the deployed service.
* run client: `python3 client.py`
