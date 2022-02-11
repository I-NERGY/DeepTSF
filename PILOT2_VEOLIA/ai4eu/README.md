# [I-NERGY](https://www.i-nergy.eu/) load forecasting

I-NERGY load forecasting service is based on a prediction model for electrical load of a boiler room in a large district heating network.
The implementation makes use of a SARIMA (Seasonal ARIMA) model.  

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
import logging
import load_prediction_pb2
import load_prediction_pb2_grpc
import numpy as np
import pandas as pd

SAMPLE_DATA = {
    "days_to_append": 6,
    "days_ahead": 1,
    "daily_steps": 24,
    "news": np.random.normal(3.0, 1.5, 6 * 24), #size parameter = days_to_append * daily_steps
}
# comment following line when providing real data ("news")
SAMPLE_DATA['news'] = [x if x > 0 else -x for x in SAMPLE_DATA["news"]]

def get_load_prediction(stub):
    return stub.GetLoadPrediction(
        load_prediction_pb2.Input(
            days_to_append=SAMPLE_DATA["days_to_append"],
            days_ahead=SAMPLE_DATA["days_ahead"],
            daily_steps=SAMPLE_DATA["daily_steps"],
            news=SAMPLE_DATA["news"],
        )
    )

def run():
    with grpc.insecure_channel(
        # update host, port values
        ("{}:{}").format("localhost", "60259")
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
                {"datetime": pd.to_datetime((list(response.datetime))) , "Forecasted Load": list(response.load) },
            )
            df['Time'] = pd.to_datetime(df['datetime']).dt.time
            df = df.drop('datetime',1)
            print(df)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run()
```

Users are able to configure the following fields of the request's payload:

* *days_to_append* : provide a zero or positive value corresponding to the number of days that the electricity load has been observed. These observations will be appended to the existing model as new ground truth leading eventually to the model adapting its predictions accordingly.
* *days_ahead* : provide value equal or greater than 1, referring to the number of days of the desired forecasting horizon.
* *news* : replace this value with the data concerning the electricity load measured. (size = days to append * daily steps)

In order to locally execute client:

* install required dependencies (i.e., `grpcio`, `pandas`, `pyyaml`).
* generate the imported classes. (classes are also available inside container and therefore can be copied, if users have access to container.)
  * install `grpcio-tools`.
  * locate file load_prediction.proto inside folder *microservice*
  * create classes: `python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. load_prediction.proto`
* configure request's payload as described above.
* replace values of host & port to the ones of the deployed service.
* run client: `python3 client.py`
