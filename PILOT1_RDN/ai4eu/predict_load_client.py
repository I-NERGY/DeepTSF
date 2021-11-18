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
