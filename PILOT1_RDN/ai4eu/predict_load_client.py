import grpc
import yaml
import logging
import load_prediction_pb2
import load_prediction_pb2_grpc
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# env variables
MODEL_NAME = os.environ.get('MODEL_NAME')
print(os.environ.get('MODELS_DIR').split('/'))
MODELS_DIR = os.path.join(*os.environ.get('MODELS_DIR').split('/'))

model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), MODELS_DIR, MODEL_NAME)
# print(model_path)

news_example = pd.read_csv(os.path.join(model_path, "news_example_series.csv"), parse_dates=True, index_col=0)

SAMPLE_DATA = {
    "forecast_horizon": 24,
    "news": news_example.values.reshape(-1).tolist(), # must be longer than input chunk
    "datetime": news_example.index.strftime('%Y%m%d %H:%M:%S').values.tolist() # must be longer than input chunk
}

# # comment following line when providing real data ("news")
# SAMPLE_DATA['news'] = [x if x > 0 else -x for x in SAMPLE_DATA["news"]]


# def get_load_prediction(stub):
#     return stub.GetLoadPrediction(
#         load_prediction_pb2.Input(
#             days_to_append=SAMPLE_DATA["days_to_append"],
#             days_ahead=SAMPLE_DATA["days_ahead"],
#             daily_steps=SAMPLE_DATA["daily_steps"],
#             news=SAMPLE_DATA["news"],
#         )
#     )

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
                # index=pd.to_datetime((list(response.datetime))),
            )
            # df.index.name = "Datetime"
            # df['Time'] = pd.to_datetime(df['datetime']).dt.time
            # df = df.drop('datetime', 1)
            print(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    run()
