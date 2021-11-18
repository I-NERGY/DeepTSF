from concurrent import futures
import yaml
import logging
import grpc
import load_prediction_pb2
import load_prediction_pb2_grpc
import predict_total_load as pr
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# env variables
MODEL_NAME = os.environ.get('MODEL_NAME')
print(os.environ.get('MODELS_DIR').split('/'))
MODELS_DIR = os.path.join(*os.environ.get('MODELS_DIR').split('/'))

model_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), MODELS_DIR, MODEL_NAME)

class PredictLoadServicer(load_prediction_pb2_grpc.PredictLoadServicer):
    def GetLoadPrediction(self, request, context):
        logging.info(
            f'GetLoadPrediction service called with request {request}')
        # hold input in required format by model
        model_input = {}

        try:
            model_input["forecast_horizon"] = request.forecast_horizon
            model_input["news"] = request.news
            model_input["datetime"] = request.datetime
        except Exception as e:
            logging.error('error occured while accessing request', e)
            context.set_details('please verify that input is valid')
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return load_prediction_pb2.Prediction()

        logging.info(f'model input is: {model_input}')

        response = pr.predict_load(input_dict=model_input, model_name=MODEL_NAME, models_path=MODELS_DIR)

        logging.info(f'response from model is {response}')
        return load_prediction_pb2.Prediction(load=response.values, datetime=response.index.values)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    load_prediction_pb2_grpc.add_PredictLoadServicer_to_server(
        PredictLoadServicer(), server)
    server.add_insecure_port(('{}:{}').format(
        config['server']['host'], config['server']['port']))
    server.start()
    logging.info('load_predict server starts listening at {}:{}'.format(
        config['server']['host'], config['server']['port']))
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    serve()
