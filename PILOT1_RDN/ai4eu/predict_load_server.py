from concurrent import futures
import yaml
import logging
import grpc
import load_prediction_pb2
import load_prediction_pb2_grpc
import predict_total_load as pr
import pandas as pd
import os

# get config variables
logging.basicConfig(level=logging.DEBUG)
with open("config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)
model_name = config['server']['model_name']
models_dir = config['server']['model_dir'].replace('/', os.path.sep)

model_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), models_dir, model_name)
print(
    f"\nLooking for model in: \n{model_path}. \nIf this is not the case modify config.yml file accordingly.")

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

        response = pr.predict_load(input_dict=model_input, model_name=model_name, models_path=models_dir)

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
    # logging.basicConfig(level=logging.DEBUG)
    # with open("config.yml", "r") as ymlfile:
    #     config = yaml.safe_load(ymlfile)
    serve()
