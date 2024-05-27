import os
from utils import load_model, load_scaler, load_ts_id, parse_uri_prediction_input, to_seconds
import pretty_errors

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from minio import Minio
disable_warnings(InsecureRequestWarning)
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_CLIENT_URL = os.environ.get("MINIO_CLIENT_URL")
client = Minio(MINIO_CLIENT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, secure=False)


class _MLflowPLDartsModelWrapper:

    def __init__(self, darts_model, transformer=None, ts_id_l=[[]]):
        self.model = darts_model
        #TODO Ask if this is right
        self.transformer = transformer if type(transformer) == list or transformer==None else [transformer]
        self.ts_id_l=ts_id_l

    def predict(self, model_input):
        """ 
        :params 
        
        model_input: Dict
        {"n": int, "history": json file, "past_covariates": json file, "future_covariates": json file, "roll_size":int, "batch_size": int}

        :outputs
        
        Dataframe
        """
        try:
            model_input = model_input[0]
            batched = True
        except:
            batched = False

        # Parse
        model_input = parse_uri_prediction_input(client, model_input, self.model, self.ts_id_l)

        # Transform
        if self.transformer is not None:
            print('\nTransforming...')
            model_input['history'] = self.transformer[model_input["predict_series_idx"]].transform(
                model_input['history'])
        # Predict 
        # TODO: Do that with inference dataset?
        predictions = self.model.predict(
            n=model_input['n'],
            roll_size=model_input['roll_size'],
            series=model_input['history'],
            future_covariates=model_input['future_covariates'],
            past_covariates=model_input['past_covariates'],
            batch_size=model_input['batch_size'])

        ## Untransform
        if self.transformer is not None:
            print('\nInverse transforming...')
            predictions = self.transformer[model_input["predict_series_idx"]].inverse_transform(predictions)

        # Return as DataFrame
        print(predictions[0].pd_dataframe())
        if batched:
            return [predictions[0].pd_dataframe()]
        else:
            return predictions[0].pd_dataframe()

def _load_pyfunc(model_folder):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    # load model from MLflow or local folder
    print(f"Local path inside _load_pyfunc: {model_folder}")
    model_folder = model_folder.replace('/', os.path.sep)
    model_folder = model_folder.replace('\\', os.path.sep)
    print(f"Local path altered for loading: {model_folder}")
    
    # TODO: Create a class for these functions instead of bringing them from utils.py
    print(model_folder)
    # Different behaviours for pl and pkl models are defined in load_model
    model = load_model(client, model_root_dir=model_folder, mode="local")
    scaler = load_scaler(scaler_uri=f"{model_folder}/scaler_series.pkl", mode="local")
    ts_id_l = load_ts_id(load_ts_id_uri=f"{model_folder}/ts_id_l.pkl", mode="local")
    return _MLflowPLDartsModelWrapper(model, scaler, ts_id_l)
