from sys import version_info
import os
import cloudpickle
from utils import load_model, load_scaler, parse_uri_prediction_input
import darts, mlflow, torch
import numpy as np



PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

mlflow_serve_conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
                'cloudpickle=={}'.format(cloudpickle.__version__),
                'darts=={}'.format(darts.__version__),
                'torch=={}'.format(torch.__version__),
                'mlflow=={}'.format(mlflow.__version__)
            ],
        },
    ],
    'name': 'darts_infer_pl_env'
}

class _MLflowPLDartsModelWrapper:

    def __init__(self, darts_model, transformer=None):
        self.model = darts_model
        self.transformer = transformer

    def predict(self, model_input):
        """ 
        :param model_input: Dict
        {"n": int, "history": json file, "past_covariates": json file, "future_covariates": json file, "roll_size":int, "batch_size": int}
        """
        # Parse
        model_input = parse_uri_prediction_input(model_input, self.model)

        # Transform
        if self.transformer is not None:
            print('\nTransforming...')
            model_input['history'] = self.transformer.transform(
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
            predictions = self.transformer.inverse_transform(predictions)

        # Return as DataFrame
        return predictions.pd_dataframe()

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
    model = load_model(model_root_dir=model_folder, mode="local")
    scaler = load_scaler(scaler_uri=f"{model_folder}/scaler_series.pkl", mode="local")

    return _MLflowPLDartsModelWrapper(model, scaler)



