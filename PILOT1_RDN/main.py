# This script should handle the mlflow workflow
mlflow_config = read_config('config.yml', 'mlflow_setings')
MLFLOW_TRACKING_URI = mlflow_config['mlflow_tracking_uri']
