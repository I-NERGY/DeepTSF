import yaml
import os

def read_config(filepath, entrypoint):

    with open(filepath, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    return config['entry_points'][entrypoint]





