import numpy as np
import pandas as pd

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)

class EmptyDataframe(Exception):
    """
    Exception raised if dataframe is empty.
    """
    def __init__(self, from_database):
        super().__init__("Dataframe provided is empty" + (" or does not exist in mongo database" if from_database else ""))


class DatetimesNotInOrder(Exception):
    """
    Exception raised if dates in series_csv are not sorted.
    """
    def __init__(self, id=0):
        super().__init__(f"Datetimes in series_csv are not sorted for time series component with id {id}. Check date format in input csv.")

class WrongColumnNames(Exception):
    """
    Exception raised if series_csv has wrong column names.
    """
    def __init__(self, columns, col_num, names):
        mames = "and".join(names)
        self.message = f'Column names provided: {columns}. series_csv must have {col_num} columns named {names}.'
        super().__init__(self.message)

class CountryDoesNotExist(Exception):
    """
    Exception raised if the country specified does not exist/have holidays.
    """
    def __init__(self):
        super().__init__("The country specified does not exist/have holidays")

class WrongIDs(Exception):
    """
    Exception raised if the IDs present in a multiple timeseries file are not consecutive integers.
    """
    def __init__(self, ids):
        self.message = f'ID names provided: {ids}. IDs in a multiple timeseries file must be consecutive integers.'
        super().__init__(self.message)

class DifferentComponentDimensions(Exception):
    """
    Exception raised if not all timeseries in a multiple timeseries file have the same number of components.
    """
    def __init__(self):
        self.message = f'Not all timeseries in multiple timeseries file have the same number of components.'
        super().__init__(self.message)

class NanInSet(Exception):
    """
    Exception raised if val or test set has nan values.
    """
    def __init__(self):
        self.message = f'Validation and test set can not have any nan values'
        super().__init__(self.message)

class MandatoryArgNotSet(Exception):
    """
    Exception raised if a mandatory argument was not set by the user.
    """
    def __init__(self, argument_name, mandatory_prerequisites):
        if mandatory_prerequisites:
            mandatory_prerequisites = "\n".join(("- " + args[0] + "=" + args[1]) for args in mandatory_prerequisites)
            self.message = f'Argument {argument_name} is mandatory since the following conditions apply: \n{mandatory_prerequisites}.\nIt was set to None / not set.'
        else:
            self.message = f'Argument {argument_name} is mandatory and set to None / not set.'
        super().__init__(self.message)

class NotValidConfig(Exception):
    """
    Exception raised if config is not an entrypoint in config file and not a valid json string.
    """
    def __init__(self):
        self.message = f'config is not an entrypoint in config file or a valid json string'
        super().__init__(self.message)

class NoUpsamplingException(Exception):
    """
    Exception raised if the user tries to convert a series to a lower resolution.
    """
    def __init__(self):
        self.message = f'Upsampling is not allowed. Change the target resolution of the series'
        super().__init__(self.message)

class TsUsedIdDoesNotExcist(Exception):
    """
    Exception raised if ts_used_id chosen by the user does not exist in the multiple time series file.
    """
    def __init__(self):
        self.message = f'This ts_used_id does not exist in the multiple time series file'
        super().__init__(self.message)