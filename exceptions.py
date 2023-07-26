import numpy as np
import pandas as pd

class EmptyDataframe(Exception):
    """
    Exception raised if dataframe is empty.
    """
    def __init__(self, from_mongo):
        super().__init__("Dataframe provided is empty" + (" or does not exist in mongo database" if from_mongo else ""))


class DatesNotInOrder(Exception):
    """
    Exception raised if dates in series_csv are not sorted.
    """
    def __init__(self, id=0):
        super().__init__(f"Dates in series_csv are not sorted for time series component with id {id}. Check date format in input csv.")

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
