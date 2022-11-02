import numpy as np
import pandas as pd

class DatesNotInOrder(Exception):
    """
    Exception raised if dates in series_csv are not sorted.
    """
    def __init__(self):
        super().__init__("Dates in series_csv are not sorted. Check date format in input csv.")

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
    Exception raised if the IDs present in a multiple timeseries file are not consecutive.
    """
    def __init__(self, ids):
        self.message = f'ID names provided: {self.ids}. IDs in multiple timeseries file must be consecutive.'
        super().__init__(self.message)
