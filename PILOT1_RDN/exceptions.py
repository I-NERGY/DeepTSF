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
    def __init__(self, columns):
        self.message = "series_csv must have 2 columns named Date and Load."
        self.columns = columns
        super().__init__(self.message)
    def __str__(self):
        return f'Column names provided: {self.columns}. {self.message}'

class CountryDoesNotExist(Exception):
    """
    Exception raised if the country specified does not exist/have holidays.
    """
    def __init__(self):
        super().__init__("The country specified does not exist/have holidays")
