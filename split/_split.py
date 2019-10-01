import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

class TrainValidateTest(BaseEstimator):
    """
    
    """
    
    # ======================================================================
    # Constructors
    # ======================================================================
    
    def __init__(self, train_size, valid_size, test_size):
        
        if self._check_data_sizes(train_size, valid_size, test_size):
            raise ValueError (
            "Ensure that the train_size, valid_size and test_size are "
            "individually less than 1, greater than 0 and sum to 1.")
        
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
    
        
    def _check_data_sizes(self, *args):
        data = np.array(args)
        return ((data < 0) | (data > 1) | (data.sum() != 1)).any()
    
    
    # ======================================================================
    # Constructors
    # ======================================================================
    
    
    def fit(self, data):
        return self

    def transform(self, data):
        
        """
        Generate the train, validation, and test dataset.

        Parameters
        ----------
        all_x : DataFrame
            All the input samples
        all_y : Pandas Series
            All the target values

        Returns
        -------
        x_train : DataFrame
            The train input samples
        x_valid : DataFrame
            The validation input samples
        x_test : DataFrame
            The test input samples
        y_train : Pandas Series
            The train target values
        y_valid : Pandas Series
            The validation target values
        y_test : Pandas Series
            The test target values
        """

        dates = data.index
        
        # define the indices for splitting data
        train_idx = int(len(dates) * self.train_size)
        valid_idx = int(len(dates) * (self.train_size + self.test_size))
        indices_or_sections = [train_idx, valid_idx]
        
        # Generate the date regions
        train_dates, valid_dates, test_dates = np.split(
            dates, 
            indices_or_sections
        )

        return (
            data.loc[train_dates, :], 
            data.loc[valid_dates, :], 
            data.loc[test_dates, :]
        )
        
    
    def fit_transform(self, data, **fit_params):
        return self.fit(X, **fit_params).transform(data)
