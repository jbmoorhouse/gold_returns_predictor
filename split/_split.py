import pandas as pd
import numpy as np

class TrainValidateTest(BaseEstimator, TransformerMixin):
    """
    train_size, 
    valid_size, 
    test_size
    """
    
    # ----------------------------------------------------------------------
    # Constructors
    
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
    
    # ----------------------------------------------------------------------
    # fit, transform methods

    def transform(self, X, y = None):
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

        dates = X.index
        train_size, valid_size, test_size = (
            self.train_size, self.valid_size, self.test_size
        )

        train_dates, valid_dates, test_dates = np.split(
            dates, [int(len(dates) * train_size), 
                    int(len(dates) * (train_size + test_size))]
        )

        return (
            X.loc[train_dates, :],
            X.loc[valid_dates, :],
            X.loc[test_dates, :],
            y.loc[train_dates],
            y.loc[valid_dates],
            y.loc[test_dates])