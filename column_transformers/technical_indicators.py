from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class StochRsiSignal(BaseEstimator, TransformerMixin):
    
    # ----------------------------------------------------------------------
    # Constructors
    
    def __init__(self, upper = 80, lower = 20, as_dataframe = True):
        self.upper = upper
        self.lower = lower
        self.as_dataframe = as_dataframe
        
        
    # ----------------------------------------------------------------------
    # fit, transform methods    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        stochrsi_fastd = X.loc[:, "stochrsi_fastd"]
        
        stochrsi_fastd_buy = (
            (stochrsi_fastd >= self.lower) & 
            (stochrsi_fastd.shift(1) <= self.lower)
        ).astype(int)
        
        stochrsi_fastd_sell = (
            (stochrsi_fastd <= self.upper) & 
            (stochrsi_fastd.shift(1) >= self.upper)
        ).astype(int)
            
        if self.as_dataframe:    
            X['stoch_fastd_buy'] = stochrsi_fastd_buy
            X['stoch_fastd_sell'] = stochrsi_fastd_sell
            
            return X
        else:
            return np.c_[X, stochrsi_fastd_buy, stochrsi_fastd_sell]