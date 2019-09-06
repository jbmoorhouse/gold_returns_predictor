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
        stoch_rsi = X.loc[:, "stochrsi_fastd"]
        
        stoch_buy = (
            (stoch_rsi > self.lower) & 
            (stoch_rsi.shift(1) < self.lower)).astype(int)
        stoch_sell = (
            (stoch_rsi < self.upper) & 
            (stoch_rsi.shift(1) > self.upper)).astype(int)
            
        if self.as_dataframe:    
            X['stoch_fastd_buy'] = stoch_buy
            X['stoch_fastd_sell'] = stoch_sell
            
            return X
        else:
            return np.c_[X, stoch_buy, stoch_sell]