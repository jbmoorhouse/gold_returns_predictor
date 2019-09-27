from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import talib


class StochasticRsi(BaseEstimator, TransformerMixin):
    
    # ======================================================================
    # Constructors
    # ======================================================================
    
    def __init__(self, price_index, timeperiod=14, k=3, d=3):
        self.price_index = price_index
        self.timeperiod = timeperiod
        self.k = k
        self.d = d
       
    # ======================================================================
    # Indicator methods
    # ======================================================================
    
    def _rsi(self, X):
        real = X[:, self.price_index]
        return talib.RSI(real, timeperiod = self.timeperiod)
    
    
    def _stoch_rsi(self, X):
        rsi = self._rsi(X) 
        
        max_rsi = talib.MAX(rsi, timeperiod = self.timeperiod)
        min_rsi = talib.MIN(rsi, timeperiod = self.timeperiod)
        
        fastk = talib.SMA(
            real = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100,
            timeperiod = self.k
        )
        fastd = talib.SMA(real=fastk, timeperiod=self.d) 
        
        return np.c_[fastk, fastd]

    
    # ======================================================================
    # Public methods
    # ======================================================================
    
    
    def fit(self, X, y=None):
        return self

    
    def transform(self, X, y=None):
        stoch_rsi = self._stoch_rsi(X)
        
        return np.c_[X, stoch_rsi]
        
        

class StochRsiSignal(BaseEstimator, TransformerMixin):
    
    # ----------------------------------------------------------------------
    # Constructors
    
    def __init__(self, upper = 80, lower = 20):
        self.upper = upper
        self.lower = lower
        
        
    # ----------------------------------------------------------------------
    # fit, transform methods    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        stochrsi_fastd = X["stochrsi_fastd"]
        
        stochrsi_fastd_buy = (
            (stochrsi_fastd >= self.lower) & 
            (stochrsi_fastd.shift(1) <= self.lower)
        ).astype(int)
        
        stochrsi_fastd_sell = (
            (stochrsi_fastd <= self.upper) & 
            (stochrsi_fastd.shift(1) >= self.upper)
        ).astype(int)
        
        X['stoch_fastd_buy'] = stochrsi_fastd_buy
        X['stoch_fastd_sell'] = stochrsi_fastd_sell
            
        return X
        
        
class MacdSignal(BaseEstimator, TransformerMixin):
    
    # ----------------------------------------------------------------------
    # Constructors
    
    def __init__(self, scale=1, lower=-40, upper=40, as_dataframe = True):
        self.scale = scale
        self.lower = lower * scale
        self.upper = upper * scale
        self.as_dataframe = as_dataframe
        
    # ----------------------------------------------------------------------
    # fit, transform methods 
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        macd_fast = X['macd']
        macd_slow = X['macd_macdsignal']
        
        uptrend = (macd_fast > macd_slow).astype(int)
    
        #macd_fast = np.ceil(macd_fast * self.scale)
        macd_fast.clip(self.lower, self.upper, inplace = True)
        
        if self.as_dataframe:
            return X.assign(
                macd_trend_signal = uptrend * macd_fast,
                macd_uptrend = uptrend
            )
        else:
            return np.c_[X, uptrend * macd_fast, uptrend]
        
        
class VolatilityDiff(BaseEstimator, TransformerMixin):
    
    # ----------------------------------------------------------------------
    # fit, transform methods
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        vol_columns = X.filter(like='vol').columns
        
        for col in vol_columns:
            name = "{}_diff".format(col)
            X[name] = X[col].diff()
        
        return X.dropna()