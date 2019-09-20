from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import talib
from tqdm import tqdm



class MacdStrategy(BaseEstimator, TransformerMixin):
    """
    Stores the parameters to constrain a moving average convergence/
    divergence (MACD) momentum based strategy. The indicator consists of
    three components (the 'macd series', the 'average series' of the macd 
    series and the 'divergence'). The 'macd series' is the difference
    between a 'fast' exponential moving average (EMA) and a 'slow' EMA. 
    The speed refers to the period of the EMA. The 'average series' is an 
    EMA of the 'macd series'. The 'divergence' is the difference between 
    these two series. 
    
    Parameters
    ----------
    fast_period : float, default 12.
        Time constant for the 'fast' price series exponential moving average
    slow_period : float, default 26.
        Time constant for the 'slow' price series exponential moving average 
    signal_period : float, default 9.
        Time constant for the macd series exponential moving average  
        
    Examples
    --------
    Constructing a MacdStrategy using the default values
    
    >>> strategy = MacdStrategy()
    
    """
    
    # ======================================================================
    # Constructors
    # ======================================================================

    def __init__(self, 
                 result = 'optimal',
                 fast_period=12.0, 
                 slow_period=26.0, 
                 signal_period=9.0):
        
        # set default model parameters if user wishes to use self.transform
        # with experimental parameters
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # set reasonable test time constants
        self.fast_period_test = np.arange(2,20, dtype = float)
        self.slow_period_test = np.arange(3, 30, dtype = float)
        self.signal_period_test = np.arange(4, 15, dtype = float)
        
        # cost
        self.ratio = 0
        
        
    # ======================================================================
    # Rendering Methods
    # ======================================================================

        
    def __repr__(self):
        return "TBD"
    
    
    # ======================================================================
    # strategy statistics/signals
    # ======================================================================
    
    def _macd(self, 
              fast_period = None, 
              slow_period = None, 
              signal_period = None,
              X = None, 
              column_name='price', 
              ):
        """
        Two-dimensional tabular data structure containing a price series and 
        claculated macd time series.
        
        Parameters
        ----------
        X : pd.DataFrame or pd.Series
             DataFrame or Series containing the price time series for a 
             single asset
        column_name : str, default price
            The name of the column containing the price time series data
            
        Returns
        -------
        X_macd : pd.DataFrame
            Returns a DataFrame consisting of the original price time
            series and the calculated macd time series data
          
        """
        
        # determine if default macd quantities shoud be used
        if fast_period is None:
            fast_period = self.fast_period
        if slow_period is None:
            slow_period = self.slow_period
        if signal_period is None:
            signal_period = self.signal_period
            
    
        if not isinstance(column_name, str):
            raise TypeError("'column_name' must be of type str")

            
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise TypeError("'X' must be of type pd.DataFrame or pd.Series")            
        elif isinstance(X, pd.DataFrame):
            
            # calculate macd time series
            price_series = X[column_name]
            
            macd, macd_signal, macd_hist = talib.MACD(
                price_series.values,
                fastperiod = fast_period,
                slowperiod = slow_period,
                signalperiod = signal_period
            )

            return X.assign(
                macd = macd,
                macd_signal = macd_signal,
                macdhist = macd_hist
            )
        elif isinstance(X, pd.Series):
        
            # calculate the macd time series
            macd, macd_signal, macd_hist = talib.MACD(
                X.values,
                fastperiod = fast_period,
                slowperiod = slow_period,
                signalperiod = signal_period
            )
            
            return pd.DataFrame(
                data = np.c_[X, macd, macd_signal, macd_hist], 
                index = X.index, 
                columns = [column_name, 'macd', 'macd_signal', 'macd_hist'])
            
            
    def _long_signal(self, X):
        self.long = (X['macd'].shift(1) > X['macd_signal'].shift(1))
        #reversal = ((X['macd'].diff().shift(1) > 0) & 
        #(df['macd'].diff().shift(2) > 0))
        
        return (self.long) * 1
    
    
    def _short_signal(self, X):
        self.short = (X['macd'].shift(1) < X['macd_signal'].shift(1)) 
        #reversal = ((df['macd'].diff().shift(1) < 0) & 
        #(df['macd'].diff().shift(2) < 0))
                
        return (self.short) * -1
    
    
    def _strategy_returns(self, X_macd):
        
        # generate strategy long/short signal and asset returns
        long_signal = self._long_signal(X_macd)
        short_signal = self._short_signal(X_macd)
        
        # calculate the assets simple returns 
        asset_returns = X_macd['price'].pct_change()
        
        return (asset_returns * long_signal) + (asset_returns * short_signal)

   
    # ======================================================================
    # Parameter estimation
    # ======================================================================

    
    def _cost_function(self, returns, annualisation_factor, risk_free_rate):
        
        excess_return = (
            (np.sqrt(annualisation_factor) * returns.mean()) - risk_free_rate
        )
        
        return excess_return / returns.std()
    
    
    def _fit(self, X, y, annualisation_factor, risk_free_rate):
        
        for f in tqdm(self.fast_period_test):
            for s in self.slow_period_test:
                for p in self.signal_period_test:
                    
                    X_macd = self._macd(f, s, p, X)
                    strategy_returns = self._strategy_returns(X_macd)
                    ratio = self._cost_function(
                        strategy_returns,
                        annualisation_factor,
                        risk_free_rate
                    )
                                    
                    if ratio > self.ratio:
                        self.ratio = ratio
                        self.fast_period = f
                        self.slow_period = s
                        self.signal_period = p
                        
        
    # ======================================================================
    # Public methods
    # ======================================================================
    
    
    def fit(self, X, y=None, annualisation_factor=252, risk_free_rate=0.00):
        self._fit(X, y, annualisation_factor, risk_free_rate)
        return self 
        
        
    def transform(self, X, y=None):
        """
        Transform the asset price series into a return time series using the 
        predefined macd momentum based strategy. 
        
        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            DataFrame or Series containing the price time series for a 
            single asset 
        y : pd.DataFrame or pd.Series, default None
            DataFrame or Series containing the price time series for a 
            single asset
            
        Returns
        -------
        X_macd :  X_macd : pd.DataFrame
            Returns a DataFrame consisting of the original price time
            series, calculated macd time series data and strategy returns
            
        Examples
        --------
        
        >>> X 
                      price
        Date	
        1979-06-15	 280.00
        1979-06-18	 278.00
        1979-06-19	 280.30
        1979-06-20	 281.35
        1979-06-21	 282.30
        1979-06-22	 283.45
        
        
        >>> strategy = MacdStrategy()
        >>> strategy.transform(X)
        
                    price  macd  macd_signal     ...  macd_strategy_returns
        Date					
        1979-06-15	  280   NaN	         NaN	 ...	                NaN
        ...           ...   ...          ...     ...                    ...
        1979-08-02	  291	4.07	     6.20	 ...	          -0.004826
        1979-08-03	  287	2.93         5.55	 ...	           0.017153
        1979-08-06	  283	1.70	    4.779	 ...	           0.013264
        
        """
        
        X_macd = self._macd(X = X)
        
        X_macd['macd_strategy_returns'] = (
            self._strategy_returns(X_macd)
        )
        
        return X_macd