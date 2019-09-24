from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import talib
from tqdm import tqdm
from column_transformers.technical_indicators import StochasticRsi



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
    result : str or int, default 'optimal'
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
    # Constants
    # ======================================================================
    
    PRICE, MACD, MACD_SIGNAL, MACD_HIST, RATIO = 0, 1, 2, 3, 3
    
    # ======================================================================
    # Constructors
    # ======================================================================

    def __init__(self, 
                 result = 'optimal',
                 fast_period=12.0, 
                 slow_period=26.0, 
                 signal_period=9.0):
        
        if result == 'optimal':
            self.optimal_result = True
            self.optimal_parameters = 0
        elif isinstance(result, (int, float)):
            self.optimal_result = False
            self.optimal_parameters = np.zeros([int(result), 4])
        else:
            raise ValueError(
                "'result' must be either a valid string or an int/float ")
        
        # set default model parameters if user wishes to use self.transform
        # with experimental parameters
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # set reasonable test time constants
        self.fast_period_test = np.arange(2,20, dtype = float)
        self.slow_period_test = np.arange(3, 30, dtype = float)
        self.signal_period_test = np.arange(4, 15, dtype = float)
        
        
    # ======================================================================
    # Rendering Methods
    # ======================================================================

        
    def __repr__(self):
        return "TBD"
    
    
    # ======================================================================
    # strategy statistics/signals
    # ======================================================================
    
    def _macd(self,
              X = None,
              fast_period = None, 
              slow_period = None, 
              signal_period = None, 
              column_name='price'):
        """
        Two-dimensional tabular data structure containing the original price 
        series and calculated macd time series.
        
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
            price_series = X[column_name].values
        elif isinstance(X, pd.Series):
            price_series = X.values
            
        # returns a tuple of macd, macd_signal and mach_hist
        macd_statistics = talib.MACD(
            price_series,
            fastperiod = fast_period,
            slowperiod = slow_period,
            signalperiod = signal_period
        )    
        
        price_macd = np.c_[price_series, np.array(macd_statistics).T]
        
        return price_macd[~np.isnan(price_macd).any(1), :]

            
    def _long_signal(self, price_macd):
        long = price_macd[:, self.MACD] > price_macd[:, self.MACD_SIGNAL]
        
        return long[:-1] * 1
    
    
    def _short_signal(self, price_macd):
        short = price_macd[:, self.MACD] < price_macd[:, self.MACD_SIGNAL]
       
        return short[:-1] * -1
    
    
    def _asset_returns(self, price_macd):
        return (
            np.diff(price_macd[:, self.PRICE]) / price_macd[:-1, self.PRICE]
        )
    
    
    def _strategy_returns(self, price_macd):
        
        # generate strategy long/short signal and asset returns
        long_signal = self._long_signal(price_macd)
        short_signal = self._short_signal(price_macd)
        
        # calculate the assets simple returns 
        asset_returns = self._asset_returns(price_macd)
        
        return (asset_returns * long_signal) + (asset_returns * short_signal)

   
    # ======================================================================
    # Parameter estimation
    # ======================================================================

    
    def _cost_function(self, 
                       strategy_returns, 
                       annualisation_factor, 
                       risk_free_rate):
        
        excess_return = (
            (np.sqrt(annualisation_factor) * strategy_returns.mean()) - risk_free_rate
        )
        
        return excess_return / strategy_returns.std()
    
    
    def _fit(self, X, y, annualisation_factor, risk_free_rate):
        
        for f in tqdm(self.fast_period_test):
            for s in self.slow_period_test:
                for p in self.signal_period_test:
                    
                    price_macd = self._macd(X, f, s, p)
                    strategy_returns = self._strategy_returns(price_macd)
                    
                    ratio = self._cost_function(
                            strategy_returns,
                            annualisation_factor,
                            risk_free_rate
                        )
                    
                    if self.optimal_result:
                        if ratio > self.optimal_ratio:
                            self.ratio = ratio
                            self.fast_period = f
                            self.slow_period = s
                            self.signal_period = p
                    else:
                        argmin = self.optimal_parameters[:, self.RATIO].argmin()
                        
                        if ratio > self.optimal_parameters[argmin, self.RATIO]:
                            self.optimal_parameters[argmin] = [
                                f, s, p, ratio
                            ]
                        
        
    # ======================================================================
    # Public methods
    # ======================================================================
    
    
    def fit(self, X, y=None, annualisation_factor=252, risk_free_rate=0.00):
        
        self._fit(X, y, annualisation_factor, risk_free_rate)
        return self 
        
        
    def transform(self, X, y=None, as_frame=False):
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
        
        price_macd = self._macd(X)
        
        index_size = X.index.size
        new_index = X.index[(index_size) - price_macd.shape[0]:]
        
        price_macd_df = pd.DataFrame(price_macd, index = new_index)
        asset_returns = np.insert(self._strategy_returns(price_macd), 0, np.nan)
      
        price_macd_df['macd_strategy_returns'] = (
            asset_returns
        )
        
        return price_macd_df
    
    
#     from abc import ABC, abstractmethod

# class BaseIndicator(ABC, BaseEstimator):
    
#     @abstractmethod
#     def __init__(self, 
#                  criterion, 
#                  optimal, 
#                  top_n, 
#                  length_min,
#                  stepsize):
        
#         self.criterion = criterion
#         self.optimal = optimal
#         self.top_n = top_n
#         self.length_min = length_min
#         self.stepsize = stepsize
        
#     # ======================================================================
#     # User implemented methods
#     # ======================================================================    
    
#     @abstractmethod
#     def _price_indicator(self, **kwargs):
#         pass
    
    
    
# class BaseStrategy(BaseIndicator, TransformerMixin):
    
#     # ======================================================================
#     # Constants
#     # ======================================================================
    
    
#     # ======================================================================
#     # Constructors
#     # ======================================================================
    
#     @abstractmethod
#     def __init__(self, 
#                  criterion, 
#                  optimal, 
#                  top_n, 
#                  length_min,
#                  stepsize,
#                  **kwargs):
        
#         self.criterion = criterion
#         self.optimal = optimal
#         self.top_n = top_n
#         self.length_min = length_min
#         self.stepsize = stepsize 
    
#     # ======================================================================
#     # User implemented methods
#     # ======================================================================
    
#     def _indicator_params(self, indicator_params):
#         for param_name, param in self.indicator_params.items():
#             param_range = range(self.length_min, param + 1)
#             yield param_range
        
        
#     def _overbought_region(self, overbought_upper, overbought_lower):
#         short_entry = long_exit = range(
#             overbought_lower, overbought_upper, stepsize)

#         return short_entry, long_exit

    
#     def _oversold_region(self, oversold_upper, oversold_lower):
#         long_entry = short_exit = range(
#             oversold_lower, oversold_upper, stepsize)

#         return long_entry, short_exit

    
#     def _parameter_grid(self):
#         ind_param_ranges = list(_indicator_params(self.indicator_params))

#         long_entry, short_exit = _oversold_region(
#             self.oversold_upper, 
#             self.oversold_lower)

#         short_entry, long_exit = _overbought_region(
#             self.overbought_upper, 
#             self.overbought_lower)

#         params = ind_param_ranges + [
#             long_entry, long_exit, short_entry, short_exit
#         ] 

#         return cartesian(params)
    
#     @abstractmethod
#     def _long_signal(self, price_indicator, long_entry, long_exit):
#         pass
    
#     @abstractmethod
#     def _short_signal(self, price_indicator, short_entry, short_exit):
#         pass
    
#     # ======================================================================
#     # Strategy methods
#     # ======================================================================
    
#     def _asset_returns(self, price_indicator):
#         return (np.diff(
#             price_indicator[:, self.PRICE]) / 
#             price_indicator[:-1, self.PRICE]
#         )
    
#     def _strategy_returns(self,
#                           price_indicator, 
#                           long_entry, 
#                           long_exit, 
#                           short_entry, 
#                           short_exit):
        
#         # generate strategy long/short signal and asset returns
#         long_signal = self._long_signal(price_indicator, long_entry, long_exit)
#         short_signal = self._short_signal(price_indicator, short_entry, short_exit)
        
#         # calculate the assets simple returns 
#         asset_returns = self._asset_returns(indicator)
        
#         return (asset_returns*long_signal) + (asset_returns*short_signal)
    
    
#     def _criterion(self, 
#                    strategy_returns, 
#                    annualisation_factor, 
#                    risk_free_rate):
        
#         excess_return = strategy_returns.mean() - risk_free_rate
        
#         return np.sqrt(annualisation_factor) * excess_return / strategy_returns.std()
    
    
#     def _fit(self, X, y, annualisation_factor, risk_free_rate):
   
#         for params in tqdm(self.parameter_grid):
#             t, k, d, l_en, l_ex, s_en, s_ex = params
                
#             indicator = self._price_indicator(X, t, k, d)
#             strategy_returns = self._strategy_returns(
#                 price_indicator, l_en, l_ex, s_en, s_ex
#             )

#             ratio = self._criterion(
#                     strategy_returns, annualisation_factor, risk_free_rate
#             )

#             if self.optimal:
#                 if ratio > self.ratio:
#                     self.ratio = ratio
#                     self.parameters =  np.array([t, k, d, l_en, l_ex, s_en, s_ex])
#             else:
#                 argmin = self.top_n_strategies[:, self.RATIO].argmin()
                
#                 if ratio > self.top_n_strategies[argmin, self.RATIO]:
#                     self.top_n_strategies[argmin] = [
#                         t, k, d, l_en, l_ex, s_en, s_ex, ratio
#                     ] 
    
    
    
    
# class t(BaseStrategy):
    
#     # ======================================================================
#     # Constants
#     # ======================================================================
    
#     PRICE, FASTK, FASTD, RATIO = 0, 1, 2, 7
    
#     def __init__(self,
#                  criterion = 'sharpe',
#                  optimal = True,
#                  top_n = False,
#                  length_min = 2,
#                  stepsize = 20,
#                  oversold_upper = 30,
#                  oversold_lower = 10,
#                  overbought_upper = 90,
#                  overbought_lower = 50,
#                  indicator_params = {}):
        
#         super().__init__(criterion = 'sharpe',
#                         optimal = True,
#                         top_n = False,
#                         length_min = 2,
#                         stepsize = 20)
        
#         self.oversold_upper = oversold_upper
#         self.oversold_lower = oversold_lower
#         self.overbought_upper = overbought_upper
#         self.overbought_lower = overbought_lower
#         self.indicator_params = indicator_params
#         self.parameter_grid = self._parameter_grid()


#     def _price_indicator(self, X, timeperiod, fastk, fastd):
#         ind = StochasticRsi(timeperiod, fastk, fastd)
        
#         return ind.fit_transform(X)
        
#     def _long_signal(self, _price_indicator, long_entry, long_exit):
        
#         # Use np.insert if shift is greater than 1
#         signal_entry = _price_indicator[:, self.FASTK] > long_entry
#         signal_hold = _price_indicator[:, self.FASTK] > long_exit
        
#         # Define the long signal
#         long = signal_entry | signal_hold
        
#         return long[:-1] * 1
    
    
#     def _short_signal(self, _price_indicator, short_entry, short_exit):
        
#         #Use np.insert if shift is greater than 1
#         signal_entry = _price_indicator[:, self.FASTK] < short_entry
#         signal_hold = _price_indicator[:, self.FASTK] < short_exit
        
#         # Define the long signal
#         short = signal_entry | signal_hold
        
#         return short[:-1] * -1
    
#     def fit(self, X, y=None, annualisation_factor=252, risk_free_rate=0.00):
        
#         super()._fit(X, y, annualisation_factor, risk_free_rate)
#         return self
    
        
# s = t(indicator_params = dict(time_period = 14, fastk = 5, fastd=5))

# s.fit(x)