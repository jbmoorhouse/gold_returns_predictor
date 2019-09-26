from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

import pandas as pd
import numpy as np
import talib

from tqdm import tqdm

from column_transformers.technical_indicators import StochasticRsi

from abc import ABC, abstractmethod
from itertools import product
from operator import itemgetter


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
    

    
class BaseIndicator(ABC, BaseEstimator):
    
    @abstractmethod
    def __init__(self, 
                 criterion = 'sharpe', 
                 optimal = True, 
                 top_n = False, 
                 **kwargs):
        
        self.criterion = criterion
        
        if optimal and not top_n:
            self.optimal = optimal
            self.top_n = top_n
            self.ratio = 0
            
        elif optimal and top_n:
            warnings.warn("Both 'optimal' and 'top_n' were detected." 
                          "'optimal' has been set to 'False'. Remove top_n"
                          " if this behaviour is not desired")
            
            self.optimal = False
            
            if isinstance(top_n, int):
                self.top_n = top_n
            else:
                raise TypeError("'top_n must be of type 'int")  
                
        elif not optimal and not top_n:
            raise ValueError("Please specify either 'optimal' or 'top_n'.")
        
            
        self.indicator_params = kwargs
        self.indicator_param_names = [k for k, v in kwargs.items()]
        
        
    # ======================================================================
    # User implemented methods
    # ======================================================================    
    
    @abstractmethod
    def _price_indicator(self, X, **kwargs):
        pass
    
    
    
class BaseStrategy(BaseIndicator, TransformerMixin):
    
    # ======================================================================
    # Constructors
    # ======================================================================
    
    @abstractmethod
    def __init__(self, 
                 annualisation_factor = 252,
                 risk_free_rate = 0.00,
                 os_region=None,
                 ob_region=None,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        if not self.indicator_params:
            raise TypeError("Please specify indicator parameters. See"
                            " Documentation for example")
        
        self.annualisation_factor = annualisation_factor
        self.risk_free_rate = risk_free_rate
        
        regions = (os_region, ob_region)
        
        if any(regions) and not all(regions):
            raise TypeError("NoneType was detected for one or more of the"
                            " overbought or oversold regions")
        elif all(regions):
            
            if all(isinstance(r, range) for r in regions):
                self.long_short_regions = True
                self.strategy_params = dict(
                    long_entry = os_region,
                    long_exit = ob_region,
                    short_entry = ob_region,
                    short_exit = os_region
                )
                
                # Define the long/short actions
                self.strategy_param_names = [
                    k for k, v in self.strategy_params.items()
                ]
            else:
                raise TypeError("os_region and ob_region must both be "
                               "of type 'range'.")
        else:
            self.long_short_regions = False
        
        self.os_region = os_region
        self.ob_region = ob_region
        self.parameter_grid = self._parameter_grid()
        
        if self.top_n:
            p = {k:0. for k, v in self.model_parameters.items()}
            r = dict(ratio = 0.)
            
            self.top_n_strategies = [{**p,**r} for n in range(self.top_n)]

            
    def _parameter_grid(self):
        self.model_parameters = self.indicator_params

        if self.long_short_regions:
            self.model_parameters = {
                **self.model_parameters, **self.strategy_params
            }
            
        param_names = [k for k, v in self.model_parameters.items()]
        param_ranges = [v for k, v in self.model_parameters.items()]

        parameter_grid = [
            {i:j for i, j in zip(param_names, row)} 
            for row in product(*param_ranges)
        ]
        
        return parameter_grid
        
    # ======================================================================
    # User implemented methods
    # ======================================================================
    
    
    @abstractmethod
    def _long_signal(self, price_indicator, **kwargs):
        pass
    
    
    @abstractmethod
    def _short_signal(self, price_indicator, **kwargs):
        pass
    
    # ======================================================================
    # Strategy methods
    # ======================================================================
    
    def _asset_returns(self, price_indicator):
        return (
            np.diff(price_indicator[:, self.PRICE]) / 
            price_indicator[:-1, self.PRICE]
        )
    
    def _strategy_returns(self,
                          price_indicator, 
                          long_entry=None, 
                          long_exit=None, 
                          short_entry=None, 
                          short_exit=None):
        
        # generate strategy long/short signal and asset returns
        if self.long_short_regions:
            long_signal = self._long_signal(
                price_indicator, long_entry, long_exit
            )
            short_signal = self._short_signal(
                price_indicator, short_entry, short_exit
            )
        else:
            long_signal = self._long_signal(price_indicator)
            short_signal = self._short_signal(price_indicator)
        
        # calculate the assets simple returns 
        asset_returns = self._asset_returns(price_indicator)
        
        return (asset_returns*long_signal) + (asset_returns*short_signal)
    
    
    def _criterion(self, 
                   strategy_returns, 
                   annualisation_factor, 
                   risk_free_rate):
        
        excess_return = strategy_returns.mean() - risk_free_rate
        sharpe_ratio = excess_return / strategy_returns.std()
        
        return np.sqrt(annualisation_factor) * sharpe_ratio 
    
    
    def fit(self, X, y=None):
        
        X = check_array(X)
   
        for p in tqdm(self.parameter_grid):
            ind_params={i:p[i] for i in self.indicator_param_names}
            price_indicator = self._price_indicator(X, **ind_params)
            
            if self.long_short_regions:
                strategy_params={i:p[i] for i in self.strategy_param_names}
                strategy_returns = self._strategy_returns(
                    price_indicator, **strategy_params
                )
            else:
                strategy_returns = self._strategy_returns(price_indicator)
            
            ratio = self._criterion(
                strategy_returns, 
                self.annualisation_factor, 
                self.risk_free_rate
            )

            if self.optimal:
                if ratio > self.ratio:
                    self.ratio = ratio
                    self.parameters = p
            else: 
                current_ratios = [d['ratio'] for d in self.top_n_strategies]
                min_strat = min(
                    enumerate(self.top_n_strategies), key=lambda k:k[1]['ratio']
                )
                
                if ratio in current_ratios:
                    continue
                elif ratio > min_strat[1]['ratio']:
                    self.top_n_strategies[min_strat[0]] = {**p, **{"ratio":ratio}}
        
        return self
    
    
    def transform(self, X, y=None, n=None):
        
        X = check_array(X)
        
        if self.top_n:
            if isinstance(n, int):
                p = self.top_n_strategies[n]
            else:
                raise ValueError("'n = {}' is invalid. Please specify number" 
                                 "between 0 and {}".format(
                                     n, len(self.top_n_strategies) - 1)
                                )
        else:
            if n is not None:
                raise ValueError("'n = {}' is invalid. 'optimal' strategy" 
                                 "paramters only. 'n' must be NoneType".format(n))
            
            p = self.parameters
        
        ind_params = {i:p[i] for i in self.indicator_param_names}
        price_indicator = self._price_indicator(X, **ind_params)

        if self.long_short_regions:
            strategy_params = {i:p[i] for i in self.strategy_param_names}
            strategy_returns = self._strategy_returns(
                price_indicator, **strategy_params
            )

        else:
            strategy_returns = self._strategy_returns(price_indicator)
                
        # strategy returns excludes np.nan from asset_returns. This is account
        # for by slicing price_indicator appropriately.
        return np.c_[price_indicator[1:], strategy_returns]