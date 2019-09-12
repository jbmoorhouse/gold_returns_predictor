import pandas as pd
from sklearn.base import TransformerMixin

class DateDummy(TransformerMixin):
    def __init__(self, *args):
        self.args = args

    def fit(self, X, y=None):
        return self
    
    def _get_dummy_variables(self, X):
        for arg in self.args:
            date_feature = getattr(X.index.to_series().dt, arg)
            
            if callable(date_feature):
                date_feature = date_feature()
                
            yield pd.get_dummies(date_feature)
    
    def transform(self, X, y=None):
        dummy_df = list(self._get_dummy_variables(X))
        X = pd.concat([X, *dummy_df], axis = 1)
        
        return X