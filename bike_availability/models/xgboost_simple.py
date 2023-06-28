from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator

class Passthrough(BaseEstimator, TransformerMixin):
    def __init__(self, cols:list):
        super().__init__()
        self.cols = cols

    def fit(self, X,y=None):
        return self
    
    def transform(self, X, y= None):
        X_ = X.copy()
        return X_[self.cols]

class XgboostSimple:
    def __init__(self, config):
        self.cat_cols = config.get_config('columns')['categorical']
        self.cont_cols = [col for col in config.get_config('columns')['useful'] if col not in self.cat_cols]
        self.model = XGBRegressor()
        self.transformer = ColumnTransformer([
            ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist'), self.cat_cols),
            ('passthrough', Passthrough(self.cont_cols), self.cont_cols)
        ])

    def get_artifacts(self):
        return self.transformer, self.model