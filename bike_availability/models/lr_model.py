from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from bike_availability.config.config import conf


class lrModel:
    def __init__(self, config):
        self.cat_cols = config.get_config('columns')['categorical']
        self.cont_cols = [col for col in config.get_config('columns')['useful'] if col not in self.cat_cols]
        self.transformer = ColumnTransformer([
            ('sc', StandardScaler(), self.cont_cols),
            ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist'), self.cat_cols)
        ])
        self.model = LinearRegression()
        
    def get_artifacts(self):
        return self.transformer, self.model