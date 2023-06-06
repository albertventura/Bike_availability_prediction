from xgboost import XGBRegressor



class XgboostBaseline:
    def __init__(self, config):
        self.model = XGBRegressor()
        self.transformer = None
    def get_artifacts(self):
        return self.transformer, self.model