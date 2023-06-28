import joblib
from bike_availability import logger
from sklearn.pipeline import Pipeline
import os
from pathlib import Path
import pandas as pd
import numpy as np
from bike_availability.config.config import conf
from bike_availability.data_manager.data_manager import data_manager
import datetime
from pydantic import BaseModel, Field

USEFUL_COLS = conf.get_config('columns')['useful']

class PredictionData(BaseModel):
    month: int
    day: int
    hour: int
    ctx_4: float = Field(..., alias='ctx-4')
    ctx_3: float = Field(..., alias='ctx-3')
    ctx_2: float = Field(..., alias='ctx-2')
    ctx_1: float = Field(..., alias='ctx-1')
    lat: float
    lon: float
    altitude: int
    icon: object
    humidity: float
    windspeed: float
    precip: float
    temp: float

    
class PredictPipeline:
    def __init__(self, conf):
        best_model_path = conf.get_config('trainer_paths')['best_model_path']
        self.transformer_path = None

        if len(os.listdir(best_model_path)) == 0:
            exc_str = 'There is no best model, please train a model'
            logger.info(exc_str)
            raise Exception(exc_str)
        
        for f in os.listdir(best_model_path):
            if f.split('_')[0] == 'model':
                self.model_path = Path.joinpath(best_model_path, f)
            elif f.split('_')[0] == 'transformer':
                self.transformer_path = Path.joinpath(best_model_path, f)

    def predict(self, X: pd.DataFrame):
        
        model = joblib.load(self.model_path)
        if self.transformer_path:
            transformer = joblib.load(self.transformer_path)
            estimator = Pipeline([
            ('transformer', transformer),
            ('model', model)
            ])
        else:
            transformer = None
            estimator = model
        X.columns = [col if 'ctx' not in col else col.replace('_', '-') for col in X.columns]
        preds = estimator.predict(X)
        logger.info(f"predicted value: {preds}")
        return preds



if __name__ == '__main__':
    from bike_availability.config.config import conf

    predictor = PredictPipeline(conf)
    data = {"month":"1","day":"2","hour":"3","ctx-4":"4","ctx-3":"5","ctx-2":"6","ctx-1":"7","lat":"8","lon":"9","altitude":"10","icon":"Partially cloudy","humidity":"1","windspeed":"1","precip":"1","temp":"10"}
    preds = predictor.predict(data)
    print(preds)

            