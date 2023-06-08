import joblib
from bike_availability import logger
from sklearn.pipeline import Pipeline
import os
from pathlib import Path

class PredictPipeline:
    def __init__(self, conf, use_best, model_path=None, transformer_path=None):
        if use_best:
            best_model_path = conf.get_config('trainer_paths')['best_model_path']
            for f in os.listdir(best_model_path):
                if f.split('_')[0] == 'model':
                    self.model_path = Path.joinpath(best_model_path, f)
                elif f.split('_')[0] == 'transformer':
                    self.trasnformer_path = Path.joinpath(best_model_path, f)
        else:
            self.model_path = model_path
            self.transformer_path = transformer_path

    def predict(self, X):
        model = joblib.load(self.model_path)

        try:
            transformer = joblib.load(self.transformer_path)
        except TypeError as e:
            logger.warning(f"{e}, defaulting transformer to None")
            transformer = None
            
        if transformer:
            estimator = Pipeline([
                ('transformer', transformer),
                ('model', model)
            ])
        else:
            estimator = model

        model.predict(X)