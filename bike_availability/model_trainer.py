import mlflow
import joblib
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from scipy import stats
from bike_availability.config.config import conf
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
from bike_availability import logger
import requests

#PLEASE FIX THIS LATER
USEFUL_COLS = conf.get_config('columns')['useful']
CATEGORICAL_COLS = conf.get_config('columns')['categorical']
CONTINUOUS_COLS = [col for col in USEFUL_COLS if col not in CATEGORICAL_COLS]
TARGET = conf.get_config('columns')['target']

def RMSE(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

class ModelTrainerBase:
    def __init__(self, transformer, model, config):
        self.transformer = transformer
        self.model = model
        self.mlflow_config = config.get_config('train')['mlflow']
        self.param_config = config.get_config('train')['tuning']
        self.train_paths = config.get_config('trainer_paths')
        self.run_name = f"{self.mlflow_config['mlflow_run']}_{datetime.now().strftime('%d%m%Y%H%M%S')}"
        self.model_name = f"model_{self.run_name}"
        self.transformer_name = f"transformer_{self.run_name}"

    def __repr__(self):
        return f"Model trainer for {self.model}"

    def setup_mlflow(self):
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['mlflow_experiment'])
        return mlflow.start_run(run_name=self.run_name)
    
    def  train(self,data):
        raise NotImplementedError('This is a base class. You need to overwrite this method')
    
    def plot_residuals(self, residuals):
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))

        fig, ax = plt.subplots(1,2, figsize = (12,6))
        #QQ plot
        ax[0].scatter(theoretical_quantiles, sorted_residuals, label = 'Residuals')
        #reference
        ax[0].plot([np.min(theoretical_quantiles), np.max(theoretical_quantiles)],
                [np.min(sorted_residuals), np.max(sorted_residuals)],
                color='red', label = 'Reference')

        ax[0].set_xlabel('Theoretical Quantiles')
        ax[0].set_ylabel('Ordered Residuals')
        ax[0].set_title('QQ Plot')
        ax[0].legend()

        #histogram
        ax[1].hist(residuals, bins = 100)

        ax[1].set_xlabel('Residuals')
        ax[1].set_ylabel('Count')
        ax[1].set_title('Residuals Histogram')

        if not os.path.exists(self.train_paths['plots_path']):
            os.makedirs(self.train_paths['plots_path'])
        plot_path = Path.joinpath(
            self.train_paths['plots_path'], f"plot_{self.mlflow_config['mlflow_experiment']}_{self.run_name}.png"
            )
        fig.savefig(plot_path)
        return fig, plot_path
    
    def model_is_best(self):
        model_tracker = pd.read_csv(self.train_paths['tracker_path'])
        if not model_tracker.empty:
            best_model_name = model_tracker[model_tracker.RMSE == model_tracker.RMSE.min()].model_name.values[0]
        else:
            logger.info("This is the first model, best model is: {self.model_name}")
            return True
        logger.info(f"Trained model is {self.model_name}")
        logger.info(f"Best model is: {best_model_name}")
        return best_model_name == self.model_name
    
    def save_model(self, is_best):
        best_path = self.train_paths['best_model_path']
        model_path = self.train_paths['models_path']
        transformers_path = self.train_paths['transformers_path']
        print("THIS MODEL IS BEST?", is_best)
        best_files = os.listdir(best_path)
        print("CURRENTLY IN BEST_FILES THERE IS: ",best_files)
        if is_best:
            #if the list is not empty move the models and transformers to the other directory
            if best_files:
                for file in best_files:
                    print(file.split('_')[0])
                    if file.split('_')[0] == 'model':
                        shutil.move(
                            Path.joinpath(best_path, file),
                            Path.joinpath(model_path, file)
                            )
                    elif file.split('_')[0] == 'transformer':
                        shutil.move(
                            Path.joinpath(best_path, file),
                            Path.joinpath(transformers_path, file)
                            )
                    else:
                        raise ValueError('File should contain either model or transformer at the start')
                logger.info(f"artifacts saved to {best_path}")
                joblib.dump(self.model, Path.joinpath(best_path, self.model_name + '.pkl'))
                if self.transformer:
                    joblib.dump(self.transformer, Path.joinpath(best_path, self.transformer_name + '.pkl'))

        else:
            logger.info(f"model saved to {model_path}")
            joblib.dump(self.model, Path.joinpath(model_path, self.model_name + '.pkl'))
            if self.transformer:
                logger.info(f"transformer saved to {transformers_path}")
                joblib.dump(self.transformer, Path.joinpath(transformers_path, self.transformer_name + '.pkl'))
                
            

    def run(self, data:pd.DataFrame):

        logger.info('Splitting year 2022 as validation and  2020 and 2021 as training')
        X_train = data[data.year.isin([2020, 2021])].copy()
        X_test = data[data.year == 2022].copy()
        y_train = X_train.pop(TARGET)
        y_test = X_test.pop(TARGET)

        logger.info(f"Starting Mlflow local server at: {self.mlflow_config.get('tracking_uri')}")

        with self.setup_mlflow():
            
            if self.param_config.get('mode') == 'standard':
                report = self.tune_params(X_train, y_train)
                
                
                if self.transformer:
                    estimator = Pipeline([
                        ('transformer', self.transformer),
                        ('model', self.model.set_params(**best_params))
                    ])
                    best_params = {k.split('__')[1]:v for k, v in report['best_params'].items()}
                else:
                    X_train = X_train[USEFUL_COLS]
                    X_test = X_test[USEFUL_COLS]

                    estimator = self.model
                    best_params = report['best_params']

                estimator.fit(X_train, y_train)
                preds = estimator.predict(X_test)
                score = RMSE(y_test, preds)
                residuals = y_test - preds
                fig, img_path = self.plot_residuals(residuals=residuals)
                
                logger.info(f"best_params: {best_params}")
                mlflow.log_params(best_params)
                logger.info(f"saving residuals figure")
                mlflow.log_figure(fig, img_path)
                logger.info(f"CV Score:  {report['best_score']}")
                logger.info(f"Test RMSE:  {score}")
                mlflow.log_metrics(dict(zip(['cv_score', 'test_score'], [report['best_score'], score])))

                with open(self.train_paths['tracker_path'], 'a') as f:
                    f.write(f"{self.model_name},{self.transformer_name},{score}\n")
                f.close()

                #train with all data
                logger.info("Starting training with all data")
                y = data.pop(TARGET)
                if self.transformer:
                    X_tran = self.transformer.fit_transform(data).copy()
                    self.model.fit(X_tran, y)
                else:
                    self.model.fit(data[USEFUL_COLS], y)

                self.save_model(self.model_is_best())

            elif self.param_config.get('mode') == 'custom':
                self.train(data)

            else: 
                raise ValueError(f"Wrong config mode. Choose either standard or custom. You chose {self.param_config.get('mode')}")
    
    def tune_params(self, X, y):
        logger.info(f"Starting hyperparameter tuning with config: {self.param_config}")
        if self.transformer:
            estimator = Pipeline([
                ('transformer', self.transformer),
                ('model', self.model)
            ])
            params = self.param_config.get('parameters')
        else:
            estimator = self.model
            X = X[USEFUL_COLS].copy()
            params = {k.split('__')[1]:v for k, v in self.param_config.get('parameters').items()}

        #forward chaining CV
        tscv = TimeSeriesSplit(n_splits=5)
        if self.param_config.get('strategy') == 'random':
            tuner = RandomizedSearchCV(
                estimator, params,
                scoring = 'neg_mean_squared_error',
                cv = tscv, verbose = 3, error_score='raise')
            
        elif self.param_config.get('strategy') == 'grid':
            tuner = GridSearchCV(
                estimator, params,
                scoring = 'neg_mean_squared_error', cv = tscv, verbose = 3)
        else:
            raise(ValueError, f"Wrong strategy, choose one of either grid or random. You chose {self.param_config.get('strategy')}")
        
        tuner.fit(X,y)

        best_params = tuner.best_params_
        best_score = tuner.best_score_
        best_rmse = np.sqrt(-best_score)
        
        return {
            'best_params':best_params,
            'best_score':best_score,
            'best_rmse': best_rmse
              }
        
