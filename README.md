# Bike_availability_prediction

This repository contains the implementation of an end-to-end machine learning pipeline for the Capstone Project of the postgraduate program in Data Science from Universitat de Barcelona.

The project focuses on developing a comprehensive machine learning pipeline that covers the entire workflow, from data preprocessing to model training and model serving. It aims to provide a practical and hands-on experience in applying data science techniques to real-world problems.

## Table of Contents
- [Project Description](#project-description)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Trainer](#model-trainer)
- [Predict](#predict)
- [Config](#config)
- [data_manager](#data-manager)
- [train.py](#trainpy)
- [main.py](#mainpy)
- [Installation](#installation)
- [In the future](#in-the-future)

## Project Description

The project is an end-to-end machine learning pipeline for bike availability prediction. It includes several modules that handle different aspects of the pipeline. The main file that launches the pipeline is `train.py`, and it is fully controlled through a `config.yaml` configuration file.

The folder structure of the project consists of the following:

- bike_availability_prediction
    - artifacts
        - best_model
            - best_model.pkl
            - best_transformer.pkl (optional)
        - models
            - list of
            - pkl models
        - transformers
            - list of
            - pkl transformers
    - data
        - raw_data
        - meteo_data
    - Notebooks
        - EDA
        - model training
    - model_tracker.csv
    - stations_to_use.csv
    - stations_info.csv
    - config.yaml
    - local_config.yaml
    - train.py
    - main.py
    - bike_availability
        - __init__.py
        - config
            - __init__.py
            - config.py
        - data_manager
            - __init__.py
            - data_manager.py
        - models
            - __init__.py
            - list of
            - models
        - model_trainer.py
        - predict.py
        - preprocessing_pipeline.py
----------

The main module of this project is `bike_availability`, which consists of:

### Preprocessing Pipeline

The `preprocessing_pipeline.py` contains the `PreprocessingPipeline` class which is responsible for downloading the data and preprocessing it into a usable dataset. It performs tasks such as data retrieval, cleaning, feature engineering, and data transformation to prepare the data for model training and prediction.

### Model Trainer

The `model_trainer.py` contains the `ModelTrainer` class which focuses on training a model, tuning hyperparameters, and evaluating the model's performance. It utilizes the preprocessed dataset to train simple machine learning models from the `models` module, and uses techniques such as cross-validation and grid search to find the best combination of hyperparameters. The models are evaluated and saved onto the artifacts directory. If the model happens to score the best it will be saved to `artifacts/best_model/` and will be served through the REST API.

In order to use the model trainer you will need to open a mlflow server and set the local url on the `config.yaml` file.

### Predict

The `predict.py` contains the `PredictPipeline` class which enables making predictions based on the trained model. It also contains the `PredictionData` class which inherits from Pydantic `BaseModel`.  It provides the necessary functionality to load the saved model from the `artifacts/best_model` directory and utilize it to make predictions on new data. This class integrates the model into an API for real-time predictions.



### Config

The config module reads the `config.yaml` and the `local_config.yaml`. The `config.yaml` file is in charge of controlling all the steps of the pipeline while the `local_config.yaml` file is in charge of setting the paths.

An example of a config.yaml file would be the following:

```yaml
config.yaml
model_to_use: xgboost_simple #model from the models module
columns: #columns to use, we set the target, the columns to use and of those which are categorical
  target: 'percentage_docks_available'
  useful: ['month', 'day', 'hour', 'ctx-4', 'ctx-3', 'ctx-2', 'ctx-1', 'lat', 'lon', 'altitude', 'icon', 'humidity', 'windspeed', 'precip', 'temp']
  categorical: ['month', 'day', 'hour', 'icon']

preprocessing: #controls the preprocessing pipeline
  bicing_cols:  # columns to use from bicing data
    - 'station_id'
    - 'last_reported'
    - 'num_bikes_available'
    - 'is_installed'
    - 'is_renting'
    - 'status'

  station_cols: # columns to use from the station info data
    - 'station_id'
    - 'name'
    - 'lat'
    - 'lon'
    - 'capacity'
    - 'altitude'

  meteo_cols: # columns to use from meteo data
    - 'year'
    - 'month'
    - 'day'
    - 'hour'
    - 'temp'
    - 'humidity'
    - 'windspeed'
    - 'precip'
    - 'icon'
  use_meteo_data: True
  use_test: True
  fetch_data: True
  save_file: True
  output_name_train: train_dataset.csv
  output_name_sub: sub_dataset.csv

train: #controls the model trainer
  mlflow:
    tracking_uri: 'http://127.0.0.1:5000'
    mlflow_experiment: 'xgboost'
    mlflow_run: 'full_pipeline_fixed_test'
  tuning: 
    mode: 'standard' # can be standard or custom. If custom, create a new class inheriting from BaseModelTrainer and write a custom train method
    strategy: random # can be either random or grid
    n_iter: 50 # Only necessary for random strategy
    parameters: # use model__ prefix so that the sklearn pipeline will detect the parameters.
      model__alpha: [0.4, 0.5, 0.6]
      model__colsample_bytree: [0.9, 1]
      model__gamma: [0,0.1]
      model__lambda: [0.4, 0.5]
      model__learning_rate: [0.1, 0.2, 0.3]
      model__max_depth: [6,7,8]

    
pipeline: # train.py, which steps of the full pipeline to use
  preprocess: True
  train: True
    
```

```yaml
local_config.yaml
data_paths:
  bicing_data: 'ROOT/data/raw_data'
  meteo_data_train: 'ROOT/data/meteo_data/train_meteo_data.csv'
  meteo_data_test: 'ROOT/data/meteo_data/test_meteo_data.csv'
  submission_data: 'ROOT/data/submission_data/metadata_sample_submission.csv'
  processed_data: 'ROOT/datasets/'
  sample_sub: 'ROOT/data/submission_data/sample_submission.csv'
  stations_to_use_path: 'ROOT'
  stations_info_path: 'ROOT/station_info.csv'
  clean_train_dataset: 'ROOT/datasets/train_dataset.csv'
  clean_test_dataset: 'ROOT/datasets/sub_dataset.csv'
  submission_directory: 'ROOT/subs'
 

trainer_paths:
  best_model_path: 'ROOT/artifacts/best_model'
  models_path: 'ROOT/artifacts/models'
  transformers_path: 'ROOT/artifacts/transformers'
  plots_path: 'ROOT/plots'
  tracker_path: 'ROOT/model_tracker.csv'
```

### data_manager

The data_manager module provides the class `DataManager` which is in charge of performing the loading and storing of the data.

---------

Other than that we have the `train.py` and `main.py` scripts, which are in charge of launching the full pipeline and deploying a REST API to serve the best model respectively.

### train.py

By running the `train.py` script, the pipeline will sequentially execute the preprocessing steps, train the model, and save the best model for future predictions.

### main.py
Serves the best model through a simple REST endpoint using FastApi. The API uses the `PredictPipeline` class to load the best model and make the predictions.
In order to get the API up and running you can run the following command:
```bash
uvicorn main:app
```

---------------
## Installation

In order to setup the project run the commands below.

```bash
# Installation steps
$ git clone https://github.com/albertventura/Bike_availability_prediction.git
$ cd Bike_availability_prediction
$ pip install -r requirements.txt
```

Meteorological data is not provided/obtained through the preprocessing_pipeline since I obtained it from scraping a meteorology API daily. If you have some meteorological data of your own, make sure to add it to the data/meteo_data directory and also make sure that it has the columns ['year', 'month', 'day', 'hour'] so it can be merged to the bicing data.

## In the future

I would like to add a few things I did not have time during the alloted time for this project.

- Change completely the model_trainer class. After fiddling with it I realized it does not make sense to follow a "let's automate everything" approach". It makes more sense to make all this experiments through a notebook and then create a pipeline for the best performing models.

- Dockerize the application.

- Deploy it to AWS. My current idea is to use S3 to store the artifacts and serve the model through an API GW + Lambda setup.
