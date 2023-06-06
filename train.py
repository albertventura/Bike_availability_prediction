from bike_availability.preprocessing_pipeline import ProcessingPipeline
from bike_availability.model_trainer import ModelTrainerBase
from bike_availability.config.config import conf
from bike_availability.data_manager.data_manager import data_manager
from bike_availability import logger
from bike_availability.models.get_model import get_model_from_config

USEFUL_COLS = conf.get_config('columns')['useful']
CATEGORICAL_COLS = conf.get_config('columns')['categorical']
CONTINUOUS_COLS = [col for col in USEFUL_COLS if col not in CATEGORICAL_COLS]
TARGET = conf.get_config('columns')['target']

if conf.get_config('pipeline')['preprocess']:
    logger.info("Starting processing pipeline: ")
    processor = ProcessingPipeline(config=conf)
    df = processor.execute()
else:
    logger.info("Skipping processing steps, reading dataset directly")
    df = data_manager.read_csv('./datasets/bicing_dataset.csv')

if conf.get_config('pipeline')['train']:
    logger.info("Starting Model Trainer: ")
    transformer, model = get_model_from_config(config=conf)

    mt = ModelTrainerBase(transformer=transformer, model=model, config=conf)
    mt.run(df)


