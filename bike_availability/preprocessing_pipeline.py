import numpy as np
import pandas as pd
import os
import pandas as pd
from pytz import timezone
import calendar
from pathlib import Path
from .config.config import Config
from .data_manager.data_manager import data_manager
from bike_availability import logger

#this is a template

class ProcessingPipeline:
    def __init__(self, config):
        self.pipeline_config = config.get_config('preprocessing')
        self.data_paths = config.get_config('data_paths')

    def preprocess_bicing_file(self, df :str) -> pd.DataFrame:
        """
        Performs some basic data cleaning and feature engineering steps to raw bicing data obtained from Ajuntament de Barcelona.
        ----------------------------------------------------------
    
        Parameters:
            file: path to the bicing file to be processed
            cols: columns that need to be selected

        Returns:
            Pandas DataFrame with the processed data
        """
        local_time = timezone('Europe/Madrid')
        utc_time = timezone('UTC')
        data = df.copy()
        #filter columns
        data = data[self.pipeline_config['bicing_cols']]
        
        #preprocess status col
        mapper = {
            'IN_SERVICE': 1,
            'NOT_IN_SERVICE': 0,
            'MAINTENANCE': 0,
            'PLANNED': 0,
            'END_OF_LIFE': 0
                    }
        data['status'] = data.status.map(mapper)
        #process timestamp and engineer datetime features
        data['last_reported'] = pd.to_datetime(data['last_reported'], unit = 's')
        data['last_reported'] = data.last_reported.dt.tz_localize(utc_time)
        data['last_reported'] = data.last_reported.dt.tz_convert(local_time)
        data['last_reported'] = pd.to_datetime(data.last_reported)
        data['month'] = data.last_reported.dt.month
        data['day'] = data.last_reported.dt.day
        data['hour'] = data.last_reported.dt.hour
        data['year'] = data.last_reported.dt.year
        data['dayofweek'] = data.last_reported.dt.dayofweek

        #resample data hourly aggregating using the mean
        resampled_data = (
            data.groupby(
                ['station_id', pd.Grouper(key = 'last_reported', freq = 'H')]
            ).mean().reset_index())
        
        #create context columns
        for ctx in range(4, 0, -1):
            resampled_data[f'ctx-{ctx}'] = resampled_data.num_bikes_available.shift(ctx)
        
        #remove the rows used to create the context
        resampled_data = resampled_data.iloc[5::4]
        return resampled_data

    def merge_station_data(self, data: pd.DataFrame, not_sub: bool = True) -> pd.DataFrame:
        """
        Performs some cleaning and transformations to the dataframe after merging it with the stations information.
        """
        data = data.copy()
        station_df = data_manager.get_station_data(self.data_paths['stations_info_path'])
        station_df = station_df[self.pipeline_config['station_cols']]
        merged_data =  data.merge(station_df, on = 'station_id', how = 'left')
        if not_sub:
            merged_data['percentage_docks_available'] = merged_data['num_bikes_available'] / merged_data['capacity']
            norm_cols = [col for col in merged_data.columns if 'ctx' in col]
            merged_data[norm_cols] = merged_data[norm_cols].div(merged_data['capacity'], axis = 0)
            merged_data[norm_cols] = merged_data[norm_cols].clip(0,1)
            merged_data['percentage_docks_available'] = merged_data['percentage_docks_available'].clip(0,1)
        return merged_data
    

    def merge_meteo_data(self, data:pd.DataFrame, train:bool = True) -> pd.DataFrame:
        logger.info("Merging meteo data")
        if train:
            meteo_data = data_manager.read_csv(self.data_paths['meteo_data_train'])
        else:
            meteo_data = data_manager.read_csv(self.data_paths['meteo_data_test'])
        meteo_data = meteo_data[self.pipeline_config['meteo_cols']]

        return data.merge(meteo_data, on = ['year', 'month', 'day', 'hour'], how = 'inner')

    def process_test_data(self, data:pd.DataFrame):
        march_2023_dict = {}
        for day in range(1, 32):
            weekday = calendar.weekday(2023, 3, day)
            march_2023_dict[day] = calendar.day_name[weekday]
        
        data['dayofweek'] = data.hour.map(march_2023_dict)
        data['year'] = 2023
        return data

    def execute(self) -> pd.DataFrame:
        """
        Main method of the pipeline. Runs an execution of the pipeline according to the config.

        Output: Cleaned dataframe with data almost ready to be used for ML.
        """
        logger.info('Starting pipeline execution')
        DATA_PATH = self.data_paths['bicing_data']
        DATA_FILENAMES = os.listdir(DATA_PATH)

        if self.pipeline_config['fetch_data']:
            logger.info('fetching data')
            data_manager.download_data()

        #processing raw bicing data
        df_bicing_processed = pd.DataFrame()
        logger.info('Processing raw data')
        logger.info("Parsing datetime data")
        logger.info("Resampling and creating sequences")
        for file in DATA_FILENAMES:
            df_temp = data_manager.read_csv(Path.joinpath(DATA_PATH, file))
            df_temp = self.preprocess_bicing_file(df_temp)
            df_bicing_processed = pd.concat([df_bicing_processed, df_temp], axis= 0)
        
        df_bicing_processed = df_bicing_processed.drop_duplicates()
        #merging station_data and selecting stations to use
        logger.info('merging with station_data')
        df_bicing_processed = self.merge_station_data(df_bicing_processed)
        stations_to_use = data_manager.load_stations_to_use(self.data_paths['stations_to_use_path'])
        logger.info('filtering stations')
        df_bicing_processed = df_bicing_processed[df_bicing_processed.station_id.isin(stations_to_use)]

        if self.pipeline_config['use_test']:
            logger.info('Processing submission data')
            df_sub = data_manager.read_csv(self.data_paths['submission_data'])
            print(df_sub)
            df_sub_processed = self.process_test_data(df_sub)
            print(df_sub_processed)
            logger.info('Merging station data for submission data')
            df_sub_processed = self.merge_station_data(df_sub_processed, not_sub=False)

        if self.pipeline_config['use_meteo_data']:
            df_bicing_processed = self.merge_meteo_data(df_bicing_processed)
            if self.pipeline_config['use_test']:
                logger.info('Merging meteo data for submission data')
                df_sub_processed = self.merge_meteo_data(df_sub_processed, train = False)

        if self.pipeline_config['save_file']:
            logger.info('saving processed train data to file')
            data_manager.save_data(
                self.data_paths['processed_data'],
                self.pipeline_config['output_name_train'],
                df_bicing_processed)
            
            if self.pipeline_config['use_test']:
                logger.info('Saving processed submission data to file')
                print(df_sub_processed)

                data_manager.save_data(
                    self.data_paths['processed_data'],
                    self.pipeline_config['output_name_sub'],
                    df_sub_processed)
                
        return df_bicing_processed


if __name__ == '__main__':
    # processingPipeline = ProcessingPipeline(config=conf)
    # processingPipeline.execute()
    conf = Config()
    print(conf.config)



 