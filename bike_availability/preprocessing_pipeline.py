import numpy as np
import pandas as pd
import os
import pandas as pd
from pytz import timezone
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
        
        #process timestamp and engineer datetime features
        data['last_reported'] = pd.to_datetime(data['last_reported'], unit = 's')
        data['last_reported'] = data.last_reported.dt.tz_localize(utc_time)
        data['last_reported'] = data.last_reported.dt.tz_convert(local_time)
        data['last_reported'] = pd.to_datetime(data.last_reported)
        data['month'] = data.last_reported.dt.month
        data['day'] = data.last_reported.dt.day
        data['hour'] = data.last_reported.dt.hour
        data['year'] = data.last_reported.dt.year
   
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
    

    def preprocess_meteo_data(self, files_path: str) -> pd.DataFrame:
        pass 

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
        for file in DATA_FILENAMES:
            df_temp = data_manager.read_csv(Path.joinpath(DATA_PATH, file))
            df_temp = self.preprocess_bicing_file(df_temp)
            df_bicing_processed = pd.concat([df_bicing_processed, df_temp], axis= 0)
        
        #merging station_data and selecting stations to use
        logger.info('merging with station_data')
        df_bicing_processed = self.merge_station_data(df_bicing_processed)
        stations_to_use = data_manager.load_stations_to_use(self.data_paths['stations_to_use_path'])
        logger.info('filtering stations')
        df_bicing_processed = df_bicing_processed[df_bicing_processed.station_id.isin(stations_to_use)]

        if self.pipeline_config['use_meteo_data']:
            #meteosteps would go here
            #...
            pass

        if self.pipeline_config['save_file']:
            logger.info('saving processed data to file')
            data_manager.save_data(self.data_paths['processed_data'], self.pipeline_config['output_name'], df_bicing_processed)

        return df_bicing_processed


if __name__ == '__main__':
    # processingPipeline = ProcessingPipeline(config=conf)
    # processingPipeline.execute()
    conf = Config()
    print(conf.config)



 