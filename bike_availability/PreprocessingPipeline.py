# from data_processing import data_processor
import numpy as np
import pandas as pd
import os
from pyunpack import Archive
import requests
import pandas as pd
from pytz import timezone
from pathlib import Path
from config.config import *

#this is a template

class ProcessingPipeline:
    def __init__(self, config):
        self.pipeline_config = config['preprocessing']
        self.data_paths = config['data_paths']

    def read_csv(self, file: str) -> pd.DataFrame:
        """
        Reads data to a pandas dataframe.
        """
        return pd.read_csv(file)
    

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
        data = data[self.pipeline_config['bicing_cols']] #placeholder
        
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

    def merge_station_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs some cleaning and transformations to the dataframe after merging it with the stations information.
        """
        data = data.copy()
        station_df = self.get_station_data()
        station_df = station_df[self.pipeline_config['station_cols']]
        merged_data =  data.merge(station_df, on = 'station_id', how = 'left')
        merged_data['percentage_docks_available'] = merged_data['num_bikes_available'] / merged_data['capacity']
        norm_cols = [col for col in merged_data.columns if 'ctx' in col]
        merged_data[norm_cols] = merged_data[norm_cols].div(merged_data['capacity'], axis = 0)
        merged_data[norm_cols] = merged_data[norm_cols].clip(0,1)
        merged_data['percentage_docks_available'] = merged_data['percentage_docks_available'].clip(0,1)
        return merged_data

    def preprocess_meteo_data(self, files_path: str) -> pd.DataFrame:
        pass 
    
    def stations_to_use(self, path_to_file):
        return np.loadtxt(Path.joinpath(path_to_file,'stations_to_use.csv')).astype('int')
    
    def download_data(self):
        """
        Downloads bicing data from opendata-ajuntament API.
        """
        if 'raw_data' not in os.listdir(Path.joinpath(ROOT_DIR ,'data')):
            os.system('cd .. & cd data & mkdir raw_data')
            i2m = list(zip(range(1,13), ['Gener','Febrer','Marc','Abril','Maig','Juny','Juliol','Agost','Setembre','Octubre','Novembre','Desembre']))

            for year in [2022, 2021, 2020, 2019]:
                for month, month_name in i2m:
                    fname = f"{year}_{month:02d}_{month_name}_BicingNou_ESTACIONS.7z"
                    if (year == 2019) and (month_name in ['Gener', 'Febrer']):
                        continue
                    os.system(f'curl --location "https://opendata-ajuntament.barcelona.cat/resources/bcn/BicingBCN/{fname}" -o {fname}"')
                    Archive(os.getcwd() +'\\' + fname).extractall(ROOT_DIR + '\\data\\raw_data')
                    os.system(f'del "{year}_{month:02d}_{month_name}_BicingNou_ESTACIONS.7z"')

        else:
            print('Data is already downloaded!')
            
    def get_station_data(self):
        """
        Makes a request to opendata-ajuntament API that returns information about the stations.
        """
        req = 'https://opendata-ajuntament.barcelona.cat/data/dataset/bd2462df-6e1e-4e37-8205-a4b8e7313b84/resource/e5adca8d-98bf-42c3-9b9c-364ef0a80494/download'
        response = requests.request("GET", req)
        if response.status_code != 200:
            print("Unexpected status code ", response.status_code)
        return pd.DataFrame(response.json()['data']['stations'])
    
    def save_data(self, file_path: Path, name: str, data: pd.DataFrame) -> None:
        """
        Saves data to the specified file path with the specified name.
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        data.to_csv(Path.joinpath(file_path, name), index= None)

    def execute(self) -> pd.DataFrame:
        """
        Main method of the pipeline. Runs an execution of the pipeline according to the config.

        Output: Cleaned dataframe with data almost ready to be used for ML.
        """
        print('Starting pipeline execution')
        DATA_PATH = self.data_paths['bicing_data']
        DATA_FILENAMES = os.listdir(DATA_PATH)

        if self.pipeline_config['fetch_data']:
            print('fetching data')
            self.download_data()

        #processing raw bicing data
        df_bicing_processed = pd.DataFrame()
        print('Processing raw data')
        for file in DATA_FILENAMES:
            df_temp = self.read_csv(Path.joinpath(DATA_PATH, file))
            df_temp = self.preprocess_bicing_file(df_temp)
            df_bicing_processed = pd.concat([df_bicing_processed, df_temp])
        
        #merging station_data and selecting stations to use
        print('mergin with station_data')
        df_bicing_processed = self.merge_station_data(df_bicing_processed)
        stations_to_use = self.stations_to_use(self.data_paths['stations_to_use_path'])
        df_bicing_processed = df_bicing_processed[df_bicing_processed.station_id.isin(stations_to_use)]

        if self.pipeline_config['use_meteo_data']:
            #meteosteps would go here
            #...
            pass

        if self.pipeline_config['save_file']:
            print('saving processed data to file')
            self.save_data(self.data_paths['processed_data'], self.pipeline_config['output_name'], df_bicing_processed)

        return df_bicing_processed


if __name__ == '__main__':
    processingPipeline = ProcessingPipeline(config=conf)
    processingPipeline.execute()




 