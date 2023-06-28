import numpy as np
import pandas as pd
import os
from pathlib import Path
from ..config.config import ROOT_DIR
from pyunpack import Archive
import requests
from bike_availability import logger

class DataManager:
    """Class that takes care of all data-related things"""

    @staticmethod
    def read_csv(origin:Path, **kwargs):
        """
        Reads data to a pandas dataframe.
        """
        return pd.read_csv(origin, **kwargs)

    @staticmethod
    def save_data(file_path: Path, name: str, data: pd.DataFrame) -> None:
        """
        Saves data to the specified file path with the specified name.
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        data.to_csv(Path.joinpath(file_path, name), index= None)
    
    @staticmethod
    def download_data():
        """
        Downloads bicing data from opendata-ajuntament API.
        """
        if 'data' not in os.listdir(ROOT_DIR):
            os.mkdir('data')
        if 'raw_data' not in os.listdir(Path.joinpath(ROOT_DIR ,'data')):
            os.system('cd data & mkdir raw_data')
            i2m = list(zip(range(1,13), ['Gener','Febrer','Marc','Abril','Maig','Juny','Juliol','Agost','Setembre','Octubre','Novembre','Desembre']))

            for year in [2022, 2021, 2020, 2019]:
                for month, month_name in i2m:
                    fname = f"{year}_{month:02d}_{month_name}_BicingNou_ESTACIONS.7z"
                    if (year == 2019) and (month_name in ['Gener', 'Febrer']):
                        continue
                    os.system(f'curl --location "https://opendata-ajuntament.barcelona.cat/resources/bcn/BicingBCN/{fname}" -o {fname}"')
                    Archive(Path.joinpath(Path(os.getcwd()),fname)).extractall(Path.joinpath(ROOT_DIR , '/data/raw_data'))
                    os.system(f'del "{year}_{month:02d}_{month_name}_BicingNou_ESTACIONS.7z"')

        else:
            logger.info('Data is already downloaded!')

    @staticmethod
    def load_stations_to_use(path_to_file):
        return np.loadtxt(Path.joinpath(path_to_file,'stations_to_use.csv')).astype('int')
    
    @staticmethod
    def request_station_data():
        """
        Makes a request to opendata-ajuntament API that returns information about the stations.
        """
        req = 'https://opendata-ajuntament.barcelona.cat/data/dataset/bd2462df-6e1e-4e37-8205-a4b8e7313b84/resource/e5adca8d-98bf-42c3-9b9c-364ef0a80494/download'
        response = requests.request("GET", req)
        if response.status_code != 200:
            logger.info("Unexpected status code ", response.status_code)
        return pd.DataFrame(response.json()['data']['stations'])
    
    @staticmethod
    def get_station_data(path_to_file):
        """Reads file with a station data snapshot that works well with the project data"""
        return pd.read_csv(path_to_file)

data_manager = DataManager()    