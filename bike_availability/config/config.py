import sys
import os
from pathlib import Path
import yaml
from bike_availability import logger

MODULE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = MODULE_DIR.parent
TARGET = 'percentage_docks_available'

class Config:
    def __init__(self):
        self.config = self.make_configs()

    def read_config(self, cfg_path: Path, make_absolute: bool = False):
        #Loading the yaml config file
        
        with open(cfg_path) as conf:
            parsed_conf = yaml.load(conf, Loader=yaml.Loader)
        #Converting to absolute paths
        if make_absolute:
            for path_type in parsed_conf.keys():
                for k,v in parsed_conf[path_type].items():
                    if v.startswith('ROOT'):
                        v = v.replace('ROOT', '.')
                        absolute_path = Path.joinpath(ROOT_DIR, v)
                        parsed_conf[path_type][k] = absolute_path
        return parsed_conf

    def make_configs(self):
        conf = self.read_config(Path.joinpath(ROOT_DIR, 'config.yaml'))
        data_conf = self.read_config(Path.joinpath(ROOT_DIR, 'local_config.yaml'), make_absolute=True)
        full_conf = {**conf, **data_conf}
        return full_conf

    def get_config(self, param):
        try:
            return self.config[param]
        except Exception as e:
            logger.info(e)
     
conf = Config()
if __name__ == '__main__':
    for cfg in conf.config.keys():
        print(cfg)
        print(conf.config[cfg])
        print('\n' * 3)