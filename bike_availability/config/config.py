# import bike_availability
import sys
import os
from pathlib import Path
import yaml
import glob

MODULE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = MODULE_DIR.parent

def read_config(cfg_path: Path, make_absolute: bool = False):
    #Loading the yaml config file
    with open(cfg_path) as conf:
        parsed_conf = yaml.load(conf, Loader=yaml.Loader)

    #Converting to absolute paths
    if make_absolute:
        for k,v in parsed_conf['data_paths'].items():
            if v.startswith('./'):
                absolute_path = Path.joinpath(ROOT_DIR, v)
                parsed_conf['data_paths'][k] = absolute_path
    return parsed_conf

def make_configs():
    conf = read_config(Path.joinpath(MODULE_DIR, 'config.yaml'))
    if conf['deployment'] == 'local':
        data_conf = read_config(Path.joinpath(MODULE_DIR, 'local_config.yaml'), make_absolute=True)
    elif conf['deployment'] == 'aws':
        data_conf = read_config(Path.joinpath(MODULE_DIR, 'aws_config.yaml'))
    else:
        raise Exception('Wrong deployment configuration set in config.yaml')
    
    full_conf = {**conf, **data_conf}
    return full_conf


conf = make_configs()        

if __name__ == '__main__':
    print(conf)