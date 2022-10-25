from confection import registry, Config
import os
# import configparser

CONFIG_PATH = os.path.join(os.path.dirname(__file__))
PROJECT_ROOT_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'data')

config = Config().from_disk(os.path.join(CONFIG_PATH,"config.cfg"))