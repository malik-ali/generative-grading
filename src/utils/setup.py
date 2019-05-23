import os
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
from pprint import pprint
from time import time, localtime, strftime

from src.utils.io_utils import load_json, save_json
from dotmap import DotMap


def makedirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir):
            raise ValueError('Directory already exists: [{}]'.format(dir))
        else:
            os.makedirs(dir)

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)

def load_config(json_file):
    config_json = load_json(json_file)
    return DotMap(config_json)


def process_config(json_file, exp_base=None, override_dotmap=None):
    """
    Processes config file:
        1) Loads the json config file
        2) Converts it to a DotMap
        3) Creates experiments path and required subdirs
        4) Set up logging
    """
    config = load_config(json_file)
    if override_dotmap is not None:
        config.update(override_dotmap)

    print("Loaded configuration: ")
    pprint(config)

    print()
    print(" *************************************** ")
    print("      Running experiment {}".format(config.exp_name))
    print(" *************************************** ")
    print()

    if exp_base is None:
        exp_base = os.getcwd()    

    timestamp = strftime('%Y-%m-%d--%H_%M_%S', localtime())
    exp_dir = os.path.join(exp_base, "experiments", config.exp_name, timestamp)

    # create some important directories to be used for the experiment.
    config.summary_dir = os.path.join(exp_dir, "summaries/")
    config.checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    config.out_dir = os.path.join(exp_dir, "out/")
    config.log_dir = os.path.join(exp_dir, "logs/")

    makedirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])

    # save config to experiment dir
    config_out = os.path.join(exp_dir, 'config.json')
    save_json(config.toDict(), config_out)

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info("Configurations and directories successfully set up.")
    return config
