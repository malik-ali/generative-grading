import argparse

from src.utils.setup import process_config
from src.agents import *


def main(config_path):
    config = process_config(config_path)


    # Create the Agent and run it with given configuration
    AgentClass = globals()[config.agent]

    agent = AgentClass(config)
    try:
        agent.run()
        agent.finalise()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'config',
        metavar='config-file',
        default='None',
        help='The path to the configuration file for the experiment')

    args = arg_parser.parse_args()
    main(args.config)
