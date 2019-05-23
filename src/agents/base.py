"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import logging
import numpy as np

from torch.utils.data import DataLoader


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def _create_dataloader(self, dataset):
        dataset_size = len(dataset)
        batch_size = self.config.batch_size
        num_workers = self.config.data_loader_workers
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        return loader, dataset_size

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def backup(self):
        """
        Backs up the model upon interrupt
        """
        raise NotImplementedError

    def finalise(self):
        """
        Do appropriate saving after model is finished training
        """
        raise NotImplementedError
