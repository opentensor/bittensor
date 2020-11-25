import argparse
import time
from munch import Munch
from torch.utils.tensorboard import SummaryWriter

class Metadata():
    def __init__(self, config):
        self.config = config

        if self.config.meta_logger.log_dir:
            self.tb_logger = SummaryWriter(log_dir=self.config.meta_logger.log_dir)

        # Instantiated time is the time that this class was instantiated, which should be at the start of bittensor operations
        self.instantiated_time = time.time()
        self.latest_up_bandwidth = None
        self.latest_down_bandwidth = None

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--meta_logger.log_dir', default='data/', type=str, 
                            help='Tensorboard logging dir.')
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        Config.validate_path_create('tb_logger.log_dir', config.meta_logger.log_dir)
        return config

    def write_loss(self, name, loss):
        if self.config.meta_logger.log_dir:
            self.tb_logger.add_scalar('loss/'+name, loss, round(time.time() - self.instantiated_time, 2))
    
    def write_accuracy(self, name, accuracy):
        if self.config.meta_logger.log_dir:
            self.tb_logger.add_scalar('accuracy/'+name, accuracy, round(time.time() - self.instantiated_time, 2))
    
    def save_dendrite_bandwidth_data(self, name, data):
        self.latest_up_bandwidth = data
        if self.config.meta_logger.log_dir:
            self.tb_logger.add_scalar('dendrite/'+name, data, round(time.time() - self.instantiated_time, 2))
    
    def write_network_data(self, name, data):
        if self.config.meta_logger.log_dir:
            self.tb_logger.add_scalar('network/'+name, data, round(time.time() - self.instantiated_time, 2))
    
    def write_axon_network_data(self, name, data):
        self.latest_down_bandwidth = data
        if self.config.meta_logger.log_dir:
            self.tb_logger.add_scalar('axon/'+name, data, round(time.time() - self.instantiated_time, 2))

    def write_custom(self, name, metric):
        if self.config.meta_logger.log_dir:
            self.tb_logger.add_scalar(name, metric, round(time.time() - self.instantiated_time, 2))