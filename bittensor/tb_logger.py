import argparse
from munch import Munch
from torch.utils.tensorboard import SummaryWriter
import time

class TBLogger():
    def __init__(self, config):
        self.config = config
        self.tb_logger = SummaryWriter(log_dir=self.config.logger.log_dir)

        # Instantiated time is the time that this class was instantiated, which should be at the start of bittensor operations
        self.instantiated_time = time.time()

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--logger.log_dir', default='/tmp/', type=str, 
                            help='Tensorboard logging dir.')
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        Config.validate_path_create('logger.log_dir', config.logger.log_dir)
        return config

    def write_loss(self, name, loss):
        self.tb_logger.add_scalar('loss/'+name, loss, round(time.time() - self.instantiated_time, 2))
    
    def write_accuracy(self, name, accuracy):
        self.tb_logger.add_scalar('accuracy/'+name, accuracy, round(time.time() - self.instantiated_time, 2))
    
    def write_dendrite_network_data(self, name, data):
        self.tb_logger.add_scalar('dendrite/'+name, data, round(time.time() - self.instantiated_time, 2))
    
    def write_network_data(self, name, data):
        self.tb_logger.add_scalar('network/'+name, data, round(time.time() - self.instantiated_time, 2))
    
    def write_axon_network_data(self, name, data):
        self.tb_logger.add_scalar('axon/'+name, data, round(time.time() - self.instantiated_time, 2))

    def write_custom(self, name, metric):
        self.tb_logger.add_scalar(name, metric, round(time.time() - self.instantiated_time, 2))