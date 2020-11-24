import argparse
from munch import Munch
import bittensor
from bittensor.session import BTSession

class Neuron(object):
    """ 
    """
    def __init__(   
                self,
                config,
        ):
        self.config = config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        pass

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        return config

    def start(self, session: BTSession):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError