import argparse
import copy
import math
import torch
import time
import sys
import os
import traceback

import threading
import multiprocessing as mp
import bittensor.utils.networking as net

from tqdm import tqdm
from munch import Munch
from termcolor import colored
from types import SimpleNamespace
from qqdm import qqdm, format_str
from typing import Tuple, List, Optional
from torch.utils.tensorboard import SummaryWriter
import bittensor

from loguru import logger
logger = logger.opt(colors=True)

class Miner( bittensor.neuron.Neuron ):

    def __init__( 
            self, 
            config: Munch = None,
            root_dir: str = None,
            **kwargs
        ):
        r""" Initializes a new base miner object.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    miner.Miner.default_config()
                root_dir (str, default '~/.bittensor/miners/'):
                    Root path to load and save data associated with each miner
        """
        if config == None:
            config = Miner.default_config()
        config = copy.deepcopy( config ); bittensor.config.Config.update_with_kwargs( config, kwargs )
        Miner.check_config( config )
        self.config = config
        self.tensorboard = SummaryWriter( log_dir = self.config.miner.full_path )
        super( Miner, self ).__init__( self.config, **kwargs )

    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Miner.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod
    def check_config(config: Munch):
        assert 'name' in config.miner, 'miners must specify a name argument.'
        bittensor.neuron.Neuron.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}'.format( config.miner.root_dir, config.wallet.name + "-" + config.wallet.hotkey, config.miner.name ))
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser ):
        bittensor.neuron.Neuron.add_args( parser )
        try:
            parser.add_argument (
                '--miner.root_dir',
                default='~/.bittensor/miners/',
                type=str,
                help='Root path to load and save data associated with each miner'
            )
        except argparse.ArgumentError:
            pass
    
    def init_logging ( self ):
        # Override Neuron init_logging.
        if self.config.record_log == True:
            filepath = self.config.miner.full_path + "/bittensor_output.log"
            logger.add (
                filepath,
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="25 MB",
                retention="10 days"
            )
            logger.info('logging is <green>ON</green> with sink: <cyan>{}</cyan>', filepath)
        else: 
            logger.info('logging is <red>OFF</red>')



    