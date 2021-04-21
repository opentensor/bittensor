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

class Neuron():

    def __init__( 
            self, 
            config: Munch = None,
            **kwargs
        ):
        r"""
            Args:
                config (:obj:`Munch`, `optional`): 
                    neuron.Neuron.default_config()
        """
        # Config: Config items for all subobjects: wallet, metagraph, nucleus, axon, dendrite.
        # This object can be instantiated by calling Neuron.default_config()
        if config == None:
            config = Neuron.default_config()
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs(config, kwargs )
        Neuron.check_config(config)
        self.config = config
        
        # --- Bittensor components ----
        # Wallet: Holds the hotkey keypair and coldkey pub which are user to sign messages 
        # and subscribe to the chain.
        self.wallet = bittensor.wallet.Wallet( config = self.config )
        
        # Subtensor: provides an interface to the subtensor chain given a wallet.
        self.subtensor = bittensor.subtensor.Subtensor( config = self.config )
        
        # Metagraph: Maintains a connection to the subtensor chain and hold chain state.
        self.metagraph = bittensor.metagraph.Metagraph( config = self.config, wallet = self.wallet, subtensor = self.subtensor  )
                
        # Axon: RPC server endpoint which serves your nucleus. Responds to Forward and Backward requests.
        self.axon = bittensor.axon.Axon( config = self.config, wallet = self.wallet )
        
        # Dendrite: RPC client makes Forward and Backward requests to downstream peers.
        self.dendrite = bittensor.dendrite.Dendrite( config = self.config, wallet = self.wallet )

    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Neuron.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser ):
        bittensor.wallet.Wallet.add_args( parser )
        bittensor.subtensor.Subtensor.add_args( parser )
        bittensor.metagraph.Metagraph.add_args( parser )
        bittensor.axon.Axon.add_args( parser )
        bittensor.nucleus.Nucleus.add_args( parser )
        bittensor.dendrite.Dendrite.add_args( parser )

    @staticmethod
    def check_config( config: Munch ):
        bittensor.wallet.Wallet.add_args( config )
        bittensor.subtensor.Subtensor.check_config( config )
        bittensor.metagraph.Metagraph.check_config( config )
        bittensor.axon.Axon.check_config( config )
        bittensor.nucleus.Nucleus.check_config( config )
        bittensor.dendrite.Dendrite.check_config( config )