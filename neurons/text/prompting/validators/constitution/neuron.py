# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import json
import math
import copy
import queue
import torch
import random
import bittensor
import argparse
import bittensor as bt

from loguru import logger
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict

class neuron:
    @classmethod
    def check_config( cls, config: 'bt.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bt.logging.check_config( config )
        bt.wallet.check_config( config )
        bt.subtensor.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/netuid{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.netuid, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser( full_path )
        if not os.path.exists( config.neuron.full_path ):
            os.makedirs( config.neuron.full_path, exist_ok = True)

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        parser.add_argument( '--netuid', type = int, help = 'Prompting network netuid', default = 1 )
        parser.add_argument( '--neuron.name', type = str, help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default = 'core_prompting_validator')
        parser.add_argument( '--neuron.device', type = str, help = 'Device to run the validator on.', default = "cuda" if torch.cuda.is_available() else "cpu" )
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.logging.add_args( parser )
        bt.axon.add_args( parser )
        return bt.config( parser )
    
    def __init__( self ):
        self.config = neuron.config()
        self.check_config( self.config )
        bt.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        print( self.config )
        self.subtensor = bt.subtensor ( config = self.config )
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = bt.metagraph( netuid = self.config.netuid, network = self.subtensor.network )
        print ('done init')

    def train( self ):
        while True:
            uids = torch.tensor( random.sample( self.metagraph.uids.tolist(), 2 ), dtype = torch.int64 )
            A = bittensor.text_prompting( keypair = self.wallet.hotkey, axon = self.metagraph.axons[uids[0]] )
            B = bittensor.text_prompting( keypair = self.wallet.hotkey, axon = self.metagraph.axons[uids[1]] )
            resp_A = A.forward( 
                roles = ['user'], 
                messages = ['ask me a random question?'], 
                timeout = 5,
            )
            resp_B = B.forward( 
                roles = ['user'], 
                messages = ['ask me a random question?'], 
                timeout = 5,
            )
            bittensor.logging.info(str(resp_A))
            bittensor.logging.info(str(resp_B))

            if resp_A.is_success and resp_B.is_success:
                bittensor.logging.info('success')
                break
            else:
                bittensor.logging.info('failure')
                continue


if __name__ == '__main__':
    bittensor.logging.info( 'neuron().train()' )
    neuron().train()
