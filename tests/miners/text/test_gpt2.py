import os, sys
from unittest.mock import MagicMock
import bittensor
import torch
import numpy

from miners.gpt2_genesis import Miner

def test_run_gpt2():
    miner = Miner()
    miner.config.miner.n_epochs = 1,
    miner.config.miner.epoch_length = 2,
    miner.config.miner.name = 'pytest_gpt2'
    miner.subtensor.connect = MagicMock(return_value = True)  
    miner.subtensor.is_connected = MagicMock(return_value = True)      
    miner.subtensor.subscribe = MagicMock(return_value = True)  
    miner.metagraph.set_weights = MagicMock(return_value = True) 
    miner.run()
test_run_gpt2()