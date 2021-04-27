import os, sys, time
from unittest.mock import MagicMock
import bittensor
import torch
import numpy

from miners.text.gpt2_genesis.gpt2_genesis import Miner

def test_run_gpt2():
    miner = Miner(
        n_epochs = 1,
        epoch_length = 1
    )
    miner.subtensor.connect = MagicMock(return_value = True)  
    miner.subtensor.is_connected = MagicMock(return_value = True)      
    miner.subtensor.subscribe = MagicMock(return_value = True)  
    miner.metagraph.set_weights = MagicMock(return_value = True) 
    miner.run()
test_run_gpt2()