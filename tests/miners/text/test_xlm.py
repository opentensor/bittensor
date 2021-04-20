import os, sys, time
from unittest.mock import MagicMock
import bittensor
import torch
import numpy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/text/")
from xlm import Miner
        
def test_run_xlm_clm():
    miner = Miner(
        miner_n_epochs = 1,
        miner_epoch_length = 1,
    )
    miner.subtensor.connect = MagicMock(return_value = True)    
    miner.subtensor.subscribe = MagicMock(return_value = True)  
    miner.metagraph.set_weights = MagicMock()   
    miner.metagraph.sync = MagicMock()  
    miner.run()

test_run_xlm_clm()