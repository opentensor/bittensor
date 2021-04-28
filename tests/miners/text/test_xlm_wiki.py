import os, sys
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/TEXT/")

from xlm_wiki.xlm_wiki import Miner
        
def test_run_xlm_clm():
    miner = Miner(
        n_epochs = 1,
        epoch_length = 1
    )
    miner.subtensor.connect = MagicMock(return_value = True)    
    miner.subtensor.is_connected = MagicMock(return_value = True)    
    miner.subtensor.subscribe = MagicMock(return_value = True) 
    miner.metagraph.set_weights = MagicMock(return_value = True) 
    miner.run()

test_run_xlm_clm()