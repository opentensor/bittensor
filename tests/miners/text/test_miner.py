import os, sys
from unittest.mock import MagicMock
import unittest.mock as mock
import bittensor
import torch
import numpy
from miners.text.template_miner import Miner,Nucleus

def test_run_template():
    bittensor.logging()

    magic = MagicMock(return_value = 1)
    def test_forward(cls,inputs_x):
        print ('call')
        return magic(inputs_x)

    # mimic the get block function
    class block():
        def __init__(self):
            self.i = 0
        def blocks(self):
            self.i += 1
            return self.i

    block_check = block()

    config = Miner.config()
    config.miner.n_epochs = 1
    config.miner.epoch_length = 1
    config.wallet.path = '/tmp/pytest'
    config.wallet.name = 'pytest'
    config.wallet.hotkey = 'pytest'
    
    wallet = bittensor.wallet(
        path = '/tmp/pytest',
        name = 'pytest',
        hotkey = 'pytest',
    )
    
    with mock.patch.object(Miner,'forward_text',new=test_forward):
        
        miner = Miner( config = config )
        miner.wallet = wallet.create(coldkey_use_password = False)
        
        with mock.patch.object(miner.subtensor, 'get_current_block', new=block_check.blocks):
            bittensor.subtensor.connect = MagicMock(return_value = True)  
            bittensor.subtensor.is_connected = MagicMock(return_value = True)      
            bittensor.subtensor.register = MagicMock(return_value = True)  

            miner.run()

            assert magic.call_count == 1
            assert torch.is_tensor(magic.call_args[0][0])

if __name__ == "__main__":
    test_run_template()