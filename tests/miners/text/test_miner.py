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
    def test_forward(cls,pubkey,inputs_x):
        print ('call')
        return magic(pubkey,inputs_x)

    # mimic the get block function
    class block():
        def __init__(self):
            self.i = 0
        def block(self):
            if self.i < 10:
                self.i += 1
                return 100
            else:
                self.i += 1
                return 101

    block = block()
    config = Miner.config()
    config.miner.n_epochs = 1
    config.miner.epoch_length = 2
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
        miner.neuron.wallet = wallet.create(coldkey_use_password = False)
        
        with mock.patch.object(miner.neuron.subtensor, 'get_current_block', new=block.block):
            bittensor.neuron.subtensor.connect = MagicMock(return_value = True)  
            bittensor.neuron.subtensor.is_connected = MagicMock(return_value = True)      
            bittensor.neuron.subtensor.subscribe = MagicMock(return_value = True)  

            miner.run()

            assert magic.call_count == 1
            assert isinstance(magic.call_args[0][0],str)
            assert torch.is_tensor(magic.call_args[0][1])

if __name__ == "__main__":
    test_run_template()