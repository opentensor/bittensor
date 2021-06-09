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

from concurrent.futures.thread import ThreadPoolExecutor
from os import name
import bittensor
import argparse
import copy

from . import dendrite_impl

class dendrite:

    def __new__(
            cls, 
            config: 'bittensor.config' = None,
            wallet: 'bittensor.Wallet' = None,
            receptor_pool: 'bittensor.ReceptorPool' = None,
        ) -> 'bittensor.Dendrite':
        r""" Creates a new Dendrite object from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.dendrite.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                receptor_pool (:obj:`bittensor.ReceptorPool`, `optional`):
                    bittensor receptor pool, maintains a pool of active TCP connections.
        """
        if config == None: config = dendrite.config().dataloader
        config = copy.deepcopy(config)
        if wallet == None:
            wallet = bittensor.wallet( config = config.wallet )
        config.wallet = copy.deepcopy( wallet.config )
        if receptor_pool == None:
            receptor_pool = bittensor.receptor_pool( config = config.receptor_pool, wallet = wallet )
        config.receptor_pool = copy.deepcopy( receptor_pool.config )
        return dendrite_impl.Dendrite ( 
            config = config,
            wallet = wallet, 
            receptor_pool = receptor_pool 
        )
        
    @staticmethod   
    def config( config: 'bittensor.Config' = None, namespace: str = 'dendrite' ) -> 'bittensor.Config':
        if config == None: config = bittensor.config()
        dendrite_config = bittensor.config()
        config[ namespace ] = dendrite_config
        if namespace != '': namespace += '.'
        bittensor.wallet.config( dendrite_config )
        bittensor.receptor_pool.config( dendrite_config )
        return config

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        assert config.batch_size > 0, 'Batch size must be larger than 0'
        assert config.block_size > 0, 'Block size must be larger than 0'
        assert config.max_corpus_size > 0, 'max_corpus_size must be larger than 0'
        assert config.num_workers >= 0, 'num_workers must be equal to or larger than 0'
        bittensor.wallet.check_config( config.wallet )
        bittensor.receptor_pool.check_config( config.receptor_pool )

