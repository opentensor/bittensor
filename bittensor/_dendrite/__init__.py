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
        if config == None: config = dendrite.config().dendrite
        config = copy.deepcopy(config)
        if wallet == None:
            wallet = bittensor.wallet( config = config.wallet )
        if receptor_pool == None:
            receptor_pool = bittensor.receptor_pool( 
                wallet = wallet,
                max_worker_threads = config.max_worker_threads,
                max_active_receptors = config.max_active_receptors
            )
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
        parser = argparse.ArgumentParser()
        parser.add_argument('--' + namespace + 'max_worker_threads', dest = 'max_worker_threads',  default=150, type=int, help='''Max number of concurrent threads used for sending RPC requests.''')
        parser.add_argument('--' + namespace + 'max_active_receptors', dest = 'max_active_receptors', default=500, type=int, help='''Max number of concurrently active receptors / tcp-connections''')
        parser.add_argument('--' + namespace + 'timeout', dest = 'timeout', type=int, help='''Default request timeout.''', default=5)
        parser.add_argument('--' + namespace + 'requires_grad', dest = 'requires_grad', type=bool, help='''If true, the dendrite passes gradients on the wire.''', default=False)
        parser.parse_known_args( namespace = dendrite_config )
        return config

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        bittensor.wallet.check_config( config.wallet )
        assert config.max_worker_threads > 0, 'max_worker_threads must be larger than 0'
        assert config.max_active_receptors > 0, 'max_active_receptors must be larger than 0'

