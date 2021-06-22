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
            timeout: int = None,
            requires_grad: bool = None,
            max_worker_threads: int = None,
            max_active_receptors: int = None,
            receptor_pool: 'bittensor.ReceptorPool' = None,
        ) -> 'bittensor.Dendrite':
        r""" Creates a new Dendrite object from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.dendrite.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                timeout (:type:`int`, `optional`):
                    Default request timeout.
                requires_grad (:type:`bool`, `optional`):
                    If true, the dendrite passes gradients on the wire by default.
                max_worker_threads (:type:`int`, `optional`):
                    Maximum number of active client threads. Does not override the
                    optionally passed receptor pool.
                max_active_receptors (:type:`int`, `optional`):
                    Maximum allowed active allocated TCP connections. Does not override the
                    optionally passed receptor pool.
                receptor_pool (:obj:`bittensor.ReceptorPool`, `optional`):
                    bittensor receptor pool, maintains a pool of active TCP connections.
        """
        if config == None: config = dendrite.config()
        config = copy.deepcopy(config)
        config.dendrite.timeout = timeout if timeout != None else config.dendrite.timeout
        config.dendrite.ip = requires_grad if requires_grad != None else config.dendrite.requires_grad
        config.dendrite.max_worker_threads = max_worker_threads if max_worker_threads != None else config.dendrite.max_worker_threads
        config.dendrite.max_active_receptors = max_active_receptors if max_active_receptors != None else config.dendrite.max_active_receptors
        dendrite.check_config( config )

        if wallet == None:
            wallet = bittensor.wallet( config = config )
        if receptor_pool == None:
            receptor_pool = bittensor.receptor_pool( 
                wallet = wallet,
                max_worker_threads = config.dendrite.max_worker_threads,
                max_active_receptors = config.dendrite.max_active_receptors
            )
        return dendrite_impl.Dendrite ( 
            config = config,
            wallet = wallet, 
            receptor_pool = receptor_pool 
        )

    @classmethod   
    def config(cls) -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        dendrite.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        try:
            parser.add_argument('--dendrite.max_worker_threads', default=150, type=int, help='''Max number of concurrent threads used for sending RPC requests.''')
            parser.add_argument('--dendrite.max_active_receptors', default=500, type=int, help='''Max number of concurrently active receptors / tcp-connections''')
            parser.add_argument('--dendrite.timeout', type=int, help='''Default request timeout.''', default=5)
            parser.add_argument('--dendrite.requires_grad', action='store_true', help='''If true, the dendrite passes gradients on the wire.''', default=True)
            parser.add_argument('--dendrite.no_requires_grad', dest='dendrite.requires_grad', action='store_false', help='''If true, the dendrite passes gradients on the wire.''')
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass
        bittensor.wallet.add_args( parser )


    @classmethod   
    def check_config( cls, config: 'bittensor.Config' ):
        assert config.dendrite
        assert 'timeout' in config.dendrite
        assert 'requires_grad' in config.dendrite
        assert config.dendrite.max_worker_threads > 0, 'max_worker_threads must be larger than 0'
        assert config.dendrite.max_active_receptors > 0, 'max_active_receptors must be larger than 0'
        bittensor.wallet.check_config( config )

    @classmethod   
    def check_config( cls, config: 'bittensor.Config' ):
        assert config.dendrite
        assert 'timeout' in config.dendrite
        assert 'requires_grad' in config.dendrite
        assert config.dendrite.max_worker_threads > 0, 'max_worker_threads must be larger than 0'
        assert config.dendrite.max_active_receptors > 0, 'max_active_receptors must be larger than 0'
        bittensor.wallet.check_config( config )