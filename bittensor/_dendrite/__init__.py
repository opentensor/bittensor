
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

from concurrent.futures import ThreadPoolExecutor
import bittensor
import argparse
import copy
from munch import Munch

from . import dendrite_impl

class dendrite:

    def __new__(
            cls, 
            config: Munch = None, 
            wallet: 'bittensor.wallet' = None,
            thread_pool: 'ThreadPoolExecutor' = None,
            max_worker_threads: int = None,
            max_active_tcp_connections: int = None,
            pass_gradients: bool = None,
            timeout: int = None,
            do_backoff : bool =  None,
            max_backoff: int = None, 
        ) -> 'bittensor.Dendrite':
        r""" Creates a new Dendrite object from passed arguments.
            Args:
                config (:obj:`Munch`, `optional`): 
                    bittensor.dendrite.default_config()
                wallet (:obj:`bittensor.wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                thread_pool (:obj:`ThreadPoolExecutor`, `optional`):
                    Threadpool used for making client queries.
                max_worker_threads (:type:`int`, `optional`):
                    Maximum number of active client threads. Does not override passed 
                    Threadpool.
                max_active_tcp_connections (:type:`int`, `optional`):
                    Maximum allowed active allocated TCP connections.
                pass_gradients (:type:`bool`, `optional`):
                    If true, the dendrite passes gradients on the wire by default.
                timeout (:type:`int`, `optional`):
                    Default request timeout.
                do_backoff (:type:`bool`, `optional`):
                    If true, non-successful requests cause a backoff from the error'd server.
                max_backoff (:type:`int`, `optional`):
                    If do_backoff, max_backoff is the number of maximum number of backed off requests
                    before another test query is sent.
        """
        if config == None:
            config = dendrite.default_config()
        config.dendrite.max_worker_threads = max_worker_threads if max_worker_threads != None else config.dendrite.max_worker_threads
        config.dendrite.max_active_tcp_connections = max_active_tcp_connections if max_active_tcp_connections != None else config.dendrite.max_active_tcp_connections
        config.receptor.pass_gradients = pass_gradients if pass_gradients != None else config.receptor.pass_gradients
        config.receptor.timeout = timeout if timeout != None else config.receptor.timeout
        config.receptor.do_backoff = do_backoff if do_backoff != None else config.receptor.do_backoff
        config.receptor.max_backoff = max_backoff if max_backoff != None else config.receptor.max_backoff
        config = copy.deepcopy(config)
        dendrite.check_config( config )

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet( config )
        wallet = wallet

        # Threadpool executor for making queries across the line.
        if thread_pool == None:
            thread_pool = ThreadPoolExecutor( max_workers = config.dendrite.max_worker_threads )

        return dendrite_impl.Dendrite( config, wallet, thread_pool )

    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        dendrite.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def check_config(config: Munch):
        bittensor.receptor.check_config( config )

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser ):
        bittensor.receptor.add_args(parser)
        parser.add_argument('--dendrite.max_worker_threads', default=20, type=int, 
                help='''Max number of concurrent threads used for sending RPC requests.''')
        parser.add_argument('--dendrite.max_active_tcp_connections', default=150, type=int, 
                help='''Max number of concurrently active receptors / tcp-connections''')
        return parser