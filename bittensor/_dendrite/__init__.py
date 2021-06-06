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
import bittensor
import argparse
import copy

from . import dendrite_impl

class dendrite:

    def __new__(
            cls, 
            config: 'bittensor.Config' = None,
            wallet: 'bittensor.Wallet' = None,
            receptor_pool: 'bittensor.ReceptorPool' = None,
            thread_pool: ThreadPoolExecutor = None,
            max_worker_threads: int = None,
            max_active_receptors: int = None,
            pass_gradients: bool = None,
            timeout: int = None,
            do_backoff: bool =  None,
            max_backoff: int = None, 
            namespace: str = ''
        ) -> 'bittensor.Dendrite':
        r""" Creates a new Dendrite object from passed arguments.
            Args:
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                receptor_pool (:obj:`bittensor.ReceptorPool`, `optional`):
                    bittensor receptor pool, maintains a pool of active TCP connections.
                thread_pool (:obj:`ThreadPoolExecutor`, `optional`):
                    thread pool executor passed the receptor pool unless defined.s
                max_worker_threads (:type:`int`, `optional`):
                    Maximum number of active client threads. Does not override passed 
                    Threadpool.
                max_active_receptors (:type:`int`, `optional`):
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
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.dendrite.config()
                namespace (:obj:`str, `optional`): 
                    config namespace.
        """
        if config == None:
            config = bittensor.config.cut_namespace(dendrite.config( namespace ), namespace ).dendrite
        config.pass_gradients = pass_gradients if pass_gradients != None else config.pass_gradients
        config.receptor_pool.max_worker_threads = max_worker_threads if max_worker_threads != None else config.receptor_pool.max_worker_threads
        config.receptor_pool.max_active_receptors = max_active_receptors if max_active_receptors != None else config.receptor_pool.max_active_receptors
        config.receptor_pool.receptor.timeout = timeout if timeout != None else config.receptor_pool.receptor.timeout
        config.receptor_pool.receptor.do_backoff = do_backoff if do_backoff != None else config.receptor_pool.receptor.do_backoff
        config.receptor_pool.receptor.max_backoff = max_backoff if max_backoff != None else config.receptor_pool.receptor.max_backoff
        config = copy.deepcopy(config)
        dendrite.check_config( config )

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet( config.wallet )

        # Threadpool executor for making queries across the line.
        if receptor_pool == None:
            receptor_pool = bittensor.receptor_pool ( config.receptor_pool, wallet, thread_pool = thread_pool )

        return dendrite_impl.Dendrite( config, wallet, receptor_pool )

    @staticmethod   
    def config(namespace: str = '') -> 'bittensor.Config':
        parser = argparse.ArgumentParser(); 
        dendrite.add_args(parser = parser, namespace = namespace) 
        config = bittensor.config( parser ); 
        return bittensor.config.cut_namespace( config, namespace )

    @staticmethod   
    def check_config( config: 'bittensor.Config', namespace: str = ''):
        bittensor.config.cut_namespace( config, namespace )
        bittensor.wallet.check_config( config.wallet )
        bittensor.receptor_pool.check_config( config.receptor_pool )

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser, namespace: str = ''):
        if namespace != '':
            namespace = namespace + 'dendrite.'
        else:
            namespace = 'dendrite.'
        bittensor.wallet.add_args( parser, namespace )
        bittensor.receptor_pool.add_args( parser, namespace )
        parser.add_argument('--' + namespace + 'pass_gradients', default=True, type=bool, 
                help='''Does this dendrite pass gradients by default.''')

    @staticmethod   
    def print_help(namespace: str = ''):
        parser = argparse.ArgumentParser(); 
        dendrite.add_args( parser, namespace ) 
        parser.print_help()

