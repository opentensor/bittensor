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

from concurrent import futures
from typing import List, Tuple, Optional, Callable
import bittensor
import argparse
import copy
import grpc

from . import axon_impl

class axon:

    def __new__(
            cls, 
            config: 'bittensor.Config' = None,
            wallet: 'bittensor.Wallet' = None,
            forward_callback: 'Callable' = None,
            backward_callback: 'Callable' = None,
            thread_pool: 'futures.ThreadPoolExecutor' = None,
            server: 'grpc._Server' = None,
            local_port: int = None,
            local_ip: str =  None,
            max_workers: int = None, 
            maximum_concurrent_rpcs: int = None,
            namespace: str = ''
        ) -> 'bittensor.Axon':
        r""" Creates a new bittensor.Axon object from passed arguments.
            Args:
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                forward_callback (:obj:`callable`, `optional`):
                    function which is called on forward requests.
                backward_callback (:obj:`callable`, `optional`):
                    function which is called on backward requests.
                thread_pool (:obj:`ThreadPoolExecutor`, `optional`):
                    Threadpool used for processing server queries.
                server (:obj:`grpc._Server`, `required`):
                    Grpc server endpoint, overrides passed threadpool.
                local_port (:type:`int`, `optional`):
                    Binding port.
                local_ip (:type:`str`, `optional`):
                    Binding ip.
                max_workers (:type:`int`, `optional`):
                    Used to create the threadpool if not passed, specifies the number of active threads servicing requests.
                maximum_concurrent_rpcs (:type:`int`, `optional`):
                    Maximum allowed concurrently processed RPCs.
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.axon.config()
                namespace (:obj:`str, `optional`): 
                    config namespace.
        """        
        if config == None:
            config = bittensor.config.cut_namespace( axon.config( namespace ), namespace ).axon
        config.local_port = local_port if local_port != None else config.local_port
        config.local_ip = local_ip if local_ip != None else config.local_ip
        config.max_workers = max_workers if max_workers != None else config.max_workers
        config.maximum_concurrent_rpcs = maximum_concurrent_rpcs if maximum_concurrent_rpcs != None else config.maximum_concurrent_rpcs
        config = copy.deepcopy( config )
        axon.check_config( config )

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet( config = config.wallet )
        wallet = wallet

        # Create threadpool if non-existent.
        # Pass this to the grpc server.
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor( max_workers = config.max_workers )

        # GRPC Server object. 
        if server == None:
            server = grpc.server( thread_pool, maximum_concurrent_rpcs = config.maximum_concurrent_rpcs )

        axon_instance = axon_impl.Axon( config, wallet, server )

        # Attach callbacks.
        if forward_callback != None:
            axon_instance.attach_forward_callback( forward_callback )
        if backward_callback != None:
            axon_instance.attach_backward_callback( backward_callback )

        return axon_instance

    @staticmethod   
    def config(namespace: str = '') -> 'bittensor.Config':
        parser = argparse.ArgumentParser(); 
        axon.add_args(parser = parser, namespace = namespace) 
        config = bittensor.config( parser ); 
        return bittensor.config.cut_namespace( config, namespace )

    @staticmethod   
    def check_config( config: 'bittensor.Config', namespace: str = ''):
        bittensor.config.cut_namespace( config, namespace )
        assert config.local_port > 1024 and config.local_port < 65535, 'axon.local_port must be in range [1024, 65535]'

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser, namespace: str = ''):
        r""" Adds this axon's command line arguments to the passed parser.
            Args:
                parser (:obj:`argparse.ArgumentParser`, `required`): 
                    parser argument to append args to.
        """
        if namespace != '':
            namespace = namespace + 'axon.'
        else:
            namespace = 'axon.'
        bittensor.wallet.add_args(parser, namespace)
        parser.add_argument('--' + namespace + 'local_port', default=8091, type=int, 
            help='''The port this axon endpoint is served on. i.e. 8091''')
        parser.add_argument('--' + namespace + 'local_ip', default='127.0.0.1', type=str, 
            help='''The local ip this axon binds to. ie. 0.0.0.0''')
        parser.add_argument('--' + namespace + 'max_workers', default=10, type=int, 
            help='''The maximum number connection handler threads working simultaneously on this endpoint. 
                    The grpc server distributes new worker threads to service requests up to this number.''')
        parser.add_argument('--' + namespace + 'maximum_concurrent_rpcs', default=400, type=int, 
            help='''Maximum number of allowed active connections''')  

    @staticmethod   
    def print_help(namespace: str = ''):
        parser = argparse.ArgumentParser(); 
        axon.add_args( parser, namespace ) 
        parser.print_help()          
