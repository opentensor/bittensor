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
            config: 'bittensor.config' = None,
            wallet: 'bittensor.Wallet' = None,
            forward_callback: 'Callable' = None,
            backward_callback: 'Callable' = None,
            thread_pool: 'futures.ThreadPoolExecutor' = None,
            server: 'grpc._Server' = None,
            local_port: int = 8091,
            local_ip: str = '127.0.0.1',
            max_workers: int = 10, 
            maximum_concurrent_rpcs: int = 400,
        ) -> 'bittensor.Axon':
        r""" Creates a new bittensor.Axon object from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.axon.config()
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
        """              
        if config == None: config = axon.config()
        config = copy.deepcopy(config)
        config.axon.local_port = local_port if local_port != None else config.axon.local_port
        config.axon.local_ip = local_ip if local_ip != None else config.axon.local_ip
        config.axon.max_workers = max_workers if max_workers != None else config.axon.max_workers
        config.axon.maximum_concurrent_rpcs = maximum_concurrent_rpcs if maximum_concurrent_rpcs != None else config.axon.maximum_concurrent_rpcs
        axon.check_config( config )

        if wallet == None:
            wallet = bittensor.wallet( config = config )
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor( max_workers = config.axon.max_workers )
        if server == None:
            server = grpc.server( thread_pool, maximum_concurrent_rpcs = config.axon.maximum_concurrent_rpcs )
        axon_instance = axon_impl.Axon( 
            wallet = wallet, 
            server = server,
            forward_callback = forward_callback,
            backward_callback = backward_callback
        )
        bittensor.grpc.add_BittensorServicer_to_server( axon_instance, server )
        full_address = str( local_ip ) + ":" + str( local_port )
        server.add_insecure_port( full_address )
        return axon_instance 

    @classmethod   
    def config(cls) -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        axon.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        try:
            parser.add_argument('--axon.local_port',default=8091, type=int, 
                    help='''The port this axon endpoint is served on. i.e. 8091''')
            parser.add_argument('--axon.local_ip', default='127.0.0.1', type=str, 
                help='''The local ip this axon binds to. ie. 0.0.0.0''')
            parser.add_argument('--axon.max_workers',default=10, type=int, 
                help='''The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.''')
            parser.add_argument('--axon.maximum_concurrent_rpcs', default=400, type=int, 
                help='''Maximum number of allowed active connections''')   
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

        bittensor.wallet.add_args( parser )

    @classmethod   
    def check_config(cls, config: 'bittensor.Config' ):
        assert config.axon.local_port > 1024 and config.axon.local_port < 65535, 'local_port must be in range [1024, 65535]'
        bittensor.wallet.check_config( config )

