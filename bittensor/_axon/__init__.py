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
import inspect

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
            port: int = None,
            ip: str = None,
            max_workers: int = None, 
            maximum_concurrent_rpcs: int = None,
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
                port (:type:`int`, `optional`):
                    Binding port.
                ip (:type:`str`, `optional`):
                    Binding ip.
                max_workers (:type:`int`, `optional`):
                    Used to create the threadpool if not passed, specifies the number of active threads servicing requests.
                maximum_concurrent_rpcs (:type:`int`, `optional`):
                    Maximum allowed concurrently processed RPCs.
        """              
        if config == None: config = axon.config()
        config = copy.deepcopy(config)
        config.axon.port = port if port != None else config.axon.port
        config.axon.ip = ip if ip != None else config.axon.ip
        config.axon.max_workers = max_workers if max_workers != None else config.axon.max_workers
        config.axon.maximum_concurrent_rpcs = maximum_concurrent_rpcs if maximum_concurrent_rpcs != None else config.axon.maximum_concurrent_rpcs
        axon.check_config( config )

        if wallet == None:
            wallet = bittensor.wallet( config = config )
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor( max_workers = config.axon.max_workers )
        if server == None:
            server = grpc.server( thread_pool, maximum_concurrent_rpcs = config.axon.maximum_concurrent_rpcs )

        if forward_callback != None:
            axon.check_forward_callback(forward_callback)
        if backward_callback != None:
            axon.check_backward_callback(backward_callback)

        axon_instance = axon_impl.Axon( 
            wallet = wallet, 
            server = server,
            ip = config.axon.ip,
            port = config.axon.port,
            forward_callback = forward_callback,
            backward_callback = backward_callback
        )
        bittensor.grpc.add_BittensorServicer_to_server( axon_instance, server )
        full_address = str( config.axon.ip ) + ":" + str( config.axon.port )
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
            parser.add_argument('--axon.port',default=8091, type=int, 
                    help='''The port this axon endpoint is served on. i.e. 8091''')
            parser.add_argument('--axon.ip', default='[::]', type=str, 
                help='''The local ip this axon binds to. ie. [::]''')
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
        assert config.axon.port > 1024 and config.axon.port < 65535, 'port must be in range [1024, 65535]'
        bittensor.wallet.check_config( config )


    @staticmethod
    def check_backward_callback( backward_callback:Callable ):
        if not inspect.ismethod(backward_callback) and not inspect.isfunction(backward_callback):
            raise ValueError('The axon backward callback must be a function with signature Callable[pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:, got {}'.format(backward_callback))        
        if len( inspect.signature(backward_callback).parameters) != 4:
            raise ValueError('The axon backward callback must have signature Callable[pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))
        if 'pubkey' not in inspect.signature(backward_callback).parameters:
            raise ValueError('The axon backward callback must have signature Callable[pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))
        if 'inputs_x' not in inspect.signature(backward_callback).parameters:
            raise ValueError('The axon backward callback must have signature Callable[pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))
        if 'grads_dy' not in inspect.signature(backward_callback).parameters:
            raise ValueError('The axon backward callback must have signature Callable[pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))
        if 'modality' not in inspect.signature(backward_callback).parameters:
            raise ValueError('The axon backward callback must have signature Callable[pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))

    @staticmethod
    def check_forward_callback( forward_callback:Callable ):
        if not inspect.ismethod(forward_callback) and not inspect.isfunction(forward_callback):
            raise ValueError('The axon forward callback must be a function with signature Callable[pubkey:str, inputs_x: torch.Tensor, modality:int] -> torch.FloatTensor:, got {}'.format(forward_callback))   
        if len( inspect.signature(forward_callback).parameters) != 3:
            raise ValueError('The axon forward callback must have signature Callable[pubkey:str, inputs_x: torch.Tensor, modality:int] -> torch.FloatTensor:, got {}'.format(inspect.signature(forward_callback)))
        if 'pubkey' not in inspect.signature(forward_callback).parameters:
            raise ValueError('The axon forward callback must have signature Callable[pubkey:str, inputs_x: torch.Tensor, modality:int] -> torch.FloatTensor:, got {}'.format(inspect.signature(forward_callback)))
        if 'inputs_x' not in inspect.signature(forward_callback).parameters:
            raise ValueError('The axon forward callback must have signature Callable[pubkey:str, inputs_x: torch.Tensor, modality:int] -> torch.FloatTensor:, got {}'.format(inspect.signature(forward_callback)))
        if 'modality' not in inspect.signature(forward_callback).parameters:
            raise ValueError('The axon forward callback must have signature Callable[pubkey:str, inputs_x: torch.Tensor, modality:int] -> torch.FloatTensor:, got {}'.format(inspect.signature(forward_callback)))
