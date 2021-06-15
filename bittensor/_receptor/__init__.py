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

import bittensor
import argparse
import copy
import grpc
import bittensor.utils.networking as net
from concurrent.futures import ThreadPoolExecutor

from . import receptor_impl

class receptor:
    def __new__( cls, endpoint: 'bittensor.Endpoint', wallet: 'bittensor.Wallet' = None) -> 'bittensor.Receptor':
        r""" Initializes a receptor grpc connection.
            Args:
                endpoint (:obj:`bittensor.Endpoint`, `required`):
                    neuron endpoint descriptor.
        """        

        if wallet == None:
            wallet = bittensor.wallet()
        try:
            external_ip = str(net.get_external_ip())
        except:
            pass
        finally:
            external_ip = None

        # Get endpoint string.
        if endpoint.ip == external_ip:
            ip = "localhost:"
            endpoint_str = ip + str(endpoint.port)
        else:
            endpoint_str = endpoint.ip + ':' + str(endpoint.port)
        
        channel = grpc.insecure_channel(
            endpoint_str,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
        stub = bittensor.grpc.BittensorStub( channel )
        return receptor_impl.Receptor( 
            endpoint = endpoint,
            channel = channel, 
            wallet = wallet,
            stub = stub
        )

class receptor_pool:

    def __new__( 
            cls, 
            config: 'bittensor.config' = None,
            wallet: 'bittensor.Wallet' = None,
            thread_pool: ThreadPoolExecutor = None,
            max_worker_threads: int = 20,
            max_active_receptors: int = 150,
        ) -> 'bittensor.ReceptorPool':
        r""" Initializes a receptor grpc connection.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.receptor_pool.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                thread_pool (:obj:`ThreadPoolExecutor`, `optional`):
                    thread pool executor passed the receptor pool unless defined.
                max_worker_threads (:type:`int`, `optional`):
                    Maximum number of active client threads. Does not override passed 
                    Threadpool.
                max_active_receptors (:type:`int`, `optional`):
                    Maximum allowed active allocated TCP connections.
        """        
        if config == None: config = receptor_pool.config().receptor_pool
        config.max_worker_threads = max_worker_threads if max_worker_threads != None else config.max_worker_threads
        config.max_active_receptors = max_active_receptors if max_active_receptors != None else config.max_active_receptors
        config = copy.deepcopy( config )
        if wallet == None:
            wallet = bittensor.wallet( config.wallet )
        if thread_pool == None:
            thread_pool = ThreadPoolExecutor( max_workers = max_worker_threads )
        return bittensor.ReceptorPool( 
            wallet = wallet,
            thread_pool = thread_pool,
            max_active_receptors = max_active_receptors
        )

    @staticmethod   
    def config( config: 'bittensor.Config' = None, namespace: str = 'receptor_pool' ) -> 'bittensor.Config':
        if config == None: config = bittensor.config()
        receptor_pool_config = bittensor.config()        
        bittensor.wallet.config( receptor_pool_config )
        config[ namespace ] = receptor_pool_config
        parser = argparse.ArgumentParser()
        parser.add_argument('--' + namespace + 'max_worker_threads', dest = 'max_worker_threads',  default=150, type=int, help='''Max number of concurrent threads used for sending RPC requests.''')
        parser.add_argument('--' + namespace + 'max_active_receptors', dest = 'max_active_receptors', default=500, type=int, help='''Max number of concurrently active receptors / tcp-connections''')
        parser.parse_known_args( namespace = receptor_pool_config )
        return config

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        assert config.max_worker_threads > 0, 'max_worker_threads must be larger than 0'
        assert config.max_active_receptors > 0, 'max_active_receptors must be larger than 0'
        bittensor.wallet.check_config( config.wallet )
