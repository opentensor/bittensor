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
            wallet: 'bittensor.Wallet',
            thread_pool: ThreadPoolExecutor = None,
            max_worker_threads: int = 150,
            max_active_receptors: int = 500,
        ) -> 'bittensor.ReceptorPool':
        r""" Initializes a receptor grpc connection.
            Args:
                wallet (:obj:`bittensor.Wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
                thread_pool (:obj:`ThreadPoolExecutor`, `optional`):
                    thread pool executor passed the receptor pool unless defined.
                max_worker_threads (:type:`int`, `optional`):
                    Maximum number of active client threads. Does not override passed 
                    Threadpool.
                max_active_receptors (:type:`int`, `optional`):
                    Maximum allowed active allocated TCP connections.
        """        
        if thread_pool == None:
            thread_pool = ThreadPoolExecutor( max_workers = max_worker_threads )
        return bittensor.ReceptorPool ( 
            wallet = wallet,
            thread_pool = thread_pool,
            max_active_receptors = max_active_receptors
        )

