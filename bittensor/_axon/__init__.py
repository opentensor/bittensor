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
        assert local_port > 1024 and local_port < 65535, 'local_port must be in range [1024, 65535]' 
        if wallet == None:
            wallet = bittensor.wallet()
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor( max_workers = max_workers )
        if server == None:
            server = grpc.server( thread_pool, maximum_concurrent_rpcs = maximum_concurrent_rpcs )
        axon_instance = axon_impl.Axon( 
            wallet = wallet, 
            server = server,
            local_ip = local_ip,
            local_port = local_port,
            forward_callback = forward_callback,
            backward_callback = backward_callback
        )
        bittensor.grpc.add_BittensorServicer_to_server( axon_instance, server )
        full_address = str( local_ip ) + ":" + str( local_port )
        server.add_insecure_port( full_address )
        return axon_instance 
