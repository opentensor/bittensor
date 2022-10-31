""" Factory class for managing grpc connections with axon endpoint
"""
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

import grpc
import json
import bittensor
from . import receptor_impl

class receptor:
    """ Create and init the receptor object, which encapsulates a grpc connection to an axon endpoint
    """
    def __new__( 
            cls,
            endpoint: 'bittensor.Endpoint',
            max_processes: 'int' = 1,
            wallet: 'bittensor.Wallet' = None,
            external_ip: 'str' = None,
            compression: str = None,
        ) -> 'bittensor.Receptor':
        r""" Initializes a receptor grpc connection.
            Args:
                endpoint (:obj:`bittensor.Endpoint`, `required`):
                    neuron endpoint descriptor.
        """        

        if wallet == None:
            wallet = bittensor.wallet()

        # Get endpoint string.
        if endpoint.ip == external_ip:
            ip = "localhost:"
            endpoint_str = ip + str(endpoint.port)
        else:
            endpoint_str = endpoint.ip + ':' + str(endpoint.port)

        # Determine the grpc compression algorithm
        if compression == 'gzip':
            compress_alg = grpc.Compression.Gzip
        elif compression == 'deflate':
            compress_alg = grpc.Compression.Deflate
        else:
            compress_alg = grpc.Compression.NoCompression

        channel = grpc.aio.insecure_channel(
            endpoint_str,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1),
                     ('grpc.keepalive_time_ms', 100000)])
        stub = bittensor.grpc.BittensorStub( channel )
        return receptor_impl.Receptor( 
            endpoint = endpoint,
            channel = channel, 
            wallet = wallet,
            stub = stub,
            max_processes=max_processes
        )

        

class receptor_pool:
    """ Create and init the receptor_pool object, which manage a pool of grpc connections 
    """
    def __new__( 
            cls, 
            wallet: 'bittensor.Wallet',
            max_active_receptors: int = 4096,
            compression: str = None,
        ) -> 'bittensor.ReceptorPool':
        r""" Initializes a receptor grpc connection.
            Args:
                wallet (:obj:`bittensor.Wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
                max_active_receptors (:type:`int`, `optional`):
                    Maximum allowed active allocated TCP connections.
        """        
        return bittensor.ReceptorPool ( 
            wallet = wallet,
            max_active_receptors = max_active_receptors,
            compression = compression
        )