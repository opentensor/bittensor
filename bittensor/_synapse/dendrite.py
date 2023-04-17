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

import grpc
import time
import torch
import asyncio
import bittensor
from typing import Union, Optional, Callable

from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class DendriteCall( ABC ):
    """ Base class for all dendrite calls."""

    is_forward: bool
    name: str

    def __init__(
            self, 
            dendrite: bittensor.Dendrite,
            timeout: float = bittensor.__blocktime__
        ):
        self.dendrite = dendrite
        self.completed = False
        self.timeout = timeout
        self.start_time = time.time()
        self.src_hotkey = self.dendrite.wallet.hotkey.ss58_address 
        self.src_version = bittensor.__version_as_int__
        self.dest_hotkey = self.dendrite.endpoint.hotkey
        self.dest_version = self.dendrite.endpoint.version  
        self.return_code: bittensor.proto.ReturnCode = bittensor.proto.ReturnCode.Success
        self.return_message: str = 'Success'

    @abstractmethod
    def get_callable(self) -> Callable: ...

    @abstractmethod
    def get_inputs_shape(self) -> torch.Shape: ...
    
    @abstractmethod
    def get_outputs_shape(self) -> torch.Shape: ...

    @abstractmethod
    def get_request_proto(self) -> object: ...

    def _get_request_proto(self) -> object:
        request_proto = self.request_proto    
        request_proto.version = self.src_version
        request_proto.timeout = self.timeout, 
        request_proto.hotkey = self.src_hotkey
        return request_proto
    
    @abstractmethod
    def apply_response_proto( self, response_proto: object ): ...

    def _apply_response_proto( self, response_proto: object ):
        self.apply_response_proto( response_proto )
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.completed = True

    def log_outbound(self):
        bittensor.logging.rpc_log(
            axon = False, 
            forward = self.is_forward, 
            is_response = True, 
            code = self.return_code, 
            call_time = self.elapsed if self.completed else 0, 
            pubkey = self.dest_hotkey, 
            uid = None,
            inputs = self.get_inputs_shape(), 
            outputs = self.get_outputs_shape(),
            message = self.return_message,
            synapse = self.name,
        )

    def log_inbound(self):
        bittensor.logging.rpc_log( 
            axon = False, 
            forward = self.is_forward, 
            is_response = False, 
            code = self.return_code, 
            call_time = 0, 
            pubkey = self.dest_hotkey, 
            uid = None, 
            inputs = self.get_inputs_shape(),
            outputs = self.get_outputs_shape(),
            message = self.return_message,
            synapse = self.name
        )      

class Dendrite( ABC, torch.nn.Module ):
    def __init__(
            self,
            wallet: 'bittensor.wallet',
            endpoint: Union[ 'bittensor.Endpoint', torch.Tensor ], 
        ):
        """ Dendrite abstract class
            Args:
                wallet (:obj:`bittensor.Wallet`, `required`):
                    bittensor wallet object.
                endpoint (:obj:`bittensor.Endpoint`, `required`):   
                    bittensor endpoint object.
        """
        super(Dendrite, self).__init__()
        self.wallet = wallet
        self.endpoint = endpoint
        self.receptor = bittensor.receptor( wallet = self.wallet, endpoint = self.endpoint )
    
    async def apply( self, dendrite_call: DendriteCall ) -> DendriteCall:
        """ Applies a dendrite call to the endpoint.
            Args:
                dendrite_call (:obj:`DendriteCall`, `required`):
                    Dendrite call to apply.
            Returns:
                DendriteCall: Dendrite call with response.
        """
        bittensor.logging.debug('Dendrite.apply()')
        try:
            dendrite_call.log_outbound()
            asyncio_future = dendrite_call.get_callable()(
                request = dendrite_call._get_request_proto(),
                timeout = dendrite_call.timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature', self.receptor.sign() ),
                    ('bittensor-version',str(bittensor.__version_as_int__)),
                )
            )
            bittensor.debug.dendrite( 'Dendrite.apply() awaiting response from: {}'.format( self.endpoint.hotkey ) )
            response_proto = await asyncio.wait_for( asyncio_future, timeout = dendrite_call.timeout )
            dendrite_call._apply_response_proto( response_proto )
            bittensor.debug.dendrite( 'Dendrite.apply() received response from: {}'.format( self.endpoint.hotkey ) )

        # Request failed with GRPC code.
        except grpc.RpcError as rpc_error_call:
            dendrite_call.return_code = rpc_error_call.code()
            dendrite_call.return_message = 'GRPC error code: {}, details: {}'.format( rpc_error_call.code(), str(rpc_error_call.details()) )
            bittensor.logging.debug( 'Dendrite.apply() rpc error: {}'.format( dendrite_call.return_message ) )
                                    
        # Catch timeout errors.
        except asyncio.TimeoutError:
            dendrite_call.return_code = bittensor.proto.ReturnCode.Timeout
            dendrite_call.return_message = 'GRPC request timeout after: {}s'.format( dendrite_call.timeout)
            bittensor.logging.debug( 'Denrite.apply() timeout error: {}'.format( dendrite_call.return_message ) )

        except Exception as e:
            # Catch unknown errors.
            dendrite_call.return_code = bittensor.proto.ReturnCode.UnknownException
            dendrite_call.return_message = str(e)   
            bittensor.logging.debug( 'Dendrite.apply() unknown error: {}'.format( dendrite_call.return_message ) )

        finally:
            dendrite_call.log_outbound()           
            return dendrite_call



    

    
    