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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SynapseCall( ABC ):
    # If it is an forward of backward
    is_forward: bool = True
    # The name of the call.
    name: str = 'synapse_call'
    
    def __init__(
        self,
        synapse: bittensor.Synapse,
        request_proto: object
    ):
        self.completed = False
        self.start_time = time.time()
        self.timeout = request_proto.timeout
        self.src_version = request_proto.version
        self.src_hotkey = request_proto.hotkey
        self.dest_hotkey = synapse.axon.wallet.hotkey.ss58_address
        self.dest_version = bittensor.__version_as_int__ 
        self.return_code: bittensor.proto.ReturnCode = bittensor.proto.ReturnCode.Success
        self.return_message: str = 'Success'

    @abstractmethod
    def get_inputs_shape( self ) -> torch.Shape: ...    

    @abstractmethod
    def get_outputs_shape( self ) -> torch.Shape: ...     

    @abstractmethod
    def get_response_proto( self ) -> object: ...

    @abstractmethod
    def apply( self ): ...

    def _apply( self ): 
        # TODO(const): measure apply time.
        self.apply()
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.completed = True

    def log_outbound( self ):
        bittensor.logging.rpc_log(
            axon = True, 
            forward = self.is_forward, 
            is_response = False, 
            code = self.return_code, 
            call_time = 0, 
            pubkey = self.src_hotkey, 
            uid = None,
            inputs = self.get_inputs_shape(), 
            outputs = self.get_outputs_shape(),
            message = self.return_message,
            synapse = self.name,
        )

    def log_outbound( self ):
        bittensor.logging.rpc_log( 
            axon = True, 
            forward = self.is_forward, 
            is_response = True, 
            code = self.return_code, 
            call_time = self.elapsed if self.completed else 0, 
            pubkey = self.src_hotkey, 
            uid = None, 
            inputs = self.get_inputs_shape(),
            outputs = self.get_outputs_shape(),
            message = self.return_message,
            synapse = self.name
        )      

class Synapse( ABC ):

    def __init__( self, axon: bittensor.axon ):
        self.axon = axon

    def apply( self, call: bittensor.SynapseCall ) -> object:
        try:
            call.log_inbound()

            # Check blacklist.
            if self.blacklist( call ):
                call.request_code = bittensor.proto.ReturnCode.Blacklisted
                call.request_message = "Blacklisted"
            
            # Make call.
            else:
                # Queue the forward call with priority.
                future = self.axon.priority_threadpool.submit(
                    call._apply,
                    priority = self.priority( call ),
                )
                future.result( timeout = call.timeout )

        # Catch timeouts
        except asyncio.TimeoutError:
            call.return_code = bittensor.proto.ReturnCode.Timeout
            call.return_message = 'GRPC request timeout after: {}s'.format( call.timeout)

        # Catch unknown exceptions.
        except Exception as e:
            call.return_code = bittensor.proto.ReturnCode.UnknownException
            call.return_message = str(e)

        # Finally return the call.
        finally:
            call.log_outbound()
            return call.get_reponse_proto()
        
   