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

import uuid
import grpc
import time
import torch
import asyncio
import bittensor

from grpc import _common
from typing import Union, Optional, Callable, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class DendriteCall( ABC ):
    """ Base class for all dendrite calls."""

    is_forward: bool
    name: str

    def __init__(
            self, 
            dendrite: 'bittensor.Dendrite',
            timeout: float = bittensor.__blocktime__
        ):
        self.dendrite = dendrite
        self.completed = False
        self.timeout = timeout
        self.start_time = time.time()
        self.src_hotkey = self.dendrite.keypair.ss58_address 
        self.src_version = bittensor.__version_as_int__
        self.dest_hotkey = self.dendrite.axon_info.hotkey
        self.dest_version = self.dendrite.axon_info.version  
        self.return_code: bittensor.proto.ReturnCode = bittensor.proto.ReturnCode.Success
        self.return_message: str = 'Success'

    def __repr__(self) -> str: 
        return f"DendriteCall( {bittensor.utils.codes.code_to_string(self.return_code)}, to:{self.dest_hotkey[:4]} + ... + {self.dest_hotkey[-4:]}, msg:{self.return_message})"
    
    def __str__(self) -> str: 
        return self.__repr__()

    @abstractmethod
    def get_callable(self) -> Callable: ...

    @abstractmethod
    def get_inputs_shape(self) -> torch.Size: ...
    
    @abstractmethod
    def get_outputs_shape(self) -> torch.Size: ...

    @abstractmethod
    def get_request_proto(self) -> object: ...

    def _get_request_proto(self) -> object:
        request_proto = self.get_request_proto()    
        request_proto.version = self.src_version
        request_proto.timeout = self.timeout
        request_proto.hotkey = self.src_hotkey
        return request_proto
    
    @abstractmethod
    def apply_response_proto( self, response_proto: object ): ...

    def _apply_response_proto( self, response_proto: object ):
        self.apply_response_proto( response_proto )
        try: self.return_message = response_proto.return_message
        except: pass
        try: self.return_code = response_proto.return_code
        except: pass
    
    def end(self):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.completed = True

    @property
    def did_timeout( self ) -> bool: return self.return_code == bittensor.proto.ReturnCode.Timeout
    @property
    def is_success( self ) -> bool: return self.return_code == bittensor.proto.ReturnCode.Success
    @property 
    def did_fail( self ) -> bool: return not self.is_success

    def log_outbound(self):
        bittensor.logging.rpc_log(
            axon = False, 
            forward = self.is_forward, 
            is_response = False, 
            code = self.return_code, 
            call_time = 0, 
            pubkey = self.dest_hotkey, 
            uid = self.dendrite.uid,
            inputs = self.get_inputs_shape(), 
            outputs = self.get_outputs_shape(),
            message = self.return_message,
            synapse = self.name,
        )

    def log_inbound(self):
        bittensor.logging.rpc_log( 
            axon = False, 
            forward = self.is_forward, 
            is_response = True, 
            code = self.return_code, 
            call_time = self.elapsed,
            pubkey = self.dest_hotkey, 
            uid = self.dendrite.uid, 
            inputs = self.get_inputs_shape(),
            outputs = self.get_outputs_shape(),
            message = self.return_message,
            synapse = self.name
        )      

class Dendrite( ABC, torch.nn.Module ):
    def __init__(
            self,
            keypair: Union[ 'bittensor.Wallet', 'bittensor.Keypair'],
            axon: Union[ 'bittensor.axon_info', 'bittensor.axon' ], 
            uid : int = 0,
            ip: str = None,
            grpc_options: List[Tuple[str,object]] = 
                    [('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1),
                     ('grpc.keepalive_time_ms', 100000) ]
        ):
        """ Dendrite abstract class
            Args:
                keypair (:obj:`Union[ 'bittensor.Wallet', 'bittensor.Keypair']`, `required`):
                    bittensor keypair used for signing messages.
                axon (:obj:Union[`bittensor.axon_info`, 'bittensor.axon'], `required`):   
                    bittensor axon object or its info used to create the connection.
                grpc_options (:obj:`List[Tuple[str,object]]`, `optional`):
                    grpc options to pass through to channel.
        """
        super(Dendrite, self).__init__()
        self.uuid = str(uuid.uuid1())
        self.uid = uid
        self.ip = ip
        self.keypair = keypair.hotkey if isinstance( keypair, bittensor.Wallet ) else keypair
        self.axon_info = axon.info() if isinstance( axon, bittensor.axon ) else axon
        if self.axon_info.ip == self.ip: 
            self.endpoint_str = "localhost:" + str(self.axon_info.port)
        else: 
            self.endpoint_str = self.axon_info.ip + ':' + str(self.axon_info.port)
        self.channel = grpc.aio.insecure_channel( self.endpoint_str, options = grpc_options )
        self.state_dict = _common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY
        self.loop = asyncio.get_event_loop()

    async def apply( self, dendrite_call: 'DendriteCall' ) -> DendriteCall:
        """ Applies a dendrite call to the endpoint.
            Args:
                dendrite_call (:obj:`DendriteCall`, `required`):
                    Dendrite call to apply.
            Returns:
                DendriteCall: Dendrite call with response.
        """
        bittensor.logging.trace('Dendrite.apply()')
        try:
            dendrite_call.log_outbound()
            asyncio_future = dendrite_call.get_callable()(
                request = dendrite_call._get_request_proto(),
                timeout = dendrite_call.timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature', self.sign() ),
                    ('bittensor-version',str(bittensor.__version_as_int__)),
                )
            )
            bittensor.logging.trace( 'Dendrite.apply() awaiting response from: {}'.format( self.axon_info.hotkey ) )
            response_proto = await asyncio.wait_for( asyncio_future, timeout = dendrite_call.timeout )
            dendrite_call._apply_response_proto( response_proto )
            bittensor.logging.trace( 'Dendrite.apply() received response from: {}'.format( self.axon_info.hotkey ) )

        # Request failed with GRPC code.
        except grpc.RpcError as rpc_error_call:
            dendrite_call.return_code = rpc_error_call.code()
            dendrite_call.return_message = 'GRPC error code: {}, details: {}'.format( rpc_error_call.code(), str(rpc_error_call.details()) )
            bittensor.logging.trace( 'Dendrite.apply() rpc error: {}'.format( dendrite_call.return_message ) )
    
        # Catch timeout errors.
        except asyncio.TimeoutError:
            dendrite_call.return_code = bittensor.proto.ReturnCode.Timeout
            dendrite_call.return_message = 'GRPC request timeout after: {}s'.format( dendrite_call.timeout)
            bittensor.logging.trace( 'Denrite.apply() timeout error: {}'.format( dendrite_call.return_message ) )

        except Exception as e:
            # Catch unknown errors.
            dendrite_call.return_code = bittensor.proto.ReturnCode.UnknownException
            dendrite_call.return_message = str(e)   
            bittensor.logging.trace( 'Dendrite.apply() unknown error: {}'.format( dendrite_call.return_message ) )

        finally:
            dendrite_call.end()         
            dendrite_call.log_inbound()  
            return dendrite_call

    def __exit__ ( self ): 
        self.__del__()

    def close ( self ): 
        self.__exit__()

    def __del__ ( self ):
        try:
            result = self.channel._channel.check_connectivity_state(True)
            if self.state_dict[result] != self.state_dict[result].SHUTDOWN: 
                self.loop.run_until_complete ( self.channel.close() )
        except:
            pass

    def nonce ( self ): 
        return time.monotonic_ns()

    def sign(self) -> str:
        """ Creates a signature for the dendrite and returns it as a string."""
        nonce = f"{self.nonce()}"
        sender_hotkey = self.keypair.ss58_address
        receiver_hotkey = self.axon_info.hotkey
        message = f"{nonce}.{sender_hotkey}.{receiver_hotkey}.{self.uuid}"
        signature = f"0x{self.keypair.sign(message).hex()}"
        return ".".join([nonce, sender_hotkey, signature, self.uuid])
        
    def state ( self ):
        """ Returns the state of the dendrite channel."""
        try: 
            return self.state_dict[self.channel._channel.check_connectivity_state(True)]
        except ValueError:
            return "Channel closed"



    

    
    
