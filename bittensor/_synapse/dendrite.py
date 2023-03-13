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

class Dendrite(torch.nn.Module):
    """ Dendrite object.
        Dendrites are the forward pass of the bittensor network. They are responsible for making the forward call to the receptor.
    """
    def __init__(
            self,
            endpoint: Union[ 'bittensor.Endpoint', torch.Tensor ], 
            wallet: Optional[ 'bittensor.wallet' ]  = None,
        ):
        """ Initializes the Dendrite
            Args:
                endpoint (:obj:Union[]`bittensor.endpoint`, `required`):
                    bittensor endpoint object.
                wallet (:obj:`bittensor.wallet`, `optional`):
                    bittensor wallet object.
        """
        super(Dendrite, self).__init__()
        if wallet is None: 
            wallet = bittensor.wallet()
        self.wallet = wallet
        if isinstance(endpoint, torch.Tensor ): 
            endpoint = bittensor.endpoint.from_tensor( endpoint )
        self.endpoint = endpoint
        self.receptor = bittensor.receptor( endpoint = self.endpoint, wallet = self.wallet )

    def __str__( self ) -> str:
        """ Returns the name of the dendrite."""
        return "Dendrite"

    def get_stub( self ) -> object:
        """ Returns the channel stub for the dendrite. """
        raise NotImplementedError('Dendrite.get_forward_stub() not implemented.')
    
    def pre_process_forward_call_to_request_proto( 
            self, 
            forward_call: 'bittensor.BittensorCall' 
        ) -> 'bittensor.ForwardRequest':
        """ Preprocesses the request proto to a forward call.
            --------------------------------------------
            Args:
                forward_call (:obj:`bittensor.BittensorCall`, `required`):
                    forward_call to preprocess.
            Returns:
                request_proto (:obj:`bittensor.ForwardRequest`, `required`):
                    bittensor forward call object.
        """
        raise NotImplementedError('Must implement pre_process_forward_call_to_request_proto() in subclass.')
    
    def post_process_response_proto_to_forward_call( 
            self, 
            forward_call: bittensor.BittensorCall,
            response_proto: 'bittensor.ForwardResponse'
        ) -> bittensor.BittensorCall:
        """ Postprocesses the response proto to fill forward call.
            --------------------------------------------
            Args:
                forward_call (:obj:`bittensor.BittensorCall`, `required`):
                    bittensor forward call object to fill.
                response_proto (:obj:`bittensor.ForwardResponse`, `required`):
                    bittensor forward response proto.
            Returns:
                forward_call (:obj:`bittensor.BittensorCall`, `required`):
                    filled bittensor forward call object.
        """
        raise NotImplementedError('Must implement post_process_response_proto_to_forward_call() in subclass.')
    

    def _forward( self, forward_call: 'bittensor.BittensorCall' ) -> 'bittensor.BittensorCall':
        """ Forward call to remote endpoint."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete( self.async_forward( forward_call = forward_call ) )

    async def async_forward( self, forward_call: 'bittensor.BittensorCall' ) -> 'bittensor.BittensorCall':
        """ The function async_forward is a coroutine function that makes an RPC call 
            to a remote endpoint to perform a forward pass. It uses a BittensorCall object which it fills 
            using the subclass inherited functions _fill_forward_request and _process_forward_response.
            It returns the BittensorCall object with the filled responses.

            The function also logs the request and response messages using bittensor.logging.rpc_log.
            Args:
                forward_call (:obj:bittensor.BittensorCall, required): 
                    The BittensorCall object containing the request to be made to the remote endpoint.
            Returns:
                forward_call (:obj:bittensor.BittensorCall, required):
                    The BittensorCall object containing the response from the remote endpoint.
        """
        forward_call.hotkey = self.wallet.hotkey.ss58_address
        forward_call.version = bittensor.__version_as_int__
        try:
            forward_call.request_proto = self.pre_process_forward_call_to_request_proto( forward_call = forward_call )
            forward_call.request_proto.hotkey = self.wallet.hotkey.ss58_address
            forward_call.request_proto.version = bittensor.__version_as_int__
            forward_call.request_proto.timeout = forward_call.timeout
        except Exception as e:
            forward_call.request_code = bittensor.proto.ReturnCode.RequestSerializationException
            forward_call.request_message = str(e)
        finally:
            # Log accepted request
            bittensor.logging.rpc_log ( 
                axon = False, 
                forward = True, 
                is_response = False, 
                code = forward_call.request_code, 
                call_time = time.time() - forward_call.start_time, 
                pubkey = self.endpoint.hotkey, 
                uid = self.endpoint.uid, 
                inputs = forward_call.get_inputs_shape() if forward_call.request_code == bittensor.proto.ReturnCode.Success else None,
                outputs = None,
                message = forward_call.request_message,
                synapse = self.__str__()
            )
            # Optionally return.
            if forward_call.request_code != bittensor.proto.ReturnCode.Success:
                forward_call.end_time = time.time()
                return forward_call

   
        # Make the call and wait for response.
        try:
            # Make asyncio call.
            asyncio_future = self.get_stub( self.receptor.channel).Forward(
                request = forward_call.request_proto,
                timeout = forward_call.timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature', self.receptor.sign() ),
                    ('bittensor-version',str(bittensor.__version_as_int__)),
                ))
        
            # Wait for response.
            forward_call.response_proto = await asyncio.wait_for( asyncio_future, timeout = forward_call.timeout )

            # Process response.
            forward_call = self.post_process_response_proto_to_forward_call( 
                forward_call = forward_call, 
                response_proto = forward_call.response_proto 
            )

        except grpc.RpcError as rpc_error_call:
            # Request failed with GRPC code.
            forward_call.response_code = rpc_error_call.code()
            forward_call.response_message = 'GRPC error code: {}, details: {}'.format( rpc_error_call.code(), str(rpc_error_call.details()) )
        except asyncio.TimeoutError:
            # Catch timeout errors.
            forward_call.response_code = bittensor.proto.ReturnCode.Timeout
            forward_call.response_message = 'GRPC request timeout after: {}s'.format( forward_call.timeout)
        except Exception as e:
            # Catch unknown errors.
            forward_call.response_code = bittensor.proto.ReturnCode.UnknownException
            forward_call.response_message = str(e)
        finally:
            # Log Response 
            bittensor.logging.rpc_log( 
                axon = False, 
                forward = True, 
                is_response = True, 
                code = forward_call.response_code, 
                call_time = time.time() - forward_call.start_time, 
                pubkey = self.endpoint.hotkey, 
                uid = self.endpoint.uid, 
                inputs = forward_call.get_inputs_shape(), 
                outputs = forward_call.get_outputs_shape() if forward_call.response_code == bittensor.proto.ReturnCode.Success else None,
                message = forward_call.response_message,
                synapse = self.__str__(),
            )
            forward_call.end_time = time.time()
            return forward_call
    

    

    
    