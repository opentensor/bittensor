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
from typing import Union, Optional

class TextLastHiddenStateDendrite(torch.nn.Module):
    """ Dendrite for the text_last_hidden_state synapse."""

    def __init__(
            self,
            endpoint: Union[ 'bittensor.Endpoint', torch.Tensor ], 
            wallet: Optional[ 'bittensor.wallet' ]  = None,
            text_inputs_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
            hidden_states_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
        ):
        """ Initializes the dendrite
            Args:
                endpoint (:obj:`bittensor.endpoint`, `required`):
                    bittensor endpoint object.
                wallet (:obj:`bittensor.wallet`, `optional`):
                    bittensor wallet object.
                text_inputs_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for text inputs.
                hidden_states_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for hidden states.
        """
        super(TextLastHiddenStateDendrite, self).__init__()
        if wallet is None: 
            wallet = bittensor.wallet()
        self.wallet = wallet
        if isinstance(endpoint, torch.Tensor ): 
            endpoint = bittensor.endpoint.from_tensor( endpoint )
        self.endpoint = endpoint
        self.receptor = bittensor.receptor( endpoint = self.endpoint, wallet = self.wallet )
        self._text_inputs_serializer_type = text_inputs_serializer_type
        self._hidden_states_serializer_type = hidden_states_serializer_type

    def _nill_response_for_inputs( self, text_inputs: torch.FloatTensor ) -> torch.FloatTensor:
        """ Returns a nill response for the given inputs.
            Args:
                text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                    torch tensor of text inputs.
            Returns:
                hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `required`):
                    torch tensor of hidden states.
        """
        return torch.zeros( text_inputs.shape[0], text_inputs.shape[1], bittensor.__network_dim__ ).to( text_inputs.device )

    def forward( 
            self, 
            text_inputs: torch.FloatTensor, 
            timeout: float = bittensor.__blocktime__ 
        ) -> torch.FloatTensor:
        """ Forward call to the receptor.
            Args:
                text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                    torch tensor of text inputs.
                timeout (:obj:`float`, `optional`, defaults to 5 seconds):  
                    timeout for the forward call.
            Returns:
                hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `required`):
                    torch tensor of hidden states.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete( 
            self.async_forward ( 
                text_inputs = text_inputs, 
                timeout = timeout
            ) )
    
    async def async_forward( 
            self, 
            text_inputs: torch.FloatTensor, 
            timeout: float = bittensor.__blocktime__ 
        ) -> torch.FloatTensor:
        """ Forward call to the receptor.
            Args:
                text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                    torch tensor of text inputs.
                timeout (:obj:`float`, `optional`, defaults to 5 seconds):
                    timeout for the forward call.
            Returns:
                hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `required`):
                    torch tensor of hidden states.
        """
        start_time = time.time()
        request_code = bittensor.proto.ReturnCode.Success
        response_code = bittensor.proto.ReturnCode.Success
        request_message = 'Success'
        response_message = 'Success'

        # ========================
        # ==== Create request ====
        # ========================
        try:
            # Serialize text inputs.
            text_serializer = bittensor.serializer( serializer_type = self._text_inputs_serializer_type )
            serialized_text = text_serializer.serialize( text_inputs, from_type = bittensor.proto.TensorType.TORCH )

            # Make request.
            request = bittensor.ForwardTextLastHiddenStateRequest(
                serialized_text_inputs = serialized_text,
                text_inputs_serializer_type = self._text_inputs_serializer_type,
                hidden_states_serializer_type = self._hidden_states_serializer_type,
            )
            
        except Exception as e:
            request_code = bittensor.proto.ReturnCode.RequestSerializationException
            request_message = str(e)
            
        # =====================
        # ==== Log request ====
        # =====================
        bittensor.logging.rpc_log ( 
            axon = False, 
            forward = True, 
            is_response = False, 
            code = request_code, 
            call_time = time.time() - start_time, 
            pubkey = self.endpoint.hotkey, 
            uid = self.endpoint.uid, 
            inputs = list(text_inputs.shape), 
            outputs = None,
            message = request_message,
            synapse = 'text_last_hidden_state'
        )
        if request_code != bittensor.proto.ReturnCode.Success:
            return self._nill_response_for_inputs( text_inputs = text_inputs )

        # ==========================
        # ==== Fire and Recieve ====
        # ==========================
        try:
            # Make call.
            asyncio_future = self.receptor.stub.ForwardTextLastHiddenState(
                    request = request, 
                    timeout = timeout,
                    metadata = (
                        ('rpc-auth-header','Bittensor'),
                        ('bittensor-signature', self.receptor.sign() ),
                        ('bittensor-version',str(bittensor.__version_as_int__)),
                    ))
            
            print ( asyncio_future )
            # Wait for response.
            grpc_response = await asyncio.wait_for( asyncio_future, timeout = timeout )

            # Catch failed code.
            if grpc_response.return_code != bittensor.proto.ReturnCode.Success:
                raise Exception( 'Remote Server Failure: '+ grpc_response.message )

            # Deserialize hidden states.
            hidden_states_serializer = bittensor.serializer( serializer_type = self._hidden_states_serializer_type )
            hidden_states = hidden_states_serializer.deserialize( grpc_response.hidden_states, to_type = bittensor.proto.TensorType.TORCH )

        # ================
        # ==== Errors ====
        # ================
        except grpc.RpcError as rpc_error_call:
            # Request failed with GRPC code.
            response_code = rpc_error_call.code()
            response_message = 'GRPC error code: {}, details: {}'.format( rpc_error_call.code(), str(rpc_error_call.details()) )
        except asyncio.TimeoutError:
            response_code = bittensor.proto.ReturnCode.Timeout
            response_message = 'GRPC request timeout after: {}s'.format(timeout)
        except Exception as e:
            response_code = bittensor.proto.ReturnCode.UnknownException
            response_message = str(e)

        # =====================
        # ==== Log Response ====
        # =====================
        bittensor.logging.rpc_log ( 
            axon = False, 
            forward = True, 
            is_response = True, 
            code = response_code, 
            call_time = time.time() - start_time, 
            pubkey = self.endpoint.hotkey, 
            uid = self.endpoint.uid, 
            inputs = list(text_inputs.shape), 
            outputs = list(hidden_states.shape) if response_code == bittensor.proto.ReturnCode.Success else None,
            message = response_message,
            synapse = 'text_last_hidden_state'
        )
        if response_code != bittensor.proto.ReturnCode.Success:
            return self._nill_response_for_inputs( text_inputs = text_inputs )

        return hidden_states
    

    

    
    