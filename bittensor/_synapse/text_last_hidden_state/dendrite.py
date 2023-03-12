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

import torch
import bittensor
from typing import Union, Optional, Callable
from .. import dendrite

class TextLastHiddenStateForwardCall( dendrite.ForwardCall ):
    """ Call state for the text_last_hidden_state synapse."""
    def __init__( self, text_inputs: torch.LongTensor, timeout: float = bittensor.__blocktime__ ):
        super().__init__(timeout = timeout)
        self.text_inputs = text_inputs
        self.hidden_states = torch.zeros( text_inputs.shape[0], text_inputs.shape[1], bittensor.__network_dim__ )

    def get_inputs_shape(self) -> torch.Size:
        if self.text_inputs is not None:
            return self.text_inputs.shape
        else: return None
    
    def get_outputs_shape(self) -> torch.Size:
        if self.hidden_states is not None:
            return self.hidden_states.shape
        else: return None

class TextLastHiddenStateDendrite( dendrite.Dendrite ):
    """ Dendrite for the text_last_hidden_state synapse."""
    
    def __str__( self ) -> str:
        return "TextLastHiddenState"

    def _stub_callable( self ) -> Callable:
        return self.receptor.stub.ForwardTextLastHiddenState

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
        super( TextLastHiddenStateDendrite, self ).__init__( wallet = wallet, endpoint = endpoint)
        self._text_inputs_serializer_type = text_inputs_serializer_type
        self._hidden_states_serializer_type = hidden_states_serializer_type

    def forward( 
            self, 
            text_inputs: torch.FloatTensor, 
            timeout: float = bittensor.__blocktime__ 
        ) -> 'TextLastHiddenStateForwardCall':
        """ Forward call to the receptor.
            Args:
                text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                    torch tensor of text inputs.
                timeout (:obj:`float`, `optional`, defaults to 5 seconds):  
                    timeout for the forward call.
            Returns:
                TextLastHiddenStateForwardCall (:obj:`TextLastHiddenStateForwardCall`, `required`):
                    bittensor forward call dataclass.
        """
        return self._forward( call_state = TextLastHiddenStateForwardCall( text_inputs = text_inputs, timeout = timeout ) )
    
    async def _preprocess( self, call_state: 'TextLastHiddenStateForwardCall' ):
        """ Fills the forward request proto on the call_state object."""
        # Serialize text inputs.
        text_serializer = bittensor.serializer( serializer_type = self._text_inputs_serializer_type )
        serialized_text = text_serializer.serialize( call_state.text_inputs, from_type = bittensor.proto.TensorType.TORCH )

        # Fill request
        call_state.request_proto = bittensor.ForwardTextLastHiddenStateRequest(
            serialized_text_inputs = serialized_text,
            text_inputs_serializer_type = self._text_inputs_serializer_type,
            hidden_states_serializer_type = self._hidden_states_serializer_type,
            timeout = call_state.timeout,
        )
    
    async def _postprocess( self, call_state: 'TextLastHiddenStateForwardCall' ):
        """ Processes the forward response proto on the call_state object."""
        # Catch failed code.
        if call_state.response_proto.return_code != bittensor.proto.ReturnCode.Success:
            raise Exception( 'Remote Server Failure: '+ call_state.response_proto.message )

        # Deserialize hidden states.
        hidden_states_serializer = bittensor.serializer( serializer_type = self._hidden_states_serializer_type )
        call_state.hidden_states = hidden_states_serializer.deserialize( call_state.response_proto.serialized_hidden_states, to_type = bittensor.proto.TensorType.TORCH )
    
    
    