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
from typing import Callable

class TextLastHiddenStateDendrite( bittensor.Dendrite ):
    """ Dendrite for the text_last_hidden_state synapse."""
    
    def __str__( self ) -> str:
        return "TextLastHiddenState"

    def _stub_callable( self ) -> Callable:
        return self.receptor.stub.ForwardTextLastHiddenState
    
    def pre_process_forward_call_to_request_proto( 
            self, 
            forward_call: 'bittensor.TextLastHiddenStateForwardCall' 
        ) -> 'bittensor.ForwardTextLastHiddenStateRequest':
        """ Preprocesses the forward call to a request proto.
            --------------------------------------------
            Args:
                forward_call (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    forward_call to preprocess.
            Returns:
                request_proto (:obj:`bittensor.ForwardTextLastHiddenStateRequest`, `required`):
                    bittensor request proto object.
        """
        text_serializer = bittensor.serializer( serializer_type = forward_call.text_inputs_serializer_type )
        serialized_text = text_serializer.serialize( forward_call.text_inputs )

        # Fill request
        return bittensor.ForwardTextLastHiddenStateRequest(
            serialized_text_inputs = serialized_text,
            text_inputs_serializer_type = forward_call.text_inputs_serializer_type,
            hidden_states_serializer_type = forward_call.hidden_states_serializer_type,
            timeout = forward_call.timeout,
        )
    
    def post_process_response_proto_to_forward_call( 
            self, 
            forward_call: bittensor.TextLastHiddenStateForwardCall,
            response_proto: bittensor.ForwardTextLastHiddenStateResponse 
        ) -> bittensor.TextLastHiddenStateForwardCall :
        """ Postprocesses the response proto to fill forward call.
            --------------------------------------------
            Args:
                forward_call (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    bittensor forward call object to fill.
                response_proto (:obj:`bittensor.ForwardTextLastHiddenStateResponse`, `required`):
                    bittensor forward response proto.
            Returns:
                forward_call (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    filled bittensor forward call object.
        """
        hidden_states_serializer = bittensor.serializer( serializer_type = forward_call.hidden_states_serializer_type )
        hidden_states = hidden_states_serializer.deserialize( response_proto.serialized_hidden_states )
        forward_call.hidden_states = hidden_states
        return forward_call

    def forward( 
            self, 
            text_inputs: torch.FloatTensor, 
            timeout: float = bittensor.__blocktime__,
            text_inputs_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
            hidden_states_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
        ) -> 'bittensor.TextLastHiddenStateForwardCall':
        """ Forward call to the receptor.
            Args:
                text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                    torch tensor of text inputs.
                timeout (:obj:`float`, `optional`, defaults to 5 seconds):  
                    timeout for the forward call.
                text_prompt_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for text inputs.
                hidden_states_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for hidden states.
            Returns:
                bittensor.TextLastHiddenStateForwardCall (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    bittensor forward call dataclass.
        """
        return self._forward( 
            forward_call = bittensor.TextLastHiddenStateForwardCall( 
                text_inputs = text_inputs, 
                timeout = timeout,
                text_inputs_serializer_type = text_inputs_serializer_type,
                hidden_states_serializer_type = hidden_states_serializer_type
            ) )

    