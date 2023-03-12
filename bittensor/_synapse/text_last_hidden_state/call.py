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
from .. import call

class TextLastHiddenStateForwardCall( call.ForwardCall ):
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

    def to_forward_response_proto( self ) -> object:
        # Serialize hidden states.
        generations_serializer = bittensor.serializer( serializer_type = self.generations_serializer_type )
        serialized_generations = generations_serializer.serialize( self.generations, from_type = bittensor.proto.TensorType.TORCH )

        # Set response.
        return bittensor.ForwardTextSeq2SeqResponse(
            serialized_generations = serialized_generations
        )

    def from_forward_response_proto( self, response_proto: bittensor.ForwardTextSeq2SeqResponse ) -> object:
        # Deserialize hidden states.
        hidden_states_serializer = bittensor.serializer( serializer_type = self.hidden_states_serializer_type )
        self.hidden_states = hidden_states_serializer.deserialize( response_proto.serialized_hidden_states, to_type = bittensor.proto.TensorType.TORCH )

    @staticmethod
    def from_forward_request_proto( self, request_proto: bittensor.ForwardTextSeq2SeqRequest ) -> object:
        # Deserialize text inputs.
        text_serializer = bittensor.serializer( serializer_type = request_proto.text_inputs_serializer_type )
        self.text_inputs = text_serializer.deserialize( request_proto.serialized_text_inputs, from_type = bittensor.proto.TensorType.TORCH )

    def to_forward_request_proto( self ) -> bittensor.ForwardTextSeq2SeqRequest:
        # Serialize text inputs.
        text_serializer = bittensor.serializer( serializer_type = self._text_inputs_serializer_type )
        serialized_text = text_serializer.serialize( self.text_inputs, from_type = bittensor.proto.TensorType.TORCH )

        # Fill request
        return bittensor.ForwardTextLastHiddenStateRequest(
            serialized_text_inputs = serialized_text,
            text_inputs_serializer_type = self._text_inputs_serializer_type,
            hidden_states_serializer_type = self._hidden_states_serializer_type,
            timeout = self.timeout,
        )