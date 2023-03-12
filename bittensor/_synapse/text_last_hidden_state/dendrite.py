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

from . import call
from .. import dendrite

class TextLastHiddenStateDendrite( dendrite.Dendrite ):
    """ Dendrite for the text_last_hidden_state synapse."""
    
    def __str__( self ) -> str:
        return "TextLastHiddenState"

    def _stub_callable( self ) -> Callable:
        return self.receptor.stub.ForwardTextLastHiddenState

    def forward( 
            self, 
            text_inputs: torch.FloatTensor, 
            timeout: float = bittensor.__blocktime__,
            text_inputs_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
            hidden_state_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
        ) -> 'call.TextLastHiddenStateForwardCall':
        """ Forward call to the receptor.
            Args:
                text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                    torch tensor of text inputs.
                timeout (:obj:`float`, `optional`, defaults to 5 seconds):  
                    timeout for the forward call.
                text_prompt_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for text inputs.
                hidden_state_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for hidden states.
            Returns:
                call.TextLastHiddenStateForwardCall (:obj:`call.TextLastHiddenStateForwardCall`, `required`):
                    bittensor forward call dataclass.
        """
        return self._forward( 
            forward_call = call.TextLastHiddenStateForwardCall( 
                text_inputs = text_inputs, 
                timeout = timeout,
                text_inputs_serializer_type = text_inputs_serializer_type,
                hidden_state_serializer_type = hidden_state_serializer_type
            ) )

    