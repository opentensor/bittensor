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

    # Dendrite name.
    name: str = 'text_last_hidden_state'

    def __str__( self ) -> str:
        return "TextLastHiddenState"

    def get_stub( self, channel ) -> Callable:
        return bittensor.grpc.TextLastHiddenStateStub( channel )
    
    def pre_process_forward_call_to_request_proto( 
            self, 
            forward_call: 'bittensor.TextLastHiddenStateBittensorCall' 
        ) -> 'bittensor.ForwardTextLastHiddenStateRequest':
        """ Preprocesses the forward call to a request proto.
            --------------------------------------------
            Args:
                forward_call (:obj:`bittensor.TextLastHiddenStateBittensorCall`, `required`):
                    forward_call to preprocess.
            Returns:
                request_proto (:obj:`bittensor.ForwardTextLastHiddenStateRequest`, `required`):
                    bittensor request proto object.
        """

        # Serialize text inputs.
        text_serializer = bittensor.serializer( serializer_type = forward_call.text_inputs_serializer_type )
        serialized_text = text_serializer.serialize( forward_call.text_inputs )

        # Optionally serialize mask.
        if forward_call.mask != None:
            mask_serializer = bittensor.serializer( serializer_type = forward_call.mask_serializer_type )
            serialized_mask = mask_serializer.serialize( forward_call.mask )
        else:
            serialized_mask = None

        # Return forward call.
        return bittensor.ForwardTextLastHiddenStateRequest(
            timeout = forward_call.timeout,
            serialized_mask = serialized_mask,
            serialized_text_inputs = serialized_text,
            mask_serializer_type = forward_call.mask_serializer_type,
            text_inputs_serializer_type = forward_call.text_inputs_serializer_type,
            hidden_states_serializer_type = forward_call.hidden_states_serializer_type,
        )

    
    def post_process_response_proto_to_forward_call( 
            self, 
            forward_call: bittensor.TextLastHiddenStateBittensorCall,
            response_proto: bittensor.ForwardTextLastHiddenStateResponse 
        ) -> bittensor.TextLastHiddenStateBittensorCall :
        """ Postprocesses the response proto to fill forward call.
            --------------------------------------------
            Args:
                forward_call (:obj:`bittensor.TextLastHiddenStateBittensorCall`, `required`):
                    bittensor forward call object to fill.
                response_proto (:obj:`bittensor.ForwardTextLastHiddenStateResponse`, `required`):
                    bittensor forward response proto.
            Returns:
                forward_call (:obj:`bittensor.TextLastHiddenStateBittensorCall`, `required`):
                    filled bittensor forward call object.
        """
        forward_call.response_code = response_proto.return_code
        forward_call.response_message = response_proto.message

        if (response_proto.return_code != bittensor.proto.ReturnCode.Success) or \
            (response_proto.return_code != bittensor.proto.ReturnCode.Success):
            forward_call.hidden_states = None
            return forward_call

        # Deserialize hidden states.
        hidden_states_serializer = bittensor.serializer( serializer_type = forward_call.hidden_states_serializer_type )
        hidden_states = hidden_states_serializer.deserialize( response_proto.serialized_hidden_states )

        # If the mask is not none, we need to expand the hidden states to the proper size.
        if forward_call.mask != None:
            # From the encode_forward_response function the forward_response_tensor is [ len(mask), net_dim ]
            # a set of rows from the stacked_forward_response_tensor = [ bs * seq, net_dim ]
            # We will load these rows into a destination tensor = [bs, seq, net_dim]
            destination = torch.zeros( [ forward_call.text_inputs.size(0) * forward_call.text_inputs.size(1), bittensor.__network_dim__ ])

            # Iterate through the mask and fill the destination tensor 
            # with the hidden states from the forward call.
            counter = 0
            for i, not_masked in enumerate(forward_call.mask.reshape(-1)):
                if not_masked:
                    destination[i, :] = hidden_states[counter, :]
                    counter += 1
            
            # Reshape the destination tensor to the proper expanded size.
            hidden_states = destination.reshape( (forward_call.text_inputs.size(0), forward_call.text_inputs.size(1), bittensor.__network_dim__) )
            
            # Fill forward call.
            forward_call.hidden_states = hidden_states

        # If the mask is none, we can just fill the forward call.
        else:
            forward_call.hidden_states = hidden_states

        # Return.
        return forward_call

    def forward( 
            self, 
            text_inputs: torch.FloatTensor, 
            mask: torch.BoolTensor = None,
            timeout: float = bittensor.__blocktime__,
            mask_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
            text_inputs_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
            hidden_states_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
        ) -> 'bittensor.TextLastHiddenStateBittensorCall':
        """ Forward call to the receptor.
            Args:
                text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                    torch tensor of text inputs.
                mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                    mask over returned hidden states.
                timeout (:obj:`float`, `optional`, defaults to 5 seconds):  
                    timeout for the forward call.
                text_prompt_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for text inputs.
                hidden_states_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for hidden states.
            Returns:
                bittensor.TextLastHiddenStateBittensorCall (:obj:`bittensor.TextLastHiddenStateBittensorCall`, `required`):
                    bittensor forward call dataclass.
        """
        return self._forward( 
            forward_call = bittensor.TextLastHiddenStateBittensorCall( 
                text_inputs = text_inputs, 
                mask = mask,
                timeout = timeout,
                mask_serializer_type = mask_serializer_type,
                text_inputs_serializer_type = text_inputs_serializer_type,
                hidden_states_serializer_type = hidden_states_serializer_type
            ) )

    