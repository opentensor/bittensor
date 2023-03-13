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
import torch
import bittensor

class TextLastHiddenStateSynapse( bittensor.Synapse ):
    """ TextLastHiddenStateSynapse: A class for servicing text_last_hidden_state requests."""

    def __init__( self ):
        r""" Initializes a new Synapse."""
        self.priority_threadpool = bittensor.prioritythreadpool()

    def __str__(self):
        return 'TextLastHiddenState'
    
    def priority( self, forward_call: 'bittensor.TextLastHiddenStateForwardCall' ) -> float:
        """ priority: Returns the priority of the synapse for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement priority() in subclass.')

    def blacklist( self, forward_call: 'bittensor.TextLastHiddenStateForwardCall'  ) -> bool:
        """ blacklist: Returns True if the synapse should not be called for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement blacklist() in subclass.')

    def forward( self, forward_call: 'bittensor.TextLastHiddenStateForwardCall' ) -> bittensor.TextLastHiddenStateForwardCall:
        """ fills in the hidden_states of the forward call.
            Args:
                forward_call (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    bittensor forward call dataclass to fill.
            Returns:
                forward_call (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    filled bittensor forward call dataclass.
        """
        raise NotImplementedError('Must implement forward() in subclass.')
    
    def pre_process_request_proto_to_forward_call( 
            self, 
            request_proto: bittensor.ForwardTextLastHiddenStateRequest 
        ) -> 'bittensor.TextLastHiddenStateForwardCall':
        """ pre_process_request_proto_to_forward_call
            ------------------------------------------
            Args:
                request_proto (bittensor.ForwardTextLastHiddenStateRequest):
                    bittensor forward request proto.
            Returns:
                bittensor.TextLastHiddenStateForwardCall (:obj:`bittensor.TextLastHiddenStateForwardCall`, `required`):
                    bittensor forward call dataclass.
        """
        # Deserialize text inputs.
        text_serializer = bittensor.serializer( serializer_type = request_proto.text_inputs_serializer_type )
        text_inputs = text_serializer.deserialize( request_proto.serialized_text_inputs )

        return bittensor.TextLastHiddenStateForwardCall(
            text_inputs = text_inputs,
            timeout = request_proto.timeout,
            text_inputs_serializer_type = request_proto.text_inputs_serializer_type,
            hidden_states_serializer_type = request_proto.hidden_states_serializer_type,
        )
    
    def post_process_forward_call_to_response_proto( 
            self, 
            forward_call: 'bittensor.TextLastHiddenStateForwardCall' 
        ) -> bittensor.ForwardTextLastHiddenStateResponse:
        """ post_process_forward_call_to_response_proto
            --------------------------------------------
            Args:
                forward_call (bittensor.TextLastHiddenStateForwardCall):
                    forward_call.text_inputs (torch.FloatTensor): text inputs.
                    forward_call.timeout (float): timeout for the request.
                    forward_call.text_inputs_serializer_type (bittensor.proto.SerializerType): text inputs serializer type.
                    forward_call.hidden_states_serializer_type (bittensor.proto.SerializerType): hidden states serializer type.
                    forward_call.hidden_states (torch.FloatTensor): hidden states.
            Returns:    
                response (bittensor.ForwardTextLastHiddenStateResponse):
                    response.serialized_hidden_states (string): serialized hidden states.
        """
        hidden_state_serializer = bittensor.serializer( serializer_type = forward_call.hidden_states_serializer_type )
        serialized_hidden_states = hidden_state_serializer.serialize( forward_call.hidden_states )

        # Return the forward response proto.
        return bittensor.ForwardTextLastHiddenStateResponse(
            serialized_hidden_states = serialized_hidden_states
        )
    
    def ForwardTextLastHiddenState( 
            self, 
            request: bittensor.ForwardTextLastHiddenStateRequest, 
            context: grpc.ServicerContext 
        ) -> bittensor.ForwardTextLastHiddenStateResponse:
        """ ForwardTextLastHiddenState
            ----------------------------
            Args:
                request (bittensor.ForwardTextLastHiddenStateRequest): 
                    request.version (int): version of the caller.
                    request.hotkey (string): hotkey of the neuron.
                    request.timeout (float): timeout for the request.
                    request.text_inputs_serializer_type (bittensor.proto.SerializerType): text inputs serializer type.
                    request.serialized_text_inputs (string): serialized text inputs.
                context (grpc.ServicerContext):
                    grpc tcp context.
            Returns:
                response (bittensor.ForwardTextLastHiddenStateResponse): 
                    response.serialized_hidden_states (string): serialized hidden states.
        """
        return self._Forward( request_proto = request )
    