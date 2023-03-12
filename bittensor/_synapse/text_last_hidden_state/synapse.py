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

import time
import grpc
import torch
import bittensor

from .. import synapse

class TextLastHiddenStateForwardCall( synapse.ForwardCall ):
    """ TextLastHiddenStateForwardCall: A class for storing the state of a forward call."""

    response_proto: bittensor.ForwardTextLastHiddenStateResponse = bittensor.ForwardTextLastHiddenStateResponse()

    def __init__( self, request_proto: bittensor.ForwardTextLastHiddenStateRequest ):
        super().__init__( request_proto = request_proto )
        self.text_inputs = None
        self.hidden_states = None

    def get_inputs_shape(self) -> torch.Size:
        if self.text_inputs is not None:
            return self.text_inputs.shape
        else: return None
    
    def get_outputs_shape(self) -> torch.Size:
        if self.hidden_states is not None:
            return self.hidden_states.shape
        else: return None

class TextLastHiddenStateSynapse( synapse.Synapse ):
    """ TextLastHiddenStateSynapse: A class for servicing text_last_hidden_state requests."""

    def __init__( self ):
        r""" Initializes a new Synapse."""
        self.priority_threadpool = bittensor.prioritythreadpool()

    def __str__(self):
        return 'TextLastHiddenState'
    
    def priority( self, forward_call: 'TextLastHiddenStateForwardCall' ) -> float:
        """ priority: Returns the priority of the synapse for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement priority() in subclass.')

    def blacklist( self, forward_call: 'TextLastHiddenStateForwardCall'  ) -> torch.FloatTensor:
        """ blacklist: Returns True if the synapse should not be called for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement blacklist() in subclass.')

    def forward( self, forward_call: 'TextLastHiddenStateForwardCall' ) -> torch.FloatTensor:
        """ forward: Returns the hidden states of the synapse for the given text_inputs."""
        raise NotImplementedError('Must implement forward() in subclass.')
    
    def _apply_forward_call(self, forward_call: TextLastHiddenStateForwardCall ):   

        # Deserialize text inputs. 
        text_deserializer = bittensor.serializer( serializer_type = forward_call.request_proto.text_inputs_serializer_type )
        forward_call.text_inputs = text_deserializer.deserialize( forward_call.request_proto.serialized_text_inputs, to_type = bittensor.proto.TensorType.TORCH )

        # Apply forward call.
        forward_call.hidden_states = self.forward( forward_call = forward_call )

        # Serialize hidden states.
        hidden_states_serializer = bittensor.serializer( serializer_type = forward_call.request_proto.hidden_states_serializer_type )
        serialized_hidden_states = hidden_states_serializer.serialize( forward_call.hidden_states, from_type = bittensor.proto.TensorType.TORCH )

        # Set response.
        forward_call.response_proto = bittensor.ForwardTextLastHiddenStateResponse(
            serialized_hidden_states = serialized_hidden_states,
            return_code = bittensor.proto.ReturnCode.Success,
            message = 'Success'
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
        forward_call = TextLastHiddenStateForwardCall( request_proto = request )
        return self._Forward( forward_call = forward_call )
    