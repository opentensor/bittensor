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

class TextCausalLMSynapse( bittensor.TextCausalLMServicer ):
    """ TextCausalLMSynapse: A class for servicing text_causal_lm requests from the axon."""

    def priority( self, hotkey:str, text_inputs: torch.FloatTensor ) -> float:
        """ priority: Returns the priority of the synapse for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement priority() in subclass.')

    def blacklist( self, hotkey:str, text_inputs: torch.FloatTensor ) -> torch.FloatTensor:
        """ blacklist: Returns True if the synapse should not be called for the given hotkey and text_inputs. """
        raise NotImplementedError('Must implement blacklist() in subclass.')

    def forward( self, text_inputs: torch.FloatTensor ) -> torch.FloatTensor:
        """ forward: Forward pass through the synapse."""
        raise NotImplementedError('Must implement forward() in subclass.')

    def _attach( self, axon: 'bittensor.axon.Axon' ):
        """ _attach: Attaches the synapse to the axon. """
        bittensor.grpc.add_TextCausalLMServicer_to_server( self, axon )
    
    def ForwardTextCausalLM( 
            self, 
            request: bittensor.ForwardTextCausalLMRequest, 
            context: grpc.ServicerContext 
        ) -> bittensor.ForwardTextCausalLMResponse:
        """ ForwardTextCausalLM: Forward pass through the synapse.

        Args:
            request (:obj:`bittensor.ForwardTextCausalLMRequest`, `required`):
                request.serialized_text_inputs (:obj:`torch.FloatTensor`, `required`):
                    text_inputs = torch.FloatTensor( request.serialized_text_inputs )
                request.text_inputs_serializer_type (:obj:`bittensor.proto.TensorType`, `required`):
                    text_inputs = bittensor.bittensor.serializer_for_type( request.text_inputs_serializer_type ).deserialize( request.serialized_text_inputs, from_type = bittensor.proto.TensorType.TORCH )
                request.hidden_states_serializer_type (:obj:`bittensor.proto.TensorType`, `required`):
                    hidden_states_serializer = bittensor.bittensor.serializer_for_type( request.hidden_states_serializer_type )
                request.timeout (:obj:`int`, `required`):
                    timeout = request.timeout
            context (:obj:`grpc.ServicerContext`, `required`):
                grpc tcp context.
        Returns:    
            response (:obj:`bittensor.ForwardTextCausalLMResponse`, `required`):
                response.serialized_hidden_states (:obj:`torch.FloatTensor`, `required`):
                    serialized_hidden_states = hidden_states_serializer.serialize( hidden_states, from_type = bittensor.proto.TensorType.TORCH )
        """
        # Deserialize text_inputs.
        text_deserialized = bittensor.bittensor.serializer_for_type( request.text_inputs_serializer_type )
        text_inputs = text_deserialized.deserialize( request.serialized_text_inputs, from_type = bittensor.proto.TensorType.TORCH )

        # Check blacklist.
        if self.blacklist( request.hotkey, text_inputs ): return bittensor.ForwardTextCausalLMResponse()

        # Get priority.
        priority = self.priority( request.hotkey, text_inputs  )

        # Submit to threadpool.
        future = self.priority_threadpool.submit(
            self.forward,
            hotkey = request.hotkey,
            text_inputs = text_inputs,
            priority = priority,
        )
        # Wait for result.
        hidden_states = future.result( timeout = request.timeout )

        # Serialize hidden_states.
        hidden_states_serializer = bittensor.bittensor.serializer_for_type( request.hidden_states_serializer_type )
        serialized_hidden_states = hidden_states_serializer.serialize( hidden_states, from_type = bittensor.proto.TensorType.TORCH )

        # Return response.
        return bittensor.ForwardTextCausalLMResponse(
            serialized_hidden_states = serialized_hidden_states,
        )