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

class TextLastHiddenStateSynapse( bittensor.TextLastHiddenStateServicer ):
    """ TextLastHiddenStateSynapse: A class for servicing text_last_hidden_state requests."""
    
    def priority( self, hotkey:str, text_inputs: torch.FloatTensor, request: bittensor.ForwardTextLastHiddenStateRequest ) -> float:
        """ priority: Returns the priority of the synapse for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement priority() in subclass.')

    def blacklist( self, hotkey:str, text_inputs: torch.FloatTensor, request: bittensor.ForwardTextLastHiddenStateRequest ) -> torch.FloatTensor:
        """ blacklist: Returns True if the synapse should not be called for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement blacklist() in subclass.')

    def forward( self, hotkey:str, text_inputs: torch.FloatTensor, request: bittensor.ForwardTextLastHiddenStateRequest  ) -> torch.FloatTensor:
        """ forward: Returns the hidden states of the synapse for the given text_inputs."""
        raise NotImplementedError('Must implement forward() in subclass.')

    def _attach( self, axon: 'bittensor.axon.Axon' ):
        """ _attach: Attaches the synapse to the axon."""
        bittensor.grpc.add_TextLastHiddenStateServicer_to_server( self, axon.server )
    
    def ForwardTextLastHiddenState( 
            self, 
            request: bittensor.ForwardTextLastHiddenStateRequest, 
            context: grpc.ServicerContext 
        ) -> bittensor.ForwardTextLastHiddenStateResponse:
        """ ForwardTextLastHiddenState
            ----------------------------
            Args:
                request (bittensor.ForwardTextLastHiddenStateRequest): 
                    request.hotkey (string): hotkey of the neuron.
                    request.serialized_text_inputs (string): serialized text inputs.
                    request.text_inputs_serializer_type (bittensor.proto.SerializerType): text inputs serializer type.
                    request.hidden_states_serializer_type (bittensor.proto.SerializerType): hidden states serializer type.
                    request.timeout (float): timeout for the request.
                context (grpc.ServicerContext):
                    grpc tcp context.
            Returns:
                response (bittensor.ForwardTextLastHiddenStateResponse): 
                    response.serialized_hidden_states (string): serialized hidden states.
        """
        # Call variables.
        start_time = time.time()
        request_code = bittensor.proto.ReturnCode.Success
        request_message = 'Success'

        response_code = bittensor.proto.ReturnCode.Success
        response_message = 'Success'

        try:
            # Deserialize text inputs.
            text_deserialized = bittensor.serializer( serializer_type = request.text_inputs_serializer_type )
            text_inputs = text_deserialized.deserialize( request.serialized_text_inputs, from_type = bittensor.proto.TensorType.TORCH )

            # Check blacklist.
            if self.blacklist( request.hotkey, text_inputs, request ): 
                raise Exception('Blacklisted')
            
            # Get priority.
            priority = self.priority( request.hotkey, text_inputs, request )
            
            # Queue the forward call.
            future = self.priority_threadpool.submit(
                self.forward,
                hotkey = request.hotkey,
                text_inputs = text_inputs,
                request = request,
                priority = priority,
            )

        except Exception as e:
            request_code = bittensor.proto.ReturnCode.UnknownException
            request_message = str(e)

        # Log request.
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
            return bittensor.ForwardTextLastHiddenStateResponse()

        # Do forward.
        try:
            # Get the result.
            hidden_states = future.result( timeout = request.timeout )
            
            # Serialize hidden states.
            hidden_states_serializer = bittensor.serializer( serializer_type = request.hidden_states_serializer_type )
            serialized_hidden_states = hidden_states_serializer.serialize( hidden_states, from_type = bittensor.proto.TensorType.TORCH )
            
            # Return response.
            response = bittensor.proto.TextLastHiddenStateResponse(
                serialized_hidden_states = serialized_hidden_states,
            )

        except Exception as e:
            response_code = bittensor.proto.ReturnCode.UnknownException
            response_message = str(e)

    
        # Log response
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
            return bittensor.ForwardTextLastHiddenStateResponse()
        else:
            return response
