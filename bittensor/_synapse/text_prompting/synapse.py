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

from typing import List, Dict, Union, Callable
from abc import ABC, abstractmethod
import json


class SynapseForwardMulti( bittensor.SynapseCall ):
    name: str = "text_prompting_forward_multi"
    is_forward: bool = True
    multi_completions: List[ str ] = [""]

    def __init__( 
            self, 
            synapse: "TextPromptingSynapseMulti", 
            request_proto: bittensor.proto.MultiForwardTextPromptingRequest,
            multi_forward_callback: Callable,
        ):
        super().__init__( synapse = synapse, request_proto = request_proto )
        self.messages: List[ Dict[str, str] ] = request_proto.messages
        self.formatted_messages = [ json.loads(message) for message in self.messages ]
        self.multi_forward_callback = multi_forward_callback

    def apply( self ):
        bittensor.logging.trace( "SynapseForward.apply()" )
        self.multi_completions = self.multi_forward_callback( messages = self.formatted_messages )
        bittensor.logging.trace( "SynapseForward.apply() = ", self.multi_completions )

    def get_response_proto( self ) -> bittensor.proto.MultiForwardTextPromptingResponse: 
        bittensor.logging.trace( "SynapseForward.get_response_proto()")
        return bittensor.MultiForwardTextPromptingResponse( multi_completions = self.multi_completions )
            
    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "SynapseForward.get_inputs_shape()" )
        return torch.Size( [ len(message) for message in self.messages ] )
    
    def get_outputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "SynapseForward.get_outputs_shape()" )
        return torch.Size( [ len(self.multi_completions) ]  )

class SynapseForward( bittensor.SynapseCall ):
    name: str = "text_prompting_forward"
    is_forward: bool = True
    completion: str = ""

    def __init__( 
            self, 
            synapse: "TextPromptingSynapse", 
            request_proto: bittensor.proto.ForwardTextPromptingRequest,
            forward_callback: Callable,
        ):
        super().__init__( synapse = synapse, request_proto = request_proto )
        self.messages = request_proto.messages
        self.formatted_messages = [ json.loads(message) for message in self.messages ]
        self.forward_callback = forward_callback

    def apply( self ):
        bittensor.logging.trace( "SynapseForward.apply()" )
        self.completion = self.forward_callback( messages = self.formatted_messages )
        bittensor.logging.trace( "SynapseForward.apply() = ", self.completion )

    def get_response_proto( self ) -> bittensor.proto.ForwardTextPromptingResponse: 
        bittensor.logging.trace( "SynapseForward.get_response_proto()" )
        return bittensor.ForwardTextPromptingResponse( response = self.completion )
    
    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "SynapseForward.get_inputs_shape()" )
        return torch.Size( [ len(message) for message in self.messages ] )
    
    def get_outputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "SynapseForward.get_outputs_shape()" )
        return torch.Size( [ len(self.completion) ]  )
    
class SynapseBackward( bittensor.SynapseCall ):
    name: str = "text_prompting_backward"
    is_forward: bool = False

    def __init__( 
            self, 
            synapse: "TextPromptingSynapse", 
            request_proto: bittensor.proto.BackwardTextPromptingRequest,
            backward_callback: Callable,
        ):
        super().__init__( synapse = synapse, request_proto = request_proto )
        self.formatted_messages = [ message for message in request_proto.messages ]
        self.formatted_rewards = torch.tensor( [ request_proto.rewards ], dtype = torch.float32 )
        self.completion = request_proto.response
        self.backward_callback = backward_callback

    def apply( self ):
        self.backward_callback(
            rewards = self.formatted_rewards,
            messages = self.formatted_messages,
            response = self.completion,
        )    
    
    def get_response_proto( self ) -> bittensor.proto.BackwardTextPromptingResponse: 
        return bittensor.BackwardTextPromptingResponse( )

    def get_inputs_shape(self) -> torch.Size: 
        return torch.Size( [ len(message) for message in self.formatted_messages ] )
    
    def get_outputs_shape(self) -> torch.Size: 
        return torch.Size( [ 0 ] )


class TextPromptingSynapse( bittensor.Synapse, bittensor.grpc.TextPromptingServicer ):
    name: str = "text_prompting_synapse"

    def __init__(self, axon: "bittensor.axon" ):
        super().__init__( axon = axon )
        self.axon = axon
        bittensor.grpc.add_TextPromptingServicer_to_server( self, self.axon.server )

    @abstractmethod
    def forward( self, messages: List[Dict[str, str]] ) -> str: ...

    @abstractmethod
    def multi_forward( self, messages: List[Dict[str, str]] ) -> List[ str ]: ...

    @abstractmethod
    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: ...

    def Forward( self, request: bittensor.proto.ForwardTextPromptingRequest, context: grpc.ServicerContext ) -> bittensor.proto.ForwardTextPromptingResponse:
        call = SynapseForward( self, request, self.forward )
        bittensor.logging.trace( 'Forward: {} '.format( call ) )
        return self.apply( call = call ) 

    def MultiForward( self, request: bittensor.proto.MultiForwardTextPromptingRequest, context: grpc.ServicerContext ) -> bittensor.proto.MultiForwardTextPromptingResponse:
        call = SynapseForwardMulti( self, request, self.multi_forward )
        bittensor.logging.trace( 'MultiForward: {} '.format( call ) )
        return self.apply( call = call ) 
                         
    def Backward( self, request: bittensor.proto.BackwardTextPromptingRequest, context: grpc.ServicerContext ) -> bittensor.proto.BackwardTextPromptingResponse:
        call = SynapseBackward( self, request, self.backward )
        bittensor.logging.trace( 'Backward: {}'.format( call ) )
        return self.apply( call = call ) 

