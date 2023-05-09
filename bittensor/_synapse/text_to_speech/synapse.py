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

import json
import grpc
import torch
import bittensor

from fastapi import FastAPI, APIRouter
from typing import List, Dict, Union, Callable
from abc import ABC, abstractmethod

class TextToSpeechForward( bittensor.SynapseCall ):
    name: str = "text_to_speech_forward"
    is_forward: bool = True
    speech: bytes = None

    def __init__( 
            self, 
            synapse: "TextToSpeechSynapse", 
            request_proto: bittensor.proto.ForwardTextToSpeechRequest,
            forward_callback: Callable,
        ):
        super().__init__( synapse = synapse, request_proto = request_proto )
        self.text = request_proto.text
        self.forward_callback = forward_callback

    def apply( self ):
        bittensor.logging.trace( "TextToSpeechForward.apply()" )
        self.speech = self.forward_callback( text = self.text )
        bittensor.logging.trace( "TextToSpeechForward.apply() = len(result)", len(self.speech) )

    def get_response_proto( self ) -> bittensor.proto.ForwardTextToSpeechResponse: 
        bittensor.logging.trace( "TextToSpeechForward.get_response_proto()" )
        return bittensor.proto.ForwardTextToSpeechResponse( speech = self.speech )
    
    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToSpeechForward.get_inputs_shape()" )
        return torch.Size( [ len(self.text) ] )
    
    def get_outputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToSpeechForward.get_outputs_shape()" )
        return torch.Size( [ len(self.speech) ]  ) if self.speech is not None else None

class TextToSpeechSynapse( bittensor.Synapse, bittensor.grpc.TextToImageServicer ):
    name: str = "text_to_speech"

    def attach( self, axon: 'bittensor.axon.Axon' ):
        self.router = APIRouter()
        self.router.add_api_route("/TextToSpeech/Forward/", self.fast_api_forward_text_to_speech, methods=["GET"])
        self.axon.fastapi_app.include_router( self.router )
        bittensor.grpc.add_TextToSpeechServicer_to_server( self, self.axon.server )

    @abstractmethod
    def forward( self, text: str ) -> bytes: 
        ...

    def fast_api_forward_text_to_speech( self, hotkey: str, timeout: int, text: str ) -> bytes:
        request_proto = bittensor.proto.ForwardTextToSpeechRequest( 
            hotkey = hotkey, 
            version = bittensor.__version_as_int__,
            timeout = timeout, 
            text = text
        )
        call = TextToSpeechForward( self, request_proto, self.forward )
        bittensor.logging.trace( 'FastTextToSpeechForward: {} '.format( call ) )
        response_proto = self.apply( call = call )
        return response_proto.speech

    def Forward( self, request: bittensor.proto.ForwardTextToSpeechRequest, context: grpc.ServicerContext ) -> bittensor.proto.ForwardTextToSpeechResponse:
        call = TextToSpeechForward( self, request, self.forward )
        bittensor.logging.trace( 'GRPCTextToSpeechForward: {} '.format( call ) )
        return self.apply( call = call ) 
