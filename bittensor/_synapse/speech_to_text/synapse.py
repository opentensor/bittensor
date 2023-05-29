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
from pydantic import BaseModel

class SpeechToTextForward( bittensor.SynapseCall ):
    name: str = "speech_to_text_forward"
    is_forward: bool = True
    text: str = None

    def __init__( 
            self, 
            synapse: "SpeechToTextSynapse", 
            request_proto: bittensor.proto.ForwardSpeechToTextRequest,
            forward_callback: Callable,
        ):
        super().__init__( synapse = synapse, request_proto = request_proto )
        self.speech = request_proto.speech
        self.forward_callback = forward_callback

    def apply( self ):
        bittensor.logging.trace( "SpeechToTextForward.apply()" )
        self.text = self.forward_callback( speech = self.speech )
        bittensor.logging.trace( "SpeechToTextForward.apply() = len(result)", len(self.text) )

    def get_response_proto( self ) -> bittensor.proto.ForwardSpeechToTextResponse: 
        bittensor.logging.trace( "SpeechToTextForward.get_response_proto()" )
        return bittensor.proto.ForwardSpeechToTextResponse( text = self.text )
    
    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "SpeechToTextForward.get_inputs_shape()" )
        return torch.Size( [ len(self.speech) ] )
    
    def get_outputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "SpeechToTextForward.get_outputs_shape()" )
        return torch.Size( [ len(self.text) ]  ) if self.text != None else None


class SpeechToText( BaseModel ):
    speech: str
    timeout: int = 12

class SpeechToTextSynapse( bittensor.Synapse, bittensor.grpc.TextToImageServicer ):
    name: str = "speech_to_text"

    def attach( self, axon: 'bittensor.axon.Axon' ):
        bittensor.grpc.add_SpeechToTextServicer_to_server( self, self.axon.server )
        self.router = APIRouter()
        self.router.add_api_route("/SpeechToText/Forward/", self.fast_api_forward_speech_to_text, methods=["GET", "POST"])
        self.axon.fastapi_app.include_router( self.router )
        
    @abstractmethod
    def forward( self, speech: bytes ) -> str: 
        ...

    def fast_api_forward_speech_to_text( self, hotkey: str, item: SpeechToText ) -> str:
        request_proto = bittensor.proto.ForwardSpeechToTextRequest( 
            hotkey = hotkey, 
            version = bittensor.__version_as_int__,
            timeout = item.timeout, 
            speech = item.speech
        )
        call = SpeechToTextForward( self, request_proto, self.forward )
        bittensor.logging.trace( 'FastSpeechToText: {} '.format( call ) )
        response_proto = self.apply( call = call )
        return response_proto.text

    def Forward( self, request: bittensor.proto.ForwardSpeechToTextRequest, context: grpc.ServicerContext ) -> bittensor.proto.ForwardSpeechToTextResponse:
        call = SpeechToTextForward( self, request, self.forward )
        bittensor.logging.trace( 'GRPCSpeechToText: {} '.format( call ) )
        return self.apply( call = call ) 
