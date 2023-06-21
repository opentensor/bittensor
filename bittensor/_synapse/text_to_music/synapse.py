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


class TextToMusicForward( bittensor.SynapseCall ):
    name: str = "text_to_music_forward"
    is_forward: bool = True
    music: str = ''

    def __init__( 
            self, 
            synapse: "TextToMusicSynapse", 
            request_proto: bittensor.proto.ForwardTextToMusicRequest,
            forward_callback: Callable,
        ):
        super().__init__( synapse = synapse, request_proto = request_proto )

        #TODO: make these optional
        self.text = request_proto.text
        self.duration = request_proto.duration
        self.forward_callback = forward_callback

    def apply( self ):
        bittensor.logging.trace( "TextToMusicForward.apply()" )
        self.music = self.forward_callback( 
            text = self.text, 
            duration = self.duration,
        )
        bittensor.logging.trace( "TextToMusicForward.apply() = len(result)", len(self.music) )

    def get_response_proto( self ) -> bittensor.proto.ForwardTextToMusicResponse: 
        bittensor.logging.trace( "TextToMusicForward.get_response_proto()" )
        return bittensor.proto.ForwardTextToMusicResponse( music = self.music )
    
    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToMusicForward.get_inputs_shape()" )
        return torch.Size( [ len(self.text) ] )
    
    def get_outputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToMusicForward.get_outputs_shape()" )
        return torch.Size( [ len(self.music) ]  )


class TextToMusic(BaseModel):
    text: str
    duration: int = 12
    timeout: int = 60
    

class TextToMusicSynapse( bittensor.Synapse, bittensor.grpc.TextToMusicServicer ):
    name: str = "text_to_music"

    def __init__( self, axon: 'bittensor.axon.Axon' ):
        super().__init__( axon = axon )
        bittensor.grpc.add_TextToMusicServicer_to_server( self, self.axon.server )
        self.router = APIRouter()
        self.router.add_api_route("/TextToMusic/Forward/", self.fast_api_forward_text_to_music, methods=["POST"])
        self.axon.fastapi_app.include_router( self.router )
        
    @abstractmethod
    def forward( self, text: str ) -> bytes: 
        ...

    def fast_api_forward_text_to_music( self, hotkey: str, item: TextToMusic ) -> bytes:
        request_proto = bittensor.proto.ForwardTextToMusicRequest( 
            hotkey = hotkey, 
            version = bittensor.__version_as_int__,
            timeout = item.timeout, 
            text = item.text,
            duration = item.duration
        )
        call = TextToMusicForward( self, request_proto, self.forward )
        bittensor.logging.trace( 'FastAPITextToMusicForward: {} '.format( call ) )
        response_proto = self.apply( call = call )
        return response_proto.music

    def Forward( self, request: bittensor.proto.ForwardTextToMusicRequest, context: grpc.ServicerContext ) -> bittensor.proto.ForwardTextToMusicResponse:
        call = TextToMusicForward( self, request, self.forward )
        bittensor.logging.trace( 'GRPCAPITextToMusicForward: {} '.format( call ) )
        return self.apply( call = call ) 