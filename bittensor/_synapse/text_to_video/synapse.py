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

class TextToVideoForward( bittensor.SynapseCall ):
    name: str = "text_to_video_forward"
    is_forward: bool = True
    video: torch.FloatTensor = None

    def __init__( 
            self, 
            synapse: "TextToVideoSynapse", 
            request_proto: bittensor.proto.ForwardTextToVideoRequest,
            forward_callback: Callable,
        ):
        super().__init__( synapse = synapse, request_proto = request_proto )
        self.text = request_proto.text
        self.forward_callback = forward_callback

    def apply( self ):
        bittensor.logging.trace( "TextToVideoForward.apply()" )
        self.video = self.forward_callback( text = self.text )
        bittensor.logging.trace( "TextToVideoForward.apply() = len(result)", len(self.video) )

    def get_response_proto( self ) -> bittensor.proto.ForwardTextToVideoResponse: 
        bittensor.logging.trace( "TextToVideoForward.get_response_proto()" )
        video_tensor = bittensor.serializer().serialize( self.video )
        return bittensor.proto.ForwardTextToVideoResponse( video = video_tensor )
    
    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToVideoForward.get_inputs_shape()" )
        return torch.Size( [ len(self.text) ] )
    
    def get_outputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToVideoForward.get_outputs_shape()" )
        return self.video.shape if self.video is not None else None

class TextToVideo(BaseModel):
    text: str
    num_inference_steps: int = 30
    frames: int = 30
    fps: int = 8

class TextToVideoSynapse( bittensor.Synapse, bittensor.grpc.TextToImageServicer ):
    name: str = "text_to_video"

    def attach( self, axon: 'bittensor.axon.Axon' ):
        self.router = APIRouter()
        self.router.add_api_route("/TextToVideo/Forward/", self.fast_api_forward_text_to_video, methods=["POST"])
        self.axon.fastapi_app.include_router( self.router )
        bittensor.grpc.add_TextToVideoServicer_to_server( self, self.axon.server )

    @abstractmethod
    def forward( self, text: str,  ) -> torch.FloatTensor: 
        ...

    def fast_api_forward_text_to_video( self, hotkey: str, timeout: int, item: TextToVideo) -> str:
        request_proto = bittensor.proto.ForwardTextToVideoRequest( 
            hotkey = hotkey, 
            version = bittensor.__version_as_int__,
            timeout = timeout, 
            text = item.text,
            num_inference_steps = item.num_inference_steps,
            frames = item.frames,
            fps = item.fps,
        )
        call = TextToVideoForward( self, request_proto, self.forward )
        bittensor.logging.trace( 'FastTextToVideoForward: {} '.format( call ) )
        self.apply( call = call )
        return call.video

    def Forward( self, request: bittensor.proto.ForwardTextToVideoRequest, context: grpc.ServicerContext ) -> bittensor.proto.ForwardTextToVideoResponse:
        call = TextToVideoForward( self, request, self.forward )
        bittensor.logging.trace( 'GRPCTextToVideoForward: {} '.format( call ) )
        return self.apply( call = call ) 
