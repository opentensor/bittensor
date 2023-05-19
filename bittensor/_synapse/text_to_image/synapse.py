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


class TextToImageForward( bittensor.SynapseCall ):
    name: str = "text_to_image_forward"
    is_forward: bool = True
    image: bytes = b''

    def __init__( 
            self, 
            synapse: "TextToImageSynapse", 
            request_proto: bittensor.proto.ForwardTextToImageRequest,
            forward_callback: Callable,
        ):
        super().__init__( synapse = synapse, request_proto = request_proto )

        #TODO: make these optional
        self.text = request_proto.text
        self.image = request_proto.image
        self.height = request_proto.height
        self.width = request_proto.width
        self.num_images_per_prompt = request_proto.num_images_per_prompt
        self.num_inference_steps = request_proto.num_inference_steps
        self.guidance_scale = request_proto.guidance_scale
        self.strength = request_proto.strength
        self.negative_prompt = request_proto.negative_prompt
        self.seed = request_proto.seed
        self.forward_callback = forward_callback

    def apply( self ):
        bittensor.logging.trace( "TextToImageForward.apply()" )
        self.image = self.forward_callback( 
            text = self.text, 
            image = self.image,
            height = self.height,
            width = self.width,
            num_images_per_prompt = self.num_images_per_prompt,
            num_inference_steps = self.num_inference_steps,
            guidance_scale = self.guidance_scale,
            strength = self.strength,
            negative_prompt = self.negative_prompt,
            seed = self.seed,
            )
        bittensor.logging.trace( "TextToImageForward.apply() = len(result)", len(self.image) )

    def get_response_proto( self ) -> bittensor.proto.ForwardTextToImageResponse: 
        bittensor.logging.trace( "TextToImageForward.get_response_proto()" )
        return bittensor.proto.ForwardTextToImageResponse( image = self.image )
    
    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToImageForward.get_inputs_shape()" )
        return torch.Size( [ len(self.text) ] )
    
    def get_outputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToImageForward.get_outputs_shape()" )
        return torch.Size( [ len(self.image) ]  )


class TextToImage(BaseModel):
    text: str
    image: bytes = ''
    height: int = 512 # performance degrades with smaller image sizes than 512
    width: int = 512
    timeout: int = 12
    num_images_per_prompt: int = 1
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.75
    negative_prompt: str = ""
    seed: int = -1
    

class TextToImageSynapse( bittensor.Synapse, bittensor.grpc.TextToImageServicer ):
    name: str = "text_to_image"

    def attach( self, axon: 'bittensor.axon.Axon' ):
        bittensor.grpc.add_TextToImageServicer_to_server( self, self.axon.server )
        self.router = APIRouter()
        self.router.add_api_route("/TextToImage/Forward/", self.fast_api_forward_text_to_image, methods=["POST"])
        self.axon.fastapi_app.include_router( self.router )
        
    @abstractmethod
    def forward( self, text: str ) -> bytes: 
        ...

    def fast_api_forward_text_to_image( self, hotkey: str, item: TextToImage ) -> bytes:
        request_proto = bittensor.proto.ForwardTextToImageRequest( 
            hotkey = hotkey, 
            version = bittensor.__version_as_int__,
            timeout = item.timeout, 
            text = item.text,
            image = item.image,
            height = item.height,
            width = item.width,
            num_images_per_prompt = item.num_images_per_prompt,
            num_inference_steps = item.num_inference_steps,
            guidance_scale = item.guidance_scale,
            strength = item.strength,
            negative_prompt = item.negative_prompt,
            seed= item.seed,
        )
        call = TextToImageForward( self, request_proto, self.forward )
        bittensor.logging.trace( 'FastAPITextToImageForward: {} '.format( call ) )
        response_proto = self.apply( call = call )
        return response_proto.image

    def Forward( self, request: bittensor.proto.ForwardTextToImageRequest, context: grpc.ServicerContext ) -> bittensor.proto.ForwardTextToImageResponse:
        call = TextToImageForward( self, request, self.forward )
        bittensor.logging.trace( 'GRPCAPITextToImageForward: {} '.format( call ) )
        return self.apply( call = call ) 
