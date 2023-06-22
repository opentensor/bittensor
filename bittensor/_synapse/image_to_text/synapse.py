# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
from pydantic import BaseModel

from fastapi import APIRouter
from typing import Union, Callable
from abc import abstractmethod

class ImageToTextForward( bittensor.SynapseCall ):
    name: str = "image_to_text_forward"
    is_forward: bool = True
    text: str = "none"

    def __init__( 
            self, 
            synapse: "ImageToTextSynapse", 
            request_proto: bittensor.proto.ForwardImageToTextRequest,
            forward_callback: Callable,
        ):
        super().__init__( synapse = synapse, request_proto = request_proto )
        self.image = request_proto.image
        self.forward_callback = forward_callback

    def apply( self ):
        bittensor.logging.trace( "ImageToTextForward.apply()" )
        self.text = self.forward_callback( image = self.image )
        bittensor.logging.trace( "ImageToTextForward.apply() = len(response)", len(self.image) )

    def get_response_proto( self ) -> bittensor.proto.ForwardImageToTextResponse: 
        bittensor.logging.trace( "ImageToTextForward.get_response_proto()" )
        return bittensor.proto.ForwardImageToTextResponse( text = self.text )
    
    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "ImageToTextForward.get_inputs_shape()" )
        return torch.Size( [ len(self.image) ] )
    
    def get_outputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "ImageToTextForward.get_outputs_shape()" )
        return torch.Size( [ len(self.text) ]  )


class ImageToText( BaseModel ):
    image: str
    timeout: int = 12

class ImageToTextSynapse( bittensor.Synapse, bittensor.grpc.TextToImageServicer ):
    name: str = "image_to_text_synapse"

    def __init__( self, axon: 'bittensor.axon.Axon' ):
        super().__init__( axon = axon )
        bittensor.grpc.add_ImageToTextServicer_to_server( self, self.axon.server )
        self.router = APIRouter()
        self.router.add_api_route("/ImageToText/Forward/", self.fast_api_forward_image_to_text, methods=["GET", "POST"])
        self.axon.fastapi_app.include_router( self.router )
        
    @abstractmethod
    def forward( self, image: str ) -> str: 
        ...

    def fast_api_forward_image_to_text( self, hotkey: str, item: ImageToText ) -> str:
        request_proto = bittensor.proto.ForwardImageToTextRequest( 
            hotkey = hotkey, 
            version = bittensor.__version_as_int__,
            timeout = item.timeout,
            image = item.image
        )
        call = ImageToTextForward( self, request_proto, self.forward )
        bittensor.logging.trace( 'FastImageToTextForward: {} '.format( call ) )
        response_proto = self.apply( call = call )
        return response_proto.text

    def Forward( self, request: bittensor.proto.ForwardImageToTextRequest, context: grpc.ServicerContext ) -> bittensor.proto.ForwardImageToTextResponse:
        call = ImageToTextForward( self, request, self.forward )
        bittensor.logging.trace( 'GRPCImageToTextForward: {} '.format( call ) )
        return self.apply( call = call ) 