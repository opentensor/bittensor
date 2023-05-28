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

# TODO: implement multi-forward for text_to_embedding
# TODO: add list of strings for text argument like text_propmting...

class TextToEmbeddingForward( bittensor.SynapseCall ):
    name: str = "text_to_embedding_forward"
    is_forward: bool = True
    embedding: torch.FloatTensor = None

    def __init__( 
            self, 
            synapse: "TextToEmbeddingSynapse", 
            request_proto: bittensor.proto.ForwardTextToEmbeddingRequest,
            forward_callback: Callable,
        ):
        super().__init__( synapse = synapse, request_proto = request_proto )
        self.text = request_proto.text
        self.forward_callback = forward_callback

    def apply( self ):
        bittensor.logging.trace( "TextToEmbeddingForward.apply()" )
        self.embedding = self.forward_callback( text = self.text )
        bittensor.logging.trace( "TextToEmbeddingForward.apply() = len(result)", len(self.embedding) )

    def get_response_proto( self ) -> bittensor.proto.ForwardTextToEmbeddingResponse: 
        bittensor.logging.trace( "TextToEmbeddingForward.get_response_proto()" )
        embedding_tensor = bittensor.serializer().serialize( self.embedding )
        return bittensor.proto.ForwardTextToEmbeddingResponse( embedding = embedding_tensor )

    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToEmbeddingForward.get_inputs_shape()" )
        return torch.Size( [ len(self.text) ] )
    
    def get_outputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToEmbeddingForward.get_outputs_shape()" )
        return self.embedding.shape if self.embedding is not None else None

class TextToEmbedding( BaseModel ):
    text: str
    timeout: int = 12

class TextToEmbeddingSynapse( bittensor.Synapse, bittensor.grpc.TextToEmbeddingServicer ):
    name: str = "text_to_embedding"

    def attach( self, axon: 'bittensor.axon.Axon' ):
        bittensor.grpc.add_TextToEmbeddingServicer_to_server( self, self.axon.server )
        self.router = APIRouter()
        self.router.add_api_route("/TextToEmbedding/Forward/", self.fast_api_forward_text_to_embedding, methods=["GET", "POST"])
        self.axon.fastapi_app.include_router( self.router )

    @abstractmethod
    def forward( self, text: str ) -> torch.FloatTensor: 
        ...

    def fast_api_forward_text_to_embedding( self, hotkey: str, item: TextToEmbedding ) -> List[List[float]]:
        request_proto = bittensor.proto.ForwardTextToEmbeddingRequest( 
            hotkey = hotkey, 
            version = bittensor.__version_as_int__,
            text = item.text,
            timeout = item.timeout, 
        )
        call = TextToEmbeddingForward( self, request_proto, self.forward )
        bittensor.logging.trace( 'FastTextToEmbeddingForward: {} '.format( call ) )
        self.apply( call = call )
        response = call.embedding.tolist() if isinstance( call.embedding, torch.Tensor) else call.embedding 
        return response

    def Forward( self, request: bittensor.proto.ForwardTextToEmbeddingRequest, context: grpc.ServicerContext ) -> bittensor.proto.ForwardTextToEmbeddingResponse:
        call = TextToEmbeddingForward( self, request, self.forward )
        bittensor.logging.trace( 'GRPCTextToEmbeddingForward: {} '.format( call ) )
        return self.apply( call = call )
