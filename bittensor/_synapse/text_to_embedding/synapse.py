import grpc
import torch
import bittensor

from fastapi import APIRouter
from typing import Union, Callable, List
from abc import abstractmethod


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
        proto = bittensor.proto.ForwardTextToEmbeddingResponse( embedding = embedding_tensor )
        return proto

    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToEmbeddingForward.get_inputs_shape()" )
        return torch.Size( [ len(self.text) ] )
    
    def get_outputs_shape(self) -> Union[torch.Size, None]: 
        bittensor.logging.trace( "TextToEmbeddingForward.get_outputs_shape()" )
        return self.embedding.shape if self.embedding is not None else None

class TextToEmbeddingSynapse( bittensor.Synapse, bittensor.grpc.TextToImageServicer ):
    name: str = "text_to_embedding"

    def attach( self, axon: 'bittensor.axon.Axon' ):
        self.router = APIRouter()
        self.router.add_api_route("/TextToEmbedding/Forward/", self.fast_api_forward_text_to_embedding, methods=["GET"])
        self.axon.fastapi_app.include_router( self.router )
        bittensor.grpc.add_TextToEmbeddingServicer_to_server( self, self.axon.server )

    @abstractmethod
    def forward( self, text: str ) -> torch.FloatTensor: 
        ...

    def fast_api_forward_text_to_embedding( self, hotkey: str, timeout: int, text: List[str] ) -> List[List[float]]:
        request_proto = bittensor.proto.ForwardTextToEmbeddingRequest( 
            hotkey = hotkey, 
            version = bittensor.__version_as_int__,
            timeout = timeout, 
            text = text
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