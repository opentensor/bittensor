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
import torch
import bittensor
from typing import Callable, Union

class TextToMusicForwardCall( bittensor.DendriteCall ):

    name: str = "text_to_image_forward"
    is_forward: bool = True
    image: bytes = b''

    def __init__(
        self,
        dendrite: 'bittensor.TextToMusicDendrite',
        text: str,
        duration: int,
        timeout: float = bittensor.__blocktime__,
    ):
        super().__init__( dendrite = dendrite, timeout = timeout )
        self.text = text
        self.duration = duration
        
    def get_callable( self ) -> Callable:
        return bittensor.grpc.TextToMusicStub( self.dendrite.channel ).Forward

    def get_request_proto( self ) -> bittensor.proto.ForwardTextToMusicRequest:
        return bittensor.proto.ForwardTextToMusicRequest( 
            text = self.text,
            duration = self.duration,
        )
    
    def apply_response_proto( self, response_proto: bittensor.proto.ForwardTextToMusicResponse ):
        self.music = response_proto.music
        
    def get_inputs_shape(self) -> torch.Size: 
        return torch.Size( [len(self.text)] )

    def get_outputs_shape(self) -> torch.Size:
        return torch.Size([ len(self.music) ] )

class TextToMusicDendrite( bittensor.Dendrite ):

    def forward(
            self,
            text: str,
            duration: int,
            timeout: float = bittensor.__blocktime__,
            return_call:bool = True,
        ) -> Union[ str, TextToMusicForwardCall ]:
        forward_call = TextToMusicForwardCall(
            dendrite = self, 
            timeout = timeout,
            text = text,
            duration = duration,
        )
        response_call = self.loop.run_until_complete( self.apply( dendrite_call = forward_call ) )
        if return_call: return response_call
        else: return response_call.music
    
    async def async_forward(
        self,
        text: str,
        duration: int = 12,
        timeout: float = bittensor.__blocktime__,
        return_call: bool = True,
    ) -> Union[ str, TextToMusicForwardCall ]:
        forward_call = TextToMusicForwardCall(
            dendrite = self, 
            timeout = timeout,
            text = text,
            duration = duration,
        )
        forward_call = await self.apply( dendrite_call = forward_call )
        if return_call: return forward_call
        else: return forward_call.music