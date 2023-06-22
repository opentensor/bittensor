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
from typing import Callable, List, Dict, Union

class ImageToTextForwardCall( bittensor.DendriteCall ):

    name: str = "image_to_text_forward"
    is_forward: bool = True
    text: str = ""

    def __init__(
        self,
        dendrite: 'bittensor.ImageToTextDendrite',
        image: str,
        timeout: float = bittensor.__blocktime__,
    ):
        super().__init__( dendrite = dendrite, timeout = timeout )
        self.image = image
        
    def get_callable( self ) -> Callable:
        return bittensor.grpc.ImageToTextStub( self.dendrite.channel ).Forward

    def get_request_proto( self ) -> bittensor.proto.ForwardImageToTextRequest:
        return bittensor.proto.ForwardImageToTextRequest( image = self.image )
    
    def apply_response_proto( self, response_proto: bittensor.proto.ForwardImageToTextResponse ):
        self.text = response_proto.text
        
    def get_inputs_shape(self) -> torch.Size: 
        return torch.Size( [len(self.image)] )

    def get_outputs_shape(self) -> torch.Size:
        return torch.Size([ len(self.text) ] )

class ImageToTextDendrite( bittensor.Dendrite ):

    def forward(
            self,
            image: str,
            timeout: float = bittensor.__blocktime__,
            return_call:bool = True,
        ) -> Union[ str, ImageToTextForwardCall ]:
        forward_call = ImageToTextForwardCall(
            dendrite = self, 
            timeout = timeout,
            image = image
        )
        response_call = self.loop.run_until_complete( self.apply( dendrite_call = forward_call ) )
        if return_call: return response_call
        else: return response_call.text
    
    async def async_forward(
        self,
        image: str,
        timeout: float = bittensor.__blocktime__,
        return_call: bool = True,
    ) -> Union[ str, ImageToTextForwardCall ]:
        forward_call = ImageToTextForwardCall(
            dendrite = self, 
            timeout = timeout,
            image = image,
        )
        forward_call = await self.apply( dendrite_call = forward_call )
        if return_call: return forward_call
        else: return forward_call.text