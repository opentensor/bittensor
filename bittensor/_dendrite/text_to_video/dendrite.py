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
import torch
import bittensor
from typing import Callable, List, Dict, Union


# class TextToVideo(BaseModel):
#     text: str
#     num_inference_steps: int = 30
#     num_frames: int = 30
#     fps: int = 8

class TextToVideoForwardCall( bittensor.DendriteCall ):

    name: str = "text_to_video_forward"
    is_forward: bool = True
    video: torch.FloatTensor = None

    def __init__(
        self,
        dendrite: 'bittensor.TextToVideoDendrite',
        text: str,
        num_inference_steps: int = 30,
        num_frames: int = 30,
        fps: int = 8,
        timeout: float = bittensor.__blocktime__,
    ):
        super().__init__( dendrite = dendrite, timeout = timeout )
        self.text = text
        self.num_inference_steps = num_inference_steps
        self.num_frames = num_frames
        self.fps = fps
        
    def get_callable( self ) -> Callable:
        return bittensor.grpc.TextToVideoStub( self.dendrite.channel ).Forward

    def get_request_proto( self ) -> bittensor.proto.ForwardTextToVideoRequest:
        return bittensor.proto.ForwardTextToVideoRequest( 
            text = self.text,
            num_inference_steps = self.num_inference_steps,
            num_frames = self.num_frames,
            fps = self.fps
        )
    
    def apply_response_proto( self, response_proto: bittensor.proto.ForwardTextToVideoResponse ):
        self.video = bittensor.serializer().deserialize( response_proto.video )
        
    def get_inputs_shape(self) -> torch.Size: 
        return torch.Size( [len(self.text)] )

    def get_outputs_shape(self) -> torch.Size:
        return torch.Size( [ len(self.video) ] ) if self.video is not None else None

class TextToVideoDendrite( bittensor.Dendrite ):

    def forward(
            self,
            text: str,
            num_inference_steps: int = 30,
            num_frames: int = 30,
            fps: int = 8,
            timeout: float = bittensor.__blocktime__,
            return_call:bool = True,
        ) -> Union[ torch.FloatTensor, TextToVideoForwardCall ]:
        forward_call = TextToVideoForwardCall(
            dendrite = self, 
            timeout = timeout,
            text=text,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            fps=fps,
        )
        response_call = self.loop.run_until_complete( self.apply( dendrite_call = forward_call ) )
        if return_call: return response_call
        else: return response_call.video
    
    async def async_forward(
        self,
        text: str,
        num_inference_steps: int = 30,
        num_frames: int = 30,
        fps: int = 8,
        timeout: float = bittensor.__blocktime__,
        return_call: bool = True,
    ) -> Union[  torch.FloatTensor, TextToVideoForwardCall ]:
        forward_call = TextToVideoForwardCall(
            dendrite = self, 
            timeout = timeout,
            text = text,
            num_inference_steps = num_inference_steps,
            num_frames = num_frames,
            fps = fps,
        )
        forward_call = await self.apply( dendrite_call = forward_call )
        if return_call: return forward_call
        else: return forward_call.video
