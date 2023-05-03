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
import grpc
import json
import torch
import asyncio
import bittensor
from typing import Callable, List, Dict, Union

class DendriteForwardCall( bittensor.DendriteCall ):

    name: str = "data_blocks_forward"
    is_forward: bool = True
    completion: str = "" # To be filled.

    def __init__(
        self,
        dendrite: 'bittensor.SubnetServerDendrite',
        messages: List[str],
        roles: List[str],
        timeout: float = bittensor.__blocktime__,
    ):
        super().__init__( dendrite = dendrite, timeout = timeout )
        self.messages = messages
        self.roles = roles
        self.packed_messages = [json.dumps({"role": role, "content": message}) for role, message in zip(self.roles, self.messages)]

    def __repr__(self) -> str:
        return f"DendriteForwardCall( {bittensor.utils.codes.code_to_string(self.return_code)}, to: {self.dest_hotkey[:4]}...{self.dest_hotkey[-4:]}, msg: {self.return_message}, completion: {self.completion.strip()})"
    
    def __str__(self) -> str: return self.__repr__()
    
    def get_callable( self ) -> Callable:
        return bittensor.grpc.TextPromptingStub( self.dendrite.channel ).Forward

    def get_request_proto( self ) -> bittensor.proto.DataBlocksRequest:
        return bittensor.ForwardTextPromptingRequest( timeout = self.timeout, messages = self.packed_messages )
    
    def apply_response_proto( self, response_proto: bittensor.DataBlocksResponse ):
        self.completion = response_proto.response
        
    def get_inputs_shape(self) -> torch.Size: 
        return torch.Size( [len(message) for message in self.packed_messages] )

    def get_outputs_shape(self) -> torch.Size:
        return torch.Size([ len(self.completion) ] )
    

class SubnetServerDendrite( bittensor.Dendrite ):

    def get_stub(self, channel) -> Callable:
        return bittensor.grpc.TextPromptingStub(channel)

    def forward(
            self,
            roles: List[ str ] ,
            messages: List[ str ],
            timeout: float = bittensor.__blocktime__,
            return_call:bool = True,
        ) -> Union[ str, DendriteForwardCall ]:
        forward_call = DendriteForwardCall(
            dendrite = self, 
            messages = messages,
            roles = roles,
            timeout = timeout,
        )
        response_call = self.loop.run_until_complete( self.apply( dendrite_call = forward_call ) )
        if return_call: return response_call
        else: return response_call.completion
    
    async def async_forward(
        self,
        roles: List[ str ],
        messages: List[ str ],
        timeout: float = bittensor.__blocktime__,
        return_call: bool = True,
    ) -> Union[ str, DendriteForwardCall ]:
        forward_call = DendriteForwardCall(
            dendrite = self, 
            messages = messages,
            roles = roles,
            timeout = timeout,
        )
        forward_call = await self.apply( dendrite_call = forward_call )
        if return_call: return forward_call
        else: return forward_call.completion




