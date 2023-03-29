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

from typing import Callable, List

import torch

import bittensor


class TextPromptingDendrite(bittensor.Dendrite):
    """Dendrite for the text_prompting synapse."""

    # Dendrite name.
    name: str = "text_prompting"

    def __str__(self) -> str:
        return "TextPrompting"

    def get_stub(self, channel) -> Callable:
        return bittensor.grpc.TextPromptingStub(channel)

    def pre_process_forward_call_to_request_proto(
        self, forward_call: "bittensor.TextPromptingForwardCall"
    ) -> "bittensor.ForwardTextPromptingRequest":
        return bittensor.ForwardTextPromptingRequest( timeout = forward_call.timeout, messages = forward_call.messages )

    def post_process_response_proto_to_forward_call(
        self,
        forward_call: bittensor.TextPromptingForwardCall,
        response_proto: bittensor.ForwardTextPromptingResponse,
    ) -> bittensor.TextPromptingForwardCall:
        forward_call.response_code = response_proto.return_code
        forward_call.response_message = response_proto.message
        forward_call.response = response_proto.response
        return forward_call

    def forward(
        self,
        messages: List[str],
        timeout: float = bittensor.__blocktime__,
    ) -> "bittensor.TextPromptingForwardCall":
        return self._forward(
            forward_call=bittensor.TextPromptingForwardCall(
                messages = messages,
                timeout = timeout,
            )
        )
