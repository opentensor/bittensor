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

import copy

import torch

import bittensor


class TextPromptingSynapse(bittensor.Synapse, bittensor.grpc.TextPromptingServicer):
    """TextPromptingSynapse: A class for servicing text_prompting requests."""

    synapse_name: str = "text_prompting"
    default_blacklist_stake: float = 10

    def __init__(
        self,
        config: "bittensor.Config" = None,
    ):
        if config is None:
            config = self.config()
        TextPromptingSynapse.check_config(config)
        super().__init__(config)
        self.config = copy.deepcopy(config)

    def _attach(self, axon: "bittensor.axon"):
        """_attach: Attaches the synapse to the axon."""
        bittensor.grpc.add_TextPromptingServicer_to_server(self, axon.server)

    def pre_process_request_proto_to_forward_call(
        self, request_proto: bittensor.ForwardTextPromptingRequest
    ) -> "bittensor.TextPromptingForwardCall":
        return bittensor.TextPromptingForwardCall( messages=request_proto.messages, timeout=request_proto.timeout )

    def post_process_forward_call_to_response_proto(
        self, forward_call: "bittensor.TextPromptingForwardCall"
    ) -> bittensor.ForwardTextPromptingResponse:
        return bittensor.ForwardTextPromptingResponse(
            version = bittensor.__version_as_int__,
            hotkey = self.axon.wallet.hotkey.ss58_address,
            responses = forward_call.responses,
            message = forward_call.request_message,
            return_code = forward_call.request_code,
        )
    
    def pre_process_request_proto_to_backward_call(
        self, request_proto: object
    ) -> "bittensor.TextPromptingBackwardCall":
        pass
