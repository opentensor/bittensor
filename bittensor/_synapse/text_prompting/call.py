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

import time
from typing import Union, List

import torch

import bittensor


class TextPromptingForwardCall(bittensor.BittensorCall):
    """Call state for the text_prompting synapse."""

    # The name of the synapse call.
    name: str = "forward_prompting"
    messages = None # To be filled by the forward call

    def __str__(self) -> str:
        return """
bittensor.TextPromptingForwardCall( 
    description: Returns the logits for the last predicted item in a given sequence.
    caller: {},
    version: {},
    timeout = {}, 
    start_time = {},
    end_time = {},
    elapsed = {},
    Args:
    \messages: List[str] = {}, 
)
""".format(
            self.hotkey,
            self.version,
            self.timeout,
            self.start_time,
            self.end_time,
            time.time() - self.start_time,
            self.messages,
            self.responses if self.responses is not None else "To be filled by the forward call.",
        )

    def __init__(
        self,
        messages: List[str],
        timeout: float = bittensor.__blocktime__,
    ):
        """Forward call to the receptor.
        Args:
            messages (:obj:`List[str]` of shape :obj:`(n)`, `required`):
                list of text prompts
            timeout (:obj:`float`, `optional`, defaults to 5 seconds):
                timeout for the forward call.
        Returns:
            call.TextPromptingForwardCall (:obj:`call.TextPromptingForwardCall`, `required`):
                bittensor forward call dataclass.
        """
        super().__init__(timeout=timeout)
        self.messages = messages
        self.responses = None

    def get_inputs_shape(self) -> Union[torch.Size, None]:
        if self.message is not None:
            return torch.Size( [len(message) for message in self.messages] )
        return None

    def get_outputs_shape(self) -> Union[torch.Size, None]:
        if self.responses is not None:
            return torch.Size( [len(message) for message in self.responses] )
        return None