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
import json
import torch
import bittensor
from typing import Union, List, Dict

class TextPromptingForwardCall(bittensor.BittensorCall):
    """
    Call state for the text_prompting synapse.
    """

    name: str = "forward_prompting"
    messages = None  # To be filled by the forward call

    def __repr__(self) -> str:
        return f"TextPromptingForwardCall( version: {self.version}, code: {self.response_code}, message: {self.response_message}, caller: {self.hotkey}, timeout: {self.timeout}, start_time: {self.start_time} end_time: {self.end_time}, messages: {self.messages}, response: {self.response} )"

    def __str__(self) -> str:
        return f"""
            bittensor.TextPromptingForwardCall( 
                description: Returns completions from the endpoint based on the passed message and prompt.
                caller: {self.hotkey},
                version: {self.version},
                timeout: {self.timeout}, 
                start_time: {self.start_time},
                end_time: {self.end_time},
                elapsed: {self.end_time - self.start_time},
                Args:
                    messages: List[str] = {self.messages}, 
                    response: List[str] = {self.response}
            )
        """

    def __init__(
        self,
        messages: List[str],
        timeout: float = bittensor.__blocktime__,
    ):
        """Forward call to the receptor.
        Args:
            messages (:obj:`str`, `required`):
                stringified list of json text prompts
            timeout (:obj:`float`, `optional`, defaults to 5 seconds):
                timeout for the forward call.
        Returns:
            call.TextPromptingForwardCall (:obj:`call.TextPromptingForwardCall`, `required`):
                bittensor forward call dataclass.
        """
        super().__init__(timeout=timeout)
        self.messages = messages
        self.response = None

    def get_inputs_shape(self) -> Union[torch.Size, None]:
        if self.messages is not None:
            return torch.Size( [len(message) for message in self.messages] )
        return None

    def get_outputs_shape(self) -> Union[torch.Size, None]:
        if self.response is not None:
            return torch.Size([len(self.response)])
        return None

class TextPromptingBackwardCall(bittensor.BittensorCall):
    """Call state for the text prompting backward call """

    # The name of the synapse call.
    name: str = "backward_prompting"

    def __repr__(self) -> str:
        return f"TextPromptingBackwardCall( version: {self.version}, code: {self.response_code}, message: {self.response_message}, caller: {self.hotkey}, timeout: {self.timeout}, start_time: {self.start_time} end_time: {self.end_time}, messages: {self.messages}, response: {self.response}, rewards: {self.rewards} )"


    def __str__(self) -> str:
        return f"""
            bittensor.TextPromptingBackwardCall( 
                description: Returns RL rewards to miner based on reward model scoring.
                caller: {self.hotkey},
                version: {self.version},
                timeout: {self.timeout}, 
                start_time: {self.start_time},
                end_time: {self.end_time},
                elapsed: {self.end_time - self.start_time},
                Args:
                    messages: List[Dict[str,str]] = {self.messages}, 
                    response: str = {self.response}, 
                    rewards: List[float] = {self.rewards}
            )
        """

    def __init__(
        self,
        messages: List[str],
        response: str,
        rewards: List[float],
        timeout: float = bittensor.__blocktime__,
    ):
        """Forward call to the receptor.
        Args:
            messages (:obj:`List[str]`, `required`):
                messages on forward call.
            response (:obj:`str`, `required`):
                response from forward call.
            rewards (:obj:`List[float]`, `required`):
                rewards vector from reward model from forward call.
            timeout (:obj:`float`, `optional`, defaults to 5 seconds):
                timeout for the backward call. (redundant.)
        Returns:
            call.TextPromptingForwardCall (:obj:`call.TextPromptingForwardCall`, `required`):
                bittensor forward call dataclass.
        """
        super().__init__(timeout=timeout)
        self.messages = messages
        self.response = response
        self.rewards = rewards

    def get_inputs_shape(self) -> Union[torch.Size, None]:
        if self.rewards is not None:
            return torch.Size( [ len(self.rewards) ] )
        return torch.Size( [] )

    def get_outputs_shape(self) -> Union[torch.Size, None]:
        return torch.Size( [] )