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
from typing import List, Dict, Union, Tuple

bittensor.logging(bittensor.logging.config())


class Synapse(bittensor.TextPromptingSynapse):
    def priority(self, forward_call: "bittensor.TextPromptingForwardCall") -> float:
        return 0.0

    def blacklist(
        self, forward_call: "bittensor.TextPromptingForwardCall"
    ) -> Union[Tuple[bool, str], bool]:
        return False

    def backward(
        self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor
    ) -> str:
        pass

    def forward(self, messages: List[Dict[str, str]]) -> str:
        return "hello im a chat bot."

    def multi_forward(self, messages: List[Dict[str, str]]) -> List[str]:
        return ["hello im a chat bot.", "my name is bob"]


# Create a mock wallet.
wallet = bittensor.wallet().create_if_non_existent()
axon = bittensor.axon(wallet=wallet, port=9090, external_ip="127.0.0.1")

dendrite = bittensor.text_prompting(axon=axon, keypair=wallet.hotkey)
synapse = Synapse(axon=axon)
axon.start()


forward_call = dendrite.forward(
    roles=["system", "assistant"],
    messages=["you are chat bot", "what is the whether"],
    timeout=1e6,
)
print(forward_call)
print(
    "success",
    forward_call.is_success,
    "failed",
    forward_call.did_fail,
    "timedout",
    forward_call.did_timeout,
)
print("completion", forward_call.completion)


multi_forward_call = dendrite.multi_forward(
    roles=["system", "assistant"],
    messages=["you are chat bot", "what is the whether"],
    timeout=1e6,
)
print(multi_forward_call)
print(
    "success",
    multi_forward_call.is_success,
    "failed",
    multi_forward_call.did_fail,
    "timedout",
    multi_forward_call.did_timeout,
)
print("completions", multi_forward_call.multi_completions)
