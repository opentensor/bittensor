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
from typing import List, Dict
bittensor.logging(debug=True)
# import openai



# Create a synapse that returns zeros.
class Synapse(bittensor.TextPromptingSynapse):
    def _priority(self, forward_call: "bittensor.TextPromptingForwardCall") -> float:
        return 0.0

    def _blacklist(self, forward_call: "bittensor.TextPromptingForwardCall") -> bool:
        return False

    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str:
        # Apply PPO.
        pass

    def forward(self, messages: List[Dict[str, str]]) -> str:
        return "hello im a chat bot."

# Create a mock wallet.
wallet = bittensor.wallet().create_if_non_existent()

# Create a local endpoint receptor grpc connection.
local_endpoint = bittensor.endpoint(
    version=bittensor.__version_as_int__,
    uid=0,
    ip="127.0.0.1",
    ip_type=4,
    port=9090,
    hotkey=wallet.hotkey.ss58_address,
    coldkey=wallet.coldkeypub.ss58_address,
    modality=0,
)

metagraph = None # Allow offline testing with unregistered keys.
axon = bittensor.axon(wallet=wallet, port=9090, ip="127.0.0.1", metagraph=metagraph)

synapse = Synapse()
axon.attach(synapse=synapse)
axon.start()

batch_size = 4
sequence_length = 32
# Create a text_prompting module and call it.
module = bittensor.text_prompting( endpoint = local_endpoint, wallet = wallet )
forward_response = module.forward(
    roles = ['user', 'assistant'],
    messages = [{ "user": "Human", "content": "hello"}],
    timeout = 1e6
)
backward_response = module.backward(
    roles = ['user', 'assistant'],
    messages = [{ "user": "Human", "content": "hello"}],
    response = forward_response.response,
    rewards = [1,2,3,4,5],
    timeout = 1e6
)


# # Delete objects.
del axon
del synapse
del module
