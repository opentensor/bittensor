# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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
import bittensor
import torch
import unittest
from unittest.mock import MagicMock


class MockTextPromptingSynapse( bittensor.TextPromptingSynapse ):
    def forward( self, messages ):
        return messages

    def multi_forward( self, messages ):
        return messages

    def backward( self, messages, response, rewards ):
        return messages, response, rewards

    def priority( self, call: bittensor.SynapseCall ) -> float:
        return 0.0

    def blacklist( self, call: bittensor.SynapseCall ) -> bool:
        return False

def test_create_text_prompting():
    mock_wallet = MagicMock(
        spec=bittensor.Wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )
    axon = bittensor.axon( wallet = mock_wallet, metagraph = None )
    synapse = MockTextPromptingSynapse( axon = axon )

# @unittest.skip("This is for convenience of testing without violating DRY too much")
def get_synapse():
    mock_wallet = MagicMock(
        spec=bittensor.Wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )
    axon = bittensor.axon( wallet = mock_wallet, metagraph = None )
    return MockTextPromptingSynapse( axon = axon )


def test_text_prompting_synapse_forward():
    synapse = get_synapse()
    messages = ['test message']
    response = synapse.forward( messages )
    assert response == messages

def test_text_prompting_synapse_multi_forward():
    synapse = get_synapse()
    messages = ['test message'] * 10
    responses = synapse.multi_forward( messages )
    assert responses == messages

def test_text_prompting_synapse_backward():
    synapse = get_synapse()
    messages = ['test message']
    response = ['test response']
    rewards = torch.tensor([1.0])
    output = synapse.backward( messages, response, rewards )
    assert len(output) == 3
    assert messages == output[0]
    assert response == output[1]
    assert torch.all(torch.eq(rewards, output[2]))

def test_text_prompting_synapse_blacklist():
    synapse = get_synapse()
    request = bittensor.proto.ForwardTextPromptingRequest(
        hotkey = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
    )
    context = MagicMock()
    call = bittensor._synapse.text_prompting.synapse.SynapseForward( synapse, request, synapse.forward )
    blacklist = synapse.blacklist( call )
    assert blacklist == False

def test_text_prompting_synapse_priority():
    synapse = get_synapse()
    request = bittensor.proto.ForwardTextPromptingRequest(
        hotkey = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
    )
    call = bittensor._synapse.text_prompting.synapse.SynapseForward( synapse, request, synapse.forward )
    priority = synapse.priority( call )
    assert priority == 0.0

