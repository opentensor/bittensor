# The MIT License (MIT)
# Copyright © 2023 Opentensor Technologies Inc

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

from unittest.mock import Mock
import pytest
import torch
import bittensor


@pytest.fixture
def mock_environment():
    # Create a Mock for subtensor
    subtensor = Mock()

    # Create a list of Mock Neurons
    neurons = [
        Mock(
            uid=i,
            trust=i + 0.5,
            consensus=i + 0.1,
            incentive=i + 0.2,
            dividends=i + 0.3,
            rank=i + 0.4,
            emission=i + 0.5,
            active=i,
            last_update=i,
            validator_permit=i % 2 == 0,
            validator_trust=i + 0.6,
            total_stake=Mock(tao=i + 0.7),
            stake=i + 0.8,
            axon_info=f"axon_info_{i}",
            weights=[(j, j + 0.1) for j in range(5)],
            bonds=[(j, j + 0.2) for j in range(5)],
        )
        for i in range(10)
    ]

    return subtensor, neurons


def test_set_metagraph_attributes(mock_environment):
    subtensor, neurons = mock_environment
    metagraph = bittensor.metagraph(1, sync=False)
    metagraph.neurons = neurons
    metagraph._set_metagraph_attributes(block=5, subtensor=subtensor)

    # Check the attributes are set as expected
    assert metagraph.n.item() == len(neurons)
    assert metagraph.block.item() == 5
    assert (
        torch.equal(
            metagraph.uids,
            torch.tensor([neuron.uid for neuron in neurons], dtype=torch.int64),
        )
        == True
    )

    assert (
        torch.equal(
            metagraph.trust,
            torch.tensor([neuron.trust for neuron in neurons], dtype=torch.float32),
        )
        == True
    )

    assert (
        torch.equal(
            metagraph.consensus,
            torch.tensor([neuron.consensus for neuron in neurons], dtype=torch.float32),
        )
        == True
    )
    # Similarly for other attributes...

    # Test the axons
    assert metagraph.axons == [n.axon_info for n in neurons]


def test_process_weights_or_bonds(mock_environment):
    _, neurons = mock_environment
    metagraph = bittensor.metagraph(1, sync=False)
    metagraph.neurons = neurons

    # Test weights processing
    weights = metagraph._process_weights_or_bonds(
        data=[neuron.weights for neuron in neurons], attribute="weights"
    )
    assert weights.shape[0] == len(
        neurons
    )  # Number of rows should be equal to number of neurons
    assert weights.shape[1] == len(
        neurons
    )  # Number of columns should be equal to number of neurons
    # TODO: Add more checks to ensure the weights have been processed correctly

    # Test bonds processing
    bonds = metagraph._process_weights_or_bonds(
        data=[neuron.bonds for neuron in neurons], attribute="bonds"
    )
    assert bonds.shape[0] == len(
        neurons
    )  # Number of rows should be equal to number of neurons
    assert bonds.shape[1] == len(
        neurons
    )  # Number of columns should be equal to number of neurons

    # TODO: Add more checks to ensure the bonds have been processed correctly
