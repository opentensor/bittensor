# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from unittest.mock import MagicMock
from unittest.mock import Mock

import numpy as np
import pytest

from bittensor.core import settings
from bittensor.core.metagraph import Metagraph


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
    metagraph = Metagraph(1, sync=False)
    metagraph.neurons = neurons
    metagraph._set_metagraph_attributes(block=5, subtensor=subtensor)

    # Check the attributes are set as expected
    assert metagraph.n.item() == len(neurons)
    assert metagraph.block.item() == 5
    assert (
        np.array_equal(
            metagraph.uids,
            np.array([neuron.uid for neuron in neurons], dtype=np.int64),
        )
        is True
    )

    assert (
        np.array_equal(
            metagraph.trust,
            np.array([neuron.trust for neuron in neurons], dtype=np.float32),
        )
        is True
    )

    assert (
        np.array_equal(
            metagraph.consensus,
            np.array([neuron.consensus for neuron in neurons], dtype=np.float32),
        )
        is True
    )
    # Similarly for other attributes...

    # Test the axons
    assert metagraph.axons == [n.axon_info for n in neurons]


def test_process_weights_or_bonds(mock_environment):
    _, neurons = mock_environment
    metagraph = Metagraph(1, sync=False)
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


# Mocking the bittensor.Subtensor class for testing purposes
@pytest.fixture
def mock_subtensor():
    subtensor = MagicMock()
    subtensor.chain_endpoint = settings.FINNEY_ENTRYPOINT
    subtensor.network = "finney"
    subtensor.get_current_block.return_value = 601
    return subtensor


# Mocking the metagraph instance for testing purposes
@pytest.fixture
def metagraph_instance():
    metagraph = Metagraph(netuid=1337, sync=False)
    metagraph._assign_neurons = MagicMock()
    metagraph._set_metagraph_attributes = MagicMock()
    metagraph._set_weights_and_bonds = MagicMock()
    metagraph._get_all_stakes_from_chain = MagicMock()
    return metagraph


@pytest.fixture
def loguru_sink():
    class LogSink:
        def __init__(self):
            self.messages = []

        def write(self, message):
            # Assuming `message` is an object, you might need to adjust how you extract the text
            self.messages.append(str(message))

        def __contains__(self, item):
            return any(item in message for message in self.messages)

    return LogSink()


@pytest.mark.parametrize(
    "block, test_id",
    [
        (300, "warning_case_block_greater_than_300"),
    ],
)
def test_sync_warning_cases(block, test_id, metagraph_instance, mock_subtensor, caplog):
    metagraph_instance.sync(block=block, lite=True, subtensor=mock_subtensor)

    expected_message = "Attempting to sync longer than 300 blocks ago on a non-archive node. Please use the 'archive' network for subtensor and retry."
    assert (
        expected_message in caplog.text
    ), f"Test ID: {test_id} - Expected warning message not found in Loguru sink."
