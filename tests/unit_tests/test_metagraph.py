import asyncio
import copy
from functools import partial
from unittest.mock import Mock

import numpy as np
import pytest

from bittensor.core import settings
from bittensor.core.metagraph import Metagraph
from bittensor.core.subtensor import Subtensor
from bittensor.utils import execute_coroutine


@pytest.fixture
def mock_environment(mocker):
    # Create a Mock for subtensor
    subtensor = mocker.AsyncMock()

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


@pytest.mark.asyncio
async def test_set_metagraph_attributes(mock_environment):
    subtensor, neurons = mock_environment
    metagraph = Metagraph(1, sync=False)
    metagraph.neurons = neurons
    await metagraph._set_metagraph_attributes(block=5, subtensor=subtensor)

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
def mock_subtensor(mocker):
    subtensor = mocker.Mock(spec=Subtensor)
    subtensor.chain_endpoint = settings.FINNEY_ENTRYPOINT
    subtensor.network = "finney"
    subtensor.async_subtensor = mocker.AsyncMock(
        get_current_block=mocker.AsyncMock(return_value=601)
    )
    subtensor.event_loop = asyncio.new_event_loop()
    subtensor.execute_coroutine = partial(
        execute_coroutine, event_loop=subtensor.event_loop
    )
    return subtensor


# Mocking the metagraph instance for testing purposes
@pytest.fixture
def metagraph_instance(mocker):
    metagraph = Metagraph(netuid=1337, sync=False)
    metagraph._assign_neurons = mocker.AsyncMock()
    metagraph._set_metagraph_attributes = mocker.AsyncMock()
    metagraph._set_weights_and_bonds = mocker.AsyncMock()
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


def test_deepcopy(mock_environment):
    subtensor, neurons = mock_environment
    metagraph = Metagraph(1, sync=False)
    metagraph.neurons = neurons
    metagraph.subtensor = subtensor

    # Do a deep copy
    copied_metagraph = copy.deepcopy(metagraph)

    # Check that the subtensor attribute is None
    assert copied_metagraph.subtensor is None

    # Check that other attributes are copied correctly
    assert copied_metagraph.n == metagraph.n
    assert copied_metagraph.block == metagraph.block
    assert np.array_equal(copied_metagraph.uids, metagraph.uids)
    assert np.array_equal(copied_metagraph.stake, metagraph.stake)
    assert np.array_equal(copied_metagraph.total_stake, metagraph.total_stake)
    assert np.array_equal(copied_metagraph.ranks, metagraph.ranks)
    assert np.array_equal(copied_metagraph.trust, metagraph.trust)
    assert np.array_equal(copied_metagraph.consensus, metagraph.consensus)
    assert np.array_equal(copied_metagraph.validator_trust, metagraph.validator_trust)
    assert np.array_equal(copied_metagraph.incentive, metagraph.incentive)
    assert np.array_equal(copied_metagraph.emission, metagraph.emission)
    assert np.array_equal(copied_metagraph.dividends, metagraph.dividends)
    assert np.array_equal(copied_metagraph.active, metagraph.active)
    assert np.array_equal(copied_metagraph.last_update, metagraph.last_update)
    assert np.array_equal(copied_metagraph.validator_permit, metagraph.validator_permit)
    assert np.array_equal(copied_metagraph.weights, metagraph.weights)
    assert np.array_equal(copied_metagraph.bonds, metagraph.bonds)

    # Check that the neurons are different objects in the original and copied metagraphs
    for original_neuron, copied_neuron in zip(
        metagraph.neurons, copied_metagraph.neurons
    ):
        assert original_neuron is not copied_neuron
        assert original_neuron.uid == copied_neuron.uid
        assert original_neuron.trust == copied_neuron.trust
        assert original_neuron.consensus == copied_neuron.consensus
        assert original_neuron.incentive == copied_neuron.incentive
        assert original_neuron.dividends == copied_neuron.dividends
        assert original_neuron.rank == copied_neuron.rank
        assert original_neuron.emission == copied_neuron.emission
        assert original_neuron.active == copied_neuron.active
        assert original_neuron.last_update == copied_neuron.last_update
        assert original_neuron.validator_permit == copied_neuron.validator_permit
        assert original_neuron.validator_trust == copied_neuron.validator_trust
        assert original_neuron.total_stake.tao == copied_neuron.total_stake.tao
        assert original_neuron.stake == copied_neuron.stake
        assert original_neuron.axon_info == copied_neuron.axon_info
        assert original_neuron.weights == copied_neuron.weights
        assert original_neuron.bonds == copied_neuron.bonds


def test_copy(mock_environment):
    subtensor, neurons = mock_environment
    metagraph = Metagraph(1, sync=False)
    metagraph.neurons = neurons
    metagraph.subtensor = subtensor

    # Do a shallow copy
    copied_metagraph = copy.copy(metagraph)

    # Check that the subtensor attribute is None in the copied object
    assert copied_metagraph.subtensor is None

    # Check that other attributes are copied correctly
    assert copied_metagraph.n == metagraph.n
    assert copied_metagraph.block == metagraph.block
    assert np.array_equal(copied_metagraph.uids, metagraph.uids)
    assert np.array_equal(copied_metagraph.stake, metagraph.stake)
    assert np.array_equal(copied_metagraph.total_stake, metagraph.total_stake)
    assert np.array_equal(copied_metagraph.ranks, metagraph.ranks)
    assert np.array_equal(copied_metagraph.trust, metagraph.trust)
    assert np.array_equal(copied_metagraph.consensus, metagraph.consensus)
    assert np.array_equal(copied_metagraph.validator_trust, metagraph.validator_trust)
    assert np.array_equal(copied_metagraph.incentive, metagraph.incentive)
    assert np.array_equal(copied_metagraph.emission, metagraph.emission)
    assert np.array_equal(copied_metagraph.dividends, metagraph.dividends)
    assert np.array_equal(copied_metagraph.active, metagraph.active)
    assert np.array_equal(copied_metagraph.last_update, metagraph.last_update)
    assert np.array_equal(copied_metagraph.validator_permit, metagraph.validator_permit)
    assert copied_metagraph.axons == metagraph.axons
    assert copied_metagraph.neurons == metagraph.neurons
    assert np.array_equal(copied_metagraph.weights, metagraph.weights)
    assert np.array_equal(copied_metagraph.bonds, metagraph.bonds)

    # Check that the neurons are the same objects in the original and copied metagraphs
    for original_neuron, copied_neuron in zip(
        metagraph.neurons, copied_metagraph.neurons
    ):
        assert original_neuron is copied_neuron
