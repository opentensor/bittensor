import pytest

from bittensor.core.chain_data import AxonInfo, NeuronInfo
from bittensor.core.subtensor import Subtensor
from bittensor.utils.balance import Balance
from tests.helpers.helpers import FakeWebsocket
from bittensor.utils.mock.subtensor_mock import MockSubtensor


@pytest.fixture
def hotkey():
    yield "5DkzsviNQr4ZePXMmEfNPDcE7cQ9cVyepmQbgUw6YT3odcwh"


@pytest.fixture
def netuid():
    yield 23


async def prepare_test(mocker, seed):
    """
    Helper function: sets up the test environment.
    """
    mocker.patch(
        "async_substrate_interface.sync_substrate.connect",
        mocker.Mock(return_value=FakeWebsocket(seed=seed)),
    )
    subtensor = Subtensor("unknown", _mock=True)
    return subtensor


@pytest.mark.asyncio
async def test_get_all_subnets_info(mocker):
    subtensor = await prepare_test(mocker, "get_all_subnets_info")
    result = subtensor.get_all_subnets_info()
    assert isinstance(result, list)
    assert result[0].owner_ss58 == "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    assert result[1].kappa == 32767
    assert result[1].max_weight_limit == 65535
    assert result[1].blocks_since_epoch == 88


@pytest.mark.asyncio
async def test_metagraph(mocker):
    subtensor = await prepare_test(mocker, "metagraph")
    result = subtensor.metagraph(1)
    assert result.n == 1
    assert result.netuid == 1
    assert result.block == 3264143


@pytest.mark.asyncio
async def test_get_netuids_for_hotkey(mocker):
    subtensor = await prepare_test(mocker, "get_netuids_for_hotkey")
    result = subtensor.get_netuids_for_hotkey(
        "5DkzsviNQr4ZePXMmEfNPDcE7cQ9cVyepmQbgUw6YT3odcwh"
    )
    assert result == [23]


@pytest.mark.asyncio
async def test_get_current_block(mocker):
    subtensor = await prepare_test(mocker, "get_current_block")
    result = subtensor.get_current_block()
    assert result == 3264143


@pytest.mark.asyncio
async def test_is_hotkey_registered_any(mocker):
    subtensor = await prepare_test(mocker, "is_hotkey_registered_any")
    result = subtensor.is_hotkey_registered_any(
        "5DkzsviNQr4ZePXMmEfNPDcE7cQ9cVyepmQbgUw6YT3odcwh"
    )
    assert result is True


@pytest.mark.asyncio
async def test_is_hotkey_registered_on_subnet(mocker):
    subtensor = await prepare_test(mocker, "is_hotkey_registered_on_subnet")
    result = subtensor.is_hotkey_registered_on_subnet(
        "5DkzsviNQr4ZePXMmEfNPDcE7cQ9cVyepmQbgUw6YT3odcwh", 23
    )
    assert result is True


@pytest.mark.asyncio
async def test_is_hotkey_registered(mocker):
    subtensor = await prepare_test(mocker, "is_hotkey_registered")
    result = subtensor.is_hotkey_registered(
        "5DkzsviNQr4ZePXMmEfNPDcE7cQ9cVyepmQbgUw6YT3odcwh"
    )
    assert result is True


@pytest.mark.asyncio
async def test_blocks_since_last_update(mocker):
    subtensor = await prepare_test(mocker, "blocks_since_last_update")
    result = subtensor.blocks_since_last_update(1, 0)
    assert result == 3264146


@pytest.mark.asyncio
async def test_get_block_hash(mocker):
    subtensor = await prepare_test(mocker, "get_block_hash")
    result = subtensor.get_block_hash(3234677)
    assert (
        result == "0xe89482ae7892ab5633f294179245f4058a99781e15f21da31eb625169da5d409"
    )


@pytest.mark.asyncio
async def test_subnetwork_n(mocker):
    subtensor = await prepare_test(mocker, "subnetwork_n")
    result = subtensor.subnetwork_n(1)
    assert result == 94


@pytest.mark.asyncio
async def test_get_neuron_for_pubkey_and_subnet(mocker):
    subtensor = await prepare_test(mocker, "get_neuron_for_pubkey_and_subnet")
    result = subtensor.get_neuron_for_pubkey_and_subnet(
        "5EU2xVWC7qffsUNGtvakp5WCj7WGJMPkwG1dsm3qnU2Kqvee", 1
    )
    assert isinstance(result, NeuronInfo)
    assert result.hotkey == "5EU2xVWC7qffsUNGtvakp5WCj7WGJMPkwG1dsm3qnU2Kqvee"
    assert isinstance(result.total_stake, Balance)
    assert isinstance(result.axon_info, AxonInfo)
    assert result.is_null is False


def test_mock_subtensor_force_register_neuron():
    """Tests the force_register_neuron method of the MockSubtensor class."""
    # Preps
    test_netuid = 1
    subtensor = MockSubtensor()
    subtensor.create_subnet(netuid=test_netuid)

    uid1 = subtensor.force_register_neuron(test_netuid, "hk1", "cc1")
    uid2 = subtensor.force_register_neuron(test_netuid, "hk2", "cc2")

    # Calls
    neurons = subtensor.neurons(test_netuid)
    neuron1 = subtensor.neuron_for_uid(uid1, test_netuid)
    neuron2 = subtensor.neuron_for_uid(uid2, test_netuid)

    # Assertions
    assert len(neurons) == 2
    assert [neuron1, neuron2] == neurons
    assert neuron1.hotkey == "hk1"
    assert neuron1.coldkey == "cc1"
    assert neuron2.hotkey == "hk2"
    assert neuron2.coldkey == "cc2"
