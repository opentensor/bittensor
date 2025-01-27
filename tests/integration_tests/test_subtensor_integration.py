import os.path

import pytest
from bt_decode import PortableRegistry, MetadataV15

from bittensor import NeuronInfo
from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.subtensor import Subtensor
from bittensor.utils.balance import Balance
from tests.helpers.helpers import FakeConnectContextManager


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
    with open(
        os.path.join(os.path.dirname(__file__), "..", "helpers", "registry"), "rb"
    ) as f:
        registry = PortableRegistry.from_metadata_v15(
            MetadataV15.decode_from_metadata_option(f.read())
        )
    subtensor = Subtensor("unknown", _mock=True)
    mocker.patch(
        "async_substrate_interface.sync_substrate.connect",
        mocker.Mock(return_value=FakeConnectContextManager(seed=seed)),
    )
    mocker.patch.object(subtensor.substrate, "registry", registry)
    return subtensor


@pytest.mark.asyncio
async def test_get_all_subnets_info(mocker):
    subtensor = await prepare_test(mocker, "get_all_subnets_info")
    result = subtensor.get_all_subnets_info()
    assert isinstance(result, list)
    assert result[0].owner_ss58 == "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    assert result[1].kappa == 32767
    assert result[1].max_weight_limit == 65535
    assert result[1].blocks_since_epoch == 1


@pytest.mark.asyncio
async def test_metagraph(mocker):
    subtensor = await prepare_test(mocker, "metagraph")
    result = subtensor.metagraph(23)
    assert result.n == 19
    assert result.netuid == 23
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
    result = subtensor.blocks_since_last_update(23, 5)
    assert result == 1293687


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
async def test_get_neuron_for_pubkey_and_subnet(hotkey, netuid, mocker):
    subtensor = await prepare_test(mocker, "get_neuron_for_pubkey_and_subnet")
    result = subtensor.get_neuron_for_pubkey_and_subnet(hotkey, netuid)
    assert isinstance(result, NeuronInfo)
    assert result.hotkey == hotkey
    assert isinstance(result.total_stake, Balance)
    assert isinstance(result.axon_info, AxonInfo)
    assert result.is_null is False
