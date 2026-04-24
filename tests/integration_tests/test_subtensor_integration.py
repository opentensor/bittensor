import pytest

from bittensor.core.chain_data import AxonInfo, NeuronInfo, CrowdloanInfo
from bittensor.core.subtensor import Subtensor
from bittensor.core.types import CrowdloansResponse
from bittensor.utils.balance import Balance
from tests.helpers.helpers import FakeWebsocket
from bittensor.utils.mock.subtensor_mock import MockSubtensor


@pytest.fixture
def hotkey():
    yield "5DkzsviNQr4ZePXMmEfNPDcE7cQ9cVyepmQbgUw6YT3odcwh"


@pytest.fixture
def netuid():
    yield 23


async def prepare_test(mocker, seed, **subtensor_args):
    """
    Helper function: sets up the test environment.
    """
    mocker.patch(
        "async_substrate_interface.sync_substrate.connect",
        mocker.Mock(return_value=FakeWebsocket(seed=seed)),
    )
    subtensor = Subtensor("unknown", mock=True, **subtensor_args)
    return subtensor


# TODO: update test after Runtime updated
# @pytest.mark.asyncio
# async def test_get_all_subnets_info(mocker):
#     subtensor = await prepare_test(mocker, "get_all_subnets_info")
#     result = subtensor.get_all_subnets_info()
#     assert isinstance(result, list)
#     assert result[0].owner_ss58 == "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
#     assert result[1].kappa == 32767
#     assert result[1].max_weight_limit == 65535
#     assert result[1].blocks_since_epoch == 30


@pytest.mark.asyncio
async def test_metagraph(mocker):
    subtensor = await prepare_test(mocker, "metagraph")
    result = subtensor.metagraph(1)
    assert result.n == 1024
    assert result.netuid == 1
    assert result.block.item() == 6852690


@pytest.mark.asyncio
async def test_get_netuids_for_hotkey(mocker):
    subtensor = await prepare_test(mocker, "get_netuids_for_hotkey")
    result = subtensor.get_netuids_for_hotkey(
        "5Cf4LPRv6tiyuFsfLRQaFYEEn3zJRGi4bAE9DwbbKmbCSHpV"
    )
    assert result == [14]


@pytest.mark.asyncio
async def test_get_current_block(mocker):
    subtensor = await prepare_test(mocker, "get_current_block")
    result = subtensor.get_current_block()
    assert result == 6522038


@pytest.mark.asyncio
async def test_is_hotkey_registered_any(mocker):
    subtensor = await prepare_test(mocker, "is_hotkey_registered_any")
    result = subtensor.is_hotkey_registered_any(
        "5Cf4LPRv6tiyuFsfLRQaFYEEn3zJRGi4bAE9DwbbKmbCSHpV"
    )
    assert result is True


@pytest.mark.asyncio
async def test_is_hotkey_registered_on_subnet(mocker):
    subtensor = await prepare_test(mocker, "is_hotkey_registered_on_subnet")
    result = subtensor.is_hotkey_registered_on_subnet(
        "5Cf4LPRv6tiyuFsfLRQaFYEEn3zJRGi4bAE9DwbbKmbCSHpV", 14
    )
    assert result is True


@pytest.mark.asyncio
async def test_is_hotkey_registered(mocker):
    subtensor = await prepare_test(mocker, "is_hotkey_registered")
    result = subtensor.is_hotkey_registered(
        "5Cf4LPRv6tiyuFsfLRQaFYEEn3zJRGi4bAE9DwbbKmbCSHpV"
    )
    assert result is True


@pytest.mark.asyncio
async def test_blocks_since_last_update(mocker):
    subtensor = await prepare_test(mocker, "blocks_since_last_update")
    result = subtensor.blocks_since_last_update(1, 0)
    assert result == 4009702


@pytest.mark.asyncio
async def test_get_block_hash(mocker):
    subtensor = await prepare_test(mocker, "get_block_hash")
    result = subtensor.get_block_hash(6522038)
    assert (
        result == "0x4f0bbe5e96a78ca4cc1df9a7e757d335165d6d71c545cf6eb4794986424bcd29"
    )


@pytest.mark.asyncio
async def test_subnetwork_n(mocker):
    subtensor = await prepare_test(mocker, "subnetwork_n")
    result = subtensor.subnetwork_n(1)
    assert result == 1024


@pytest.mark.asyncio
async def test_get_neuron_for_pubkey_and_subnet(mocker):
    subtensor = await prepare_test(mocker, "get_neuron_for_pubkey_and_subnet")
    result = subtensor.get_neuron_for_pubkey_and_subnet(
        "5Cf4LPRv6tiyuFsfLRQaFYEEn3zJRGi4bAE9DwbbKmbCSHpV", 14
    )
    assert isinstance(result, NeuronInfo)
    assert result.hotkey == "5Cf4LPRv6tiyuFsfLRQaFYEEn3zJRGi4bAE9DwbbKmbCSHpV"
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


@pytest.mark.asyncio
async def test_archive_node_retry(mocker):
    subtensor = await prepare_test(
        mocker, "retry_archive", archive_endpoints=["ws://fake-endpoi.nt"]
    )
    current_block = subtensor.substrate.get_block_number()
    old_block = current_block - 1000
    assert isinstance((subtensor.substrate.get_block(block_number=old_block)), dict)


@pytest.mark.asyncio
async def test_decode_crowdloan_entry(mocker):
    entry = CrowdloansResponse(
        **{
            "creator": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "deposit": 10000000000,
            "min_contribution": 1000000000,
            "end": 10055,
            "cap": 100000000000,
            "funds_account": "5EYCAe5fvncWtwXjyNBBHFPvNVDH5LPQ2harKS7KdAGbezkb",
            "raised": 10000000000,
            "target_address": None,
            "call": {
                "Inline": "0x0500008eaf04151687736326c9fea17e25fc5287613693c912909cb226aa4794f26a4802286bee"
            },
            "finalized": False,
            "contributors_count": 1,
        }
    )
    subtensor = await prepare_test(mocker, "decode_crowdloan_entry")
    actual = subtensor._decode_crowdloan_entry(17, entry)
    expected = CrowdloanInfo(
        id=17,
        creator="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        deposit=Balance.from_tao(10.0),
        min_contribution=Balance.from_tao(1.0),
        end=10055,
        cap=Balance.from_tao(100.0),
        funds_account="5EYCAe5fvncWtwXjyNBBHFPvNVDH5LPQ2harKS7KdAGbezkb",
        raised=Balance.from_tao(10.0),
        target_address=None,
        call={
            "call_index": "0x0500",
            "call_function": "transfer_allow_death",
            "call_module": "Balances",
            "call_args": [
                {
                    "name": "dest",
                    "type": "AccountIdLookupOf",
                    "value": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                },
                {"name": "value", "type": "Balance", "value": 1000000000},
            ],
            "call_hash": "0x117349ae93488150fa503b1ff7a0a94bfaa3ba193950a3d812ce32b9bb69fb02",
        },
        finalized=False,
        contributors_count=1,
    )
    assert actual == expected
