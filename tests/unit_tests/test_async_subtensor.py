import datetime
import unittest.mock as mock
import numpy as np
import pytest
from async_substrate_interface.types import ScaleObj
from bittensor_wallet import Wallet

from bittensor import u64_normalized_float
from bittensor.core import async_subtensor
from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.core.chain_data import (
    proposal_vote_data,
    ChainIdentity,
    NeuronInfo,
    StakeInfo,
    SelectiveMetagraphIndex,
)
from bittensor.utils import U64_MAX
from bittensor.utils.balance import Balance
from tests.helpers.helpers import assert_submit_signed_extrinsic


@pytest.fixture
def mock_substrate(mocker):
    mocked = mocker.patch(
        "bittensor.core.async_subtensor.AsyncSubstrateInterface",
        autospec=True,
    )
    mocked.return_value.get_block_hash = mocker.AsyncMock()

    return mocked.return_value


@pytest.fixture
def subtensor(mock_substrate):
    return async_subtensor.AsyncSubtensor()


def test_decode_ss58_tuples_in_proposal_vote_data(mocker):
    """Tests that ProposalVoteData instance instantiation works properly,"""
    # Preps
    mocked_decode_account_id = mocker.patch.object(
        proposal_vote_data, "decode_account_id"
    )
    fake_proposal_dict = {
        "index": "0",
        "threshold": 1,
        "ayes": ("0 line", "1 line"),
        "nays": ("2 line", "3 line"),
        "end": 123,
    }

    # Call
    async_subtensor.ProposalVoteData.from_dict(fake_proposal_dict)

    # Asserts
    assert mocked_decode_account_id.call_count == len(fake_proposal_dict["ayes"]) + len(
        fake_proposal_dict["nays"]
    )
    assert mocked_decode_account_id.mock_calls == [
        mocker.call("0 line"),
        mocker.call("1 line"),
        mocker.call("2 line"),
        mocker.call("3 line"),
    ]


def test_decode_hex_identity_dict_with_non_tuple_value():
    """Tests _decode_hex_identity_dict when value is not a tuple."""
    info_dict = {"info": "regular_string"}
    result = async_subtensor.decode_hex_identity_dict(info_dict)
    assert result["info"] == "regular_string"


@pytest.mark.asyncio
async def test_init_if_unknown_network_is_valid(mock_substrate):
    """Tests __init__ if passed network unknown and is valid."""
    # Preps
    fake_valid_endpoint = "wss://blabla.net"

    # Call
    subtensor = AsyncSubtensor(fake_valid_endpoint)

    # Asserts
    assert subtensor.chain_endpoint == fake_valid_endpoint
    assert subtensor.network == "unknown"


@pytest.mark.asyncio
async def test_init_if_unknown_network_is_known_endpoint(mock_substrate):
    """Tests __init__ if passed network unknown and is valid."""
    # Preps
    fake_valid_endpoint = "ws://127.0.0.1:9944"

    # Call
    subtensor = AsyncSubtensor(fake_valid_endpoint)

    # Asserts
    assert subtensor.chain_endpoint == fake_valid_endpoint
    assert subtensor.network == "local"


@pytest.mark.asyncio
async def test_init_if_unknown_network_is_not_valid(mock_substrate):
    """Tests __init__ if passed network unknown and isn't valid."""

    # Call
    subtensor = AsyncSubtensor("blabla-net")

    # Asserts
    assert subtensor.chain_endpoint == "ws://blabla-net"
    assert subtensor.network == "unknown"


def test__str__return(subtensor):
    """Simply tests the result if printing subtensor instance."""
    # Asserts
    assert (
        str(subtensor)
        == "Network: finney, Chain: wss://entrypoint-finney.opentensor.ai:443"
    )


@pytest.mark.asyncio
async def test_async_subtensor_magic_methods(mock_substrate):
    """Tests async magic methods of AsyncSubtensor class."""

    # Call
    subtensor = async_subtensor.AsyncSubtensor(network="local")
    async with subtensor:
        pass

    # Asserts
    mock_substrate.initialize.assert_called_once()
    mock_substrate.close.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [ConnectionRefusedError, async_subtensor.ssl.SSLError, TimeoutError],
)
@pytest.mark.asyncio
async def test_async_subtensor_aenter_connection_refused_error(
    subtensor, mocker, error
):
    """Tests __aenter__ method handling all errors."""
    # Preps
    fake_async_substrate = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface,
        initialize=mocker.AsyncMock(side_effect=error),
    )
    mocker.patch.object(
        async_subtensor, "AsyncSubstrateInterface", return_value=fake_async_substrate
    )
    # Call
    subtensor = async_subtensor.AsyncSubtensor(network="local")

    with pytest.raises(ConnectionError):
        async with subtensor:
            pass

    # Asserts
    fake_async_substrate.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_burned_register(mock_substrate, subtensor, fake_wallet, mocker):
    mock_substrate.submit_extrinsic.return_value = mocker.AsyncMock(
        is_success=mocker.AsyncMock(return_value=True)(),
    )
    mock_substrate.get_payment_info.return_value = {"partial_fee": 10}
    mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=NeuronInfo.get_null_neuron(),
    )
    mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=Balance(1),
    )

    success = await subtensor.burned_register(
        fake_wallet,
        netuid=1,
    )

    assert success is True

    subtensor.get_neuron_for_pubkey_and_subnet.assert_called_once_with(
        fake_wallet.hotkey.ss58_address,
        netuid=1,
        block_hash=mock_substrate.get_chain_head.return_value,
    )

    assert_submit_signed_extrinsic(
        mock_substrate,
        fake_wallet.coldkey,
        call_module="SubtensorModule",
        call_function="burned_register",
        call_params={
            "netuid": 1,
            "hotkey": fake_wallet.hotkey.ss58_address,
        },
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )


@pytest.mark.asyncio
async def test_burned_register_on_root(mock_substrate, subtensor, fake_wallet, mocker):
    mock_substrate.submit_extrinsic.return_value = mocker.AsyncMock(
        is_success=mocker.AsyncMock(return_value=True)(),
    )
    mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=Balance(1),
    )
    mocker.patch.object(
        subtensor,
        "is_hotkey_registered",
        return_value=False,
    )

    success = await subtensor.burned_register(
        fake_wallet,
        netuid=0,
    )

    assert success is True

    subtensor.is_hotkey_registered.assert_called_once_with(
        netuid=0,
        hotkey_ss58=fake_wallet.hotkey.ss58_address,
    )

    assert_submit_signed_extrinsic(
        mock_substrate,
        fake_wallet.coldkey,
        call_module="SubtensorModule",
        call_function="root_register",
        call_params={
            "hotkey": fake_wallet.hotkey.ss58_address,
        },
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )


@pytest.mark.asyncio
async def test_encode_params(subtensor, mocker):
    """Tests encode_params happy path."""
    # Preps
    subtensor.substrate.create_scale_object = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.create_scale_object
    )
    subtensor.substrate.create_scale_object.return_value.encode = mocker.Mock(
        return_value=b""
    )

    call_definition = {
        "params": [
            {"name": "coldkey", "type": "Vec<u8>"},
            {"name": "uid", "type": "u16"},
        ]
    }
    params = ["coldkey", "uid"]

    # Call
    decoded_params = await subtensor.encode_params(
        call_definition=call_definition, params=params
    )

    # Asserts
    subtensor.substrate.create_scale_object.call_args(
        mocker.call("coldkey"),
        mocker.call("Vec<u8>"),
        mocker.call("uid"),
        mocker.call("u16"),
    )
    assert decoded_params == "0x"


@pytest.mark.asyncio
async def test_encode_params_raises_error(subtensor, mocker):
    """Tests encode_params with raised error."""
    # Preps
    subtensor.substrate.create_scale_object = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.create_scale_object
    )
    subtensor.substrate.create_scale_object.return_value.encode = mocker.Mock(
        return_value=b""
    )

    call_definition = {
        "params": [
            {"name": "coldkey", "type": "Vec<u8>"},
        ]
    }
    params = {"undefined param": "some value"}

    # Call and assert
    with pytest.raises(ValueError):
        await subtensor.encode_params(call_definition=call_definition, params=params)

        subtensor.substrate.create_scale_object.return_value.encode.assert_not_called()


@pytest.mark.asyncio
async def test_get_current_block(subtensor):
    """Tests get_current_block method."""
    # Call
    result = await subtensor.get_current_block()

    # Asserts
    subtensor.substrate.get_block_number.assert_called_once()
    assert result == subtensor.substrate.get_block_number.return_value


@pytest.mark.asyncio
async def test_get_block_hash_without_block_id_aka_none(subtensor):
    """Tests get_block_hash method without passed block_id."""
    # Call
    result = await subtensor.get_block_hash()

    # Asserts
    assert result == subtensor.substrate.get_chain_head.return_value


@pytest.mark.asyncio
async def test_get_block_hash_with_block_id(subtensor):
    """Tests get_block_hash method with passed block_id."""
    # Call
    result = await subtensor.get_block_hash(block=1)

    # Asserts
    assert result == subtensor.substrate.get_block_hash.return_value


@pytest.mark.asyncio
async def test_is_hotkey_registered_any(subtensor, mocker):
    """Tests is_hotkey_registered_any method."""
    # Preps
    mocked_get_netuids_for_hotkey = mocker.AsyncMock(
        return_value=[1, 2], autospec=subtensor.get_netuids_for_hotkey
    )
    subtensor.get_netuids_for_hotkey = mocked_get_netuids_for_hotkey

    # Call
    result = await subtensor.is_hotkey_registered_any(
        hotkey_ss58="hotkey", block_hash="FAKE_HASH"
    )

    # Asserts
    assert result is (len(mocked_get_netuids_for_hotkey.return_value) > 0)


@pytest.mark.asyncio
async def test_get_subnet_burn_cost(subtensor, mocker):
    """Tests get_subnet_burn_cost method."""
    # Preps
    mocked_query_runtime_api = mocker.AsyncMock(
        autospec=subtensor.query_runtime_api, return_value=1000
    )
    subtensor.query_runtime_api = mocked_query_runtime_api
    fake_block_hash = None

    # Call
    result = await subtensor.get_subnet_burn_cost(block_hash=fake_block_hash)

    # Assert
    assert result == mocked_query_runtime_api.return_value
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="SubnetRegistrationRuntimeApi",
        method="get_network_registration_cost",
        params=[],
        block=None,
        block_hash=fake_block_hash,
        reuse_block=False,
    )


@pytest.mark.asyncio
async def test_get_total_subnets(subtensor, mocker):
    """Tests get_total_subnets method."""
    # Preps
    mocked_substrate_query = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.query
    )
    subtensor.substrate.query = mocked_substrate_query
    fake_block_hash = None

    # Call
    result = await subtensor.get_total_subnets(block_hash=fake_block_hash)

    # Assert
    assert result == mocked_substrate_query.return_value.value
    mocked_substrate_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="TotalNetworks",
        params=[],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )


@pytest.mark.parametrize(
    "records, response",
    [([(0, True), (1, False), (3, False), (3, True)], [0, 3]), ([], [])],
    ids=["with records", "empty-records"],
)
@pytest.mark.asyncio
async def test_get_subnets(subtensor, mocker, records, response):
    """Tests get_subnets method with any return."""
    # Preps
    fake_result = mocker.AsyncMock(autospec=list)
    fake_result.records = records
    fake_result.__aiter__.return_value = iter(records)

    mocked_substrate_query_map = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.query_map,
        return_value=fake_result,
    )

    subtensor.substrate.query_map = mocked_substrate_query_map
    fake_block_hash = None

    # Call
    result = await subtensor.get_subnets(block_hash=fake_block_hash)

    # Asserts
    mocked_substrate_query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result == response


@pytest.mark.parametrize(
    "hotkey_ss58_in_result",
    [True, False],
    ids=["hotkey-exists", "hotkey-doesnt-exist"],
)
@pytest.mark.asyncio
async def test_is_hotkey_delegate(subtensor, mocker, hotkey_ss58_in_result):
    """Tests is_hotkey_delegate method with any return."""
    # Preps
    fake_hotkey_ss58 = "hotkey_58"
    mocked_get_delegates = mocker.AsyncMock(
        return_value=[
            mocker.Mock(hotkey_ss58=fake_hotkey_ss58 if hotkey_ss58_in_result else "")
        ]
    )
    subtensor.get_delegates = mocked_get_delegates

    # Call
    result = await subtensor.is_hotkey_delegate(
        hotkey_ss58=fake_hotkey_ss58, block_hash=None, reuse_block=True
    )

    # Asserts
    assert result == hotkey_ss58_in_result
    mocked_get_delegates.assert_called_once_with(block_hash=None, reuse_block=True)


@pytest.mark.parametrize(
    "fake_result, response", [(None, []), ([mock.Mock()], [mock.Mock()])]
)
@pytest.mark.asyncio
async def test_get_delegates(subtensor, mocker, fake_result, response):
    """Tests get_delegates method."""
    # Preps
    mocked_query_runtime_api = mocker.AsyncMock(
        autospec=subtensor.query_runtime_api, return_value=fake_result
    )
    subtensor.query_runtime_api = mocked_query_runtime_api
    mocked_delegate_info_list_from_dicts = mocker.patch.object(
        async_subtensor.DelegateInfo,
        "list_from_dicts",
    )

    # Call
    result = await subtensor.get_delegates(block_hash=None, reuse_block=False)

    # Asserts
    if fake_result:
        assert result == mocked_delegate_info_list_from_dicts.return_value
        mocked_delegate_info_list_from_dicts.assert_called_once_with(fake_result)
    else:
        assert result == response

    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="DelegateInfoRuntimeApi",
        method="get_delegates",
        params=[],
        block=None,
        block_hash=None,
        reuse_block=False,
    )


@pytest.mark.parametrize(
    "fake_result, response", [(None, []), ([mock.Mock()], [mock.Mock()])]
)
@pytest.mark.asyncio
async def test_get_stake_info_for_coldkey(subtensor, mocker, fake_result, response):
    """Tests get_stake_info_for_coldkey method."""
    # Preps
    fake_coldkey_ss58 = "fake_coldkey_58"

    mocked_query_runtime_api = mocker.AsyncMock(
        autospec=subtensor.query_runtime_api, return_value=fake_result
    )
    subtensor.query_runtime_api = mocked_query_runtime_api

    mock_stake_info = mocker.Mock(
        spec=async_subtensor.StakeInfo, stake=Balance.from_rao(100)
    )
    mocked_stake_info_list_from_dicts = mocker.Mock(
        return_value=[mock_stake_info] if fake_result else []
    )
    mocker.patch.object(
        async_subtensor.StakeInfo,
        "list_from_dicts",
        mocked_stake_info_list_from_dicts,
    )

    # Call
    result = await subtensor.get_stake_info_for_coldkey(
        coldkey_ss58=fake_coldkey_ss58, block_hash=None, reuse_block=True
    )

    # Asserts
    if fake_result:
        mocked_stake_info_list_from_dicts.assert_called_once_with(fake_result)
        assert result == mocked_stake_info_list_from_dicts.return_value
    else:
        assert result == []

    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="StakeInfoRuntimeApi",
        method="get_stake_info_for_coldkey",
        params=[fake_coldkey_ss58],
        block=None,
        block_hash=None,
        reuse_block=True,
    )


@pytest.mark.asyncio
async def test_get_stake_for_coldkey_and_hotkey(subtensor, mocker):
    netuids = [1, 2, 3]
    block_hash = "valid_block_hash"
    stake_info_dict = {
        "netuid": 1,
        "hotkey": b"\x16:\xech\r\xde,g\x03R1\xb9\x88q\xe79\xb8\x88\x93\xae\xd2)?*\rp\xb2\xe62\xads\x1c",
        "coldkey": b"\x16:\xech\r\xde,g\x03R1\xb9\x88q\xe79\xb8\x88\x93\xae\xd2)?*\rp\xb2\xe62\xads\x1c",
        "stake": 1,
        "locked": False,
        "emission": 1,
        "drain": 1,
        "is_registered": True,
    }
    query_result = stake_info_dict
    expected_result = {
        netuid: StakeInfo.from_dict(stake_info_dict) for netuid in netuids
    }

    query_fetcher = mocker.AsyncMock(return_value=query_result)

    mocked_query_runtime_api = mocker.patch.object(
        subtensor, "query_runtime_api", side_effect=query_fetcher
    )
    mocked_determine_block_hash = mocker.patch.object(
        subtensor, "determine_block_hash", return_value=block_hash
    )
    mocked_get_chain_head = mocker.patch.object(
        subtensor.substrate, "get_chain_head", return_value=block_hash
    )
    mocked_get_subnets = mocker.patch.object(
        subtensor, "get_subnets", return_value=netuids
    )

    result = await subtensor.get_stake_for_coldkey_and_hotkey(
        hotkey_ss58="hotkey", coldkey_ss58="coldkey", block_hash=None, netuids=None
    )

    assert result == expected_result

    # validate that mocked functions were called with the right arguments
    mocked_query_runtime_api.assert_has_calls(
        [
            mock.call(
                "StakeInfoRuntimeApi",
                "get_stake_info_for_hotkey_coldkey_netuid",
                params=["hotkey", "coldkey", netuid],
                block_hash=block_hash,
            )
            for netuid in netuids
        ]
    )
    mocked_determine_block_hash.assert_called_once()
    mocked_get_chain_head.assert_not_called()
    mocked_get_subnets.assert_called_once_with(block_hash=block_hash)


@pytest.mark.asyncio
async def test_query_runtime_api(subtensor, mocker):
    """Tests query_runtime_api method."""
    # Preps
    fake_runtime_api = "DelegateInfoRuntimeApi"
    fake_method = "get_delegated"
    fake_params = [1, 2, 3]
    fake_block_hash = None
    reuse_block = False

    mocked_runtime_call = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.runtime_call
    )
    subtensor.substrate.runtime_call = mocked_runtime_call

    mocked_scalecodec = mocker.Mock(autospec=async_subtensor.scalecodec.ScaleBytes)
    mocker.patch.object(async_subtensor.scalecodec, "ScaleBytes", mocked_scalecodec)

    # Call
    result = await subtensor.query_runtime_api(
        runtime_api=fake_runtime_api,
        method=fake_method,
        params=fake_params,
        block_hash=fake_block_hash,
        reuse_block=reuse_block,
    )

    # Asserts
    mocked_runtime_call.assert_called_once_with(
        fake_runtime_api,
        fake_method,
        fake_params,
        fake_block_hash,
    )

    assert result == mocked_runtime_call.return_value.value


@pytest.mark.asyncio
async def test_get_balance(subtensor, mocker):
    """Tests get_balance method."""
    # Preps
    fake_address = "a1"
    fake_block = 123
    fake_block_hash = None
    reuse_block = True

    mocked_determine_block_hash = mocker.AsyncMock()
    mocker.patch.object(
        async_subtensor.AsyncSubtensor,
        "determine_block_hash",
        mocked_determine_block_hash,
    )

    mocked_balance = mocker.patch.object(async_subtensor, "Balance")

    # Call
    result = await subtensor.get_balance(
        fake_address, fake_block, fake_block_hash, reuse_block
    )

    mocked_determine_block_hash.assert_awaited_once_with(
        fake_block, fake_block_hash, reuse_block
    )
    subtensor.substrate.query.assert_awaited_once_with(
        module="System",
        storage_function="Account",
        params=[fake_address],
        block_hash=mocked_determine_block_hash.return_value,
        reuse_block_hash=reuse_block,
    )
    mocked_balance.assert_called_once_with(
        subtensor.substrate.query.return_value.__getitem__.return_value.__getitem__.return_value
    )
    assert result == mocked_balance.return_value


@pytest.mark.parametrize("balance", [100, 100.1])
@pytest.mark.asyncio
async def test_get_transfer_fee(subtensor, fake_wallet, mocker, balance):
    """Tests get_transfer_fee method."""
    # Preps
    fake_wallet.coldkeypub = "coldkeypub"
    fake_dest = "fake_dest"
    fake_value = Balance(balance)

    mocked_compose_call = mocker.AsyncMock()
    subtensor.substrate.compose_call = mocked_compose_call

    mocked_get_payment_info = mocker.AsyncMock(return_value={"partial_fee": 100})
    subtensor.substrate.get_payment_info = mocked_get_payment_info

    # Call
    result = await subtensor.get_transfer_fee(
        wallet=fake_wallet, dest=fake_dest, value=fake_value
    )

    # Assertions
    mocked_compose_call.assert_awaited_once()
    mocked_compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_keep_alive",
        call_params={
            "dest": fake_dest,
            "value": fake_value.rao,
        },
    )

    assert isinstance(result, async_subtensor.Balance)
    mocked_get_payment_info.assert_awaited_once()
    mocked_get_payment_info.assert_called_once_with(
        call=mocked_compose_call.return_value, keypair="coldkeypub"
    )


@pytest.mark.asyncio
async def test_get_transfer_with_exception(subtensor, mocker):
    """Tests get_transfer_fee method handle Exception properly."""
    # Preps
    fake_value = 123

    mocked_compose_call = mocker.AsyncMock()
    subtensor.substrate.compose_call = mocked_compose_call
    subtensor.substrate.get_payment_info.side_effect = Exception

    # Call
    result = await subtensor.get_transfer_fee(
        wallet=mocker.Mock(), dest=mocker.Mock(), value=fake_value
    )

    # Assertions
    assert result == async_subtensor.Balance.from_rao(int(2e7))


@pytest.mark.asyncio
async def test_get_netuids_for_hotkey_with_records(subtensor, mocker):
    """Tests get_netuids_for_hotkey method handle records properly."""
    # Preps
    records = []
    expected_response = []
    fake_result = mocker.AsyncMock(autospec=list)
    fake_result.records = records
    fake_result.__aiter__.return_value = iter(records)

    mocked_substrate_query_map = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.query_map,
        return_value=fake_result,
    )

    subtensor.substrate.query_map = mocked_substrate_query_map
    fake_hotkey_ss58 = "hotkey_58"
    fake_block_hash = None

    # Call
    result = await subtensor.get_netuids_for_hotkey(
        hotkey_ss58=fake_hotkey_ss58, block_hash=fake_block_hash, reuse_block=True
    )

    # Assertions
    mocked_substrate_query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="IsNetworkMember",
        params=[fake_hotkey_ss58],
        block_hash=fake_block_hash,
        reuse_block_hash=True,
    )
    assert result == expected_response


@pytest.mark.asyncio
async def test_get_netuids_for_hotkey_without_records(subtensor, mocker):
    """Tests get_netuids_for_hotkey method handle empty records properly."""
    # Preps
    records = []
    expected_response = []
    fake_result = mocker.AsyncMock(autospec=list)
    fake_result.records = records
    fake_result.__aiter__.return_value = iter(records)

    mocked_substrate_query_map = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.query_map,
        return_value=fake_result,
    )

    subtensor.substrate.query_map = mocked_substrate_query_map
    fake_hotkey_ss58 = "hotkey_58"
    fake_block_hash = None

    # Call
    result = await subtensor.get_netuids_for_hotkey(
        hotkey_ss58=fake_hotkey_ss58, block_hash=fake_block_hash, reuse_block=True
    )

    # Assertions
    mocked_substrate_query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="IsNetworkMember",
        params=[fake_hotkey_ss58],
        block_hash=fake_block_hash,
        reuse_block_hash=True,
    )
    assert result == expected_response


@pytest.mark.asyncio
async def test_subnet_exists(subtensor, mocker):
    """Tests subnet_exists method ."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_reuse_block_hash = False

    mocked_substrate_query = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.query
    )
    subtensor.substrate.query = mocked_substrate_query

    # Call
    result = await subtensor.subnet_exists(
        netuid=fake_netuid,
        block_hash=fake_block_hash,
        reuse_block=fake_reuse_block_hash,
    )

    # Asserts
    mocked_substrate_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        params=[fake_netuid],
        block_hash=fake_block_hash,
        reuse_block_hash=fake_reuse_block_hash,
    )
    assert result == mocked_substrate_query.return_value.value


@pytest.mark.asyncio
async def test_get_hyperparameter_happy_path(subtensor, mocker):
    """Tests get_hyperparameter method with happy path."""
    # Preps
    fake_param_name = "param_name"
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_reuse_block_hash = False

    # kind of fake subnet exists
    mocked_subtensor_subnet_exists = mocker.AsyncMock(return_value=True)
    subtensor.subnet_exists = mocked_subtensor_subnet_exists

    mocked_substrate_query = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.query
    )
    subtensor.substrate.query = mocked_substrate_query

    # Call
    result = await subtensor.get_hyperparameter(
        param_name=fake_param_name,
        netuid=fake_netuid,
        block_hash=fake_block_hash,
        reuse_block=fake_reuse_block_hash,
    )

    # Assertions
    mocked_subtensor_subnet_exists.assert_called_once()
    mocked_substrate_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function=fake_param_name,
        params=[fake_netuid],
        block_hash=fake_block_hash,
        reuse_block_hash=fake_reuse_block_hash,
    )
    assert result == mocked_substrate_query.return_value.value


@pytest.mark.asyncio
async def test_get_hyperparameter_if_subnet_does_not_exist(subtensor, mocker):
    """Tests get_hyperparameter method if subnet does not exist."""
    # Preps
    # kind of fake subnet doesn't exist
    mocked_subtensor_subnet_exists = mocker.AsyncMock(return_value=False)
    subtensor.subnet_exists = mocked_subtensor_subnet_exists

    mocked_substrate_query = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.query
    )
    subtensor.substrate.query = mocked_substrate_query

    # Call
    result = await subtensor.get_hyperparameter(mocker.Mock(), mocker.Mock())

    # Assertions
    mocked_subtensor_subnet_exists.assert_called_once()
    mocked_substrate_query.assert_not_called()
    assert result is None


@pytest.mark.parametrize(
    "all_netuids, filter_for_netuids, response",
    [([1, 2], [3, 4], []), ([1, 2], [1, 3], [1]), ([1, 2], None, [1, 2])],
    ids=[
        "all arguments -> no comparison",
        "all arguments -> is comparison",
        "not filter_for_netuids",
    ],
)
@pytest.mark.asyncio
async def test_filter_netuids_by_registered_hotkeys(
    subtensor, mocker, all_netuids, filter_for_netuids, response
):
    """Tests filter_netuids_by_registered_hotkeys method."""
    # Preps
    fake_wallet_1 = mocker.Mock(spec_set=Wallet)
    fake_wallet_1.hotkey.ss58_address = "ss58_address_1"
    fake_wallet_2 = mocker.Mock(spec_set=Wallet)
    fake_wallet_2.hotkey.ss58_address = "ss58_address_2"

    fake_all_netuids = all_netuids
    fake_filter_for_netuids = filter_for_netuids
    fake_all_hotkeys = [fake_wallet_1, fake_wallet_2]
    fake_block_hash = "fake_block_hash"
    fake_reuse_block = False

    mocked_get_netuids_for_hotkey = mocker.AsyncMock(
        # returned subnets list
        return_value=[1, 2]
    )
    subtensor.get_netuids_for_hotkey = mocked_get_netuids_for_hotkey

    # Call

    result = await subtensor.filter_netuids_by_registered_hotkeys(
        all_netuids=fake_all_netuids,
        filter_for_netuids=fake_filter_for_netuids,
        all_hotkeys=fake_all_hotkeys,
        block_hash=fake_block_hash,
        reuse_block=fake_reuse_block,
    )

    # Asserts
    mocked_get_netuids_for_hotkey.call_count = len(fake_all_netuids)
    assert mocked_get_netuids_for_hotkey.mock_calls == [
        mocker.call(
            w.hotkey.ss58_address,
            block_hash=fake_block_hash,
            reuse_block=fake_reuse_block,
        )
        for w in fake_all_hotkeys
    ]
    assert result == response


@pytest.mark.asyncio
async def test_get_existential_deposit_happy_path(subtensor, mocker):
    """Tests get_existential_deposit method."""
    # Preps
    fake_block_hash = "block_hash"
    fake_reuse_block_hash = False

    mocked_substrate_get_constant = mocker.AsyncMock(return_value=mocker.Mock(value=1))
    subtensor.substrate.get_constant = mocked_substrate_get_constant

    spy_balance_from_rao = mocker.spy(async_subtensor.Balance, "from_rao")

    # Call
    result = await subtensor.get_existential_deposit(
        block_hash=fake_block_hash, reuse_block=fake_reuse_block_hash
    )

    # Asserts
    mocked_substrate_get_constant.assert_awaited_once()
    mocked_substrate_get_constant.assert_called_once_with(
        module_name="Balances",
        constant_name="ExistentialDeposit",
        block_hash=fake_block_hash,
        reuse_block_hash=fake_reuse_block_hash,
    )
    spy_balance_from_rao.assert_called_once_with(
        mocked_substrate_get_constant.return_value.value
    )
    assert result == async_subtensor.Balance(
        mocked_substrate_get_constant.return_value.value
    )


@pytest.mark.asyncio
async def test_get_existential_deposit_raise_exception(subtensor, mocker):
    """Tests get_existential_deposit method raise Exception."""
    # Preps
    fake_block_hash = "block_hash"
    fake_reuse_block_hash = False

    mocked_substrate_get_constant = mocker.AsyncMock(return_value=None)
    subtensor.substrate.get_constant = mocked_substrate_get_constant

    spy_balance_from_rao = mocker.spy(async_subtensor.Balance, "from_rao")

    # Call
    with pytest.raises(Exception):
        await subtensor.get_existential_deposit(
            block_hash=fake_block_hash, reuse_block=fake_reuse_block_hash
        )

    # Asserts
    mocked_substrate_get_constant.assert_awaited_once()
    mocked_substrate_get_constant.assert_called_once_with(
        module_name="Balances",
        constant_name="ExistentialDeposit",
        block_hash=fake_block_hash,
        reuse_block_hash=fake_reuse_block_hash,
    )
    spy_balance_from_rao.assert_not_called()


@pytest.mark.asyncio
async def test_neurons(subtensor, mocker):
    """Tests neurons method."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_reuse_block_hash = False

    mocked_query_runtime_api = mocker.patch.object(
        subtensor, "query_runtime_api", return_value="NOT NONE"
    )
    mocked_neuron_info_list_from_dicts = mocker.patch.object(
        async_subtensor.NeuronInfo, "list_from_dicts"
    )
    # Call
    result = await subtensor.neurons(
        netuid=fake_netuid,
        block_hash=fake_block_hash,
        reuse_block=fake_reuse_block_hash,
    )

    # Asserts
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="NeuronInfoRuntimeApi",
        method="get_neurons",
        params=[fake_netuid],
        block=None,
        block_hash=fake_block_hash,
        reuse_block=fake_reuse_block_hash,
    )
    assert result == mocked_neuron_info_list_from_dicts.return_value


@pytest.mark.parametrize(
    "fake_result, response",
    [(None, []), (mock.Mock(), mock.Mock())],
    ids=["none", "with data"],
)
@pytest.mark.asyncio
async def test_neurons_lite(subtensor, mocker, fake_result, response):
    """Tests neurons_lite method."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_reuse_block_hash = False

    mocked_query_runtime_api = mocker.AsyncMock(return_value=fake_result)
    subtensor.query_runtime_api = mocked_query_runtime_api

    mocked_neuron_info_lite_list_from_dicts = mocker.patch.object(
        async_subtensor.NeuronInfoLite, "list_from_dicts"
    )

    # Call
    result = await subtensor.neurons_lite(
        netuid=fake_netuid,
        block_hash=fake_block_hash,
        reuse_block=fake_reuse_block_hash,
    )

    # Assertions
    mocked_query_runtime_api.assert_awaited_once()
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="NeuronInfoRuntimeApi",
        method="get_neurons_lite",
        params=[fake_netuid],
        block=None,
        block_hash=fake_block_hash,
        reuse_block=fake_reuse_block_hash,
    )
    if fake_result:
        mocked_neuron_info_lite_list_from_dicts.assert_called_once_with(fake_result)
        assert result == mocked_neuron_info_lite_list_from_dicts.return_value
    else:
        mocked_neuron_info_lite_list_from_dicts.assert_not_called()
        assert result == []


@pytest.mark.asyncio
async def test_get_neuron_for_pubkey_and_subnet_success(subtensor, mocker):
    """Tests successful retrieval of neuron information."""
    # Preps
    fake_hotkey = "fake_ss58_address"
    fake_netuid = 1
    fake_uid = mocker.Mock(value=123)
    fake_result = b"fake_neuron_data"

    mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=fake_uid,
    )
    mocker.patch.object(
        subtensor.substrate,
        "runtime_call",
        return_value=mocker.Mock(value=fake_result),
    )
    mocked_neuron_info = mocker.patch.object(
        async_subtensor.NeuronInfo, "from_dict", return_value="fake_neuron_info"
    )

    # Call
    result = await subtensor.get_neuron_for_pubkey_and_subnet(
        hotkey_ss58=fake_hotkey, netuid=fake_netuid
    )

    # Asserts
    subtensor.substrate.query.assert_awaited_once()
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[fake_netuid, fake_hotkey],
        block_hash=None,
        reuse_block_hash=False,
    )
    subtensor.substrate.runtime_call.assert_awaited_once()
    subtensor.substrate.runtime_call.assert_called_once_with(
        "NeuronInfoRuntimeApi",
        "get_neuron",
        [fake_netuid, fake_uid.value],
        None,
    )
    mocked_neuron_info.assert_called_once_with(fake_result)
    assert result == "fake_neuron_info"


@pytest.mark.asyncio
async def test_get_neuron_for_pubkey_and_subnet_uid_not_found(subtensor, mocker):
    """Tests the case where UID is not found."""
    # Preps
    fake_hotkey = "fake_ss58_address"
    fake_netuid = 1

    mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=None,
    )
    mocked_get_null_neuron = mocker.patch.object(
        async_subtensor.NeuronInfo, "get_null_neuron", return_value="null_neuron"
    )

    # Call
    result = await subtensor.get_neuron_for_pubkey_and_subnet(
        hotkey_ss58=fake_hotkey, netuid=fake_netuid
    )

    # Asserts
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[fake_netuid, fake_hotkey],
        block_hash=None,
        reuse_block_hash=False,
    )
    mocked_get_null_neuron.assert_called_once()
    assert result == "null_neuron"


@pytest.mark.asyncio
async def test_get_neuron_for_pubkey_and_subnet_rpc_result_empty(subtensor, mocker):
    """Tests the case where RPC result is empty."""
    # Preps
    fake_hotkey = "fake_ss58_address"
    fake_netuid = 1
    fake_uid = 123

    mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=mocker.Mock(value=fake_uid),
    )
    mocker.patch.object(
        subtensor.substrate,
        "runtime_call",
        return_value=mocker.Mock(value=None),
    )
    mocked_get_null_neuron = mocker.patch.object(
        async_subtensor.NeuronInfo, "get_null_neuron", return_value="null_neuron"
    )

    # Call
    result = await subtensor.get_neuron_for_pubkey_and_subnet(
        hotkey_ss58=fake_hotkey, netuid=fake_netuid
    )

    # Asserts
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[fake_netuid, fake_hotkey],
        block_hash=None,
        reuse_block_hash=False,
    )
    subtensor.substrate.runtime_call.assert_called_once_with(
        "NeuronInfoRuntimeApi",
        "get_neuron",
        [fake_netuid, fake_uid],
        None,
    )
    mocked_get_null_neuron.assert_called_once()
    assert result == "null_neuron"


@pytest.mark.asyncio
async def test_neuron_for_uid_happy_path(subtensor, mocker):
    """Tests neuron_for_uid method with happy path."""
    # Preps
    fake_uid = 1
    fake_netuid = 2
    fake_block_hash = "block_hash"

    mocked_null_neuron = mocker.patch.object(
        async_subtensor.NeuronInfo,
        "get_null_neuron",
    )
    mocked_neuron_info_from_dict = mocker.patch.object(
        async_subtensor.NeuronInfo,
        "from_dict",
    )

    # Call
    result = await subtensor.neuron_for_uid(
        uid=fake_uid, netuid=fake_netuid, block_hash=fake_block_hash
    )

    # Asserts
    mocked_null_neuron.assert_not_called()
    mocked_neuron_info_from_dict.assert_called_once_with(
        subtensor.substrate.runtime_call.return_value.value
    )
    assert result == mocked_neuron_info_from_dict.return_value


@pytest.mark.asyncio
async def test_neuron_for_uid_with_none_uid(subtensor, mocker):
    """Tests neuron_for_uid method when uid is None."""
    # Preps
    fake_uid = None
    fake_netuid = 1
    fake_block_hash = "block_hash"

    mocked_null_neuron = mocker.patch.object(
        async_subtensor.NeuronInfo,
        "get_null_neuron",
    )

    # Call
    result = await subtensor.neuron_for_uid(
        uid=fake_uid, netuid=fake_netuid, block_hash=fake_block_hash
    )

    # Asserts
    mocked_null_neuron.assert_called_once()
    assert result == mocked_null_neuron.return_value


@pytest.mark.asyncio
async def test_neuron_for_uid(subtensor, mocker):
    """Tests neuron_for_uid method."""
    # Preps
    fake_uid = 1
    fake_netuid = 2
    fake_block_hash = "block_hash"

    mocked_null_neuron = mocker.patch.object(
        async_subtensor.NeuronInfo,
        "get_null_neuron",
    )

    # no result in response
    mocked_substrate_runtime_call = mocker.AsyncMock(
        return_value=mocker.Mock(
            value=None,
        ),
    )
    subtensor.substrate.runtime_call = mocked_substrate_runtime_call

    mocked_neuron_info_from_dict = mocker.patch.object(
        async_subtensor.NeuronInfo,
        "from_dict",
    )

    # Call
    result = await subtensor.neuron_for_uid(
        uid=fake_uid, netuid=fake_netuid, block_hash=fake_block_hash
    )

    # Asserts
    mocked_null_neuron.assert_called_once()
    mocked_neuron_info_from_dict.assert_not_called()
    assert result == mocked_null_neuron.return_value


@pytest.mark.asyncio
async def test_get_delegated_no_block_hash_no_reuse(subtensor, mocker):
    """Tests get_delegated method with no block_hash and reuse_block=False."""
    # Preps
    fake_coldkey_ss58 = "fake_ss58_address"

    mocked_delegated_list_from_dicts = mocker.patch.object(
        async_subtensor.DelegatedInfo,
        "list_from_dicts",
    )

    # Call
    result = await subtensor.get_delegated(coldkey_ss58=fake_coldkey_ss58)

    # Asserts
    subtensor.substrate.runtime_call.assert_called_once_with(
        "DelegateInfoRuntimeApi",
        "get_delegated",
        [fake_coldkey_ss58],
        None,
    )
    mocked_delegated_list_from_dicts.assert_called_once_with(
        subtensor.substrate.runtime_call.return_value.value
    )
    assert result == mocked_delegated_list_from_dicts.return_value


@pytest.mark.asyncio
async def test_get_delegated_with_block_hash(subtensor, mocker):
    """Tests get_delegated method with specified block_hash."""
    # Preps
    fake_coldkey_ss58 = "fake_ss58_address"
    fake_block_hash = "fake_block_hash"

    mocked_delegated_list_from_dicts = mocker.patch.object(
        async_subtensor.DelegatedInfo,
        "list_from_dicts",
    )

    # Call
    result = await subtensor.get_delegated(
        coldkey_ss58=fake_coldkey_ss58, block_hash=fake_block_hash
    )

    # Asserts
    subtensor.substrate.runtime_call.assert_called_once_with(
        "DelegateInfoRuntimeApi",
        "get_delegated",
        [fake_coldkey_ss58],
        fake_block_hash,
    )
    mocked_delegated_list_from_dicts.assert_called_once_with(
        subtensor.substrate.runtime_call.return_value.value
    )
    assert result == mocked_delegated_list_from_dicts.return_value


@pytest.mark.asyncio
async def test_get_delegated_with_reuse_block(subtensor, mocker):
    """Tests get_delegated method with reuse_block=True."""
    # Preps
    fake_coldkey_ss58 = "fake_ss58_address"
    reuse_block = True

    mocked_delegated_list_from_dicts = mocker.patch.object(
        async_subtensor.DelegatedInfo,
        "list_from_dicts",
    )

    # Call
    result = await subtensor.get_delegated(
        coldkey_ss58=fake_coldkey_ss58, reuse_block=reuse_block
    )

    # Asserts
    subtensor.substrate.runtime_call.assert_called_once_with(
        "DelegateInfoRuntimeApi",
        "get_delegated",
        [fake_coldkey_ss58],
        subtensor.substrate.last_block_hash,
    )
    mocked_delegated_list_from_dicts.assert_called_once_with(
        subtensor.substrate.runtime_call.return_value.value
    )
    assert result == mocked_delegated_list_from_dicts.return_value


@pytest.mark.asyncio
async def test_get_delegated_with_empty_result(subtensor, mocker):
    """Tests get_delegated method when RPC request returns an empty result."""
    # Preps
    fake_coldkey_ss58 = "fake_ss58_address"

    mocked_runtime_call = mocker.AsyncMock(
        return_value=mocker.Mock(
            value=None,
        ),
    )
    subtensor.substrate.runtime_call = mocked_runtime_call

    # Call
    result = await subtensor.get_delegated(coldkey_ss58=fake_coldkey_ss58)

    # Asserts
    mocked_runtime_call.assert_called_once_with(
        "DelegateInfoRuntimeApi",
        "get_delegated",
        [fake_coldkey_ss58],
        None,
    )
    assert result == []


@pytest.mark.asyncio
async def test_query_identity_successful(subtensor, mocker):
    """Tests query_identity method with successful identity query."""
    # Preps
    fake_coldkey_ss58 = "test_key"
    fake_block_hash = "block_hash"
    fake_identity_info = {
        "additional": "Additional",
        "description": "Description",
        "discord": "",
        "github_repo": "https://github.com/opentensor/bittensor",
        "image": "",
        "name": "Name",
        "url": "https://www.example.com",
    }

    mocked_query = mocker.AsyncMock(return_value=fake_identity_info)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.query_identity(
        coldkey_ss58=fake_coldkey_ss58, block_hash=fake_block_hash
    )

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="IdentitiesV2",
        params=[fake_coldkey_ss58],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result == ChainIdentity(
        additional="Additional",
        description="Description",
        discord="",
        github="https://github.com/opentensor/bittensor",
        image="",
        name="Name",
        url="https://www.example.com",
    )


@pytest.mark.asyncio
async def test_query_identity_no_info(subtensor, mocker):
    """Tests query_identity method when no identity info is returned."""
    # Preps
    fake_coldkey_ss58 = "test_key"

    mocked_query = mocker.AsyncMock(return_value=None)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.query_identity(coldkey_ss58=fake_coldkey_ss58)

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="IdentitiesV2",
        params=[fake_coldkey_ss58],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result is None


@pytest.mark.asyncio
async def test_query_identity_type_error(subtensor, mocker):
    """Tests query_identity method when a TypeError occurs during decoding."""
    # Preps
    fake_coldkey_ss58 = "test_key"
    fake_identity_info = {"info": {"rank": (b"\xff\xfe",)}}

    mocked_query = mocker.AsyncMock(return_value=fake_identity_info)
    subtensor.substrate.query = mocked_query

    mocker.patch.object(
        async_subtensor,
        "decode_hex_identity_dict",
        side_effect=TypeError,
    )

    # Call
    result = await subtensor.query_identity(coldkey_ss58=fake_coldkey_ss58)

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="IdentitiesV2",
        params=[fake_coldkey_ss58],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result is None


@pytest.mark.asyncio
async def test_weights_successful(subtensor, mocker):
    """Tests weights method with successful weight distribution retrieval."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_weights = [
        (0, mocker.AsyncMock(value=[(1, 10), (2, 20)])),
        (1, mocker.AsyncMock(value=[(0, 15), (2, 25)])),
    ]

    async def mock_query_map(**_):
        for uid, w in fake_weights:
            yield uid, w

    mocker.patch.object(subtensor.substrate, "query_map", side_effect=mock_query_map)

    # Call
    result = await subtensor.weights(netuid=fake_netuid, block_hash=fake_block_hash)

    # Asserts
    subtensor.substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Weights",
        params=[fake_netuid],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result == [(0, [(1, 10), (2, 20)]), (1, [(0, 15), (2, 25)])]


@pytest.mark.asyncio
async def test_bonds(subtensor, mocker):
    """Tests bonds method with successful bond distribution retrieval."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_bonds = [
        (0, mocker.Mock(value=[(1, 100), (2, 200)])),
        (1, mocker.Mock(value=[(0, 150), (2, 250)])),
    ]

    async def mock_query_map(**_):
        for uid, b in fake_bonds:
            yield uid, b

    mocker.patch.object(subtensor.substrate, "query_map", side_effect=mock_query_map)

    # Call
    result = await subtensor.bonds(netuid=fake_netuid, block_hash=fake_block_hash)

    # Asserts
    subtensor.substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Bonds",
        params=[fake_netuid],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result == [(0, [(1, 100), (2, 200)]), (1, [(0, 150), (2, 250)])]


@pytest.mark.asyncio
async def test_does_hotkey_exist_true(subtensor, mocker):
    """Tests does_hotkey_exist method when the hotkey exists and is valid."""
    # Preps
    fake_hotkey_ss58 = "valid_hotkey"
    fake_block_hash = "block_hash"
    fake_query_result = ["decoded_account_id"]

    mocked_query = mocker.AsyncMock(value=fake_query_result)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.does_hotkey_exist(
        hotkey_ss58=fake_hotkey_ss58, block_hash=fake_block_hash
    )

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result is True


@pytest.mark.asyncio
async def test_does_hotkey_exist_false_for_specific_account(subtensor, mocker):
    """Tests does_hotkey_exist method when the hotkey exists but matches the specific account ID to ignore."""
    # Preps
    fake_hotkey_ss58 = "fake_hotkey"
    fake_query_result = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"

    mocked_query = mocker.patch.object(
        subtensor.substrate, "query", return_value=fake_query_result
    )

    # Call
    result = await subtensor.does_hotkey_exist(hotkey_ss58=fake_hotkey_ss58)

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result is False


@pytest.mark.asyncio
async def test_get_hotkey_owner_successful(subtensor, mocker):
    """Tests get_hotkey_owner method when the hotkey exists and has an owner."""
    # Preps
    fake_hotkey_ss58 = "valid_hotkey"
    fake_block_hash = "block_hash"

    mocked_query = mocker.AsyncMock(return_value="decoded_owner_account_id")
    subtensor.substrate.query = mocked_query

    mocked_does_hotkey_exist = mocker.AsyncMock(return_value=True)
    subtensor.does_hotkey_exist = mocked_does_hotkey_exist

    # Call
    result = await subtensor.get_hotkey_owner(
        hotkey_ss58=fake_hotkey_ss58, block_hash=fake_block_hash
    )

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    mocked_does_hotkey_exist.assert_awaited_once_with(
        fake_hotkey_ss58, block_hash=fake_block_hash
    )
    assert result == "decoded_owner_account_id"


@pytest.mark.asyncio
async def test_get_hotkey_owner_non_existent_hotkey(subtensor, mocker):
    """Tests get_hotkey_owner method when the hotkey does not exist in the query result."""
    # Preps
    fake_hotkey_ss58 = "non_existent_hotkey"
    fake_block_hash = "block_hash"

    mocked_query = mocker.AsyncMock(return_value=None)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.get_hotkey_owner(
        hotkey_ss58=fake_hotkey_ss58, block_hash=fake_block_hash
    )

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result is None


@pytest.mark.asyncio
async def test_sign_and_send_extrinsic_success_finalization(
    subtensor, fake_wallet, mocker
):
    """Tests sign_and_send_extrinsic when the extrinsic is successfully finalized."""
    # Preps
    fake_call = mocker.Mock()
    fake_extrinsic = mocker.Mock()
    fake_response = mocker.Mock()

    mocked_create_signed_extrinsic = mocker.AsyncMock(return_value=fake_extrinsic)
    subtensor.substrate.create_signed_extrinsic = mocked_create_signed_extrinsic

    mocked_submit_extrinsic = mocker.AsyncMock(return_value=fake_response)
    subtensor.substrate.submit_extrinsic = mocked_submit_extrinsic

    fake_response.process_events = mocker.AsyncMock()

    async def fake_is_success():
        return True

    fake_response.is_success = fake_is_success()

    # Call
    result = await subtensor.sign_and_send_extrinsic(
        call=fake_call,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=fake_call, keypair=fake_wallet.coldkey
    )
    mocked_submit_extrinsic.assert_called_once_with(
        fake_extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result == (True, "")


@pytest.mark.asyncio
async def test_sign_and_send_extrinsic_error_finalization(
    subtensor, fake_wallet, mocker
):
    """Tests sign_and_send_extrinsic when the extrinsic is error finalized."""
    # Preps
    fake_call = mocker.Mock()
    fake_extrinsic = mocker.Mock()
    fake_response = mocker.Mock()

    mocked_create_signed_extrinsic = mocker.AsyncMock(return_value=fake_extrinsic)
    subtensor.substrate.create_signed_extrinsic = mocked_create_signed_extrinsic

    mocked_submit_extrinsic = mocker.AsyncMock(return_value=fake_response)
    subtensor.substrate.submit_extrinsic = mocked_submit_extrinsic

    fake_response.process_events = mocker.AsyncMock()

    async def fake_is_success():
        return False

    fake_response.is_success = fake_is_success()

    async def fake_error_message():
        return {"some error": "message"}

    fake_response.error_message = fake_error_message()

    mocked_format_error_message = mocker.Mock()
    async_subtensor.format_error_message = mocked_format_error_message

    # Call
    result = await subtensor.sign_and_send_extrinsic(
        call=fake_call,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=fake_call, keypair=fake_wallet.coldkey
    )
    mocked_submit_extrinsic.assert_called_once_with(
        fake_extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result == (False, mocked_format_error_message.return_value)


@pytest.mark.asyncio
async def test_sign_and_send_extrinsic_success_without_inclusion_finalization(
    subtensor, fake_wallet, mocker
):
    """Tests sign_and_send_extrinsic when extrinsic is submitted without waiting for inclusion or finalization."""
    # Preps
    fake_call = mocker.Mock()
    fake_extrinsic = mocker.Mock()

    mocked_create_signed_extrinsic = mocker.AsyncMock(return_value=fake_extrinsic)
    subtensor.substrate.create_signed_extrinsic = mocked_create_signed_extrinsic

    mocked_submit_extrinsic = mocker.AsyncMock()
    subtensor.substrate.submit_extrinsic = mocked_submit_extrinsic

    # Call
    result = await subtensor.sign_and_send_extrinsic(
        call=fake_call,
        wallet=fake_wallet,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )

    # Asserts
    mocked_create_signed_extrinsic.assert_awaited_once()
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=fake_call, keypair=fake_wallet.coldkey
    )
    mocked_submit_extrinsic.assert_awaited_once()
    mocked_submit_extrinsic.assert_called_once_with(
        fake_extrinsic,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )
    assert result == (True, "Not waiting for finalization or inclusion.")


@pytest.mark.asyncio
async def test_sign_and_send_extrinsic_substrate_request_exception(
    subtensor, fake_wallet, mocker
):
    """Tests sign_and_send_extrinsic when SubstrateRequestException is raised."""
    # Preps
    fake_call = mocker.Mock()
    fake_extrinsic = mocker.Mock()
    fake_exception = async_subtensor.SubstrateRequestException("Test Exception")

    mocked_create_signed_extrinsic = mocker.AsyncMock(return_value=fake_extrinsic)
    subtensor.substrate.create_signed_extrinsic = mocked_create_signed_extrinsic

    mocked_submit_extrinsic = mocker.AsyncMock(side_effect=fake_exception)
    subtensor.substrate.submit_extrinsic = mocked_submit_extrinsic

    mocker.patch.object(
        async_subtensor,
        "format_error_message",
        return_value=str(fake_exception),
    )

    # Call
    result = await subtensor.sign_and_send_extrinsic(
        call=fake_call,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    assert result == (False, str(fake_exception))


@pytest.mark.asyncio
async def test_sign_and_send_extrinsic_raises_error(
    mock_substrate, subtensor, fake_wallet, mocker
):
    mock_substrate.submit_extrinsic.return_value = mocker.AsyncMock(
        error_message=mocker.AsyncMock(
            return_value={
                "name": "Exception",
            },
        )(),
        is_success=mocker.AsyncMock(return_value=False)(),
    )

    with pytest.raises(
        async_subtensor.SubstrateRequestException,
        match="{'name': 'Exception'}",
    ):
        await subtensor.sign_and_send_extrinsic(
            call=mocker.Mock(),
            wallet=fake_wallet,
            raise_error=True,
        )


@pytest.mark.asyncio
async def test_get_children_success(subtensor, mocker):
    """Tests get_children when children are successfully retrieved and formatted."""
    # Preps
    fake_hotkey = "valid_hotkey"
    fake_netuid = 1
    fake_children = mocker.Mock(
        value=[
            (1000, ["child_key_1"]),
            (2000, ["child_key_2"]),
        ]
    )

    mocked_query = mocker.AsyncMock(return_value=fake_children)
    subtensor.substrate.query = mocked_query

    mocked_decode_account_id = mocker.Mock(
        side_effect=["decoded_child_key_1", "decoded_child_key_2"]
    )
    mocker.patch.object(async_subtensor, "decode_account_id", mocked_decode_account_id)

    expected_formatted_children = [
        (u64_normalized_float(1000), "decoded_child_key_1"),
        (u64_normalized_float(2000), "decoded_child_key_2"),
    ]

    # Call
    result = await subtensor.get_children(hotkey=fake_hotkey, netuid=fake_netuid)

    # Asserts
    mocked_query.assert_called_once_with(
        block_hash=None,
        module="SubtensorModule",
        storage_function="ChildKeys",
        params=[fake_hotkey, fake_netuid],
        reuse_block_hash=False,
    )
    mocked_decode_account_id.assert_has_calls(
        [mocker.call("child_key_1"), mocker.call("child_key_2")]
    )
    assert result == (True, expected_formatted_children, "")


@pytest.mark.asyncio
async def test_get_children_no_children(subtensor, mocker):
    """Tests get_children when there are no children to retrieve."""
    # Preps
    fake_hotkey = "valid_hotkey"
    fake_netuid = 1
    fake_children = []

    mocked_query = mocker.AsyncMock(return_value=fake_children)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.get_children(hotkey=fake_hotkey, netuid=fake_netuid)

    # Asserts
    mocked_query.assert_called_once_with(
        block_hash=None,
        module="SubtensorModule",
        storage_function="ChildKeys",
        params=[fake_hotkey, fake_netuid],
        reuse_block_hash=False,
    )
    assert result == (True, [], "")


@pytest.mark.asyncio
async def test_get_children_substrate_request_exception(subtensor, mocker):
    """Tests get_children when SubstrateRequestException is raised."""
    # Preps
    fake_hotkey = "valid_hotkey"
    fake_netuid = 1
    fake_exception = async_subtensor.SubstrateRequestException("Test Exception")

    mocked_query = mocker.AsyncMock(side_effect=fake_exception)
    subtensor.substrate.query = mocked_query

    mocked_format_error_message = mocker.Mock(return_value="Formatted error message")
    mocker.patch.object(
        async_subtensor, "format_error_message", mocked_format_error_message
    )

    # Call
    result = await subtensor.get_children(hotkey=fake_hotkey, netuid=fake_netuid)

    # Asserts
    mocked_query.assert_called_once_with(
        block_hash=None,
        module="SubtensorModule",
        storage_function="ChildKeys",
        params=[fake_hotkey, fake_netuid],
        reuse_block_hash=False,
    )
    mocked_format_error_message.assert_called_once_with(fake_exception)
    assert result == (False, [], "Formatted error message")


@pytest.mark.asyncio
async def test_get_parents_success(subtensor, mocker):
    """Tests get_parents when parents are successfully retrieved and formatted."""
    # Preps
    fake_hotkey = "valid_hotkey"
    fake_netuid = 1
    fake_parents = mocker.Mock(
        value=[
            (1000, ["parent_key_1"]),
            (2000, ["parent_key_2"]),
        ]
    )

    mocked_query = mocker.AsyncMock(return_value=fake_parents)
    subtensor.substrate.query = mocked_query

    mocked_decode_account_id = mocker.Mock(
        side_effect=["decoded_parent_key_1", "decoded_parent_key_2"]
    )
    mocker.patch.object(async_subtensor, "decode_account_id", mocked_decode_account_id)

    expected_formatted_parents = [
        (u64_normalized_float(1000), "decoded_parent_key_1"),
        (u64_normalized_float(2000), "decoded_parent_key_2"),
    ]

    # Call
    result = await subtensor.get_parents(hotkey=fake_hotkey, netuid=fake_netuid)

    # Asserts
    mocked_query.assert_called_once_with(
        block_hash=None,
        module="SubtensorModule",
        storage_function="ParentKeys",
        params=[fake_hotkey, fake_netuid],
        reuse_block_hash=False,
    )
    mocked_decode_account_id.assert_has_calls(
        [mocker.call("parent_key_1"), mocker.call("parent_key_2")]
    )
    assert result == expected_formatted_parents


@pytest.mark.asyncio
async def test_get_parents_no_parents(subtensor, mocker):
    """Tests get_parents when there are no parents to retrieve."""
    # Preps
    fake_hotkey = "valid_hotkey"
    fake_netuid = 1
    fake_parents = []

    mocked_query = mocker.AsyncMock(return_value=fake_parents)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.get_parents(hotkey=fake_hotkey, netuid=fake_netuid)

    # Asserts
    mocked_query.assert_called_once_with(
        block_hash=None,
        module="SubtensorModule",
        storage_function="ParentKeys",
        params=[fake_hotkey, fake_netuid],
        reuse_block_hash=False,
    )
    assert result == []


@pytest.mark.asyncio
async def test_get_parents_substrate_request_exception(subtensor, mocker):
    """Tests get_parents when SubstrateRequestException is raised."""
    # Preps
    fake_hotkey = "valid_hotkey"
    fake_netuid = 1
    fake_exception = async_subtensor.SubstrateRequestException("Test Exception")

    mocked_query = mocker.AsyncMock(side_effect=fake_exception)
    subtensor.substrate.query = mocked_query

    # Call
    with pytest.raises(async_subtensor.SubstrateRequestException):
        await subtensor.get_parents(hotkey=fake_hotkey, netuid=fake_netuid)


@pytest.mark.asyncio
async def test_get_children_pending(mock_substrate, subtensor):
    mock_substrate.query.return_value.value = [
        [
            (
                U64_MAX,
                (tuple(bytearray(32)),),
            ),
        ],
        123,
    ]

    children, cooldown = await subtensor.get_children_pending(
        "hotkey_ss58",
        netuid=1,
    )

    assert children == [
        (
            1.0,
            "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        ),
    ]
    assert cooldown == 123

    mock_substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="PendingChildKeys",
        params=[1, "hotkey_ss58"],
        block_hash=None,
        reuse_block_hash=False,
    )


@pytest.mark.asyncio
async def test_get_subnet_hyperparameters_success(subtensor, mocker):
    """Tests get_subnet_hyperparameters with successful hyperparameter retrieval."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_result = object()

    mocked_query_runtime_api = mocker.AsyncMock(return_value=fake_result)
    subtensor.query_runtime_api = mocked_query_runtime_api

    mocked_from_dict = mocker.Mock()
    mocker.patch.object(
        async_subtensor.SubnetHyperparameters, "from_dict", mocked_from_dict
    )

    # Call
    result = await subtensor.get_subnet_hyperparameters(
        netuid=fake_netuid, block_hash=fake_block_hash
    )

    # Asserts
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams_v2",
        params=[fake_netuid],
        block=None,
        block_hash=fake_block_hash,
        reuse_block=False,
    )
    assert result == mocked_from_dict.return_value


@pytest.mark.asyncio
async def test_get_subnet_hyperparameters_no_data(subtensor, mocker):
    """Tests get_subnet_hyperparameters when no hyperparameters data is returned."""
    # Preps
    fake_netuid = 1

    mocked_query_runtime_api = mocker.AsyncMock(return_value=None)
    subtensor.query_runtime_api = mocked_query_runtime_api

    # Call
    result = await subtensor.get_subnet_hyperparameters(netuid=fake_netuid)

    # Asserts
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams_v2",
        params=[fake_netuid],
        block=None,
        block_hash=None,
        reuse_block=False,
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_subnet_hyperparameters_without_0x_prefix(subtensor, mocker):
    """Tests get_subnet_hyperparameters when hex_bytes_result is without 0x prefix."""
    # Preps
    fake_netuid = 1
    fake_result = object()

    mocked_query_runtime_api = mocker.AsyncMock(return_value=fake_result)
    subtensor.query_runtime_api = mocked_query_runtime_api

    mocked_from_dict = mocker.Mock()
    mocker.patch.object(
        async_subtensor.SubnetHyperparameters, "from_dict", mocked_from_dict
    )

    # Call
    result = await subtensor.get_subnet_hyperparameters(netuid=fake_netuid)

    # Asserts
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams_v2",
        params=[fake_netuid],
        block=None,
        block_hash=None,
        reuse_block=False,
    )
    mocked_from_dict.assert_called_once_with(fake_result)
    assert result == mocked_from_dict.return_value


@pytest.mark.asyncio
async def test_get_vote_data_success(subtensor, mocker):
    """Tests get_vote_data when voting data is successfully retrieved."""
    # Preps
    fake_proposal_hash = "valid_proposal_hash"
    fake_block_hash = "block_hash"
    fake_vote_data = {"ayes": ["senate_member_1"], "nays": ["senate_member_2"]}

    mocked_query = mocker.AsyncMock(return_value=fake_vote_data)
    subtensor.substrate.query = mocked_query

    mocked_proposal_vote_data = mocker.Mock()
    mocker.patch.object(
        async_subtensor.ProposalVoteData,
        "from_dict",
        return_value=mocked_proposal_vote_data,
    )

    # Call
    result = await subtensor.get_vote_data(
        proposal_hash=fake_proposal_hash, block_hash=fake_block_hash
    )

    # Asserts
    mocked_query.assert_called_once_with(
        module="Triumvirate",
        storage_function="Voting",
        params=[fake_proposal_hash],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result == mocked_proposal_vote_data


@pytest.mark.asyncio
async def test_get_vote_data_no_data(subtensor, mocker):
    """Tests get_vote_data when no voting data is available."""
    # Preps
    fake_proposal_hash = "invalid_proposal_hash"
    fake_block_hash = "block_hash"

    mocked_query = mocker.AsyncMock(return_value=None)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.get_vote_data(
        proposal_hash=fake_proposal_hash, block_hash=fake_block_hash
    )

    # Asserts
    mocked_query.assert_called_once_with(
        module="Triumvirate",
        storage_function="Voting",
        params=[fake_proposal_hash],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_delegate_identities(subtensor, mocker):
    """Tests get_delegate_identities with successful data retrieval from both chain and GitHub."""
    # Preps
    fake_block_hash = "block_hash"
    fake_chain_data = [
        (
            ["delegate1_ss58"],
            mocker.Mock(
                value={
                    "additional": "",
                    "description": "",
                    "discord": "",
                    "github_repo": "",
                    "image": "",
                    "name": "Chain Delegate 1",
                    "url": "",
                },
            ),
        ),
        (
            ["delegate2_ss58"],
            mocker.Mock(
                value={
                    "additional": "",
                    "description": "",
                    "discord": "",
                    "github_repo": "",
                    "image": "",
                    "name": "Chain Delegate 2",
                    "url": "",
                },
            ),
        ),
    ]

    mocked_query_map = mocker.AsyncMock(
        **{"return_value.__aiter__.return_value": iter(fake_chain_data)},
    )
    subtensor.substrate.query_map = mocked_query_map

    mocked_decode_account_id = mocker.Mock(side_effect=lambda ss58: ss58)
    mocker.patch.object(async_subtensor, "decode_account_id", mocked_decode_account_id)

    mocked_decode_hex_identity_dict = mocker.Mock(side_effect=lambda data: data)
    mocker.patch.object(
        async_subtensor, "decode_hex_identity_dict", mocked_decode_hex_identity_dict
    )

    # Call
    result = await subtensor.get_delegate_identities(block_hash=fake_block_hash)

    # Asserts
    mocked_query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="IdentitiesV2",
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )

    assert result["delegate1_ss58"].name == "Chain Delegate 1"
    assert result["delegate2_ss58"].name == "Chain Delegate 2"


@pytest.mark.asyncio
async def test_is_hotkey_registered_true(subtensor, mocker):
    """Tests is_hotkey_registered when the hotkey is registered on the netuid."""
    # Preps
    fake_netuid = 1
    fake_hotkey_ss58 = "registered_hotkey"
    fake_result = "some_value"
    mocked_query = mocker.AsyncMock(return_value=fake_result)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.is_hotkey_registered(
        netuid=fake_netuid, hotkey_ss58=fake_hotkey_ss58
    )

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[fake_netuid, fake_hotkey_ss58],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result is True


@pytest.mark.asyncio
async def test_is_hotkey_registered_false(subtensor, mocker):
    """Tests is_hotkey_registered when the hotkey is not registered on the netuid."""
    # Preps
    fake_netuid = 1
    fake_hotkey_ss58 = "unregistered_hotkey"
    fake_result = None

    mocked_query = mocker.AsyncMock(return_value=fake_result)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.is_hotkey_registered(
        netuid=fake_netuid, hotkey_ss58=fake_hotkey_ss58
    )

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[fake_netuid, fake_hotkey_ss58],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result is False


@pytest.mark.asyncio
async def test_get_uid_for_hotkey_on_subnet_registered(subtensor, mocker):
    """Tests get_uid_for_hotkey_on_subnet when the hotkey is registered and has a UID."""
    # Preps
    fake_hotkey_ss58 = "registered_hotkey"
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_uid = 123

    mocked_query = mocker.AsyncMock(return_value=fake_uid)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=fake_hotkey_ss58, netuid=fake_netuid, block_hash=fake_block_hash
    )

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[fake_netuid, fake_hotkey_ss58],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result == fake_uid


@pytest.mark.asyncio
async def test_get_uid_for_hotkey_on_subnet_not_registered(subtensor, mocker):
    """Tests get_uid_for_hotkey_on_subnet when the hotkey is not registered on the subnet."""
    # Preps
    fake_hotkey_ss58 = "unregistered_hotkey"
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_result = None

    mocked_query = mocker.AsyncMock(return_value=fake_result)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=fake_hotkey_ss58, netuid=fake_netuid, block_hash=fake_block_hash
    )

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[fake_netuid, fake_hotkey_ss58],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result is None


@pytest.mark.asyncio
async def test_weights_rate_limit_success(subtensor, mocker):
    """Tests weights_rate_limit when the hyperparameter value is successfully retrieved."""
    # Preps
    fake_netuid = 1
    fake_rate_limit = 10

    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=fake_rate_limit,
    )

    # Call
    result = await subtensor.weights_rate_limit(netuid=fake_netuid)

    # Asserts
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="WeightsSetRateLimit",
        netuid=fake_netuid,
        block_hash=None,
        reuse_block=False,
    )
    assert result == fake_rate_limit


@pytest.mark.asyncio
async def test_weights_rate_limit_none(subtensor, mocker):
    """Tests weights_rate_limit when the hyperparameter value is not found."""
    # Preps
    fake_netuid = 1
    fake_result = None

    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=fake_result,
    )

    # Call
    result = await subtensor.weights_rate_limit(netuid=fake_netuid)

    # Asserts
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="WeightsSetRateLimit",
        netuid=fake_netuid,
        block_hash=None,
        reuse_block=False,
    )
    assert result is None


@pytest.mark.asyncio
async def test_blocks_since_last_update_success(subtensor, mocker):
    """Tests blocks_since_last_update when the data is successfully retrieved."""
    # Preps
    fake_netuid = 1
    fake_uid = 5
    last_update_block = 50
    current_block = 100
    fake_blocks_since_update = current_block - last_update_block

    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value={fake_uid: last_update_block},
    )

    mocked_get_current_block = mocker.AsyncMock(return_value=current_block)
    subtensor.get_current_block = mocked_get_current_block

    # Call
    result = await subtensor.blocks_since_last_update(netuid=fake_netuid, uid=fake_uid)

    # Asserts
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="LastUpdate", netuid=fake_netuid
    )
    mocked_get_current_block.assert_called_once()
    assert result == fake_blocks_since_update


@pytest.mark.asyncio
async def test_blocks_since_last_update_no_last_update(subtensor, mocker):
    """Tests blocks_since_last_update when the last update data is not found."""
    # Preps
    fake_netuid = 1
    fake_uid = 5
    fake_result = None

    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=fake_result,
    )

    # Call
    result = await subtensor.blocks_since_last_update(netuid=fake_netuid, uid=fake_uid)

    # Asserts
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="LastUpdate", netuid=fake_netuid
    )
    assert result is None


@pytest.mark.asyncio
async def test_commit_reveal_enabled(subtensor, mocker):
    """Test commit_reveal_enabled."""
    # Preps
    netuid = 1
    block_hash = "block_hash"
    mocked_get_hyperparameter = mocker.patch.object(
        subtensor, "get_hyperparameter", return_value=mocker.AsyncMock()
    )

    # Call
    result = await subtensor.commit_reveal_enabled(netuid, block_hash=block_hash)

    # Assertions
    mocked_get_hyperparameter.assert_awaited_once_with(
        param_name="CommitRevealWeightsEnabled",
        block_hash=block_hash,
        netuid=netuid,
        reuse_block=False,
    )
    assert result is False


@pytest.mark.asyncio
async def test_get_subnet_reveal_period_epochs(subtensor, mocker):
    """Test get_subnet_reveal_period_epochs."""
    # Preps
    netuid = 1
    block_hash = "block_hash"
    mocked_get_hyperparameter = mocker.patch.object(
        subtensor, "get_hyperparameter", return_value=mocker.AsyncMock()
    )

    # Call
    result = await subtensor.get_subnet_reveal_period_epochs(
        netuid, block_hash=block_hash
    )

    # Assertions
    mocked_get_hyperparameter.assert_awaited_once_with(
        param_name="RevealPeriodEpochs", block_hash=block_hash, netuid=netuid
    )
    assert result == mocked_get_hyperparameter.return_value


@pytest.mark.asyncio
async def test_transfer_success(subtensor, fake_wallet, mocker):
    """Tests transfer when the transfer is successful."""
    # Preps
    fake_destination = "destination_address"
    fake_amount = Balance.from_tao(100.0)
    fake_transfer_all = False

    mocked_transfer_extrinsic = mocker.AsyncMock(return_value=True)
    mocker.patch.object(
        async_subtensor, "transfer_extrinsic", mocked_transfer_extrinsic
    )

    # Call
    result = await subtensor.transfer(
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        transfer_all=fake_transfer_all,
    )

    # Asserts
    mocked_transfer_extrinsic.assert_awaited_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        transfer_all=fake_transfer_all,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        keep_alive=True,
        period=None,
    )
    assert result == mocked_transfer_extrinsic.return_value


@pytest.mark.asyncio
async def test_register_success(subtensor, fake_wallet, mocker):
    """Tests register when there is enough balance and registration succeeds."""
    # Preps
    fake_netuid = 1

    mocked_register_extrinsic = mocker.AsyncMock()
    mocker.patch.object(
        async_subtensor, "register_extrinsic", mocked_register_extrinsic
    )

    # Call
    result = await subtensor.register(wallet=fake_wallet, netuid=fake_netuid)

    # Asserts
    mocked_register_extrinsic.assert_awaited_once_with(
        cuda=False,
        dev_id=0,
        log_verbose=False,
        max_allowed_attempts=3,
        netuid=1,
        num_processes=None,
        output_in_place=False,
        subtensor=subtensor,
        tpb=256,
        update_interval=None,
        wait_for_finalization=True,
        wait_for_inclusion=False,
        wallet=fake_wallet,
        period=None,
    )
    assert result == mocked_register_extrinsic.return_value


@pytest.mark.asyncio
async def test_set_children(subtensor, fake_wallet, mocker):
    """Tests set_children extrinsic calls properly."""
    # Preps
    mocked_set_children_extrinsic = mocker.AsyncMock()
    mocker.patch.object(
        async_subtensor, "set_children_extrinsic", mocked_set_children_extrinsic
    )
    fake_children = [
        (
            1.0,
            "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        ),
    ]

    # Call
    result = await subtensor.set_children(
        fake_wallet,
        fake_wallet.hotkey.ss58_address,
        netuid=1,
        children=fake_children,
    )

    # Asserts
    mocked_set_children_extrinsic.assert_awaited_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey=fake_wallet.hotkey.ss58_address,
        netuid=1,
        children=fake_children,
        wait_for_finalization=True,
        wait_for_inclusion=True,
        raise_error=False,
        period=None,
    )
    assert result == mocked_set_children_extrinsic.return_value


@pytest.mark.asyncio
async def test_set_delegate_take_equal(subtensor, fake_wallet, mocker):
    mocker.patch.object(subtensor, "get_delegate_take", return_value=0.18)

    await subtensor.set_delegate_take(
        fake_wallet,
        fake_wallet.hotkey.ss58_address,
        0.18,
    )

    subtensor.substrate.submit_extrinsic.assert_not_called()


@pytest.mark.asyncio
async def test_set_delegate_take_increase(
    mock_substrate, subtensor, fake_wallet, mocker
):
    mock_substrate.submit_extrinsic.return_value = mocker.Mock(
        is_success=mocker.AsyncMock(return_value=True)(),
    )
    mocker.patch.object(subtensor, "get_delegate_take", return_value=0.18)

    await subtensor.set_delegate_take(
        fake_wallet,
        fake_wallet.hotkey.ss58_address,
        0.2,
    )

    assert_submit_signed_extrinsic(
        mock_substrate,
        fake_wallet.coldkey,
        call_module="SubtensorModule",
        call_function="increase_take",
        call_params={
            "hotkey": fake_wallet.hotkey.ss58_address,
            "take": 13107,
        },
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )


@pytest.mark.asyncio
async def test_set_delegate_take_decrease(
    mock_substrate, subtensor, fake_wallet, mocker
):
    mock_substrate.submit_extrinsic.return_value = mocker.Mock(
        is_success=mocker.AsyncMock(return_value=True)(),
    )
    mocker.patch.object(subtensor, "get_delegate_take", return_value=0.18)

    await subtensor.set_delegate_take(
        fake_wallet,
        fake_wallet.hotkey.ss58_address,
        0.1,
    )

    assert_submit_signed_extrinsic(
        mock_substrate,
        fake_wallet.coldkey,
        call_module="SubtensorModule",
        call_function="decrease_take",
        call_params={
            "hotkey": fake_wallet.hotkey.ss58_address,
            "take": 6553,
        },
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )


@pytest.mark.asyncio
async def test_set_weights_success(subtensor, fake_wallet, mocker):
    """Tests set_weights with the successful weight setting on the first try."""
    # Preps
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.3, 0.5, 0.2]
    max_retries = 1

    mocked_get_uid_for_hotkey_on_subnet = mocker.patch.object(
        subtensor, "get_uid_for_hotkey_on_subnet"
    )
    subtensor.get_uid_for_hotkey_on_subnet = mocked_get_uid_for_hotkey_on_subnet

    mocked_blocks_since_last_update = mocker.AsyncMock(return_value=2)
    subtensor.blocks_since_last_update = mocked_blocks_since_last_update

    mocked_weights_rate_limit = mocker.AsyncMock(return_value=1)
    subtensor.weights_rate_limit = mocked_weights_rate_limit

    mocked_set_weights_extrinsic = mocker.AsyncMock(return_value=(True, "Success"))
    mocker.patch.object(
        async_subtensor, "set_mechanism_weights_extrinsic", mocked_set_weights_extrinsic
    )

    # Call
    result, message = await subtensor.set_weights(
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        max_retries=max_retries,
    )

    # Asserts
    mocked_get_uid_for_hotkey_on_subnet.assert_called_once_with(
        fake_wallet.hotkey.ss58_address, fake_netuid
    )
    mocked_blocks_since_last_update.assert_called_once_with(
        fake_netuid, mocked_get_uid_for_hotkey_on_subnet.return_value
    )
    mocked_weights_rate_limit.assert_called_once_with(fake_netuid)
    mocked_set_weights_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        version_key=async_subtensor.version_as_int,
        wait_for_finalization=True,
        wait_for_inclusion=True,
        weights=fake_weights,
        period=8,
        mechid=0,
    )
    mocked_weights_rate_limit.assert_called_once_with(fake_netuid)
    assert result is True
    assert message == "Success"


@pytest.mark.asyncio
async def test_set_weights_with_exception(subtensor, fake_wallet, mocker):
    """Tests set_weights when set_weights_extrinsic raises an exception."""
    # Preps
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.3, 0.5, 0.2]
    fake_uid = 10
    max_retries = 1

    mocked_get_uid_for_hotkey_on_subnet = mocker.AsyncMock(return_value=fake_uid)
    subtensor.get_uid_for_hotkey_on_subnet = mocked_get_uid_for_hotkey_on_subnet

    mocked_blocks_since_last_update = mocker.AsyncMock(return_value=10)
    subtensor.blocks_since_last_update = mocked_blocks_since_last_update

    mocked_weights_rate_limit = mocker.AsyncMock(return_value=5)
    subtensor.weights_rate_limit = mocked_weights_rate_limit

    mocked_set_weights_extrinsic = mocker.AsyncMock(
        side_effect=Exception("Test exception")
    )
    mocker.patch.object(
        async_subtensor, "set_mechanism_weights_extrinsic", mocked_set_weights_extrinsic
    )

    # Call
    result, message = await subtensor.set_weights(
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        max_retries=max_retries,
    )

    # Asserts
    assert mocked_get_uid_for_hotkey_on_subnet.call_count == 1
    assert mocked_blocks_since_last_update.call_count == 1
    assert mocked_weights_rate_limit.call_count == 1
    assert mocked_set_weights_extrinsic.call_count == max_retries
    assert result is False
    assert message == "No attempt made. Perhaps it is too soon to set weights!"


@pytest.mark.asyncio
async def test_commit_weights_success(subtensor, fake_wallet, mocker):
    """Tests commit_weights when the weights are committed successfully."""
    # Preps
    fake_netuid = 1
    fake_salt = [12345, 67890]
    fake_uids = [1, 2, 3]
    fake_weights = [100, 200, 300]
    max_retries = 3

    mocked_commit_weights_extrinsic = mocker.AsyncMock(return_value=(True, "Success"))
    mocker.patch.object(
        async_subtensor,
        "commit_mechanism_weights_extrinsic",
        mocked_commit_weights_extrinsic,
    )

    # Call
    result, message = await subtensor.commit_weights(
        wallet=fake_wallet,
        netuid=fake_netuid,
        salt=fake_salt,
        uids=fake_uids,
        weights=fake_weights,
        max_retries=max_retries,
    )

    # Asserts
    mocked_commit_weights_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        salt=fake_salt,
        uids=fake_uids,
        weights=fake_weights,
        wait_for_inclusion=False,
        wait_for_finalization=False,
        period=16,
        mechid=0,
        raise_error=True,
    )
    assert result is True
    assert message == "Success"


@pytest.mark.asyncio
async def test_commit_weights_with_exception(subtensor, fake_wallet, mocker):
    """Tests commit_weights when an exception is raised during weight commitment."""
    # Preps
    fake_netuid = 1
    fake_salt = [12345, 67890]
    fake_uids = [1, 2, 3]
    fake_weights = [100, 200, 300]
    max_retries = 1

    mocked_commit_weights_extrinsic = mocker.AsyncMock(
        side_effect=Exception("Test exception")
    )
    mocker.patch.object(
        async_subtensor,
        "commit_mechanism_weights_extrinsic",
        mocked_commit_weights_extrinsic,
    )

    # Call
    result, message = await subtensor.commit_weights(
        wallet=fake_wallet,
        netuid=fake_netuid,
        salt=fake_salt,
        uids=fake_uids,
        weights=fake_weights,
        max_retries=max_retries,
    )

    # Asserts
    assert mocked_commit_weights_extrinsic.call_count == max_retries
    assert result is False
    assert "No attempt made. Perhaps it is too soon to commit weights!" in message


@pytest.mark.asyncio
async def test_get_all_subnets_info_success(mocker, subtensor):
    """Test get_all_subnets_info returns correct data when subnet information is found."""
    # Prep
    block = 123

    mocker.patch.object(subtensor, "query_runtime_api")
    mocker.patch.object(
        async_subtensor.SubnetInfo,
        "list_from_dicts",
    )

    # Call
    await subtensor.get_all_subnets_info(block)

    # Asserts
    subtensor.query_runtime_api.assert_awaited_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnets_info_v2",
        params=[],
        block=block,
        block_hash=None,
        reuse_block=False,
    )
    async_subtensor.SubnetInfo.list_from_dicts.assert_called_once_with(
        subtensor.query_runtime_api.return_value,
    )


@pytest.mark.asyncio
async def test_set_subnet_identity(mocker, subtensor, fake_wallet):
    """Verify that subtensor method `set_subnet_identity` calls proper function with proper arguments."""
    # Preps
    fake_netuid = 123
    fake_subnet_identity = mocker.MagicMock()

    mocked_extrinsic = mocker.patch.object(
        async_subtensor, "set_subnet_identity_extrinsic"
    )

    # Call
    result = await subtensor.set_subnet_identity(
        wallet=fake_wallet, netuid=fake_netuid, subnet_identity=fake_subnet_identity
    )

    # Asserts
    mocked_extrinsic.assert_awaited_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        subnet_name=fake_subnet_identity.subnet_name,
        github_repo=fake_subnet_identity.github_repo,
        subnet_contact=fake_subnet_identity.subnet_contact,
        subnet_url=fake_subnet_identity.subnet_url,
        logo_url=fake_subnet_identity.logo_url,
        discord=fake_subnet_identity.discord,
        description=fake_subnet_identity.description,
        additional=fake_subnet_identity.additional,
        wait_for_finalization=True,
        wait_for_inclusion=False,
        period=None,
    )
    assert result == mocked_extrinsic.return_value


@pytest.mark.asyncio
async def test_get_all_neuron_certificates(mocker, subtensor):
    fake_netuid = 12
    mocked_query_map_subtensor = mocker.AsyncMock()
    mocker.patch.object(subtensor.substrate, "query_map", mocked_query_map_subtensor)
    await subtensor.get_all_neuron_certificates(fake_netuid)
    mocked_query_map_subtensor.assert_awaited_once_with(
        module="SubtensorModule",
        storage_function="NeuronCertificates",
        params=[fake_netuid],
        block_hash=None,
        reuse_block_hash=False,
    )


@pytest.mark.asyncio
async def test_get_timestamp(mocker, subtensor):
    fake_block = 1000
    mocked_query = mocker.AsyncMock(return_value=ScaleObj(1740586018 * 1000))
    mocker.patch.object(subtensor.substrate, "query", mocked_query)
    expected_result = datetime.datetime(
        2025, 2, 26, 16, 6, 58, tzinfo=datetime.timezone.utc
    )
    actual_result = await subtensor.get_timestamp(block=fake_block)
    assert expected_result == actual_result


@pytest.mark.asyncio
async def test_get_owned_hotkeys_happy_path(subtensor, mocker):
    """Tests that the output of get_owned_hotkeys."""
    # Prep
    fake_coldkey = "fake_hotkey"
    fake_hotkey = "fake_hotkey"
    fake_hotkeys = [
        [
            fake_hotkey,
        ]
    ]
    mocked_subtensor = mocker.AsyncMock(return_value=fake_hotkeys)
    mocker.patch.object(subtensor.substrate, "query", new=mocked_subtensor)

    mocked_decode_account_id = mocker.Mock()
    mocker.patch.object(
        async_subtensor, "decode_account_id", new=mocked_decode_account_id
    )

    # Call
    result = await subtensor.get_owned_hotkeys(fake_coldkey)

    # Asserts
    mocked_subtensor.assert_awaited_once_with(
        module="SubtensorModule",
        storage_function="OwnedHotkeys",
        params=[fake_coldkey],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result == [mocked_decode_account_id.return_value]
    mocked_decode_account_id.assert_called_once_with(fake_hotkey)


@pytest.mark.asyncio
async def test_get_owned_hotkeys_return_empty(subtensor, mocker):
    """Tests that the output of get_owned_hotkeys is empty."""
    # Prep
    fake_coldkey = "fake_hotkey"
    mocked_subtensor = mocker.AsyncMock(return_value=[])
    mocker.patch.object(subtensor.substrate, "query", new=mocked_subtensor)

    # Call
    result = await subtensor.get_owned_hotkeys(fake_coldkey)

    # Asserts
    mocked_subtensor.assert_awaited_once_with(
        module="SubtensorModule",
        storage_function="OwnedHotkeys",
        params=[fake_coldkey],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result == []


@pytest.mark.asyncio
async def test_start_call(subtensor, mocker):
    """Test start_call extrinsic calls properly."""
    # preps
    wallet_name = mocker.Mock(spec=Wallet)
    netuid = 123
    mocked_extrinsic = mocker.patch.object(async_subtensor, "start_call_extrinsic")

    # Call
    result = await subtensor.start_call(wallet_name, netuid)

    # Asserts
    mocked_extrinsic.assert_awaited_once_with(
        subtensor=subtensor,
        wallet=wallet_name,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        period=None,
    )
    assert result == mocked_extrinsic.return_value


@pytest.mark.asyncio
async def test_get_metagraph_info_all_fields(subtensor, mocker):
    """Test get_metagraph_info with all fields (default behavior)."""
    # Preps
    netuid = 1
    default_mechid = 0
    mock_value = {"mock": "data"}

    mock_runtime_call = mocker.patch.object(
        subtensor.substrate,
        "runtime_call",
        return_value=mocker.AsyncMock(value=mock_value),
    )
    mock_from_dict = mocker.patch.object(
        async_subtensor.MetagraphInfo, "from_dict", return_value="parsed_metagraph"
    )

    # Call
    result = await subtensor.get_metagraph_info(
        netuid=netuid, field_indices=[f for f in range(len(SelectiveMetagraphIndex))]
    )

    # Asserts
    assert result == "parsed_metagraph"
    mock_runtime_call.assert_awaited_once_with(
        api="SubnetInfoRuntimeApi",
        method="get_selective_mechagraph",
        params=[netuid, default_mechid, SelectiveMetagraphIndex.all_indices()],
        block_hash=await subtensor.determine_block_hash(None),
    )
    mock_from_dict.assert_called_once_with(mock_value)


@pytest.mark.asyncio
async def test_get_metagraph_info_specific_fields(subtensor, mocker):
    """Test get_metagraph_info with specific fields."""
    # Preps
    netuid = 1
    default_mechid = 0
    mock_value = {"mock": "data"}
    fields = [SelectiveMetagraphIndex.Name, 5]

    mock_runtime_call = mocker.patch.object(
        subtensor.substrate,
        "runtime_call",
        return_value=mocker.AsyncMock(value=mock_value),
    )
    mock_from_dict = mocker.patch.object(
        async_subtensor.MetagraphInfo, "from_dict", return_value="parsed_metagraph"
    )

    # Call
    result = await subtensor.get_metagraph_info(netuid=netuid, field_indices=fields)

    # Asserts
    assert result == "parsed_metagraph"
    mock_runtime_call.assert_awaited_once_with(
        api="SubnetInfoRuntimeApi",
        method="get_selective_mechagraph",
        params=[
            netuid,
            default_mechid,
            [0]
            + [
                f.value if isinstance(f, SelectiveMetagraphIndex) else f for f in fields
            ],
        ],
        block_hash=await subtensor.determine_block_hash(None),
    )
    mock_from_dict.assert_called_once_with(mock_value)


@pytest.mark.asyncio
async def test_get_metagraph_info_subnet_not_exist(subtensor, mocker):
    """Test get_metagraph_info returns None when subnet doesn't exist."""
    netuid = 1
    default_mechid = 0
    mocker.patch.object(
        subtensor.substrate,
        "runtime_call",
        return_value=None,
    )

    mocked_logger = mocker.Mock()
    mocker.patch("bittensor.core.subtensor.logging.error", new=mocked_logger)

    result = await subtensor.get_metagraph_info(netuid=netuid)

    assert result is None
    mocked_logger.assert_called_once_with(
        f"Subnet mechanism {netuid}.{default_mechid} does not exist."
    )


@pytest.mark.asyncio
async def test_blocks_since_last_step_with_value(subtensor, mocker):
    """Test blocks_since_last_step returns correct value."""
    # preps
    netuid = 1
    block = 123
    mocked_query_subtensor = mocker.AsyncMock()
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = await subtensor.blocks_since_last_step(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_awaited_once_with(
        name="BlocksSinceLastStep",
        block=block,
        block_hash=None,
        reuse_block=False,
        params=[netuid],
    )

    assert result == mocked_query_subtensor.return_value.value


@pytest.mark.asyncio
async def test_blocks_since_last_step_is_none(subtensor, mocker):
    """Test blocks_since_last_step returns None correctly."""
    # preps
    netuid = 1
    block = 123
    mocked_query_subtensor = mocker.AsyncMock(return_value=None)
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = await subtensor.blocks_since_last_step(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_awaited_once_with(
        name="BlocksSinceLastStep",
        block=block,
        block_hash=None,
        reuse_block=False,
        params=[netuid],
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_subnet_owner_hotkey_has_return(subtensor, mocker):
    """Test get_subnet_owner_hotkey returns correct value."""
    # preps
    netuid = 14
    block = 123
    expected_owner_hotkey = "owner_hotkey"
    mocked_query_subtensor = mocker.AsyncMock(return_value=expected_owner_hotkey)
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = await subtensor.get_subnet_owner_hotkey(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_awaited_once_with(
        name="SubnetOwnerHotkey",
        block=block,
        params=[netuid],
    )

    assert result == expected_owner_hotkey


@pytest.mark.asyncio
async def test_get_subnet_owner_hotkey_is_none(subtensor, mocker):
    """Test get_subnet_owner_hotkey returns None correctly."""
    # preps
    netuid = 14
    block = 123
    mocked_query_subtensor = mocker.AsyncMock(return_value=None)
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = await subtensor.get_subnet_owner_hotkey(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_awaited_once_with(
        name="SubnetOwnerHotkey",
        block=block,
        params=[netuid],
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_subnet_validator_permits_has_values(subtensor, mocker):
    """Test get_subnet_validator_permits returns correct value."""
    # preps
    netuid = 14
    block = 123
    expected_validator_permits = [False, True, False]
    mocked_query_subtensor = mocker.AsyncMock(return_value=expected_validator_permits)
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = await subtensor.get_subnet_validator_permits(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_awaited_once_with(
        name="ValidatorPermit",
        block=block,
        params=[netuid],
    )

    assert result == expected_validator_permits


@pytest.mark.asyncio
async def test_get_subnet_validator_permits_is_none(subtensor, mocker):
    """Test get_subnet_validator_permits returns correct value."""
    # preps
    netuid = 14
    block = 123

    mocked_query_subtensor = mocker.AsyncMock(return_value=None)
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = await subtensor.get_subnet_validator_permits(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_awaited_once_with(
        name="ValidatorPermit",
        block=block,
        params=[netuid],
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_subnet_info_success(mocker, subtensor):
    """Test get_subnet_info returns correct data when subnet information is found."""
    # Prep
    netuid = mocker.Mock()
    block = mocker.Mock()

    mocker.patch.object(subtensor, "query_runtime_api")
    mocker.patch.object(
        async_subtensor.SubnetInfo,
        "from_dict",
    )

    # Call
    result = await subtensor.get_subnet_info(netuid=netuid, block=block)

    # Asserts
    subtensor.query_runtime_api.assert_awaited_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_info_v2",
        params=[netuid],
        block=block,
        block_hash=None,
        reuse_block=False,
    )
    async_subtensor.SubnetInfo.from_dict.assert_called_once_with(
        subtensor.query_runtime_api.return_value,
    )
    assert result == async_subtensor.SubnetInfo.from_dict.return_value


@pytest.mark.asyncio
async def test_get_subnet_info_no_data(mocker, subtensor):
    """Test get_subnet_info returns None."""
    # Prep
    netuid = mocker.Mock()
    block = mocker.Mock()
    mocker.patch.object(async_subtensor.SubnetInfo, "from_dict")
    mocker.patch.object(subtensor, "query_runtime_api", return_value=None)

    # Call
    result = await subtensor.get_subnet_info(netuid=netuid, block=block)

    # Asserts
    subtensor.query_runtime_api.assert_awaited_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_info_v2",
        params=[netuid],
        block=block,
        block_hash=None,
        reuse_block=False,
    )
    async_subtensor.SubnetInfo.from_dict.assert_not_called()
    assert result is None


@pytest.mark.parametrize(
    "call_return, expected",
    [[10, 111], [None, None], [0, 121]],
)
@pytest.mark.asyncio
async def test_get_next_epoch_start_block(mocker, subtensor, call_return, expected):
    """Check that get_next_epoch_start_block returns the correct value."""
    # Prep
    netuid = mocker.Mock()
    block = 20

    fake_block_hash = mocker.Mock()
    mocker.patch.object(subtensor, "get_block_hash", return_value=fake_block_hash)

    mocked_blocks_since_last_step = mocker.AsyncMock(return_value=call_return)
    subtensor.blocks_since_last_step = mocked_blocks_since_last_step

    mocker.patch.object(subtensor, "tempo", return_value=100)

    # Call
    result = await subtensor.get_next_epoch_start_block(netuid=netuid, block=block)

    # Asserts
    mocked_blocks_since_last_step.assert_called_once_with(
        netuid=netuid,
        block=block,
        block_hash=fake_block_hash,
        reuse_block=False,
    )
    subtensor.tempo.assert_awaited_once_with(
        netuid=netuid, block=block, block_hash=fake_block_hash, reuse_block=False
    )
    assert result == expected


@pytest.mark.asyncio
async def test_unstake_all(subtensor, fake_wallet, mocker):
    """Verifies unstake_all calls properly."""
    # Preps
    fake_unstake_all_extrinsic = mocker.AsyncMock()
    mocker.patch.object(
        async_subtensor, "unstake_all_extrinsic", fake_unstake_all_extrinsic
    )
    # Call
    result = await subtensor.unstake_all(
        wallet=fake_wallet,
        hotkey=fake_wallet.hotkey.ss58_address,
        netuid=1,
    )
    # Asserts
    fake_unstake_all_extrinsic.assert_awaited_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey=fake_wallet.hotkey.ss58_address,
        netuid=1,
        rate_tolerance=0.005,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        period=None,
    )
    assert result == fake_unstake_all_extrinsic.return_value


@pytest.mark.asyncio
async def test_get_liquidity_list_subnet_does_not_exits(subtensor, mocker):
    """Test get_liquidity_list returns None when subnet doesn't exist."""
    # Preps
    mocker.patch.object(subtensor, "subnet_exists", return_value=False)

    # Call
    result = await subtensor.get_liquidity_list(wallet=mocker.Mock(), netuid=1)

    # Asserts
    subtensor.subnet_exists.assert_awaited_once_with(netuid=1)
    assert result is None


@pytest.mark.asyncio
async def test_get_liquidity_list_subnet_is_not_active(subtensor, mocker):
    """Test get_liquidity_list returns None when subnet is not active."""
    # Preps
    mocker.patch.object(subtensor, "subnet_exists", return_value=True)
    mocker.patch.object(subtensor, "is_subnet_active", return_value=False)

    # Call
    result = await subtensor.get_liquidity_list(wallet=mocker.Mock(), netuid=1)

    # Asserts
    subtensor.subnet_exists.assert_awaited_once_with(netuid=1)
    subtensor.is_subnet_active.assert_awaited_once_with(netuid=1)
    assert result is None


@pytest.mark.asyncio
async def test_get_liquidity_list_happy_path(subtensor, fake_wallet, mocker):
    """Tests `get_liquidity_list` returns the correct value."""
    # Preps
    netuid = 2

    mocker.patch.object(subtensor, "subnet_exists", return_value=True)
    mocker.patch.object(subtensor, "is_subnet_active", return_value=True)
    mocker.patch.object(subtensor, "determine_block_hash")

    mocker.patch.object(
        async_subtensor, "price_to_tick", return_value=Balance.from_tao(1.0, netuid)
    )
    mocker.patch.object(
        async_subtensor,
        "calculate_fees",
        return_value=(Balance.from_tao(0.0), Balance.from_tao(0.0, netuid)),
    )

    mocked_substrate_query_multi = mocker.AsyncMock(
        side_effect=[
            [
                (None, {"bits": 0}),
                (None, {"bits": 0}),
                (None, {"bits": 18446744073709551616}),
            ],
            [
                (
                    None,
                    {
                        "liquidity_net": 1000000000000,
                        "liquidity_gross": 1000000000000,
                        "fees_out_tao": {"bits": 0},
                        "fees_out_alpha": {"bits": 0},
                    },
                ),
                (
                    None,
                    {
                        "liquidity_net": -1000000000000,
                        "liquidity_gross": 1000000000000,
                        "fees_out_tao": {"bits": 0},
                        "fees_out_alpha": {"bits": 0},
                    },
                ),
                (
                    None,
                    {
                        "liquidity_net": 1000000000000,
                        "liquidity_gross": 1000000000000,
                        "fees_out_tao": {"bits": 0},
                        "fees_out_alpha": {"bits": 0},
                    },
                ),
                (
                    None,
                    {
                        "liquidity_net": -1000000000000,
                        "liquidity_gross": 1000000000000,
                        "fees_out_tao": {"bits": 0},
                        "fees_out_alpha": {"bits": 0},
                    },
                ),
                (
                    None,
                    {
                        "liquidity_net": 1000000000000,
                        "liquidity_gross": 1000000000000,
                        "fees_out_tao": {"bits": 0},
                        "fees_out_alpha": {"bits": 0},
                    },
                ),
                (
                    None,
                    {
                        "liquidity_net": -1000000000000,
                        "liquidity_gross": 1000000000000,
                        "fees_out_tao": {"bits": 0},
                        "fees_out_alpha": {"bits": 0},
                    },
                ),
            ],
        ]
    )

    mocker.patch.object(
        subtensor.substrate, "query_multi", mocked_substrate_query_multi
    )

    fake_positions = [
        [
            (2,),
            mocker.Mock(
                value={
                    "id": (2,),
                    "netuid": 2,
                    "tick_low": (206189,),
                    "tick_high": (208196,),
                    "liquidity": 1000000000000,
                    "fees_tao": {"bits": 0},
                    "fees_alpha": {"bits": 0},
                }
            ),
        ],
        [
            (2,),
            mocker.Mock(
                value={
                    "id": (2,),
                    "netuid": 2,
                    "tick_low": (216189,),
                    "tick_high": (198196,),
                    "liquidity": 2000000000000,
                    "fees_tao": {"bits": 0},
                    "fees_alpha": {"bits": 0},
                }
            ),
        ],
        [
            (2,),
            mocker.Mock(
                value={
                    "id": (2,),
                    "netuid": 2,
                    "tick_low": (226189,),
                    "tick_high": (188196,),
                    "liquidity": 3000000000000,
                    "fees_tao": {"bits": 0},
                    "fees_alpha": {"bits": 0},
                }
            ),
        ],
    ]

    fake_result = mocker.AsyncMock(autospec=list)
    fake_result.__aiter__.return_value = iter(fake_positions)

    mocked_query_map = mocker.AsyncMock(return_value=fake_result)
    mocker.patch.object(subtensor, "query_map", new=mocked_query_map)

    # Call

    result = await subtensor.get_liquidity_list(wallet=fake_wallet, netuid=netuid)

    # Asserts
    subtensor.determine_block_hash.assert_awaited_once_with(
        block=None, block_hash=None, reuse_block=False
    )
    assert async_subtensor.price_to_tick.call_count == 1
    assert async_subtensor.calculate_fees.call_count == len(fake_positions)

    mocked_query_map.assert_awaited_once_with(
        module="Swap",
        name="Positions",
        block=None,
        params=[netuid, fake_wallet.coldkeypub.ss58_address],
    )
    assert len(result) == len(fake_positions)
    assert all([isinstance(p, async_subtensor.LiquidityPosition) for p in result])


@pytest.mark.asyncio
async def test_add_liquidity(subtensor, fake_wallet, mocker):
    """Test add_liquidity extrinsic calls properly."""
    # preps
    netuid = 123
    mocked_extrinsic = mocker.patch.object(async_subtensor, "add_liquidity_extrinsic")

    # Call
    result = await subtensor.add_liquidity(
        wallet=fake_wallet,
        netuid=netuid,
        liquidity=Balance.from_tao(150),
        price_low=Balance.from_tao(180).rao,
        price_high=Balance.from_tao(130).rao,
    )

    # Asserts
    mocked_extrinsic.assert_awaited_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        liquidity=Balance.from_tao(150),
        price_low=Balance.from_tao(180).rao,
        price_high=Balance.from_tao(130).rao,
        hotkey=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert result == mocked_extrinsic.return_value


@pytest.mark.asyncio
async def test_modify_liquidity(subtensor, fake_wallet, mocker):
    """Test modify_liquidity extrinsic calls properly."""
    # preps
    netuid = 123
    mocked_extrinsic = mocker.patch.object(
        async_subtensor, "modify_liquidity_extrinsic"
    )
    position_id = 2

    # Call
    result = await subtensor.modify_liquidity(
        wallet=fake_wallet,
        netuid=netuid,
        position_id=position_id,
        liquidity_delta=Balance.from_tao(150),
    )

    # Asserts
    mocked_extrinsic.assert_awaited_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        position_id=position_id,
        liquidity_delta=Balance.from_tao(150),
        hotkey=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert result == mocked_extrinsic.return_value


@pytest.mark.asyncio
async def test_remove_liquidity(subtensor, fake_wallet, mocker):
    """Test remove_liquidity extrinsic calls properly."""
    # preps
    netuid = 123
    mocked_extrinsic = mocker.patch.object(
        async_subtensor, "remove_liquidity_extrinsic"
    )
    position_id = 2

    # Call
    result = await subtensor.remove_liquidity(
        wallet=fake_wallet,
        netuid=netuid,
        position_id=position_id,
    )

    # Asserts
    mocked_extrinsic.assert_awaited_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        position_id=position_id,
        hotkey=None,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert result == mocked_extrinsic.return_value


@pytest.mark.asyncio
async def test_toggle_user_liquidity(subtensor, fake_wallet, mocker):
    """Test toggle_user_liquidity extrinsic calls properly."""
    # preps
    netuid = 123
    mocked_extrinsic = mocker.patch.object(
        async_subtensor, "toggle_user_liquidity_extrinsic"
    )
    enable = mocker.Mock()

    # Call
    result = await subtensor.toggle_user_liquidity(
        wallet=fake_wallet,
        netuid=netuid,
        enable=enable,
    )

    # Asserts
    mocked_extrinsic.assert_awaited_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        enable=enable,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert result == mocked_extrinsic.return_value


@pytest.mark.asyncio
async def test_get_subnet_price(subtensor, mocker):
    """Test get_subnet_price returns the correct value."""
    # preps
    netuid = 123
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    fake_price = 29258617
    expected_price = Balance.from_tao(0.029258617)
    mocked_query = mocker.patch.object(
        subtensor.substrate, "runtime_call", return_value=mocker.Mock(value=fake_price)
    )

    # Call
    result = await subtensor.get_subnet_price(
        netuid=netuid,
    )

    # Asserts
    mocked_determine_block_hash.assert_awaited_once()
    mocked_query.assert_awaited_once_with(
        api="SwapRuntimeApi",
        method="current_alpha_price",
        params=[netuid],
        block_hash=mocked_determine_block_hash.return_value,
    )

    assert result == expected_price


@pytest.mark.asyncio
async def test_get_subnet_prices(subtensor, mocker):
    """Test get_subnet_prices returns the correct value."""
    # preps
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")

    async def fake_current_sqrt_prices():
        yield [0, {"bits": 0}]
        yield [1, {"bits": 3155343338053956962}]

    expected_prices = {0: Balance.from_tao(1), 1: Balance.from_tao(0.029258617)}
    mocked_query_map = mocker.patch.object(
        subtensor.substrate, "query_map", return_value=fake_current_sqrt_prices()
    )

    # Call
    result = await subtensor.get_subnet_prices()

    # Asserts
    mocked_determine_block_hash.assert_awaited_once_with(
        block=None, block_hash=None, reuse_block=False
    )
    mocked_query_map.assert_awaited_once_with(
        module="Swap",
        storage_function="AlphaSqrtPrice",
        block_hash=mocked_determine_block_hash.return_value,
        page_size=129,  # total number of subnets
    )
    assert result == expected_prices


@pytest.mark.asyncio
async def test_all_subnets(subtensor, mocker):
    """Verify that `all_subnets` calls proper methods and returns the correct value."""
    # Preps
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_di_list_from_dicts = mocker.patch.object(
        async_subtensor.DynamicInfo, "list_from_dicts"
    )
    mocked_get_subnet_prices = mocker.patch.object(
        subtensor,
        "get_subnet_prices",
        return_value={0: Balance.from_tao(1), 1: Balance.from_tao(0.029258617)},
    )
    mocked_decode = mocker.Mock(return_value=[{"netuid": 0}, {"netuid": 1}])
    mocked_runtime_call = mocker.Mock(decode=mocked_decode)
    mocker.patch.object(
        subtensor.substrate, "runtime_call", return_value=mocked_runtime_call
    )

    # Call
    result = await subtensor.all_subnets()

    # Asserts
    mocked_determine_block_hash.assert_awaited_once_with(
        block=None, block_hash=None, reuse_block=False
    )
    subtensor.substrate.runtime_call.assert_called_once_with(
        api="SubnetInfoRuntimeApi",
        method="get_all_dynamic_info",
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_get_subnet_prices.assert_called_once()
    mocked_di_list_from_dicts.assert_called_once_with(
        [
            {"netuid": 0, "price": Balance.from_tao(1)},
            {"netuid": 1, "price": Balance.from_tao(0.029258617)},
        ]
    )
    assert result == mocked_di_list_from_dicts.return_value


@pytest.mark.asyncio
async def test_subnet(subtensor, mocker):
    """Verify that `subnet` calls proper methods and returns the correct value."""
    # Preps
    netuid = 14
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_di_from_dict = mocker.patch.object(async_subtensor.DynamicInfo, "from_dict")
    mocked_get_subnet_price = mocker.patch.object(
        subtensor, "get_subnet_price", return_value=Balance.from_tao(100.0)
    )
    mocked_decode = mocker.Mock(return_value={"netuid": netuid})
    mocked_runtime_call = mocker.Mock(decode=mocked_decode)
    mocker.patch.object(
        subtensor.substrate, "runtime_call", return_value=mocked_runtime_call
    )

    # Call
    result = await subtensor.subnet(netuid=netuid)

    # Asserts
    mocked_determine_block_hash.assert_awaited_once_with(
        block=None, block_hash=None, reuse_block=False
    )
    subtensor.substrate.runtime_call.assert_awaited_once_with(
        "SubnetInfoRuntimeApi",
        "get_dynamic_info",
        params=[netuid],
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_get_subnet_price.assert_awaited_once_with(
        netuid=netuid,
        block=None,
        block_hash=mocked_determine_block_hash.return_value,
        reuse_block=False,
    )
    mocked_di_from_dict.assert_called_once_with(
        {"netuid": netuid, "price": Balance.from_tao(100.0)}
    )
    assert result == mocked_di_from_dict.return_value


@pytest.mark.asyncio
async def test_get_stake_operations_fee(subtensor, mocker):
    """Verify that `get_stake_operations_fee` calls proper methods and returns the correct value."""
    # Preps
    netuid = 1
    amount = Balance.from_rao(100_000_000_000)  # 100 Tao
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query_map = mocker.patch.object(
        subtensor.substrate, "query", return_value=mocker.Mock(value=196)
    )

    # Call
    result = await subtensor.get_stake_operations_fee(netuid=netuid, amount=amount)

    # Assert
    mocked_determine_block_hash.assert_awaited_once_with(
        block=None, block_hash=None, reuse_block=False
    )
    mocked_query_map.assert_awaited_once_with(
        module="Swap",
        storage_function="FeeRate",
        params=[netuid],
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result == Balance.from_rao(299076829).set_unit(netuid)


@pytest.mark.asyncio
async def test_get_stake_add_fee(subtensor, mocker):
    """Verify that `get_stake_add_fee` calls proper methods and returns the correct value."""
    # Preps
    netuid = mocker.Mock()
    amount = mocker.Mock()
    mocked_get_stake_operations_fee = mocker.patch.object(
        subtensor, "get_stake_operations_fee"
    )

    # Call
    result = await subtensor.get_stake_add_fee(
        amount=amount,
        netuid=netuid,
        coldkey_ss58=mocker.Mock(),
        hotkey_ss58=mocker.Mock(),
    )

    # Asserts
    mocked_get_stake_operations_fee.assert_awaited_once_with(
        amount=amount, netuid=netuid, block=None
    )
    assert result == mocked_get_stake_operations_fee.return_value


@pytest.mark.asyncio
async def test_get_unstake_fee(subtensor, mocker):
    """Verify that `get_unstake_fee` calls proper methods and returns the correct value."""
    # Preps
    netuid = mocker.Mock()
    amount = mocker.Mock()
    mocked_get_stake_operations_fee = mocker.patch.object(
        subtensor, "get_stake_operations_fee"
    )

    # Call
    result = await subtensor.get_unstake_fee(
        amount=amount,
        netuid=netuid,
        coldkey_ss58=mocker.Mock(),
        hotkey_ss58=mocker.Mock(),
    )

    # Asserts
    mocked_get_stake_operations_fee.assert_awaited_once_with(
        amount=amount, netuid=netuid, block=None
    )
    assert result == mocked_get_stake_operations_fee.return_value


@pytest.mark.asyncio
async def test_get_stake_movement_fee(subtensor, mocker):
    """Verify that `get_stake_movement_fee` calls proper methods and returns the correct value."""
    # Preps
    netuid = mocker.Mock()
    amount = mocker.Mock()
    mocked_get_stake_operations_fee = mocker.patch.object(
        subtensor, "get_stake_operations_fee"
    )

    # Call
    result = await subtensor.get_stake_movement_fee(
        amount=amount,
        origin_netuid=netuid,
        origin_hotkey_ss58=mocker.Mock(),
        origin_coldkey_ss58=mocker.Mock(),
        destination_netuid=mocker.Mock(),
        destination_hotkey_ss58=mocker.Mock(),
        destination_coldkey_ss58=mocker.Mock(),
    )

    # Asserts
    mocked_get_stake_operations_fee.assert_awaited_once_with(
        amount=amount, netuid=netuid, block=None
    )
    assert result == mocked_get_stake_operations_fee.return_value


@pytest.mark.asyncio
async def test_get_stake_weight(subtensor, mocker):
    """Verify that `get_stake_weight` method calls proper methods and returns the correct value."""
    # Preps
    netuid = mocker.Mock()
    fake_weights = [0, 100, 15000]
    expected_result = [0.0, 0.0015259021896696422, 0.22888532845044632]

    mock_determine_block_hash = mocker.patch.object(
        subtensor,
        "determine_block_hash",
    )
    mocked_query = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=fake_weights,
    )

    # Call
    result = await subtensor.get_stake_weight(netuid=netuid)

    # Asserts
    mock_determine_block_hash.assert_awaited_once_with(
        block=None,
        block_hash=None,
        reuse_block=False,
    )
    mocked_query.assert_awaited_once_with(
        module="SubtensorModule",
        storage_function="StakeWeight",
        params=[netuid],
        block_hash=mock_determine_block_hash.return_value,
    )
    assert result == expected_result


@pytest.mark.asyncio
async def test_get_timelocked_weight_commits(subtensor, mocker):
    """Verify that `get_timelocked_weight_commits` method calls proper methods and returns the correct value."""
    # Preps
    netuid = 14

    mock_determine_block_hash = mocker.patch.object(
        subtensor,
        "determine_block_hash",
    )
    mocked_query_map = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.query_map,
    )
    subtensor.substrate.query_map = mocked_query_map

    # Call
    result = await subtensor.get_timelocked_weight_commits(netuid=netuid)

    # Asserts
    mock_determine_block_hash.assert_awaited_once_with(
        block=None, block_hash=None, reuse_block=False
    )
    mocked_query_map.assert_awaited_once_with(
        module="SubtensorModule",
        storage_function="TimelockedWeightCommits",
        params=[netuid],
        block_hash=mock_determine_block_hash.return_value,
    )
    assert result == []


@pytest.mark.parametrize(
    "query_return, expected_result",
    (
        ["value", [10, 90]],
        [None, None],
    ),
)
@pytest.mark.asyncio
async def test_get_mechanism_emission_split(
    subtensor, mocker, query_return, expected_result
):
    """Verify that get_mechanism_emission_split calls the correct methods."""
    # Preps
    netuid = mocker.Mock()
    query_return = (
        mocker.Mock(value=[6553, 58982]) if query_return == "value" else query_return
    )
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query = mocker.patch.object(
        subtensor.substrate, "query", return_value=query_return
    )

    # Call

    result = await subtensor.get_mechanism_emission_split(netuid)

    # Asserts
    mocked_determine_block_hash.assert_awaited_once()
    mocked_query.assert_awaited_once_with(
        module="SubtensorModule",
        storage_function="MechanismEmissionSplit",
        params=[netuid],
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result == expected_result


@pytest.mark.asyncio
async def test_get_mechanism_count(subtensor, mocker):
    """Verify that `get_mechanism_count` method processed the data correctly."""
    # Preps
    netuid = 14

    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_result = mocker.MagicMock()
    mocker.patch.object(subtensor.substrate, "runtime_call", return_value=mocked_result)
    mocked_query = mocker.patch.object(subtensor.substrate, "query")

    # Call
    result = await subtensor.get_mechanism_count(netuid=netuid)

    # Asserts
    mocked_determine_block_hash.assert_called_once()
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="MechanismCountCurrent",
        params=[netuid],
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result is mocked_query.return_value.value


@pytest.mark.asyncio
async def test_is_in_admin_freeze_window_root_net(subtensor, mocker):
    """Verify that root net has no admin freeze window."""
    # Preps
    netuid = 0
    mocked_get_next_epoch_start_block = mocker.patch.object(
        subtensor, "get_next_epoch_start_block"
    )

    # Call
    result = await subtensor.is_in_admin_freeze_window(netuid=netuid)

    # Asserts
    mocked_get_next_epoch_start_block.assert_not_called()
    assert result is False


@pytest.mark.parametrize(
    "block, next_esb, expected_result",
    (
        [89, 100, False],
        [90, 100, False],
        [91, 100, True],
    ),
)
@pytest.mark.asyncio
async def test_is_in_admin_freeze_window(
    subtensor, mocker, block, next_esb, expected_result
):
    """Verify that `is_in_admin_freeze_window` method processed the data correctly."""
    # Preps
    netuid = 14
    mocker.patch.object(subtensor, "get_current_block", return_value=block)
    mocker.patch.object(subtensor, "get_next_epoch_start_block", return_value=next_esb)
    mocker.patch.object(subtensor, "get_admin_freeze_window", return_value=10)

    # Call

    result = await subtensor.is_in_admin_freeze_window(netuid=netuid)

    # Asserts
    assert result is expected_result


@pytest.mark.asyncio
async def test_get_admin_freeze_window(subtensor, mocker):
    """Verify that `get_admin_freeze_window` calls proper methods."""
    # Preps
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query = mocker.patch.object(subtensor.substrate, "query")

    # Call
    result = await subtensor.get_admin_freeze_window()

    # Asserts
    mocked_query.assert_awaited_once_with(
        module="SubtensorModule",
        storage_function="AdminFreezeWindow",
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result == mocked_query.return_value.value


@pytest.mark.asyncio
async def test_get_auto_stakes(subtensor, mocker):
    """Tests that `get_auto_stakes` calls proper methods and returns the correct value."""
    # Preps
    fake_coldkey = mocker.Mock()
    mock_determine_block_hash = mocker.patch.object(
        subtensor,
        "determine_block_hash",
    )
    fake_hk_1 = mocker.Mock()
    fake_hk_2 = mocker.Mock()

    dest_value_1 = mocker.Mock(value=[fake_hk_1])
    dest_value_2 = mocker.Mock(value=[fake_hk_2])

    mock_result = mocker.MagicMock()
    mock_result.__aiter__.return_value = iter([(0, dest_value_1), (1, dest_value_2)])
    mocked_query_map = mocker.patch.object(
        subtensor.substrate, "query_map", return_value=mock_result
    )

    mocked_decode_account_id = mocker.patch.object(
        async_subtensor,
        "decode_account_id",
        side_effect=[fake_hk_1, fake_hk_2],
    )

    # Call
    result = await subtensor.get_auto_stakes(coldkey_ss58=fake_coldkey)

    # Asserts
    mock_determine_block_hash.assert_awaited_once()
    mocked_query_map.assert_awaited_once_with(
        module="SubtensorModule",
        storage_function="AutoStakeDestination",
        params=[fake_coldkey],
        block_hash=mock_determine_block_hash.return_value,
    )
    mocked_decode_account_id.assert_has_calls(
        [mocker.call(dest_value_1.value[0]), mocker.call(dest_value_2.value[0])]
    )
    assert result == {0: fake_hk_1, 1: fake_hk_2}


@pytest.mark.asyncio
async def test_set_auto_stake(subtensor, mocker):
    """Tests that `set_auto_stake` calls proper methods and returns the correct value."""
    # Preps
    wallet = mocker.Mock()
    netuid = mocker.Mock()
    hotkey = mocker.Mock()
    mocked_extrinsic = mocker.patch.object(async_subtensor, "set_auto_stake_extrinsic")

    # Call
    result = await subtensor.set_auto_stake(
        wallet=wallet,
        netuid=netuid,
        hotkey_ss58=hotkey,
    )

    # Asserts
    mocked_extrinsic.assert_awaited_once_with(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        hotkey_ss58=hotkey,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert result == mocked_extrinsic.return_value


@pytest.mark.asyncio
async def test_determine_block_hash(subtensor, mocker):
    """Tests that `determine_block_hash` calls proper methods and returns the correct value."""

    async def fake_get_block_hash(block: int) -> str:
        d = {
            1: "0xfake1",
            2: "0xfake2",
        }
        return d[block]

    subtensor.get_block_hash = fake_get_block_hash

    # Call
    mocked_hash = await subtensor.get_block_hash(block=1)

    expected_hash_1 = await subtensor.determine_block_hash(block_hash=mocked_hash)
    assert mocked_hash == expected_hash_1

    expected_hash_2 = await subtensor.determine_block_hash(
        block=1, block_hash=mocked_hash
    )
    assert expected_hash_1 == expected_hash_2

    with pytest.raises(ValueError):
        await subtensor.determine_block_hash(
            block_hash=mocked_hash, block=1, reuse_block=True
        )

    with pytest.raises(ValueError):
        await subtensor.determine_block_hash(block=2, block_hash=mocked_hash)
