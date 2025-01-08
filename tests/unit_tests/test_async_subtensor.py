import pytest

from bittensor import AsyncSubtensor
from bittensor.core import async_subtensor


@pytest.fixture(autouse=True)
def subtensor(mocker):
    fake_async_substrate = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface
    )
    mocker.patch.object(
        async_subtensor, "AsyncSubstrateInterface", return_value=fake_async_substrate
    )
    return async_subtensor.AsyncSubtensor()


def test_decode_ss58_tuples_in_proposal_vote_data(mocker):
    """Tests that ProposalVoteData instance instantiation works properly,"""
    # Preps
    mocked_decode_account_id = mocker.patch.object(async_subtensor, "decode_account_id")
    fake_proposal_dict = {
        "index": "0",
        "threshold": 1,
        "ayes": ("0 line", "1 line"),
        "nays": ("2 line", "3 line"),
        "end": 123,
    }

    # Call
    async_subtensor.ProposalVoteData(fake_proposal_dict)

    # Asserts
    assert mocked_decode_account_id.call_count == len(fake_proposal_dict["ayes"]) + len(
        fake_proposal_dict["nays"]
    )
    assert mocked_decode_account_id.mock_calls == [
        mocker.call("0"),
        mocker.call("1"),
        mocker.call("2"),
        mocker.call("3"),
    ]


def test_decode_hex_identity_dict_with_single_byte_utf8():
    """Tests _decode_hex_identity_dict when value is a single utf-8 decodable byte."""
    info_dict = {"name": (b"Neuron",)}
    result = async_subtensor._decode_hex_identity_dict(info_dict)
    assert result["name"] == "Neuron"


def test_decode_hex_identity_dict_with_non_utf8_data():
    """Tests _decode_hex_identity_dict when value cannot be decoded as utf-8."""
    info_dict = {"data": (b"\xff\xfe",)}
    result = async_subtensor._decode_hex_identity_dict(info_dict)
    assert result["data"] == (b"\xff\xfe",)


def test_decode_hex_identity_dict_with_non_tuple_value():
    """Tests _decode_hex_identity_dict when value is not a tuple."""
    info_dict = {"info": "regular_string"}
    result = async_subtensor._decode_hex_identity_dict(info_dict)
    assert result["info"] == "regular_string"


def test_decode_hex_identity_dict_with_nested_dict():
    """Tests _decode_hex_identity_dict with a nested dictionary."""
    info_dict = {"identity": {"rank": (65, 66, 67)}}
    result = async_subtensor._decode_hex_identity_dict(info_dict)
    assert result["identity"] == "41 4243"


@pytest.mark.asyncio
async def test_init_if_unknown_network_is_valid(mocker):
    """Tests __init__ if passed network unknown and is valid."""
    # Preps
    fake_valid_endpoint = "wss://blabla.net"
    mocker.patch.object(async_subtensor, "AsyncSubstrateInterface")

    # Call
    subtensor = AsyncSubtensor(fake_valid_endpoint)

    # Asserts
    assert subtensor.chain_endpoint == fake_valid_endpoint
    assert subtensor.network == "custom"


@pytest.mark.asyncio
async def test_init_if_unknown_network_is_known_endpoint(mocker):
    """Tests __init__ if passed network unknown and is valid."""
    # Preps
    fake_valid_endpoint = "ws://127.0.0.1:9944"
    mocker.patch.object(async_subtensor, "AsyncSubstrateInterface")

    # Call
    subtensor = AsyncSubtensor(fake_valid_endpoint)

    # Asserts
    assert subtensor.chain_endpoint == fake_valid_endpoint
    assert subtensor.network == "local"


@pytest.mark.asyncio
async def test_init_if_unknown_network_is_not_valid(mocker):
    """Tests __init__ if passed network unknown and isn't valid."""
    # Preps
    mocker.patch.object(async_subtensor, "AsyncSubstrateInterface")

    # Call
    subtensor = AsyncSubtensor("blabla-net")

    # Asserts
    assert (
        subtensor.chain_endpoint
        == async_subtensor.NETWORK_MAP[async_subtensor.DEFAULTS.subtensor.network]
    )
    assert subtensor.network == async_subtensor.DEFAULTS.subtensor.network


def test__str__return(subtensor):
    """Simply tests the result if printing subtensor instance."""
    # Asserts
    assert (
        str(subtensor)
        == f"Network: {async_subtensor.DEFAULTS.subtensor.network}, Chain: {async_subtensor.DEFAULTS.subtensor.chain_endpoint}"
    )


@pytest.mark.asyncio
async def test_async_subtensor_magic_methods(mocker):
    """Tests async magic methods of AsyncSubtensor class."""
    # Preps
    fake_async_substrate = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface
    )
    mocker.patch.object(
        async_subtensor, "AsyncSubstrateInterface", return_value=fake_async_substrate
    )

    # Call
    subtensor = async_subtensor.AsyncSubtensor(network="local")
    async with subtensor:
        pass

    # Asserts
    fake_async_substrate.__aenter__.assert_called_once()
    fake_async_substrate.__aexit__.assert_called_once()
    fake_async_substrate.close.assert_awaited_once()


@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError,
        async_subtensor.ssl.SSLError,
        async_subtensor.TimeoutException,
    ],
)
@pytest.mark.asyncio
async def test_async_subtensor_aenter_connection_refused_error(
    subtensor, mocker, error
):
    """Tests __aenter__ method handling all errors."""
    # Preps
    fake_async_substrate = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface,
        __aenter__=mocker.AsyncMock(side_effect=error),
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
    fake_async_substrate.__aenter__.assert_called_once()


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
    result = await subtensor.get_block_hash(block_id=1)

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
    mocked_query_runtime_api = mocker.AsyncMock(autospec=subtensor.query_runtime_api)
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
    assert result == mocked_substrate_query.return_value
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
    "fake_hex_bytes_result, response", [(None, []), ("0xaabbccdd", b"\xaa\xbb\xcc\xdd")]
)
@pytest.mark.asyncio
async def test_get_delegates(subtensor, mocker, fake_hex_bytes_result, response):
    """Tests get_delegates method."""
    # Preps
    mocked_query_runtime_api = mocker.AsyncMock(
        autospec=subtensor.query_runtime_api, return_value=fake_hex_bytes_result
    )
    subtensor.query_runtime_api = mocked_query_runtime_api
    mocked_delegate_info_list_from_vec_u8 = mocker.Mock()
    async_subtensor.DelegateInfo.list_from_vec_u8 = (
        mocked_delegate_info_list_from_vec_u8
    )

    # Call
    result = await subtensor.get_delegates(block_hash=None, reuse_block=True)

    # Asserts
    if fake_hex_bytes_result:
        assert result == mocked_delegate_info_list_from_vec_u8.return_value
        mocked_delegate_info_list_from_vec_u8.assert_called_once_with(
            bytes.fromhex(fake_hex_bytes_result[2:])
        )
    else:
        assert result == response

    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="DelegateInfoRuntimeApi",
        method="get_delegates",
        params=[],
        block_hash=None,
        reuse_block=True,
    )


@pytest.mark.parametrize(
    "fake_hex_bytes_result, response", [(None, []), ("0x001122", b"\xaa\xbb\xcc\xdd")]
)
@pytest.mark.asyncio
async def test_get_stake_info_for_coldkey(
    subtensor, mocker, fake_hex_bytes_result, response
):
    """Tests get_stake_info_for_coldkey method."""
    # Preps
    fake_coldkey_ss58 = "fake_coldkey_58"

    mocked_ss58_to_vec_u8 = mocker.Mock()
    async_subtensor.ss58_to_vec_u8 = mocked_ss58_to_vec_u8

    mocked_query_runtime_api = mocker.AsyncMock(
        autospec=subtensor.query_runtime_api, return_value=fake_hex_bytes_result
    )
    subtensor.query_runtime_api = mocked_query_runtime_api

    mocked_stake_info_list_from_vec_u8 = mocker.Mock()
    async_subtensor.StakeInfo.list_from_vec_u8 = mocked_stake_info_list_from_vec_u8

    # Call
    result = await subtensor.get_stake_info_for_coldkey(
        coldkey_ss58=fake_coldkey_ss58, block_hash=None, reuse_block=True
    )

    # Asserts
    if fake_hex_bytes_result:
        assert result == mocked_stake_info_list_from_vec_u8.return_value
        mocked_stake_info_list_from_vec_u8.assert_called_once_with(
            bytes.fromhex(fake_hex_bytes_result[2:])
        )
    else:
        assert result == response

    mocked_ss58_to_vec_u8.assert_called_once_with(fake_coldkey_ss58)
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="StakeInfoRuntimeApi",
        method="get_stake_info_for_coldkey",
        params=[mocked_ss58_to_vec_u8.return_value],
        block_hash=None,
        reuse_block=True,
    )


@pytest.mark.asyncio
async def test_get_stake_for_coldkey_and_hotkey(subtensor, mocker):
    """Tests get_stake_for_coldkey_and_hotkey method."""
    # Preps
    mocked_substrate_query = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.query
    )
    subtensor.substrate.query = mocked_substrate_query

    spy_balance = mocker.spy(async_subtensor, "Balance")

    # Call
    result = await subtensor.get_stake_for_coldkey_and_hotkey(
        hotkey_ss58="hotkey", coldkey_ss58="coldkey", block_hash=None
    )

    # Asserts
    mocked_substrate_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Stake",
        params=["hotkey", "coldkey"],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result == spy_balance.from_rao.return_value
    spy_balance.from_rao.assert_called_once_with(mocked_substrate_query.return_value)


@pytest.mark.asyncio
async def test_query_runtime_api(subtensor, mocker):
    """Tests query_runtime_api method."""
    # Preps
    fake_runtime_api = "DelegateInfoRuntimeApi"
    fake_method = "get_delegated"
    fake_params = [1, 2, 3]
    fake_block_hash = None
    reuse_block = False

    mocked_encode_params = mocker.AsyncMock()
    subtensor.encode_params = mocked_encode_params

    mocked_rpc_request = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface.rpc_request
    )
    subtensor.substrate.rpc_request = mocked_rpc_request

    mocked_scalecodec = mocker.Mock(autospec=async_subtensor.scalecodec.ScaleBytes)
    async_subtensor.scalecodec.ScaleBytes = mocked_scalecodec

    mocked_runtime_configuration = mocker.Mock(
        autospec=async_subtensor.RuntimeConfiguration
    )
    async_subtensor.RuntimeConfiguration = mocked_runtime_configuration

    mocked_load_type_registry_preset = mocker.Mock()
    async_subtensor.load_type_registry_preset = mocked_load_type_registry_preset

    # Call
    result = await subtensor.query_runtime_api(
        runtime_api=fake_runtime_api,
        method=fake_method,
        params=fake_params,
        block_hash=fake_block_hash,
        reuse_block=reuse_block,
    )

    # Asserts

    mocked_encode_params.assert_called_once_with(
        call_definition={
            "params": [{"name": "coldkey", "type": "Vec<u8>"}],
            "type": "Vec<u8>",
        },
        params=[1, 2, 3],
    )
    mocked_rpc_request.assert_called_once_with(
        method="state_call",
        params=[f"{fake_runtime_api}_{fake_method}", mocked_encode_params.return_value],
        reuse_block_hash=reuse_block,
    )
    mocked_runtime_configuration.assert_called_once()
    assert (
        mocked_runtime_configuration.return_value.update_type_registry.call_count == 2
    )

    mocked_runtime_configuration.return_value.create_scale_object.assert_called_once_with(
        "Vec<u8>", mocked_scalecodec.return_value
    )

    assert (
        result
        == mocked_runtime_configuration.return_value.create_scale_object.return_value.decode.return_value
    )


@pytest.mark.asyncio
async def test_get_balance(subtensor, mocker):
    """Tests get_balance method."""
    # Preps
    fake_addresses = ("a1", "a2")
    fake_block_hash = None

    mocked_substrate_create_storage_key = mocker.AsyncMock()
    subtensor.substrate.create_storage_key = mocked_substrate_create_storage_key

    mocked_batch_0_call = mocker.Mock(
        params=[
            0,
        ]
    )
    mocked_batch_1_call = {"data": {"free": 1000}}
    mocked_substrate_query_multi = mocker.AsyncMock(
        return_value=[
            (mocked_batch_0_call, mocked_batch_1_call),
        ]
    )

    subtensor.substrate.query_multi = mocked_substrate_query_multi

    # Call
    result = await subtensor.get_balance(*fake_addresses, block_hash=fake_block_hash)

    assert mocked_substrate_create_storage_key.call_count == len(fake_addresses)
    mocked_substrate_query_multi.assert_called_once()
    assert result == {0: async_subtensor.Balance(1000)}


@pytest.mark.parametrize("balance", [100, 100.1])
@pytest.mark.asyncio
async def test_get_transfer_fee(subtensor, mocker, balance):
    """Tests get_transfer_fee method."""
    # Preps
    fake_wallet = mocker.Mock(coldkeypub="coldkeypub", autospec=async_subtensor.Wallet)
    fake_dest = "fake_dest"
    fake_value = balance

    mocked_compose_call = mocker.AsyncMock()
    subtensor.substrate.compose_call = mocked_compose_call

    mocked_get_payment_info = mocker.AsyncMock(return_value={"partialFee": 100})
    subtensor.substrate.get_payment_info = mocked_get_payment_info

    # Call
    result = await subtensor.get_transfer_fee(
        wallet=fake_wallet, dest=fake_dest, value=fake_value
    )

    # Assertions
    mocked_compose_call.assert_awaited_once()
    mocked_compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_allow_death",
        call_params={
            "dest": fake_dest,
            "value": async_subtensor.Balance.from_rao(fake_value),
        },
    )

    assert isinstance(result, async_subtensor.Balance)
    mocked_get_payment_info.assert_awaited_once()
    mocked_get_payment_info.assert_called_once_with(
        call=mocked_compose_call.return_value, keypair="coldkeypub"
    )


@pytest.mark.asyncio
async def test_get_transfer_fee_with_non_balance_accepted_value_type(subtensor, mocker):
    """Tests get_transfer_fee method with non balance accepted value type."""
    # Preps
    fake_wallet = mocker.Mock(coldkeypub="coldkeypub", autospec=async_subtensor.Wallet)
    fake_dest = "fake_dest"
    fake_value = "1000"

    # Call
    result = await subtensor.get_transfer_fee(
        wallet=fake_wallet, dest=fake_dest, value=fake_value
    )

    # Assertions
    assert result == async_subtensor.Balance.from_rao(int(2e7))


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
async def test_get_total_stake_for_coldkey(subtensor, mocker):
    """Tests get_total_stake_for_coldkey method."""
    # Preps
    fake_addresses = ("a1", "a2")
    fake_block_hash = None

    mocked_substrate_create_storage_key = mocker.AsyncMock()
    subtensor.substrate.create_storage_key = mocked_substrate_create_storage_key

    mocked_batch_0_call = mocker.Mock(
        params=[
            0,
        ]
    )
    mocked_batch_1_call = 0
    mocked_substrate_query_multi = mocker.AsyncMock(
        return_value=[
            (mocked_batch_0_call, mocked_batch_1_call),
        ]
    )

    subtensor.substrate.query_multi = mocked_substrate_query_multi

    # Call
    result = await subtensor.get_total_stake_for_coldkey(
        *fake_addresses, block_hash=fake_block_hash
    )

    assert mocked_substrate_create_storage_key.call_count == len(fake_addresses)
    mocked_substrate_query_multi.assert_called_once()
    assert result == {0: async_subtensor.Balance(mocked_batch_1_call)}


@pytest.mark.asyncio
async def test_get_total_stake_for_hotkey(subtensor, mocker):
    """Tests get_total_stake_for_hotkey method."""
    # Preps
    fake_addresses = ("a1", "a2")
    fake_block_hash = None
    reuse_block = True

    mocked_substrate_query_multiple = mocker.AsyncMock(return_value={0: 1})

    subtensor.substrate.query_multiple = mocked_substrate_query_multiple

    # Call
    result = await subtensor.get_total_stake_for_hotkey(
        *fake_addresses, block_hash=fake_block_hash, reuse_block=reuse_block
    )

    # Assertions
    mocked_substrate_query_multiple.assert_called_once_with(
        params=list(fake_addresses),
        module="SubtensorModule",
        storage_function="TotalHotkeyStake",
        block_hash=fake_block_hash,
        reuse_block_hash=reuse_block,
    )
    mocked_substrate_query_multiple.assert_called_once()
    assert result == {0: async_subtensor.Balance(1)}


@pytest.mark.parametrize(
    "records, response",
    [([(0, True), (1, False), (3, False), (3, True)], [0, 3]), ([], [])],
    ids=["with records", "empty-records"],
)
@pytest.mark.asyncio
async def test_get_netuids_for_hotkey(subtensor, mocker, records, response):
    """Tests get_netuids_for_hotkey method."""
    # Preps
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
    assert result == response


@pytest.mark.asyncio
async def test_subnet_exists(subtensor, mocker):
    """Tests subnet_exists method ."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_reuse_block_hash = True

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
    assert result == mocked_substrate_query.return_value


@pytest.mark.asyncio
async def test_get_hyperparameter_happy_path(subtensor, mocker):
    """Tests get_hyperparameter method with happy path."""
    # Preps
    fake_param_name = "param_name"
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_reuse_block_hash = True

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
    assert result == mocked_substrate_query.return_value


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
    fake_wallet_1 = mocker.Mock(autospec=async_subtensor.Wallet)
    fake_wallet_1.hotkey.ss58_address = "ss58_address_1"
    fake_wallet_2 = mocker.Mock(autospec=async_subtensor.Wallet)
    fake_wallet_2.hotkey.ss58_address = "ss58_address_2"

    fake_all_netuids = all_netuids
    fake_filter_for_netuids = filter_for_netuids
    fake_all_hotkeys = [fake_wallet_1, fake_wallet_2]
    fake_block_hash = "fake_block_hash"
    fake_reuse_block = True

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
    fake_reuse_block_hash = True

    mocked_substrate_get_constant = mocker.AsyncMock(return_value=1)
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
        mocked_substrate_get_constant.return_value
    )
    assert result == async_subtensor.Balance(mocked_substrate_get_constant.return_value)


@pytest.mark.asyncio
async def test_get_existential_deposit_raise_exception(subtensor, mocker):
    """Tests get_existential_deposit method raise Exception."""
    # Preps
    fake_block_hash = "block_hash"
    fake_reuse_block_hash = True

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
    fake_reuse_block_hash = True

    mocked_query_runtime_api = mocker.patch.object(
        subtensor, "query_runtime_api", return_value="NOT NONE"
    )
    mocked_hex_to_bytes = mocker.patch.object(async_subtensor, "hex_to_bytes")
    mocked_neuron_info_list_from_vec_u8 = mocker.patch.object(
        async_subtensor.NeuronInfo, "list_from_vec_u8"
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
        block_hash=fake_block_hash,
        reuse_block=fake_reuse_block_hash,
    )
    mocked_hex_to_bytes.assert_called_once_with(mocked_query_runtime_api.return_value)
    assert result == mocked_neuron_info_list_from_vec_u8.return_value


@pytest.mark.parametrize(
    "fake_hex_bytes_result, response",
    [(None, []), ("0xaabbccdd", b"\xaa\xbb\xcc\xdd")],
    ids=["none", "with data"],
)
@pytest.mark.asyncio
async def test_neurons_lite(subtensor, mocker, fake_hex_bytes_result, response):
    """Tests neurons_lite method."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_reuse_block_hash = True

    mocked_query_runtime_api = mocker.AsyncMock(return_value=fake_hex_bytes_result)
    subtensor.query_runtime_api = mocked_query_runtime_api

    mocked_neuron_info_lite_list_from_vec_u8 = mocker.Mock()
    async_subtensor.NeuronInfoLite.list_from_vec_u8 = (
        mocked_neuron_info_lite_list_from_vec_u8
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
        block_hash=fake_block_hash,
        reuse_block=fake_reuse_block_hash,
    )
    if fake_hex_bytes_result:
        mocked_neuron_info_lite_list_from_vec_u8.assert_called_once_with(
            bytes.fromhex(fake_hex_bytes_result[2:])
        )
        assert result == mocked_neuron_info_lite_list_from_vec_u8.return_value
    else:
        mocked_neuron_info_lite_list_from_vec_u8.assert_not_called()
        assert result == []


@pytest.mark.asyncio
async def test_get_neuron_for_pubkey_and_subnet_success(subtensor, mocker):
    """Tests successful retrieval of neuron information."""
    # Preps
    fake_hotkey = "fake_ss58_address"
    fake_netuid = 1
    fake_uid = 123
    fake_result = b"fake_neuron_data"

    mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=fake_uid,
    )
    mocker.patch.object(
        subtensor.substrate,
        "rpc_request",
        return_value={"result": fake_result},
    )
    mocked_neuron_info = mocker.patch.object(
        async_subtensor.NeuronInfo, "from_vec_u8", return_value="fake_neuron_info"
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
    subtensor.substrate.rpc_request.assert_awaited_once()
    subtensor.substrate.rpc_request.assert_called_once_with(
        method="neuronInfo_getNeuron", params=[fake_netuid, fake_uid]
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
        return_value=fake_uid,
    )
    mocker.patch.object(
        subtensor.substrate,
        "rpc_request",
        return_value={"result": None},
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
    subtensor.substrate.rpc_request.assert_called_once_with(
        method="neuronInfo_getNeuron", params=[fake_netuid, fake_uid]
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

    mocked_null_neuron = mocker.Mock()
    async_subtensor.NeuronInfo.get_null_neuron = mocked_null_neuron

    # no result in response
    mocked_substrate_rpc_request = mocker.AsyncMock(
        return_value={"result": b"some_result"}
    )
    subtensor.substrate.rpc_request = mocked_substrate_rpc_request

    mocked_neuron_info_from_vec_u8 = mocker.Mock()
    async_subtensor.NeuronInfo.from_vec_u8 = mocked_neuron_info_from_vec_u8

    # Call
    result = await subtensor.neuron_for_uid(
        uid=fake_uid, netuid=fake_netuid, block_hash=fake_block_hash
    )

    # Asserts
    mocked_null_neuron.assert_not_called()
    mocked_neuron_info_from_vec_u8.assert_called_once_with(
        bytes(mocked_substrate_rpc_request.return_value.get("result"))
    )
    assert result == mocked_neuron_info_from_vec_u8.return_value


@pytest.mark.asyncio
async def test_neuron_for_uid_with_none_uid(subtensor, mocker):
    """Tests neuron_for_uid method when uid is None."""
    # Preps
    fake_uid = None
    fake_netuid = 1
    fake_block_hash = "block_hash"

    mocked_null_neuron = mocker.Mock()
    async_subtensor.NeuronInfo.get_null_neuron = mocked_null_neuron

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

    mocked_null_neuron = mocker.Mock()
    async_subtensor.NeuronInfo.get_null_neuron = mocked_null_neuron

    # no result in response
    mocked_substrate_rpc_request = mocker.AsyncMock(return_value={})
    subtensor.substrate.rpc_request = mocked_substrate_rpc_request

    mocked_neuron_info_from_vec_u8 = mocker.Mock()
    async_subtensor.NeuronInfo.from_vec_u8 = mocked_neuron_info_from_vec_u8

    # Call
    result = await subtensor.neuron_for_uid(
        uid=fake_uid, netuid=fake_netuid, block_hash=fake_block_hash
    )

    # Asserts
    mocked_null_neuron.assert_called_once()
    mocked_neuron_info_from_vec_u8.assert_not_called()
    assert result == mocked_null_neuron.return_value


@pytest.mark.asyncio
async def test_get_delegated_no_block_hash_no_reuse(subtensor, mocker):
    """Tests get_delegated method with no block_hash and reuse_block=False."""
    # Preps
    fake_coldkey_ss58 = "fake_ss58_address"

    mocked_ss58_to_vec_u8 = mocker.Mock(return_value=b"encoded_coldkey")
    mocker.patch.object(async_subtensor, "ss58_to_vec_u8", mocked_ss58_to_vec_u8)

    mocked_rpc_request = mocker.AsyncMock(return_value={"result": b"mocked_result"})
    subtensor.substrate.rpc_request = mocked_rpc_request

    mocked_delegated_list_from_vec_u8 = mocker.Mock()
    async_subtensor.DelegateInfo.delegated_list_from_vec_u8 = (
        mocked_delegated_list_from_vec_u8
    )

    # Call
    result = await subtensor.get_delegated(coldkey_ss58=fake_coldkey_ss58)

    # Asserts
    mocked_ss58_to_vec_u8.assert_called_once_with(fake_coldkey_ss58)
    mocked_rpc_request.assert_called_once_with(
        method="delegateInfo_getDelegated", params=[b"encoded_coldkey"]
    )
    mocked_delegated_list_from_vec_u8.assert_called_once_with(b"mocked_result")
    assert result == mocked_delegated_list_from_vec_u8.return_value


@pytest.mark.asyncio
async def test_get_delegated_with_block_hash(subtensor, mocker):
    """Tests get_delegated method with specified block_hash."""
    # Preps
    fake_coldkey_ss58 = "fake_ss58_address"
    fake_block_hash = "fake_block_hash"

    mocked_ss58_to_vec_u8 = mocker.Mock(return_value=b"encoded_coldkey")
    mocker.patch.object(async_subtensor, "ss58_to_vec_u8", mocked_ss58_to_vec_u8)

    mocked_rpc_request = mocker.AsyncMock(return_value={"result": b"mocked_result"})
    subtensor.substrate.rpc_request = mocked_rpc_request

    mocked_delegated_list_from_vec_u8 = mocker.Mock()
    async_subtensor.DelegateInfo.delegated_list_from_vec_u8 = (
        mocked_delegated_list_from_vec_u8
    )

    # Call
    result = await subtensor.get_delegated(
        coldkey_ss58=fake_coldkey_ss58, block_hash=fake_block_hash
    )

    # Asserts
    mocked_ss58_to_vec_u8.assert_called_once_with(fake_coldkey_ss58)
    mocked_rpc_request.assert_called_once_with(
        method="delegateInfo_getDelegated", params=[fake_block_hash, b"encoded_coldkey"]
    )
    mocked_delegated_list_from_vec_u8.assert_called_once_with(b"mocked_result")
    assert result == mocked_delegated_list_from_vec_u8.return_value


@pytest.mark.asyncio
async def test_get_delegated_with_reuse_block(subtensor, mocker):
    """Tests get_delegated method with reuse_block=True."""
    # Preps
    fake_coldkey_ss58 = "fake_ss58_address"
    subtensor.substrate.last_block_hash = "last_block_hash"

    mocked_ss58_to_vec_u8 = mocker.Mock(return_value=b"encoded_coldkey")
    mocker.patch.object(async_subtensor, "ss58_to_vec_u8", mocked_ss58_to_vec_u8)

    mocked_rpc_request = mocker.AsyncMock(return_value={"result": b"mocked_result"})
    subtensor.substrate.rpc_request = mocked_rpc_request

    mocked_delegated_list_from_vec_u8 = mocker.Mock()
    async_subtensor.DelegateInfo.delegated_list_from_vec_u8 = (
        mocked_delegated_list_from_vec_u8
    )

    # Call
    result = await subtensor.get_delegated(
        coldkey_ss58=fake_coldkey_ss58, reuse_block=True
    )

    # Asserts
    mocked_ss58_to_vec_u8.assert_called_once_with(fake_coldkey_ss58)
    mocked_rpc_request.assert_called_once_with(
        method="delegateInfo_getDelegated",
        params=["last_block_hash", b"encoded_coldkey"],
    )
    mocked_delegated_list_from_vec_u8.assert_called_once_with(b"mocked_result")
    assert result == mocked_delegated_list_from_vec_u8.return_value


@pytest.mark.asyncio
async def test_get_delegated_with_empty_result(subtensor, mocker):
    """Tests get_delegated method when RPC request returns an empty result."""
    # Preps
    fake_coldkey_ss58 = "fake_ss58_address"

    mocked_ss58_to_vec_u8 = mocker.Mock(return_value=b"encoded_coldkey")
    mocker.patch.object(async_subtensor, "ss58_to_vec_u8", mocked_ss58_to_vec_u8)

    mocked_rpc_request = mocker.AsyncMock(return_value={})
    subtensor.substrate.rpc_request = mocked_rpc_request

    # Call
    result = await subtensor.get_delegated(coldkey_ss58=fake_coldkey_ss58)

    # Asserts
    mocked_ss58_to_vec_u8.assert_called_once_with(fake_coldkey_ss58)
    mocked_rpc_request.assert_called_once_with(
        method="delegateInfo_getDelegated", params=[b"encoded_coldkey"]
    )
    assert result == []


@pytest.mark.asyncio
async def test_query_identity_successful(subtensor, mocker):
    """Tests query_identity method with successful identity query."""
    # Preps
    fake_key = "test_key"
    fake_block_hash = "block_hash"
    fake_identity_info = {"info": {"stake": (b"\x01\x02",)}}

    mocked_query = mocker.AsyncMock(return_value=fake_identity_info)
    subtensor.substrate.query = mocked_query

    mocker.patch.object(
        async_subtensor,
        "_decode_hex_identity_dict",
        return_value={"stake": "01 02"},
    )

    # Call
    result = await subtensor.query_identity(key=fake_key, block_hash=fake_block_hash)

    # Asserts
    mocked_query.assert_called_once_with(
        module="Registry",
        storage_function="IdentityOf",
        params=[fake_key],
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    assert result == {"stake": "01 02"}


@pytest.mark.asyncio
async def test_query_identity_no_info(subtensor, mocker):
    """Tests query_identity method when no identity info is returned."""
    # Preps
    fake_key = "test_key"

    mocked_query = mocker.AsyncMock(return_value=None)
    subtensor.substrate.query = mocked_query

    # Call
    result = await subtensor.query_identity(key=fake_key)

    # Asserts
    mocked_query.assert_called_once_with(
        module="Registry",
        storage_function="IdentityOf",
        params=[fake_key],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result == {}


@pytest.mark.asyncio
async def test_query_identity_type_error(subtensor, mocker):
    """Tests query_identity method when a TypeError occurs during decoding."""
    # Preps
    fake_key = "test_key"
    fake_identity_info = {"info": {"rank": (b"\xff\xfe",)}}

    mocked_query = mocker.AsyncMock(return_value=fake_identity_info)
    subtensor.substrate.query = mocked_query

    mocker.patch.object(
        async_subtensor,
        "_decode_hex_identity_dict",
        side_effect=TypeError,
    )

    # Call
    result = await subtensor.query_identity(key=fake_key)

    # Asserts
    mocked_query.assert_called_once_with(
        module="Registry",
        storage_function="IdentityOf",
        params=[fake_key],
        block_hash=None,
        reuse_block_hash=False,
    )
    assert result == {}


@pytest.mark.asyncio
async def test_weights_successful(subtensor, mocker):
    """Tests weights method with successful weight distribution retrieval."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_weights = [
        (0, [(1, 10), (2, 20)]),
        (1, [(0, 15), (2, 25)]),
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
    assert result == fake_weights


@pytest.mark.asyncio
async def test_bonds(subtensor, mocker):
    """Tests bonds method with successful bond distribution retrieval."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_bonds = [
        (0, [(1, 100), (2, 200)]),
        (1, [(0, 150), (2, 250)]),
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
    assert result == fake_bonds


@pytest.mark.asyncio
async def test_does_hotkey_exist_true(subtensor, mocker):
    """Tests does_hotkey_exist method when the hotkey exists and is valid."""
    # Preps
    fake_hotkey_ss58 = "valid_hotkey"
    fake_block_hash = "block_hash"
    fake_query_result = ["decoded_account_id"]

    mocked_query = mocker.AsyncMock(return_value=fake_query_result)
    subtensor.substrate.query = mocked_query

    mocked_decode_account_id = mocker.Mock(return_value="another_account_id")
    mocker.patch.object(async_subtensor, "decode_account_id", mocked_decode_account_id)

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
    mocked_decode_account_id.assert_called_once_with(fake_query_result[0])
    assert result is True


@pytest.mark.asyncio
async def test_does_hotkey_exist_false_for_specific_account(subtensor, mocker):
    """Tests does_hotkey_exist method when the hotkey exists but matches the specific account ID to ignore."""
    # Preps
    fake_hotkey_ss58 = "ignored_hotkey"
    fake_query_result = ["ignored_account_id"]

    mocked_query = mocker.AsyncMock(return_value=fake_query_result)
    subtensor.substrate.query = mocked_query

    # Mock the decode_account_id function to return the specific account ID that should be ignored
    mocked_decode_account_id = mocker.Mock(
        return_value="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    )
    mocker.patch.object(async_subtensor, "decode_account_id", mocked_decode_account_id)

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
    mocked_decode_account_id.assert_called_once_with(fake_query_result[0])
    assert result is False


@pytest.mark.asyncio
async def test_get_hotkey_owner_successful(subtensor, mocker):
    """Tests get_hotkey_owner method when the hotkey exists and has an owner."""
    # Preps
    fake_hotkey_ss58 = "valid_hotkey"
    fake_block_hash = "block_hash"
    fake_owner_account_id = "owner_account_id"

    mocked_query = mocker.AsyncMock(return_value=[fake_owner_account_id])
    subtensor.substrate.query = mocked_query

    mocked_decode_account_id = mocker.Mock(return_value="decoded_owner_account_id")
    mocker.patch.object(async_subtensor, "decode_account_id", mocked_decode_account_id)

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
    mocked_decode_account_id.assert_called_once_with(fake_owner_account_id)
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

    mocked_query = mocker.AsyncMock(return_value=[None])
    subtensor.substrate.query = mocked_query

    mocked_decode_account_id = mocker.Mock(return_value=None)
    mocker.patch.object(async_subtensor, "decode_account_id", mocked_decode_account_id)

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
    mocked_decode_account_id.assert_called_once_with(None)
    assert result is None


@pytest.mark.asyncio
async def test_get_hotkey_owner_exists_but_does_not_exist_flag_false(subtensor, mocker):
    """Tests get_hotkey_owner method when decode_account_id returns a value but does_hotkey_exist returns False."""
    # Preps
    fake_hotkey_ss58 = "valid_hotkey"
    fake_block_hash = "block_hash"
    fake_owner_account_id = "owner_account_id"

    mocked_query = mocker.AsyncMock(return_value=[fake_owner_account_id])
    subtensor.substrate.query = mocked_query

    mocked_decode_account_id = mocker.Mock(return_value="decoded_owner_account_id")
    mocker.patch.object(async_subtensor, "decode_account_id", mocked_decode_account_id)

    mocked_does_hotkey_exist = mocker.AsyncMock(return_value=False)
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
    mocked_decode_account_id.assert_called_once_with(fake_owner_account_id)
    mocked_does_hotkey_exist.assert_awaited_once_with(
        fake_hotkey_ss58, block_hash=fake_block_hash
    )
    assert result is None


@pytest.mark.asyncio
async def test_sign_and_send_extrinsic_success_finalization(subtensor, mocker):
    """Tests sign_and_send_extrinsic when the extrinsic is successfully finalized."""
    # Preps
    fake_call = mocker.Mock()
    fake_wallet = mocker.Mock()
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
    fake_response.process_events.assert_awaited_once()
    assert result == (True, "")


@pytest.mark.asyncio
async def test_sign_and_send_extrinsic_error_finalization(subtensor, mocker):
    """Tests sign_and_send_extrinsic when the extrinsic is error finalized."""
    # Preps
    fake_call = mocker.Mock()
    fake_wallet = mocker.Mock()
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
    fake_response.process_events.assert_awaited_once()
    assert result == (False, mocked_format_error_message.return_value)


@pytest.mark.asyncio
async def test_sign_and_send_extrinsic_success_without_inclusion_finalization(
    subtensor, mocker
):
    """Tests sign_and_send_extrinsic when extrinsic is submitted without waiting for inclusion or finalization."""
    # Preps
    fake_call = mocker.Mock()
    fake_wallet = mocker.Mock()
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
    assert result == (True, "")


@pytest.mark.asyncio
async def test_sign_and_send_extrinsic_substrate_request_exception(subtensor, mocker):
    """Tests sign_and_send_extrinsic when SubstrateRequestException is raised."""
    # Preps
    fake_call = mocker.Mock()
    fake_wallet = mocker.Mock()
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
async def test_get_children_success(subtensor, mocker):
    """Tests get_children when children are successfully retrieved and formatted."""
    # Preps
    fake_hotkey = "valid_hotkey"
    fake_netuid = 1
    fake_children = [
        (1000, ["child_key_1"]),
        (2000, ["child_key_2"]),
    ]

    mocked_query = mocker.AsyncMock(return_value=fake_children)
    subtensor.substrate.query = mocked_query

    mocked_decode_account_id = mocker.Mock(
        side_effect=["decoded_child_key_1", "decoded_child_key_2"]
    )
    mocker.patch.object(async_subtensor, "decode_account_id", mocked_decode_account_id)

    expected_formatted_children = [
        (1000, "decoded_child_key_1"),
        (2000, "decoded_child_key_2"),
    ]

    # Call
    result = await subtensor.get_children(hotkey=fake_hotkey, netuid=fake_netuid)

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="ChildKeys",
        params=[fake_hotkey, fake_netuid],
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
        module="SubtensorModule",
        storage_function="ChildKeys",
        params=[fake_hotkey, fake_netuid],
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
        module="SubtensorModule",
        storage_function="ChildKeys",
        params=[fake_hotkey, fake_netuid],
    )
    mocked_format_error_message.assert_called_once_with(fake_exception)
    assert result == (False, [], "Formatted error message")


@pytest.mark.asyncio
async def test_get_subnet_hyperparameters_success(subtensor, mocker):
    """Tests get_subnet_hyperparameters with successful hyperparameter retrieval."""
    # Preps
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_hex_bytes_result = "0xaabbccdd"

    mocked_query_runtime_api = mocker.AsyncMock(return_value=fake_hex_bytes_result)
    subtensor.query_runtime_api = mocked_query_runtime_api

    mocked_from_vec_u8 = mocker.Mock()
    mocker.patch.object(
        async_subtensor.SubnetHyperparameters, "from_vec_u8", mocked_from_vec_u8
    )

    # Call
    result = await subtensor.get_subnet_hyperparameters(
        netuid=fake_netuid, block_hash=fake_block_hash
    )

    # Asserts
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams",
        params=[fake_netuid],
        block_hash=fake_block_hash,
        reuse_block=False,
    )
    bytes_result = bytes.fromhex(fake_hex_bytes_result[2:])
    mocked_from_vec_u8.assert_called_once_with(bytes_result)
    assert result == mocked_from_vec_u8.return_value


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
        method="get_subnet_hyperparams",
        params=[fake_netuid],
        block_hash=None,
        reuse_block=False,
    )
    assert result == []


@pytest.mark.asyncio
async def test_get_subnet_hyperparameters_without_0x_prefix(subtensor, mocker):
    """Tests get_subnet_hyperparameters when hex_bytes_result is without 0x prefix."""
    # Preps
    fake_netuid = 1
    fake_hex_bytes_result = "aabbccdd"  # without "0x" prefix

    mocked_query_runtime_api = mocker.AsyncMock(return_value=fake_hex_bytes_result)
    subtensor.query_runtime_api = mocked_query_runtime_api

    mocked_from_vec_u8 = mocker.Mock()
    mocker.patch.object(
        async_subtensor.SubnetHyperparameters, "from_vec_u8", mocked_from_vec_u8
    )

    # Call
    result = await subtensor.get_subnet_hyperparameters(netuid=fake_netuid)

    # Asserts
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams",
        params=[fake_netuid],
        block_hash=None,
        reuse_block=False,
    )
    bytes_result = bytes.fromhex(fake_hex_bytes_result)
    mocked_from_vec_u8.assert_called_once_with(bytes_result)
    assert result == mocked_from_vec_u8.return_value


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
        async_subtensor, "ProposalVoteData", return_value=mocked_proposal_vote_data
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
        (["delegate1_ss58"], {"info": {"name": "Chain Delegate 1"}}),
        (["delegate2_ss58"], {"info": {"name": "Chain Delegate 2"}}),
    ]
    fake_github_data = {
        "delegate1_ss58": {
            "name": "GitHub Delegate 1",
            "url": "https://delegate1.com",
            "description": "GitHub description 1",
            "fingerprint": "fingerprint1",
        },
        "delegate3_ss58": {
            "name": "GitHub Delegate 3",
            "url": "https://delegate3.com",
            "description": "GitHub description 3",
            "fingerprint": "fingerprint3",
        },
    }

    mocked_query_map = mocker.AsyncMock(return_value=fake_chain_data)
    subtensor.substrate.query_map = mocked_query_map

    mocked_decode_account_id = mocker.Mock(side_effect=lambda ss58: ss58)
    mocker.patch.object(async_subtensor, "decode_account_id", mocked_decode_account_id)

    mocked_decode_hex_identity_dict = mocker.Mock(side_effect=lambda data: data)
    mocker.patch.object(
        async_subtensor, "decode_hex_identity_dict", mocked_decode_hex_identity_dict
    )

    mock_response = mocker.Mock()
    mock_response.ok = True
    mock_response.json = mocker.AsyncMock(return_value=fake_github_data)

    mock_session_get = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("aiohttp.ClientSession.get", mock_session_get)

    # Call
    result = await subtensor.get_delegate_identities(block_hash=fake_block_hash)

    # Asserts
    mocked_query_map.assert_called_once_with(
        module="Registry",
        storage_function="IdentityOf",
        block_hash=fake_block_hash,
        reuse_block_hash=False,
    )
    mock_session_get.assert_called_once_with(async_subtensor.DELEGATES_DETAILS_URL)

    assert result["delegate1_ss58"].display == "GitHub Delegate 1"
    assert result["delegate2_ss58"].display == ""
    assert result["delegate3_ss58"].display == "GitHub Delegate 3"


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

    mocked_get_hyperparameter = mocker.AsyncMock(return_value=fake_rate_limit)
    subtensor.get_hyperparameter = mocked_get_hyperparameter

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

    mocked_get_hyperparameter = mocker.AsyncMock(return_value=fake_result)
    subtensor.get_hyperparameter = mocked_get_hyperparameter

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

    mocked_get_hyperparameter = mocker.AsyncMock(
        return_value={fake_uid: last_update_block}
    )
    subtensor.get_hyperparameter = mocked_get_hyperparameter

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

    mocked_get_hyperparameter = mocker.AsyncMock(return_value=fake_result)
    subtensor.get_hyperparameter = mocked_get_hyperparameter

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
    result = await subtensor.commit_reveal_enabled(netuid, block_hash)

    # Assertions
    mocked_get_hyperparameter.assert_awaited_once_with(
        param_name="CommitRevealWeightsEnabled", block_hash=block_hash, netuid=netuid
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
    result = await subtensor.get_subnet_reveal_period_epochs(netuid, block_hash)

    # Assertions
    mocked_get_hyperparameter.assert_awaited_once_with(
        param_name="RevealPeriodEpochs", block_hash=block_hash, netuid=netuid
    )
    assert result == mocked_get_hyperparameter.return_value


@pytest.mark.asyncio
async def test_transfer_success(subtensor, mocker):
    """Tests transfer when the transfer is successful."""
    # Preps
    fake_wallet = mocker.Mock()
    fake_destination = "destination_address"
    fake_amount = 100.0
    fake_transfer_all = False

    mocked_transfer_extrinsic = mocker.AsyncMock(return_value=True)
    mocker.patch.object(
        async_subtensor, "transfer_extrinsic", mocked_transfer_extrinsic
    )

    mocked_balance_from_tao = mocker.Mock()
    mocker.patch.object(
        async_subtensor.Balance, "from_tao", return_value=mocked_balance_from_tao
    )

    # Call
    result = await subtensor.transfer(
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        transfer_all=fake_transfer_all,
    )

    # Asserts
    mocked_transfer_extrinsic.assert_awaited_once()
    mocked_transfer_extrinsic.assert_called_once_with(
        subtensor,
        fake_wallet,
        fake_destination,
        mocked_balance_from_tao,
        fake_transfer_all,
    )
    assert result == mocked_transfer_extrinsic.return_value


@pytest.mark.asyncio
async def test_register_success(subtensor, mocker):
    """Tests register when there is enough balance and registration succeeds."""
    # Preps
    fake_wallet = mocker.Mock(autospec=async_subtensor.Wallet)
    fake_wallet.coldkeypub.ss58_address = "wallet_address"
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_recycle_amount = 100
    fake_balance = 200

    mocked_get_block_hash = mocker.AsyncMock(return_value=fake_block_hash)
    subtensor.get_block_hash = mocked_get_block_hash

    mocked_get_hyperparameter = mocker.AsyncMock(return_value=str(fake_recycle_amount))
    subtensor.get_hyperparameter = mocked_get_hyperparameter

    mocked_get_balance = mocker.AsyncMock(
        return_value={fake_wallet.coldkeypub.ss58_address: fake_balance}
    )
    subtensor.get_balance = mocked_get_balance

    mocked_balance_from_rao = mocker.Mock(return_value=fake_recycle_amount)
    mocker.patch.object(async_subtensor.Balance, "from_rao", mocked_balance_from_rao)

    # Call
    result = await subtensor.register(wallet=fake_wallet, netuid=fake_netuid)

    # Asserts
    mocked_get_block_hash.assert_called_once()
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="Burn", netuid=fake_netuid, reuse_block=True
    )
    mocked_get_balance.assert_called_once_with(
        fake_wallet.coldkeypub.ss58_address, block_hash=fake_block_hash
    )
    assert result is True


@pytest.mark.asyncio
async def test_register_insufficient_balance(subtensor, mocker):
    """Tests register when the wallet balance is insufficient."""
    # Preps
    fake_wallet = mocker.Mock(autospec=async_subtensor.Wallet)
    fake_wallet.coldkeypub.ss58_address = "wallet_address"
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_recycle_amount = 200
    fake_balance = 100

    mocked_get_block_hash = mocker.AsyncMock(return_value=fake_block_hash)
    subtensor.get_block_hash = mocked_get_block_hash

    mocked_get_hyperparameter = mocker.AsyncMock(return_value=str(fake_recycle_amount))
    subtensor.get_hyperparameter = mocked_get_hyperparameter

    mocked_get_balance = mocker.AsyncMock(
        return_value={fake_wallet.coldkeypub.ss58_address: fake_balance}
    )
    subtensor.get_balance = mocked_get_balance

    mocked_balance_from_rao = mocker.Mock(return_value=fake_recycle_amount)
    mocker.patch.object(async_subtensor.Balance, "from_rao", mocked_balance_from_rao)

    # Call
    result = await subtensor.register(wallet=fake_wallet, netuid=fake_netuid)

    # Asserts
    mocked_get_block_hash.assert_called_once()
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="Burn", netuid=fake_netuid, reuse_block=True
    )
    mocked_get_balance.assert_called_once_with(
        fake_wallet.coldkeypub.ss58_address, block_hash=fake_block_hash
    )
    assert result is False


@pytest.mark.asyncio
async def test_register_balance_retrieval_error(subtensor, mocker):
    """Tests register when there is an error retrieving the balance."""
    # Preps
    fake_wallet = mocker.Mock(autospec=async_subtensor.Wallet)
    fake_wallet.coldkeypub.ss58_address = "wallet_address"
    fake_netuid = 1
    fake_block_hash = "block_hash"
    fake_recycle_amount = 100

    mocked_get_block_hash = mocker.AsyncMock(return_value=fake_block_hash)
    subtensor.get_block_hash = mocked_get_block_hash

    mocked_get_hyperparameter = mocker.AsyncMock(return_value=str(fake_recycle_amount))
    subtensor.get_hyperparameter = mocked_get_hyperparameter

    mocked_get_balance = mocker.AsyncMock(return_value={})
    subtensor.get_balance = mocked_get_balance

    # Call
    result = await subtensor.register(wallet=fake_wallet, netuid=fake_netuid)

    # Asserts
    mocked_get_block_hash.assert_called_once()
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="Burn", netuid=fake_netuid, reuse_block=True
    )
    mocked_get_balance.assert_called_once_with(
        fake_wallet.coldkeypub.ss58_address, block_hash=fake_block_hash
    )
    assert result is False


@pytest.mark.asyncio
async def test_pow_register_success(subtensor, mocker):
    """Tests pow_register when the registration is successful."""
    # Preps
    fake_wallet = mocker.Mock(autospec=async_subtensor.Wallet)
    fake_netuid = 1
    fake_processors = 4
    fake_update_interval = 10
    fake_output_in_place = True
    fake_verbose = True
    fake_use_cuda = False
    fake_dev_id = 0
    fake_threads_per_block = 128

    mocked_register_extrinsic = mocker.AsyncMock(return_value=True)
    mocker.patch.object(
        async_subtensor, "register_extrinsic", mocked_register_extrinsic
    )

    # Call
    result = await subtensor.pow_register(
        wallet=fake_wallet,
        netuid=fake_netuid,
        processors=fake_processors,
        update_interval=fake_update_interval,
        output_in_place=fake_output_in_place,
        verbose=fake_verbose,
        use_cuda=fake_use_cuda,
        dev_id=fake_dev_id,
        threads_per_block=fake_threads_per_block,
    )

    # Asserts
    mocked_register_extrinsic.assert_awaited_once()
    mocked_register_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        tpb=fake_threads_per_block,
        update_interval=fake_update_interval,
        num_processes=fake_processors,
        cuda=fake_use_cuda,
        dev_id=fake_dev_id,
        output_in_place=fake_output_in_place,
        log_verbose=fake_verbose,
    )
    assert result == mocked_register_extrinsic.return_value


@pytest.mark.asyncio
async def test_set_weights_success(subtensor, mocker):
    """Tests set_weights with successful weight setting on the first try."""
    # Preps
    fake_wallet = mocker.Mock(autospec=async_subtensor.Wallet)
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.3, 0.5, 0.2]
    max_retries = 1

    mocked_get_uid_for_hotkey_on_subnet = mocker.AsyncMock(return_value=fake_netuid)
    subtensor.get_uid_for_hotkey_on_subnet = mocked_get_uid_for_hotkey_on_subnet

    mocked_blocks_since_last_update = mocker.AsyncMock(return_value=2)
    subtensor.blocks_since_last_update = mocked_blocks_since_last_update

    mocked_weights_rate_limit = mocker.AsyncMock(return_value=1)
    subtensor.weights_rate_limit = mocked_weights_rate_limit

    mocked_set_weights_extrinsic = mocker.AsyncMock(return_value=(True, "Success"))
    mocker.patch.object(
        async_subtensor, "set_weights_extrinsic", mocked_set_weights_extrinsic
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
        wait_for_finalization=False,
        wait_for_inclusion=False,
        weights=fake_weights,
    )
    mocked_weights_rate_limit.assert_called_once_with(fake_netuid)
    assert result is True
    assert message == "Success"


@pytest.mark.asyncio
async def test_set_weights_with_exception(subtensor, mocker):
    """Tests set_weights when set_weights_extrinsic raises an exception."""
    # Preps
    fake_wallet = mocker.Mock(autospec=async_subtensor.Wallet)
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
        async_subtensor, "set_weights_extrinsic", mocked_set_weights_extrinsic
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
async def test_root_set_weights_success(subtensor, mocker):
    """Tests root_set_weights when the setting of weights is successful."""
    # Preps
    fake_wallet = mocker.Mock(autospec=async_subtensor.Wallet)
    fake_netuids = [1, 2, 3]
    fake_weights = [0.3, 0.5, 0.2]

    mocked_set_root_weights_extrinsic = mocker.AsyncMock()
    mocker.patch.object(
        async_subtensor, "set_root_weights_extrinsic", mocked_set_root_weights_extrinsic
    )

    mocked_np_array_netuids = mocker.Mock(autospec=async_subtensor.np.ndarray)
    mocked_np_array_weights = mocker.Mock(autospec=async_subtensor.np.ndarray)
    mocker.patch.object(
        async_subtensor.np,
        "array",
        side_effect=[mocked_np_array_netuids, mocked_np_array_weights],
    )

    # Call
    result = await subtensor.root_set_weights(
        wallet=fake_wallet,
        netuids=fake_netuids,
        weights=fake_weights,
    )

    # Asserts
    mocked_set_root_weights_extrinsic.assert_awaited_once()
    mocked_set_root_weights_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuids=mocked_np_array_netuids,
        weights=mocked_np_array_weights,
        version_key=0,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    assert result == mocked_set_root_weights_extrinsic.return_value


@pytest.mark.asyncio
async def test_commit_weights_success(subtensor, mocker):
    """Tests commit_weights when the weights are committed successfully."""
    # Preps
    fake_wallet = mocker.Mock(autospec=async_subtensor.Wallet)
    fake_netuid = 1
    fake_salt = [12345, 67890]
    fake_uids = [1, 2, 3]
    fake_weights = [100, 200, 300]
    max_retries = 3

    mocked_generate_weight_hash = mocker.Mock(return_value="fake_commit_hash")
    mocker.patch.object(
        async_subtensor, "generate_weight_hash", mocked_generate_weight_hash
    )

    mocked_commit_weights_extrinsic = mocker.AsyncMock(return_value=(True, "Success"))
    mocker.patch.object(
        async_subtensor, "commit_weights_extrinsic", mocked_commit_weights_extrinsic
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
    mocked_generate_weight_hash.assert_called_once_with(
        address=fake_wallet.hotkey.ss58_address,
        netuid=fake_netuid,
        uids=fake_uids,
        values=fake_weights,
        salt=fake_salt,
        version_key=async_subtensor.version_as_int,
    )
    mocked_commit_weights_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        commit_hash="fake_commit_hash",
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )
    assert result is True
    assert message == "Success"


@pytest.mark.asyncio
async def test_commit_weights_with_exception(subtensor, mocker):
    """Tests commit_weights when an exception is raised during weight commitment."""
    # Preps
    fake_wallet = mocker.Mock(autospec=async_subtensor.Wallet)
    fake_netuid = 1
    fake_salt = [12345, 67890]
    fake_uids = [1, 2, 3]
    fake_weights = [100, 200, 300]
    max_retries = 1

    mocked_generate_weight_hash = mocker.Mock(return_value="fake_commit_hash")
    mocker.patch.object(
        async_subtensor, "generate_weight_hash", mocked_generate_weight_hash
    )

    mocked_commit_weights_extrinsic = mocker.AsyncMock(
        side_effect=Exception("Test exception")
    )
    mocker.patch.object(
        async_subtensor, "commit_weights_extrinsic", mocked_commit_weights_extrinsic
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
