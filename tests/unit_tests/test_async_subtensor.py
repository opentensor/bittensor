import pytest

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


def test__str__return(subtensor):
    """Simply tests the result if printing subtensor instance."""
    # Asserts
    assert (
        str(subtensor)
        == "Network: finney, Chain: wss://entrypoint-finney.opentensor.ai:443"
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
        reuse_block_hash=True,
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
    "fake_hex_bytes_result, response", [(None, []), ("zz001122", b"\xaa\xbb\xcc\xdd")]
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
    fake_neurons = [mocker.Mock(), mocker.Mock()]
    fake_weights = [(1, [(10, 20), (30, 40)]), (2, [(50, 60), (70, 80)])]
    fake_bonds = [(1, [(10, 20), (30, 40)]), (2, [(50, 60), (70, 80)])]

    mocked_neurons_lite = mocker.AsyncMock(return_value=fake_neurons)
    subtensor.neurons_lite = mocked_neurons_lite

    mocked_weights = mocker.AsyncMock(return_value=fake_weights)
    subtensor.weights = mocked_weights

    mocked_bonds = mocker.AsyncMock(return_value=fake_bonds)
    subtensor.bonds = mocked_bonds

    mocked_neuron_info_method = mocker.Mock()
    async_subtensor.NeuronInfo.from_weights_bonds_and_neuron_lite = (
        mocked_neuron_info_method
    )

    # Call
    result = await subtensor.neurons(netuid=fake_netuid, block_hash=fake_block_hash)

    # Asserts
    mocked_neurons_lite.assert_awaited_once()
    mocked_neurons_lite.assert_called_once_with(
        netuid=fake_netuid, block_hash=fake_block_hash
    )
    mocked_weights.assert_awaited_once()
    mocked_weights.assert_called_once_with(
        netuid=fake_netuid, block_hash=fake_block_hash
    )
    mocked_bonds.assert_awaited_once()
    mocked_bonds.assert_called_once_with(netuid=fake_netuid, block_hash=fake_block_hash)
    assert result == [
        mocked_neuron_info_method.return_value for _ in range(len(fake_neurons))
    ]


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
    )
    mocked_decode_account_id.assert_called_once_with(fake_owner_account_id)
    mocked_does_hotkey_exist.assert_awaited_once_with(
        fake_hotkey_ss58, block_hash=fake_block_hash
    )
    assert result is None
