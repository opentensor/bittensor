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
