import pytest
from substrateinterface import SubstrateInterface

from bittensor.core import async_subtensor


@pytest.fixture
def subtensor(mocker):
    fake_async_substrate = mocker.AsyncMock(
        spec=async_subtensor.AsyncSubstrateInterface
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


@pytest.mark.asyncio
async def test_encode_params(subtensor, mocker):
    """Tests encode_params happy path."""
    # Preps
    subtensor.substrate.create_scale_object = mocker.AsyncMock()
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
    subtensor.substrate.create_scale_object = mocker.AsyncMock()
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
    mocked_get_netuids_for_hotkey = mocker.AsyncMock(return_value=[1, 2])
    subtensor.get_netuids_for_hotkey = mocked_get_netuids_for_hotkey

    # Call
    result = await subtensor.is_hotkey_registered_any(
        hotkey_ss58="hotkey", block_hash="FAKE_HASH"
    )

    # Asserts
    assert result == (len(mocked_get_netuids_for_hotkey.return_value) > 0)

