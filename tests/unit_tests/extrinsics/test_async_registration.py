import pytest
from bittensor_wallet import Wallet

from bittensor.core import async_subtensor
from bittensor.core.extrinsics import async_registration


@pytest.fixture(autouse=True)
def subtensor(mocker):
    fake_async_substrate = mocker.AsyncMock(
        autospec=async_subtensor.AsyncSubstrateInterface
    )
    mocker.patch.object(
        async_subtensor, "AsyncSubstrateInterface", return_value=fake_async_substrate
    )
    return async_subtensor.AsyncSubtensor()


@pytest.mark.asyncio
async def test_do_pow_register_success(subtensor, mocker):
    """Tests successful PoW registration."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkeypub.ss58_address = "coldkey_ss58"
    fake_pow_result = mocker.Mock(
        block_number=12345,
        nonce=67890,
        seal=b"fake_seal",
    )

    fake_call = mocker.AsyncMock()
    fake_extrinsic = mocker.AsyncMock()
    fake_response = mocker.Mock()
    fake_response.is_success = mocker.AsyncMock(return_value=True)()
    fake_response.process_events = mocker.AsyncMock()

    mocker.patch.object(subtensor.substrate, "compose_call", return_value=fake_call)
    mocker.patch.object(
        subtensor.substrate, "create_signed_extrinsic", return_value=fake_extrinsic
    )
    mocker.patch.object(
        subtensor.substrate, "submit_extrinsic", return_value=fake_response
    )

    # Call
    result, error_message = await async_registration._do_pow_register(
        subtensor=subtensor,
        netuid=1,
        wallet=fake_wallet,
        pow_result=fake_pow_result,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    subtensor.substrate.compose_call.asseert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="register",
        call_params={
            "netuid": 1,
            "block_number": 12345,
            "nonce": 67890,
            "work": [int(byte_) for byte_ in b"fake_seal"],
            "hotkey": "hotkey_ss58",
            "coldkey": "coldkey_ss58",
        },
    )
    subtensor.substrate.create_signed_extrinsic.asseert_awaited_once_with(
        call=fake_call, keypair=fake_wallet.hotkey
    )
    subtensor.substrate.submit_extrinsic.asseert_awaited_once_with(
        fake_extrinsic, wait_for_inclusion=True, wait_for_finalization=True
    )
    fake_response.process_events.assert_called_once()
    assert result is True
    assert error_message is None


@pytest.mark.asyncio
async def test_do_pow_register_failure(subtensor, mocker):
    """Tests failed PoW registration."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkeypub.ss58_address = "coldkey_ss58"
    fake_pow_result = mocker.Mock(
        block_number=12345,
        nonce=67890,
        seal=b"fake_seal",
    )
    fake_err_message = mocker.Mock(autospec=str)

    fake_call = mocker.AsyncMock()
    fake_extrinsic = mocker.AsyncMock()
    fake_response = mocker.Mock()
    fake_response.is_success = mocker.AsyncMock(return_value=False)()
    fake_response.process_events = mocker.AsyncMock()
    fake_response.error_message = mocker.AsyncMock(return_value=fake_err_message)()

    mocker.patch.object(subtensor.substrate, "compose_call", return_value=fake_call)
    mocker.patch.object(
        subtensor.substrate, "create_signed_extrinsic", return_value=fake_extrinsic
    )
    mocker.patch.object(
        subtensor.substrate, "submit_extrinsic", return_value=fake_response
    )
    mocked_format_error_message = mocker.patch.object(async_registration, "format_error_message")

    # Call
    result_error_message = await async_registration._do_pow_register(
        subtensor=subtensor,
        netuid=1,
        wallet=fake_wallet,
        pow_result=fake_pow_result,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    subtensor.substrate.compose_call.asseert_awaited_once_with(

    )
    subtensor.substrate.create_signed_extrinsic.asseert_awaited_once_with(
        fake_call, fake_wallet.hotkey
    )
    subtensor.substrate.submit_extrinsic.asseert_awaited_once_with(
        fake_extrinsic, wait_for_inclusion=True, wait_for_finalization=True
    )

    mocked_format_error_message.assert_called_once_with(
        fake_err_message, substrate=subtensor.substrate
    )
    assert result_error_message == (False, mocked_format_error_message.return_value)


@pytest.mark.asyncio
async def test_do_pow_register_no_waiting(subtensor, mocker):
    """Tests PoW registration without waiting for inclusion or finalization."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkeypub.ss58_address = "coldkey_ss58"
    fake_pow_result = mocker.Mock(
        block_number=12345,
        nonce=67890,
        seal=b"fake_seal",
    )

    fake_call = mocker.AsyncMock()
    fake_extrinsic = mocker.AsyncMock()
    fake_response = mocker.Mock()

    mocker.patch.object(subtensor.substrate, "compose_call", return_value=fake_call)
    mocker.patch.object(
        subtensor.substrate, "create_signed_extrinsic", return_value=fake_extrinsic
    )
    mocker.patch.object(
        subtensor.substrate, "submit_extrinsic", return_value=fake_response
    )

    # Call
    result, error_message = await async_registration._do_pow_register(
        subtensor=subtensor,
        netuid=1,
        wallet=fake_wallet,
        pow_result=fake_pow_result,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )

    # Asserts
    subtensor.substrate.compose_call.asseert_awaited_once_with()
    subtensor.substrate.create_signed_extrinsic.asseert_awaited_once_with()
    subtensor.substrate.submit_extrinsic.asseert_awaited_once_with(
        fake_extrinsic, wait_for_inclusion=False, wait_for_finalization=False
    )
    assert result is True
    assert error_message is None
