import pytest
from bittensor_wallet import Wallet

from bittensor.core import async_subtensor
from bittensor.core.extrinsics.asyncio import registration as async_registration


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
    subtensor.substrate.submit_extrinsic.assert_awaited_once_with(
        fake_extrinsic, wait_for_inclusion=True, wait_for_finalization=True
    )
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
    mocked_format_error_message = mocker.patch.object(
        async_registration, "format_error_message"
    )

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
    subtensor.substrate.compose_call.asseert_awaited_once_with()
    subtensor.substrate.create_signed_extrinsic.asseert_awaited_once_with(
        call=fake_call, keypair=fake_wallet.hotkey
    )
    subtensor.substrate.submit_extrinsic.asseert_awaited_once_with(
        extrinsic=fake_extrinsic, wait_for_inclusion=True, wait_for_finalization=True
    )

    mocked_format_error_message.assert_called_once_with(error_message=fake_err_message)
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


@pytest.mark.asyncio
async def test_register_extrinsic_success(subtensor, mocker):
    """Tests successful registration."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkey.ss58_address = "coldkey_ss58"

    mocked_subnet_exists = mocker.patch.object(
        subtensor, "subnet_exists", return_value=True
    )
    mocked_get_neuron = mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=True),
    )
    mocked_create_pow = mocker.patch.object(
        async_registration,
        "create_pow_async",
        return_value=mocker.Mock(is_stale_async=mocker.AsyncMock(return_value=False)),
    )
    mocked_do_pow_register = mocker.patch.object(
        async_registration, "_do_pow_register", return_value=(True, None)
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor, "is_hotkey_registered", return_value=True
    )

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=1,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_subnet_exists.assert_called_once_with(1)
    mocked_get_neuron.assert_called_once_with(hotkey_ss58="hotkey_ss58", netuid=1)
    mocked_create_pow.assert_called_once()
    mocked_do_pow_register.assert_called_once()
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=1, hotkey_ss58="hotkey_ss58"
    )
    assert result is True


@pytest.mark.asyncio
async def test_register_extrinsic_success_with_cuda(subtensor, mocker):
    """Tests successful registration with CUDA enabled."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkey.ss58_address = "coldkey_ss58"

    mocked_subnet_exists = mocker.patch.object(
        subtensor, "subnet_exists", return_value=True
    )
    mocked_get_neuron = mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=True),
    )
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocked_create_pow = mocker.patch.object(
        async_registration,
        "create_pow_async",
        return_value=mocker.Mock(is_stale_async=mocker.AsyncMock(return_value=False)),
    )
    mocked_do_pow_register = mocker.patch.object(
        async_registration, "_do_pow_register", return_value=(True, None)
    )
    mocked_is_hotkey_registered = mocker.patch.object(
        subtensor, "is_hotkey_registered", return_value=True
    )

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=1,
        cuda=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_subnet_exists.assert_called_once_with(1)
    mocked_get_neuron.assert_called_once_with(hotkey_ss58="hotkey_ss58", netuid=1)
    mocked_create_pow.assert_called_once()
    mocked_do_pow_register.assert_called_once()
    mocked_is_hotkey_registered.assert_called_once_with(
        netuid=1, hotkey_ss58="hotkey_ss58"
    )
    assert result is True


@pytest.mark.asyncio
async def test_register_extrinsic_failed_with_cuda(subtensor, mocker):
    """Tests failed registration with CUDA enabled."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkey.ss58_address = "coldkey_ss58"

    mocked_subnet_exists = mocker.patch.object(
        subtensor, "subnet_exists", return_value=True
    )
    mocked_get_neuron = mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=True),
    )
    mocker.patch("torch.cuda.is_available", return_value=False)

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=1,
        cuda=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_subnet_exists.assert_called_once_with(1)
    mocked_get_neuron.assert_called_once_with(hotkey_ss58="hotkey_ss58", netuid=1)
    assert result is False


@pytest.mark.asyncio
async def test_register_extrinsic_subnet_not_exists(subtensor, mocker):
    """Tests registration when subnet does not exist."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)

    mocked_subnet_exists = mocker.patch.object(
        subtensor, "subnet_exists", return_value=False
    )

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=1,
    )

    # Asserts
    mocked_subnet_exists.assert_called_once_with(1)
    assert result is False


@pytest.mark.asyncio
async def test_register_extrinsic_already_registered(subtensor, mocker):
    """Tests registration when the key is already registered."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    mocked_get_neuron = mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=False),
    )

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=1,
    )

    # Asserts
    mocked_get_neuron.assert_called_once_with(
        hotkey_ss58=fake_wallet.hotkey.ss58_address, netuid=1
    )
    assert result is True


@pytest.mark.asyncio
async def test_register_extrinsic_max_attempts_reached(subtensor, mocker):
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.hotkey.ss58_address = "hotkey_ss58"
    fake_wallet.coldkey.ss58_address = "coldkey_ss58"

    stale_responses = iter([False, False, False, True])

    async def is_stale_side_effect(*_, **__):
        return next(stale_responses, True)

    fake_pow_result = mocker.Mock()
    fake_pow_result.is_stale_async = mocker.AsyncMock(side_effect=is_stale_side_effect)

    mocked_subnet_exists = mocker.patch.object(
        subtensor, "subnet_exists", return_value=True
    )
    mocked_get_neuron = mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=mocker.Mock(is_null=True),
    )
    mocked_create_pow = mocker.patch.object(
        async_registration,
        "create_pow_async",
        return_value=fake_pow_result,
    )
    mocked_do_pow_register = mocker.patch.object(
        async_registration,
        "_do_pow_register",
        return_value=(False, "Test Error"),
    )

    # Call
    result = await async_registration.register_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=1,
        max_allowed_attempts=3,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_subnet_exists.assert_called_once_with(1)
    mocked_get_neuron.assert_called_once_with(hotkey_ss58="hotkey_ss58", netuid=1)
    assert mocked_create_pow.call_count == 3
    assert mocked_do_pow_register.call_count == 3

    mocked_do_pow_register.assert_called_with(
        subtensor=subtensor,
        netuid=1,
        wallet=fake_wallet,
        pow_result=fake_pow_result,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result is False
