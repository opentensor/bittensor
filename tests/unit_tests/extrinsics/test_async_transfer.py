import pytest
from bittensor.core import async_subtensor
from bittensor_wallet import Wallet
from bittensor.core.extrinsics import async_transfer
from bittensor.utils.balance import Balance


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
async def test_do_transfer_success(subtensor, mocker):
    """Tests _do_transfer when the transfer is successful."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_destination = "destination_address"
    fake_amount = mocker.Mock(autospec=Balance, rao=1000)

    fake_call = mocker.AsyncMock()
    fake_extrinsic = mocker.AsyncMock()
    fake_response = mocker.Mock()

    fake_response.is_success = mocker.AsyncMock(return_value=True)()
    fake_response.process_events = mocker.AsyncMock()
    fake_response.block_hash = "fake_block_hash"

    mocker.patch.object(subtensor.substrate, "compose_call", return_value=fake_call)
    mocker.patch.object(
        subtensor.substrate, "create_signed_extrinsic", return_value=fake_extrinsic
    )
    mocker.patch.object(
        subtensor.substrate, "submit_extrinsic", return_value=fake_response
    )

    # Call
    success, block_hash, error_message = await async_transfer._do_transfer(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_allow_death",
        call_params={"dest": fake_destination, "value": fake_amount.rao},
    )
    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value, keypair=fake_wallet.coldkey
    )
    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is True
    assert block_hash == "fake_block_hash"
    assert error_message == "Success with response."


@pytest.mark.asyncio
async def test_do_transfer_failure(subtensor, mocker):
    """Tests _do_transfer when the transfer fails."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_destination = "destination_address"
    fake_amount = mocker.Mock(autospec=Balance, rao=1000)

    fake_call = mocker.AsyncMock()
    fake_extrinsic = mocker.AsyncMock()
    fake_response = mocker.Mock()

    fake_response.is_success = mocker.AsyncMock(return_value=False)()
    fake_response.process_events = mocker.AsyncMock()
    fake_response.error_message = mocker.AsyncMock(return_value="Fake error message")()

    mocker.patch.object(subtensor.substrate, "compose_call", return_value=fake_call)
    mocker.patch.object(
        subtensor.substrate, "create_signed_extrinsic", return_value=fake_extrinsic
    )
    mocker.patch.object(
        subtensor.substrate, "submit_extrinsic", return_value=fake_response
    )

    mocked_format_error_message = mocker.patch.object(
        async_transfer,
        "format_error_message",
        return_value="Formatted error message",
    )

    # Call
    success, block_hash, error_message = await async_transfer._do_transfer(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_allow_death",
        call_params={"dest": fake_destination, "value": fake_amount.rao},
    )
    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value, keypair=fake_wallet.coldkey
    )
    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is False
    assert block_hash == ""
    mocked_format_error_message.assert_called_once_with(
        "Fake error message", substrate=subtensor.substrate
    )
    assert error_message == "Formatted error message"


@pytest.mark.asyncio
async def test_do_transfer_no_waiting(subtensor, mocker):
    """Tests _do_transfer when no waiting for inclusion or finalization."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_destination = "destination_address"
    fake_amount = mocker.Mock(autospec=Balance, rao=1000)

    fake_call = mocker.AsyncMock()
    fake_extrinsic = mocker.AsyncMock()

    mocker.patch.object(subtensor.substrate, "compose_call", return_value=fake_call)
    mocker.patch.object(
        subtensor.substrate, "create_signed_extrinsic", return_value=fake_extrinsic
    )
    mocker.patch.object(
        subtensor.substrate,
        "submit_extrinsic",
        return_value=mocker.Mock(),
    )

    # Call
    success, block_hash, error_message = await async_transfer._do_transfer(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_allow_death",
        call_params={"dest": fake_destination, "value": fake_amount.rao},
    )
    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value, keypair=fake_wallet.coldkey
    )
    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )
    assert success is True
    assert block_hash == ""
    assert error_message == "Success, extrinsic submitted without waiting."


@pytest.mark.asyncio
async def test_transfer_extrinsic_success(subtensor, mocker):
    """Tests successful transfer."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "valid_ss58_address"
    fake_amount = Balance(15)

    mocked_is_valid_address = mocker.patch.object(
        async_transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=True,
    )
    mocked_unlock_key = mocker.patch.object(
        async_transfer,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_get_chain_head = mocker.patch.object(
        subtensor.substrate, "get_chain_head", return_value="some_block_hash"
    )
    mocked_get_balance = mocker.patch.object(
        subtensor,
        "get_balance",
        return_value={fake_wallet.coldkeypub.ss58_address: 10000},
    )
    mocked_get_existential_deposit = mocker.patch.object(
        subtensor, "get_existential_deposit", return_value=1
    )
    subtensor.get_transfer_fee = mocker.patch.object(
        subtensor, "get_transfer_fee", return_value=2
    )
    mocked_do_transfer = mocker.patch.object(
        async_transfer, "_do_transfer", return_value=(True, "fake_block_hash", "")
    )

    # Call
    result = await async_transfer.transfer_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        transfer_all=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        keep_alive=True,
    )

    # Asserts
    mocked_is_valid_address.assert_called_once_with(fake_destination)
    mocked_unlock_key.assert_called_once_with(fake_wallet)
    mocked_get_chain_head.assert_called_once()
    mocked_get_balance.assert_called_with(
        fake_wallet.coldkeypub.ss58_address,
    )
    mocked_get_existential_deposit.assert_called_once_with(
        block_hash=mocked_get_chain_head.return_value
    )
    mocked_do_transfer.assert_called_once()
    assert result is True


@pytest.mark.asyncio
async def test_transfer_extrinsic_call_successful_with_failed_response(
    subtensor, mocker
):
    """Tests successful transfer call is successful with failed response."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "valid_ss58_address"
    fake_amount = Balance(15)

    mocked_is_valid_address = mocker.patch.object(
        async_transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=True,
    )
    mocked_unlock_key = mocker.patch.object(
        async_transfer,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_get_chain_head = mocker.patch.object(
        subtensor.substrate, "get_chain_head", return_value="some_block_hash"
    )
    mocked_get_balance = mocker.patch.object(
        subtensor,
        "get_balance",
        return_value={fake_wallet.coldkeypub.ss58_address: 10000},
    )
    mocked_get_existential_deposit = mocker.patch.object(
        subtensor, "get_existential_deposit", return_value=1
    )
    subtensor.get_transfer_fee = mocker.patch.object(
        subtensor, "get_transfer_fee", return_value=2
    )
    mocked_do_transfer = mocker.patch.object(
        async_transfer, "_do_transfer", return_value=(False, "", "")
    )

    # Call
    result = await async_transfer.transfer_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        transfer_all=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        keep_alive=True,
    )

    # Asserts
    mocked_is_valid_address.assert_called_once_with(fake_destination)
    mocked_unlock_key.assert_called_once_with(fake_wallet)
    mocked_get_chain_head.assert_called_once()
    mocked_get_balance.assert_called_with(
        fake_wallet.coldkeypub.ss58_address,
        block_hash=mocked_get_chain_head.return_value,
    )
    mocked_get_existential_deposit.assert_called_once_with(
        block_hash=mocked_get_chain_head.return_value
    )
    mocked_do_transfer.assert_called_once()
    assert result is False


@pytest.mark.asyncio
async def test_transfer_extrinsic_insufficient_balance(subtensor, mocker):
    """Tests transfer when balance is insufficient."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "valid_ss58_address"
    fake_amount = Balance(5000)

    mocked_is_valid_address = mocker.patch.object(
        async_transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=True,
    )
    mocked_unlock_key = mocker.patch.object(
        async_transfer,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_get_chain_head = mocker.patch.object(
        subtensor.substrate, "get_chain_head", return_value="some_block_hash"
    )
    mocked_get_balance = mocker.patch.object(
        subtensor,
        "get_balance",
        return_value={
            fake_wallet.coldkeypub.ss58_address: 1000
        },  # Insufficient balance
    )
    mocked_get_existential_deposit = mocker.patch.object(
        subtensor, "get_existential_deposit", return_value=1
    )
    subtensor.get_transfer_fee = mocker.patch.object(
        subtensor, "get_transfer_fee", return_value=2
    )

    # Call
    result = await async_transfer.transfer_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        transfer_all=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        keep_alive=True,
    )

    # Asserts
    mocked_is_valid_address.assert_called_once_with(fake_destination)
    mocked_unlock_key.assert_called_once_with(fake_wallet)
    mocked_get_chain_head.assert_called_once()
    mocked_get_balance.assert_called_once()
    mocked_get_existential_deposit.assert_called_once_with(
        block_hash=mocked_get_chain_head.return_value
    )
    assert result is False


@pytest.mark.asyncio
async def test_transfer_extrinsic_invalid_destination(subtensor, mocker):
    """Tests transfer with invalid destination address."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "invalid_address"
    fake_amount = Balance(15)

    mocked_is_valid_address = mocker.patch.object(
        async_transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=False,
    )

    # Call
    result = await async_transfer.transfer_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        transfer_all=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        keep_alive=True,
    )

    # Asserts
    mocked_is_valid_address.assert_called_once_with(fake_destination)
    assert result is False


@pytest.mark.asyncio
async def test_transfer_extrinsic_unlock_key_false(subtensor, mocker):
    """Tests transfer failed unlock_key."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "invalid_address"
    fake_amount = Balance(15)

    mocked_is_valid_address = mocker.patch.object(
        async_transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=True,
    )

    mocked_unlock_key = mocker.patch.object(
        async_transfer,
        "unlock_key",
        return_value=mocker.Mock(success=False, message=""),
    )

    # Call
    result = await async_transfer.transfer_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        transfer_all=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        keep_alive=True,
    )

    # Asserts
    mocked_is_valid_address.assert_called_once_with(fake_destination)
    mocked_unlock_key.assert_called_once_with(fake_wallet)
    assert result is False


@pytest.mark.asyncio
async def test_transfer_extrinsic_keep_alive_false_and_transfer_all_true(
    subtensor, mocker
):
    """Tests transfer with keep_alive flag set to False and transfer_all flag set to True."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "valid_ss58_address"
    fake_amount = Balance(15)

    mocked_is_valid_address = mocker.patch.object(
        async_transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=True,
    )
    mocked_unlock_key = mocker.patch.object(
        async_transfer,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_get_chain_head = mocker.patch.object(
        subtensor.substrate, "get_chain_head", return_value="some_block_hash"
    )
    mocked_get_balance = mocker.patch.object(
        subtensor,
        "get_balance",
        return_value={fake_wallet.coldkeypub.ss58_address: 1},
    )
    mocked_get_existential_deposit = mocker.patch.object(
        subtensor, "get_existential_deposit", return_value=1
    )
    subtensor.get_transfer_fee = mocker.patch.object(
        subtensor, "get_transfer_fee", return_value=2
    )
    mocked_do_transfer = mocker.patch.object(
        async_transfer, "_do_transfer", return_value=(True, "fake_block_hash", "")
    )

    # Call
    result = await async_transfer.transfer_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination=fake_destination,
        amount=fake_amount,
        transfer_all=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        keep_alive=False,
    )

    # Asserts
    mocked_is_valid_address.assert_called_once_with(fake_destination)
    mocked_unlock_key.assert_called_once_with(fake_wallet)
    mocked_get_chain_head.assert_called_once()

    mocked_get_existential_deposit.assert_called_once_with(
        block_hash=mocked_get_chain_head.return_value
    )
    mocked_do_transfer.assert_not_called()
    assert result is False
