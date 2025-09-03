import pytest
from bittensor.core.extrinsics import transfer
from bittensor.utils.balance import Balance


def test_transfer_extrinsic_success(subtensor, fake_wallet, mocker):
    """Tests successful transfer."""
    # Preps
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "valid_ss58_address"
    fake_amount = Balance(15)

    mocked_is_valid_address = mocker.patch.object(
        transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=True,
    )
    mocked_unlock_key = mocker.patch.object(
        transfer,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_get_chain_head = mocker.patch.object(
        subtensor.substrate, "get_chain_head", return_value="some_block_hash"
    )
    mocked_get_balance = mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=10000,
    )
    mocked_get_existential_deposit = mocker.patch.object(
        subtensor, "get_existential_deposit", return_value=1
    )
    subtensor.get_transfer_fee = mocker.patch.object(
        subtensor, "get_transfer_fee", return_value=2
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(True, "")
    )

    # Call
    result = transfer.transfer_extrinsic(
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
    assert mocked_get_chain_head.call_count == 1
    mocked_get_balance.assert_called_with(
        fake_wallet.coldkeypub.ss58_address,
    )
    mocked_get_existential_deposit.assert_called_once_with(
        block=subtensor.substrate.get_block_number.return_value
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
    )
    assert result is True


def test_transfer_extrinsic_call_successful_with_failed_response(
    subtensor, fake_wallet, mocker
):
    """Tests successful transfer call is successful with failed response."""
    # Preps
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "valid_ss58_address"
    fake_amount = Balance(15)

    mocked_is_valid_address = mocker.patch.object(
        transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=True,
    )
    mocked_unlock_key = mocker.patch.object(
        transfer,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_get_chain_head = mocker.patch.object(
        subtensor.substrate, "get_chain_head", return_value="some_block_hash"
    )
    mocked_get_balance = mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=10000,
    )
    mocked_get_existential_deposit = mocker.patch.object(
        subtensor, "get_existential_deposit", return_value=1
    )
    subtensor.get_transfer_fee = mocker.patch.object(
        subtensor, "get_transfer_fee", return_value=2
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(False, "")
    )

    # Call
    result = transfer.transfer_extrinsic(
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
    mocked_get_balance.assert_called_with(
        fake_wallet.coldkeypub.ss58_address,
        block=subtensor.substrate.get_block_number.return_value
    )
    mocked_get_existential_deposit.assert_called_once_with(
        block=subtensor.substrate.get_block_number.return_value
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
    )
    assert result is False


def test_transfer_extrinsic_insufficient_balance(subtensor, fake_wallet, mocker):
    """Tests transfer when balance is insufficient."""
    # Preps
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "valid_ss58_address"
    fake_amount = Balance(5000)

    mocked_is_valid_address = mocker.patch.object(
        transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=True,
    )
    mocked_unlock_key = mocker.patch.object(
        transfer,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_get_chain_head = mocker.patch.object(
        subtensor.substrate, "get_chain_head", return_value="some_block_hash"
    )
    mocked_get_balance = mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=1000,  # Insufficient balance
    )
    mocked_get_existential_deposit = mocker.patch.object(
        subtensor, "get_existential_deposit", return_value=1
    )
    subtensor.get_transfer_fee = mocker.patch.object(
        subtensor, "get_transfer_fee", return_value=2
    )

    # Call
    result = transfer.transfer_extrinsic(
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
    mocked_get_balance.assert_called_once()
    mocked_get_existential_deposit.assert_called_once_with(
        block=subtensor.substrate.get_block_number.return_value
    )
    assert result is False


def test_transfer_extrinsic_invalid_destination(subtensor, fake_wallet, mocker):
    """Tests transfer with invalid destination address."""
    # Preps
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "invalid_address"
    fake_amount = Balance(15)

    mocked_is_valid_address = mocker.patch.object(
        transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=False,
    )

    # Call
    result = transfer.transfer_extrinsic(
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


def test_transfer_extrinsic_unlock_key_false(subtensor, fake_wallet, mocker):
    """Tests transfer failed unlock_key."""
    # Preps
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "invalid_address"
    fake_amount = Balance(15)

    mocked_is_valid_address = mocker.patch.object(
        transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=True,
    )

    mocked_unlock_key = mocker.patch.object(
        transfer,
        "unlock_key",
        return_value=mocker.Mock(success=False, message=""),
    )

    # Call
    result = transfer.transfer_extrinsic(
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


def test_transfer_extrinsic_keep_alive_false_and_transfer_all_true(
    subtensor, fake_wallet, mocker
):
    """Tests transfer with keep_alive flag set to False and transfer_all flag set to True."""
    # Preps
    fake_wallet.coldkeypub.ss58_address = "fake_ss58_address"
    fake_destination = "valid_ss58_address"
    fake_amount = Balance(15)

    mocked_is_valid_address = mocker.patch.object(
        transfer,
        "is_valid_bittensor_address_or_public_key",
        return_value=True,
    )
    mocked_unlock_key = mocker.patch.object(
        transfer,
        "unlock_key",
        return_value=mocker.Mock(success=True, message="Unlocked"),
    )
    mocked_get_chain_head = mocker.patch.object(
        subtensor.substrate, "get_chain_head", return_value="some_block_hash"
    )
    mocker.patch.object(
        subtensor,
        "get_balance",
        return_value=1,
    )
    mocked_get_existential_deposit = mocker.patch.object(
        subtensor, "get_existential_deposit", return_value=1
    )
    subtensor.get_transfer_fee = mocker.patch.object(
        subtensor, "get_transfer_fee", return_value=2
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(True, "")
    )

    # Call
    result = transfer.transfer_extrinsic(
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
    assert mocked_compose_call.call_count == 0
    assert mocked_sign_and_send_extrinsic.call_count == 0
    assert result is False
