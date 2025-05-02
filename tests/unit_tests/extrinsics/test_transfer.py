from bittensor.core.extrinsics.transfer import _do_transfer
from bittensor.utils.balance import Balance


def test_do_transfer_is_success_true(subtensor, fake_wallet, mocker):
    """Successful do_transfer call."""
    # Prep
    fake_dest = "SS58PUBLICKEY"
    fake_transfer_balance = Balance(1)
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = True

    subtensor.substrate.submit_extrinsic.return_value.is_success = True

    # Call
    result = _do_transfer(
        subtensor,
        fake_wallet,
        fake_dest,
        fake_transfer_balance,
        fake_wait_for_inclusion,
        fake_wait_for_finalization,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_allow_death",
        call_params={"dest": fake_dest, "value": fake_transfer_balance.rao},
    )
    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value, keypair=fake_wallet.coldkey
    )
    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        extrinsic=subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )
    # subtensor.substrate.submit_extrinsic.return_value.process_events.assert_called_once()
    assert result == (
        True,
        subtensor.substrate.get_chain_head.return_value,
        "Success with response.",
    )


def test_do_transfer_is_success_false(subtensor, fake_wallet, mocker):
    """Successful do_transfer call."""
    # Prep
    fake_dest = "SS58PUBLICKEY"
    fake_transfer_balance = Balance(1)
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = True

    subtensor.substrate.submit_extrinsic.return_value.is_success = False

    # Call
    result = _do_transfer(
        subtensor,
        fake_wallet,
        fake_dest,
        fake_transfer_balance,
        fake_wait_for_inclusion,
        fake_wait_for_finalization,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_allow_death",
        call_params={"dest": fake_dest, "value": fake_transfer_balance.rao},
    )
    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value, keypair=fake_wallet.coldkey
    )
    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        extrinsic=subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    assert result == (
        False,
        "",
        "Subtensor returned `UnknownError(UnknownType)` error. This means: `Unknown Description`.",
    )


def test_do_transfer_no_waits(subtensor, fake_wallet, mocker):
    """Successful do_transfer call."""
    # Prep
    fake_dest = "SS58PUBLICKEY"
    fake_transfer_balance = Balance(1)
    fake_wait_for_inclusion = False
    fake_wait_for_finalization = False

    # Call
    result = _do_transfer(
        subtensor,
        fake_wallet,
        fake_dest,
        fake_transfer_balance,
        fake_wait_for_inclusion,
        fake_wait_for_finalization,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_allow_death",
        call_params={"dest": fake_dest, "value": fake_transfer_balance.rao},
    )
    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value, keypair=fake_wallet.coldkey
    )
    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        extrinsic=subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )
    assert result == (True, "", "Not waiting for finalization or inclusion.")
