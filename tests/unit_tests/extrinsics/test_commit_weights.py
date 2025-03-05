from bittensor_wallet import Wallet

from bittensor.core.extrinsics.commit_weights import (
    _do_commit_weights,
    _do_reveal_weights,
)
from bittensor.core.settings import version_as_int


def test_do_commit_weights(subtensor, mocker):
    """Successful _do_commit_weights call."""
    # Preps
    fake_wallet = mocker.MagicMock()
    netuid = 1
    commit_hash = "fake_commit_hash"
    wait_for_inclusion = True
    wait_for_finalization = True

    subtensor.substrate.submit_extrinsic.return_value.is_success = None

    mocked_format_error_message = mocker.Mock()
    mocker.patch(
        "bittensor.core.subtensor.format_error_message",
        mocked_format_error_message,
    )

    # Call
    result = _do_commit_weights(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        commit_hash=commit_hash,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Assertions
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="commit_weights",
        call_params={
            "netuid": netuid,
            "commit_hash": commit_hash,
        },
    )

    subtensor.substrate.create_signed_extrinsic.assert_called_once()
    _, kwargs = subtensor.substrate.create_signed_extrinsic.call_args
    assert kwargs["call"] == subtensor.substrate.compose_call.return_value
    assert kwargs["keypair"] == fake_wallet.hotkey

    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    mocked_format_error_message.assert_called_once_with(
        subtensor.substrate.submit_extrinsic.return_value.error_message,
    )

    assert result == (
        False,
        mocked_format_error_message.return_value,
    )


def test_do_reveal_weights(subtensor, mocker):
    """Verifies that the `_do_reveal_weights` method interacts with the right substrate methods."""
    # Preps
    fake_wallet = mocker.MagicMock(autospec=Wallet)
    fake_wallet.hotkey.ss58_address = "hotkey"

    netuid = 1
    uids = [1, 2, 3, 4]
    values = [1, 2, 3, 4]
    salt = [4, 2, 2, 1]
    wait_for_inclusion = True
    wait_for_finalization = True

    subtensor.substrate.submit_extrinsic.return_value.is_success = None

    mocked_format_error_message = mocker.Mock()
    mocker.patch(
        "bittensor.core.subtensor.format_error_message",
        mocked_format_error_message,
    )

    # Call
    result = _do_reveal_weights(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        uids=uids,
        values=values,
        salt=salt,
        version_key=version_as_int,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="reveal_weights",
        call_params={
            "netuid": netuid,
            "uids": uids,
            "values": values,
            "salt": salt,
            "version_key": version_as_int,
        },
    )

    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value,
        keypair=fake_wallet.hotkey,
        nonce=subtensor.substrate.get_account_next_index.return_value,
    )

    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    mocked_format_error_message.assert_called_once_with(
        subtensor.substrate.submit_extrinsic.return_value.error_message,
    )

    assert result == (
        False,
        mocked_format_error_message.return_value,
    )
