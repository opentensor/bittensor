import pytest

from bittensor.core import subtensor as subtensor_module
from bittensor.core.settings import version_as_int
from bittensor.core.subtensor import Subtensor
from bittensor.core.extrinsics.commit_weights import (
    do_commit_weights,
    do_reveal_weights,
)


@pytest.fixture
def subtensor(mocker):
    fake_substrate = mocker.MagicMock()
    fake_substrate.websocket.sock.getsockopt.return_value = 0
    mocker.patch.object(
        subtensor_module, "SubstrateInterface", return_value=fake_substrate
    )
    return Subtensor()


def test_do_commit_weights(subtensor, mocker):
    """Successful _do_commit_weights call."""
    # Preps
    fake_wallet = mocker.MagicMock()
    netuid = 1
    commit_hash = "fake_commit_hash"
    wait_for_inclusion = True
    wait_for_finalization = True

    subtensor.substrate.submit_extrinsic.return_value.is_success = None

    mocked_format_error_message = mocker.MagicMock()
    subtensor_module.format_error_message = mocked_format_error_message

    # Call
    result = do_commit_weights(
        self=subtensor,
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

    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value, keypair=fake_wallet.hotkey
    )

    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    subtensor.substrate.submit_extrinsic.return_value.process_events.assert_called_once()

    assert result == (
        False,
        subtensor.substrate.submit_extrinsic.return_value.error_message,
    )


def test_do_reveal_weights(subtensor, mocker):
    """Verifies that the `_do_reveal_weights` method interacts with the right substrate methods."""
    # Preps
    fake_wallet = mocker.MagicMock()
    fake_wallet.hotkey = "hotkey"

    netuid = 1
    uids = [1, 2, 3, 4]
    values = [1, 2, 3, 4]
    salt = [4, 2, 2, 1]
    wait_for_inclusion = True
    wait_for_finalization = True

    subtensor.substrate.submit_extrinsic.return_value.is_success = None

    mocked_format_error_message = mocker.MagicMock()
    subtensor_module.format_error_message = mocked_format_error_message

    # Call
    result = do_reveal_weights(
        self=subtensor,
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
        call=subtensor.substrate.compose_call.return_value, keypair=fake_wallet.hotkey
    )

    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    subtensor.substrate.submit_extrinsic.return_value.process_events.assert_called_once()

    assert result == (
        False,
        subtensor.substrate.submit_extrinsic.return_value.error_message,
    )
