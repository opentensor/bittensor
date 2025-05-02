from bittensor.core.extrinsics.commit_weights import (
    _do_commit_weights,
    _do_reveal_weights,
)
from bittensor.core.settings import version_as_int


def test_do_commit_weights(subtensor, fake_wallet, mocker):
    """Successful _do_commit_weights call."""
    # Preps
    netuid = 1
    commit_hash = "fake_commit_hash"
    wait_for_inclusion = True
    wait_for_finalization = True

    mocker.patch.object(subtensor, "sign_and_send_extrinsic", return_value=(False, ""))
    mocker.patch.object(subtensor, "get_block_hash", return_value=1)

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

    subtensor.sign_and_send_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=None,
        nonce_key="hotkey",
        sign_with="hotkey",
        use_nonce=True,
    )

    assert result == (False, "")


def test_do_reveal_weights(subtensor, fake_wallet, mocker):
    """Verifies that the `_do_reveal_weights` method interacts with the right substrate methods."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey"

    netuid = 1
    uids = [1, 2, 3, 4]
    values = [1, 2, 3, 4]
    salt = [4, 2, 2, 1]
    wait_for_inclusion = True
    wait_for_finalization = True

    mocker.patch.object(subtensor, "sign_and_send_extrinsic")

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

    subtensor.sign_and_send_extrinsic(
        call=subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=None,
        nonce_key="hotkey",
        sign_with="hotkey",
        use_nonce=True,
    )

    assert result == subtensor.sign_and_send_extrinsic.return_value
