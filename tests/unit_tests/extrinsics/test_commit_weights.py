import pytest

from bittensor.core.extrinsics.commit_weights import (
    commit_weights_extrinsic,
    reveal_weights_extrinsic,
)


@pytest.mark.parametrize(
    "sign_and_send_return",
    [
        (True, "Success"),
        (False, "Failure"),
    ],
    ids=["success", "failure"],
)
def test_commit_weights_extrinsic(subtensor, fake_wallet, mocker, sign_and_send_return):
    """Tests `commit_weights_extrinsic` calls proper methods."""
    # Preps
    fake_netuid = 1
    fake_weights = mocker.Mock()
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=sign_and_send_return
    )

    # Call
    result = commit_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        commit_hash=fake_weights,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="commit_weights",
        call_params={"netuid": fake_netuid, "commit_hash": fake_weights},
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        nonce_key="hotkey",
        period=None,
        raise_error=False,
        sign_with="hotkey",
        use_nonce=True,
    )
    assert result == sign_and_send_return


@pytest.mark.parametrize(
    "sign_and_send_return",
    [
        (True, "Success"),
        (False, "Failure"),
    ],
    ids=["success", "failure"],
)
def test_reveal_weights_extrinsic(subtensor, fake_wallet, mocker, sign_and_send_return):
    """Tests `reveal_weights_extrinsic` calls proper methods."""
    # Preps
    fake_netuid = 1
    fake_uids = mocker.Mock()
    fake_weights = mocker.Mock()
    fake_salt = mocker.Mock()
    fake_version_key = mocker.Mock()

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=sign_and_send_return
    )

    # Call
    result = reveal_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        salt=fake_salt,
        version_key=fake_version_key,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="reveal_weights",
        call_params={
            "netuid": fake_netuid,
            "uids": fake_uids,
            "values": fake_weights,
            "salt": fake_salt,
            "version_key": fake_version_key,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        nonce_key="hotkey",
        period=None,
        raise_error=False,
        sign_with="hotkey",
        use_nonce=True,
    )
    assert result == sign_and_send_return
