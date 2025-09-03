import pytest
from bittensor.core import async_subtensor
from bittensor.core.extrinsics.asyncex import weights as async_weights


@pytest.mark.asyncio
async def test_set_weights_extrinsic_success_with_finalization(
    subtensor, fake_wallet, mocker
):
    """Tests set_weights_extrinsic when weights are successfully set with finalization."""
    # Preps
    fake_netuid = 1
    fake_uids = mocker.Mock()
    fake_weights = mocker.Mock()

    mocked_convert_types = mocker.patch.object(
        async_weights,
        "convert_uids_and_weights",
        return_value=(mocker.Mock(), mocker.Mock()),
    )
    mocker_converter_normalize = mocker.patch.object(
        async_weights,
        "convert_and_normalize_weights_and_uids",
        return_value=(mocker.Mock(), mocker.Mock()),
    )

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(True, "")
    )

    # Call
    result, message = await async_weights.set_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_convert_types.assert_called_once_with(fake_uids, fake_weights)
    mocker_converter_normalize.assert_called_once_with(
        mocked_convert_types.return_value[0], mocked_convert_types.return_value[1]
    )
    mocked_compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="set_weights",
        call_params={
            "dests": mocker_converter_normalize.return_value[0],
            "weights": mocker_converter_normalize.return_value[1],
            "netuid": fake_netuid,
            "version_key": 0,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_finalization=True,
        wait_for_inclusion=True,
        period=8,
        use_nonce=True,
        nonce_key="hotkey",
        sign_with="hotkey",
        raise_error=False,
    )
    assert result is True
    assert message == ""


@pytest.mark.asyncio
async def test_set_weights_extrinsic_no_waiting(subtensor, fake_wallet, mocker):
    """Tests set_weights_extrinsic when no waiting for inclusion or finalization."""
    # Preps
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocked_convert_types = mocker.patch.object(
        async_weights,
        "convert_uids_and_weights",
        return_value=(mocker.Mock(), mocker.Mock()),
    )
    mocker_converter_normalize = mocker.patch.object(
        async_weights,
        "convert_and_normalize_weights_and_uids",
        return_value=(mocker.Mock(), mocker.Mock()),
    )

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=(True, "Not waiting for finalization or inclusion."),
    )

    # Call
    result, message = await async_weights.set_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )

    # Asserts
    mocked_convert_types.assert_called_once_with(fake_uids, fake_weights)
    mocker_converter_normalize.assert_called_once_with(
        mocked_convert_types.return_value[0], mocked_convert_types.return_value[1]
    )
    mocked_compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="set_weights",
        call_params={
            "dests": mocker_converter_normalize.return_value[0],
            "weights": mocker_converter_normalize.return_value[1],
            "netuid": fake_netuid,
            "version_key": 0,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_finalization=False,
        wait_for_inclusion=False,
        period=8,
        use_nonce=True,
        nonce_key="hotkey",
        sign_with="hotkey",
        raise_error=False,
    )
    assert result is True
    assert message == "Not waiting for finalization or inclusion."


@pytest.mark.asyncio
async def test_set_weights_extrinsic_failure(subtensor, fake_wallet, mocker):
    """Tests set_weights_extrinsic when setting weights fails."""
    # Preps
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocked_convert_types = mocker.patch.object(
        async_weights,
        "convert_uids_and_weights",
        return_value=(mocker.Mock(), mocker.Mock()),
    )
    mocker_converter_normalize = mocker.patch.object(
        async_weights,
        "convert_and_normalize_weights_and_uids",
        return_value=(mocker.Mock(), mocker.Mock()),
    )

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(False, "Test error message")
    )

    # Call
    result, message = await async_weights.set_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_convert_types.assert_called_once_with(fake_uids, fake_weights)
    mocker_converter_normalize.assert_called_once_with(
        mocked_convert_types.return_value[0], mocked_convert_types.return_value[1]
    )
    mocked_compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="set_weights",
        call_params={
            "dests": mocker_converter_normalize.return_value[0],
            "weights": mocker_converter_normalize.return_value[1],
            "netuid": fake_netuid,
            "version_key": 0,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_finalization=True,
        wait_for_inclusion=True,
        period=8,
        use_nonce=True,
        nonce_key="hotkey",
        sign_with="hotkey",
        raise_error=False,
    )
    assert result is False
    assert message == "Test error message"


@pytest.mark.asyncio
async def test_set_weights_extrinsic_exception(subtensor, fake_wallet, mocker):
    """Tests set_weights_extrinsic when an exception is raised."""
    # Preps
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocker.patch.object(
        async_weights,
        "convert_uids_and_weights",
        return_value=(mocker.Mock(), mocker.Mock()),
    )
    mocker.patch.object(
        async_weights,
        "convert_and_normalize_weights_and_uids",
        return_value=(mocker.Mock(), mocker.Mock()),
    )

    mocker.patch.object(subtensor.substrate, "compose_call")
    mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", side_effect=Exception("Unexpected error")
    )

    # Call
    with pytest.raises(Exception):
        await async_weights.set_weights_extrinsic(
            subtensor=subtensor,
            wallet=fake_wallet,
            netuid=fake_netuid,
            uids=fake_uids,
            weights=fake_weights,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )


@pytest.mark.asyncio
async def test_commit_weights_extrinsic_success(subtensor, fake_wallet, mocker):
    """Tests commit_weights_extrinsic when the commit is successful."""
    # Preps
    fake_netuid = 1
    fake_commit_hash = "test_hash"

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(True, "")
    )

    # Call
    result, message = await async_weights.commit_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        commit_hash=fake_commit_hash,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="commit_weights",
        call_params={"netuid": fake_netuid, "commit_hash": fake_commit_hash},
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        use_nonce=True,
        period=None,
        raise_error=False,
        nonce_key="hotkey",
        sign_with="hotkey",
    )
    assert result is True
    assert message == ""


@pytest.mark.asyncio
async def test_commit_weights_extrinsic_failure(subtensor, fake_wallet, mocker):
    """Tests commit_weights_extrinsic when the commit fails."""
    # Preps
    fake_netuid = 1
    fake_commit_hash = "test_hash"

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(False, "Commit failed.")
    )

    # Call
    result, message = await async_weights.commit_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        commit_hash=fake_commit_hash,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="commit_weights",
        call_params={"netuid": fake_netuid, "commit_hash": fake_commit_hash},
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        use_nonce=True,
        period=None,
        raise_error=False,
        nonce_key="hotkey",
        sign_with="hotkey",
    )
    assert result is False
    assert message == "Commit failed."
