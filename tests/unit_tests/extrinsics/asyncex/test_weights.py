import pytest
from bittensor.core import async_subtensor
from bittensor.core.extrinsics.asyncex import weights as async_weights


@pytest.mark.asyncio
async def test_do_set_weights_success(subtensor, fake_wallet, mocker):
    """Tests _do_set_weights when weights are set successfully."""
    # Preps
    fake_uids = [1, 2, 3]
    fake_vals = [100, 200, 300]
    fake_netuid = 0

    fake_call = mocker.AsyncMock()
    fake_extrinsic = mocker.AsyncMock()
    fake_response = mocker.Mock()

    async def fake_is_success():
        return True

    fake_response.is_success = fake_is_success()

    fake_response.process_events = mocker.AsyncMock()

    mocker.patch.object(subtensor.substrate, "compose_call", fake_call)
    mocker.patch.object(subtensor.substrate, "create_signed_extrinsic", fake_extrinsic)
    mocker.patch.object(
        subtensor.substrate,
        "submit_extrinsic",
        mocker.AsyncMock(return_value=fake_response),
    )

    # Call
    result, message = await async_weights._do_set_weights(
        subtensor=subtensor,
        wallet=fake_wallet,
        uids=fake_uids,
        vals=fake_vals,
        netuid=fake_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    assert result is True
    assert message == "Successfully set weights."


@pytest.mark.asyncio
async def test_do_set_weights_failure(subtensor, fake_wallet, mocker):
    """Tests _do_set_weights when setting weights fails."""
    # Preps
    fake_uids = [1, 2, 3]
    fake_vals = [100, 200, 300]
    fake_netuid = 0

    fake_call = mocker.AsyncMock()
    fake_extrinsic = mocker.AsyncMock()

    async def fake_is_success():
        return False

    fake_response = mocker.Mock()
    fake_response.is_success = fake_is_success()

    fake_response.process_events = mocker.AsyncMock()

    fake_response.error_message = mocker.AsyncMock(return_value="Error occurred")()
    fake_response.process_events = mocker.AsyncMock()

    mocked_format_error_message = mocker.Mock()
    mocker.patch.object(
        async_subtensor, "format_error_message", mocked_format_error_message
    )

    mocker.patch.object(subtensor.substrate, "compose_call", return_value=fake_call)
    mocker.patch.object(
        subtensor.substrate, "create_signed_extrinsic", return_value=fake_extrinsic
    )
    mocker.patch.object(
        subtensor.substrate, "submit_extrinsic", return_value=fake_response
    )

    # Call
    result, message = await async_weights._do_set_weights(
        subtensor=subtensor,
        wallet=fake_wallet,
        uids=fake_uids,
        vals=fake_vals,
        netuid=fake_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    assert result is False
    mocked_format_error_message.assert_called_once_with("Error occurred")
    assert message == mocked_format_error_message.return_value


@pytest.mark.asyncio
async def test_do_set_weights_no_waiting(subtensor, fake_wallet, mocker):
    """Tests _do_set_weights when not waiting for inclusion or finalization."""
    # Preps
    fake_uids = [1, 2, 3]
    fake_vals = [100, 200, 300]
    fake_netuid = 0

    fake_call = mocker.AsyncMock()
    fake_extrinsic = mocker.AsyncMock()
    fake_response = mocker.Mock()

    mocker.patch.object(subtensor.substrate, "compose_call", fake_call)
    mocker.patch.object(subtensor.substrate, "create_signed_extrinsic", fake_extrinsic)
    mocker.patch.object(
        subtensor.substrate,
        "submit_extrinsic",
        mocker.AsyncMock(return_value=fake_response),
    )

    # Call
    result, message = await async_weights._do_set_weights(
        subtensor=subtensor,
        wallet=fake_wallet,
        uids=fake_uids,
        vals=fake_vals,
        netuid=fake_netuid,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )

    # Asserts
    assert result is True
    assert message == "Not waiting for finalization or inclusion."


@pytest.mark.asyncio
async def test_set_weights_extrinsic_success_with_finalization(
    subtensor, fake_wallet, mocker
):
    """Tests set_weights_extrinsic when weights are successfully set with finalization."""
    # Preps
    fake_netuid = 1
    fake_uids = mocker.Mock()
    fake_weights = mocker.Mock()

    mocked_do_set_weights = mocker.patch.object(
        async_weights, "_do_set_weights", return_value=(True, "")
    )
    mocker_converter = mocker.patch.object(
        async_weights, "convert_and_normalize_weights_and_uids"
    )
    mocker_converter.return_value = (mocker.Mock(), mocker.Mock())

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
    mocker_converter.assert_called_once_with(fake_uids, fake_weights)

    mocked_do_set_weights.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=mocker_converter.return_value[0],
        vals=mocker_converter.return_value[1],
        version_key=0,
        wait_for_finalization=True,
        wait_for_inclusion=True,
        period=8,
    )
    assert result is True
    assert message == "Successfully set weights and Finalized."


@pytest.mark.asyncio
async def test_set_weights_extrinsic_no_waiting(subtensor, fake_wallet, mocker):
    """Tests set_weights_extrinsic when no waiting for inclusion or finalization."""
    # Preps
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocked_do_set_weights = mocker.patch.object(
        async_weights,
        "_do_set_weights",
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
    mocked_do_set_weights.assert_called_once()
    assert result is True
    assert message == "Not waiting for finalization or inclusion."


@pytest.mark.asyncio
async def test_set_weights_extrinsic_failure(subtensor, fake_wallet, mocker):
    """Tests set_weights_extrinsic when setting weights fails."""
    # Preps
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocked_do_set_weights = mocker.patch.object(
        async_weights, "_do_set_weights", return_value=(False, "Test error message")
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
    mocked_do_set_weights.assert_called_once()
    assert result is False
    assert message == "Test error message"


@pytest.mark.asyncio
async def test_set_weights_extrinsic_exception(subtensor, fake_wallet, mocker):
    """Tests set_weights_extrinsic when an exception is raised."""
    # Preps
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocked_do_set_weights = mocker.patch.object(
        async_weights, "_do_set_weights", side_effect=Exception("Unexpected error")
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
    mocked_do_set_weights.assert_called_once()
    assert result is False
    assert message == "Unexpected error"


@pytest.mark.asyncio
async def test_commit_weights_extrinsic_success(subtensor, fake_wallet, mocker):
    """Tests commit_weights_extrinsic when the commit is successful."""
    # Preps
    fake_netuid = 1
    fake_commit_hash = "test_hash"

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(True, None)
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
    assert message == "✅ [green]Successfully committed weights.[green]"


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
