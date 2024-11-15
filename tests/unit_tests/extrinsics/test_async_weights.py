import pytest
from bittensor.core import async_subtensor
from bittensor_wallet import Wallet
from bittensor.core.extrinsics import async_weights


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
async def test_do_set_weights_success(subtensor, mocker):
    """Tests _do_set_weights when weights are set successfully."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
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
async def test_do_set_weights_failure(subtensor, mocker):
    """Tests _do_set_weights when setting weights fails."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
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

    fake_response.error_message = mocker.Mock()
    fake_response.process_events = mocker.AsyncMock()

    mocked_format_error_message = mocker.Mock()
    mocker.patch.object(
        async_weights, "format_error_message", mocked_format_error_message
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
    mocked_format_error_message.assert_called_once_with(
        fake_response.error_message, substrate=subtensor.substrate
    )
    assert message == mocked_format_error_message.return_value


@pytest.mark.asyncio
async def test_do_set_weights_no_waiting(subtensor, mocker):
    """Tests _do_set_weights when not waiting for inclusion or finalization."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
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
async def test_set_weights_extrinsic_success_with_finalization(subtensor, mocker):
    """Tests set_weights_extrinsic when weights are successfully set with finalization."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocked_do_set_weights = mocker.patch.object(
        async_weights, "_do_set_weights", return_value=(True, "")
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
    mocked_do_set_weights.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=mocker.ANY,
        vals=mocker.ANY,
        version_key=0,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    assert result is True
    assert message == "Successfully set weights and Finalized."


@pytest.mark.asyncio
async def test_set_weights_extrinsic_no_waiting(subtensor, mocker):
    """Tests set_weights_extrinsic when no waiting for inclusion or finalization."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocked_do_set_weights = mocker.patch.object(
        async_weights, "_do_set_weights", return_value=(True, "")
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
async def test_set_weights_extrinsic_failure(subtensor, mocker):
    """Tests set_weights_extrinsic when setting weights fails."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
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
async def test_set_weights_extrinsic_exception(subtensor, mocker):
    """Tests set_weights_extrinsic when an exception is raised."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
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
async def test_set_weights_extrinsic_if_use_torch(subtensor, mocker):
    """Tests set_weights_extrinsic when use_torch is True."""
    # Preps
    fake_wallet = mocker.Mock(autospec=Wallet)
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocked_use_torch = mocker.patch.object(
        async_weights, "use_torch", return_value=True
    )
    mocked_torch_tensor = mocker.patch.object(
        async_weights.torch, "tensor", return_value=mocker.Mock()
    )

    mocked_do_set_weights = mocker.patch.object(
        async_weights, "_do_set_weights", return_value=(False, "Test error message")
    )
    mocked_convert_weights_and_uids_for_emit = mocker.patch.object(
        async_weights.weight_utils,
        "convert_weights_and_uids_for_emit",
        return_value=(mocker.Mock(), mocker.Mock()),
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
    mocked_use_torch.assert_called_once()
    mocked_convert_weights_and_uids_for_emit.assert_called()
    mocked_torch_tensor.assert_called_with(
        fake_weights, dtype=async_weights.torch.float32
    )
    assert result is False
    assert message == "Test error message"
