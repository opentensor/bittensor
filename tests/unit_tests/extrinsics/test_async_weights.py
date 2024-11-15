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


