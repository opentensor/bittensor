from bittensor.core.extrinsics.asyncex import start_call
import pytest


@pytest.mark.asyncio
async def test_start_call_extrinsics(subtensor, mocker, fake_wallet):
    """Test that start_call_extrinsic correctly constructs and submits the extrinsic."""
    # Preps
    netuid = 123
    wallet = fake_wallet
    wallet.name = "fake_wallet"
    wallet.coldkey = "fake_coldkey"
    substrate = subtensor.substrate.__aenter__.return_value
    substrate.compose_call = mocker.AsyncMock()
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(True, "")
    )

    # Call
    success, message = await start_call.start_call_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
    )

    # Assertions
    substrate.compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="start_call",
        call_params={"netuid": netuid},
    )

    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=substrate.compose_call.return_value,
        wallet=wallet,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        period=None,
    )

    assert success is True
    assert "Success" in message
