from bittensor.core.extrinsics import start_call


def test_start_call_extrinsics(subtensor, mocker, fake_wallet):
    """Test that start_call_extrinsic correctly constructs and submits the extrinsic."""
    # Preps
    netuid = 123
    wallet = fake_wallet
    wallet.name = "fake_wallet"
    wallet.coldkey = "fake_coldkey"

    subtensor.substrate.compose_call = mocker.Mock()
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(True, "Success")
    )

    # Call
    success, message = start_call.start_call_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
    )

    # Assertions
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="start_call",
        call_params={"netuid": netuid},
    )

    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value,
        wallet=wallet,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        period=None,
        raise_error=False,
    )

    assert success is True
    assert "Success" in message
