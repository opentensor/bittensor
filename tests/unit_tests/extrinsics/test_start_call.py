from bittensor.core.extrinsics import start_call


def test_start_call_extrinsics(subtensor, mocker, fake_wallet):
    """Test that start_call_extrinsic correctly constructs and submits the extrinsic."""

    # Preps
    netuid = 123
    wallet = fake_wallet
    wallet.name = "fake_wallet"
    wallet.coldkey = "fake_coldkey"

    substrate = subtensor.substrate.__enter__.return_value
    substrate.compose_call.return_value = "mock_call"
    substrate.create_signed_extrinsic.return_value = "signed_ext"
    substrate.submit_extrinsic.return_value = mocker.MagicMock(
        is_success=True, error_message=""
    )

    # Call
    success, message = start_call.start_call_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
    )

    # Assertions
    substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="start_call",
        call_params={"netuid": netuid},
    )

    substrate.create_signed_extrinsic.assert_called_once_with(
        call="mock_call",
        keypair=wallet.coldkey,
    )

    substrate.submit_extrinsic.assert_called_once_with(
        extrinsic="signed_ext",
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    assert success is True
    assert "Success" in message
