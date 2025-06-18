from bittensor.core.extrinsics import children


def test_set_children_extrinsic(subtensor, mocker, fake_wallet):
    """Test that set_children_extrinsic correctly constructs and submits the extrinsic."""
    # Preps
    hotkey = "fake hotkey"
    netuid = 123
    fake_children = [
        (
            1.0,
            "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        ),
    ]

    subtensor.substrate.compose_call = mocker.Mock()
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(True, "")
    )

    # Call
    success, message = children.set_children_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey=hotkey,
        netuid=netuid,
        children=fake_children,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="set_children",
        call_params={
            "children": [
                (
                    18446744073709551615,
                    "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
                ),
            ],
            "hotkey": "fake hotkey",
            "netuid": netuid,
        },
    )

    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        period=None,
        raise_error=False,
    )

    assert success is True
    assert "Success" in message


def test_root_set_pending_childkey_cooldown_extrinsic(subtensor, mocker, fake_wallet):
    """Verify root_set_pending_childkey_cooldown_extrinsic extrinsic."""
    # Preps
    cooldown = 100

    subtensor.substrate.compose_call = mocker.Mock()
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(True, "")
    )

    # Call
    success, message = children.root_set_pending_childkey_cooldown_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        cooldown=cooldown,
    )
    # Asserts

    subtensor.substrate.compose_call.call_count == 2
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        period=None,
    )
    assert success is True
    assert "Success" in message
