from bittensor.core.extrinsics import liquidity


def test_add_liquidity_extrinsic(subtensor, fake_wallet, mocker):
    """Test that the add `add_liquidity_extrinsic` executes correct calls."""
    # Preps
    fake_netuid = 1
    fake_liquidity = mocker.Mock()
    fake_price_low = mocker.Mock()
    fake_price_high = mocker.Mock()

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )
    mocked_price_to_tick = mocker.patch.object(liquidity, "price_to_tick")

    # Call
    result = liquidity.add_liquidity_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        liquidity=fake_liquidity,
        price_low=fake_price_low,
        price_high=fake_price_high,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="Swap",
        call_function="add_liquidity",
        call_params={
            "hotkey": fake_wallet.hotkey.ss58_address,
            "netuid": fake_netuid,
            "tick_low": mocked_price_to_tick.return_value,
            "tick_high": mocked_price_to_tick.return_value,
            "liquidity": fake_liquidity.rao,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        use_nonce=True,
        period=None,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value


def test_modify_liquidity_extrinsic(subtensor, fake_wallet, mocker):
    """Test that the add `modify_liquidity_extrinsic` executes correct calls."""
    # Preps
    fake_netuid = 1
    fake_position_id = 2
    fake_liquidity_delta = mocker.Mock()

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    result = liquidity.modify_liquidity_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        position_id=fake_position_id,
        liquidity_delta=fake_liquidity_delta,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="Swap",
        call_function="modify_position",
        call_params={
            "hotkey": fake_wallet.hotkey.ss58_address,
            "netuid": fake_netuid,
            "position_id": fake_position_id,
            "liquidity_delta": fake_liquidity_delta.rao,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        use_nonce=True,
        period=None,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value


def test_remove_liquidity_extrinsic(subtensor, fake_wallet, mocker):
    """Test that the add `remove_liquidity_extrinsic` executes correct calls."""
    # Preps
    fake_netuid = 1
    fake_position_id = 2

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    result = liquidity.remove_liquidity_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        position_id=fake_position_id,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="Swap",
        call_function="remove_liquidity",
        call_params={
            "hotkey": fake_wallet.hotkey.ss58_address,
            "netuid": fake_netuid,
            "position_id": fake_position_id,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        use_nonce=True,
        period=None,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value


def test_toggle_user_liquidity_extrinsic(subtensor, fake_wallet, mocker):
    """Test that the add `toggle_user_liquidity_extrinsic` executes correct calls."""
    # Preps
    fake_netuid = 1
    fake_enable = mocker.Mock()

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    result = liquidity.toggle_user_liquidity_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        enable=fake_enable,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="Swap",
        call_function="toggle_user_liquidity",
        call_params={
            "netuid": fake_netuid,
            "enable": fake_enable,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        period=None,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value
