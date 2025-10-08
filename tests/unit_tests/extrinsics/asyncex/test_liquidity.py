import pytest
from bittensor.core.extrinsics.asyncex import liquidity


@pytest.mark.asyncio
async def test_add_liquidity_extrinsic(subtensor, fake_wallet, mocker):
    """Test that the add `add_liquidity_extrinsic` executes correct calls."""
    # Preps
    fake_netuid = mocker.Mock()
    fake_liquidity = mocker.Mock()
    fake_price_low = mocker.Mock()
    fake_price_high = mocker.Mock()

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )
    mocked_param_add_liquidity = mocker.patch.object(
        liquidity.LiquidityParams, "add_liquidity"
    )

    # Call
    result = await liquidity.add_liquidity_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        liquidity=fake_liquidity,
        price_low=fake_price_low,
        price_high=fake_price_high,
    )

    # Asserts
    mocked_param_add_liquidity.assert_called_once_with(
        netuid=fake_netuid,
        hotkey_ss58=fake_wallet.hotkey.ss58_address,
        liquidity=fake_liquidity,
        price_low=fake_price_low,
        price_high=fake_price_high,
    )
    mocked_compose_call.assert_awaited_once_with(
        call_module="Swap",
        call_function="add_liquidity",
        call_params=mocked_param_add_liquidity.return_value,
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value


@pytest.mark.asyncio
async def test_modify_liquidity_extrinsic(subtensor, fake_wallet, mocker):
    """Test that the add `modify_liquidity_extrinsic` executes correct calls."""
    # Preps
    fake_netuid = 1
    fake_position_id = 2
    fake_liquidity_delta = mocker.Mock()

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    result = await liquidity.modify_liquidity_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        position_id=fake_position_id,
        liquidity_delta=fake_liquidity_delta,
    )

    # Asserts
    mocked_compose_call.assert_awaited_once_with(
        call_module="Swap",
        call_function="modify_position",
        call_params={
            "hotkey": fake_wallet.hotkey.ss58_address,
            "netuid": fake_netuid,
            "position_id": fake_position_id,
            "liquidity_delta": fake_liquidity_delta.rao,
        },
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value


@pytest.mark.asyncio
async def test_remove_liquidity_extrinsic(subtensor, fake_wallet, mocker):
    """Test that the add `remove_liquidity_extrinsic` executes correct calls."""
    # Preps
    fake_netuid = 1
    fake_position_id = 2

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    result = await liquidity.remove_liquidity_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        position_id=fake_position_id,
    )

    # Asserts
    mocked_compose_call.assert_awaited_once_with(
        call_module="Swap",
        call_function="remove_liquidity",
        call_params={
            "hotkey": fake_wallet.hotkey.ss58_address,
            "netuid": fake_netuid,
            "position_id": fake_position_id,
        },
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value


@pytest.mark.asyncio
async def test_toggle_user_liquidity_extrinsic(subtensor, fake_wallet, mocker):
    """Test that the add `toggle_user_liquidity_extrinsic` executes correct calls."""
    # Preps
    fake_netuid = 1
    fake_enable = mocker.Mock()

    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    result = await liquidity.toggle_user_liquidity_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        enable=fake_enable,
    )

    # Asserts
    mocked_compose_call.assert_awaited_once_with(
        call_module="Swap",
        call_function="toggle_user_liquidity",
        call_params={
            "netuid": fake_netuid,
            "enable": fake_enable,
        },
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value
