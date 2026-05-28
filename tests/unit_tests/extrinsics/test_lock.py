from bittensor.core.extrinsics import lock
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance


def test_lock_stake_extrinsic(mocker):
    """Verify that lock_stake_extrinsic composes correct call and submits it."""
    fake_subtensor = mocker.Mock(
        **{
            "sign_and_send_extrinsic.return_value": ExtrinsicResponse(True, "Success"),
        }
    )
    fake_wallet = mocker.Mock()
    hotkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    netuid = 1
    amount = Balance.from_tao(5)

    result = lock.lock_stake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        amount=amount,
        mev_protection=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert result.success is True
    fake_subtensor.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="lock_stake",
        call_params={
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "amount": amount.rao,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_called_once_with(
        call=fake_subtensor.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )


def test_lock_stake_extrinsic_mev_protection(mocker):
    """Verify that lock_stake_extrinsic uses submit_encrypted_extrinsic when mev_protection=True."""
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    hotkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    netuid = 1
    amount = Balance.from_tao(5)

    mock_submit = mocker.patch(
        "bittensor.core.extrinsics.lock.submit_encrypted_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    result = lock.lock_stake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        amount=amount,
        mev_protection=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )

    assert result.success is True
    mock_submit.assert_called_once_with(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        call=fake_subtensor.compose_call.return_value,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )
    fake_subtensor.sign_and_send_extrinsic.assert_not_called()


def test_move_lock_extrinsic(mocker):
    """Verify that move_lock_extrinsic composes correct call and submits it."""
    fake_subtensor = mocker.Mock(
        **{
            "sign_and_send_extrinsic.return_value": ExtrinsicResponse(True, "Success"),
        }
    )
    fake_wallet = mocker.Mock()
    destination_hotkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    netuid = 2

    result = lock.move_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        destination_hotkey_ss58=destination_hotkey_ss58,
        netuid=netuid,
        mev_protection=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert result.success is True
    fake_subtensor.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="move_lock",
        call_params={
            "destination_hotkey": destination_hotkey_ss58,
            "netuid": netuid,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_called_once_with(
        call=fake_subtensor.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )


def test_move_lock_extrinsic_mev_protection(mocker):
    """Verify that move_lock_extrinsic uses submit_encrypted_extrinsic when mev_protection=True."""
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    destination_hotkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    netuid = 2

    mock_submit = mocker.patch(
        "bittensor.core.extrinsics.lock.submit_encrypted_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    result = lock.move_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        destination_hotkey_ss58=destination_hotkey_ss58,
        netuid=netuid,
        mev_protection=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )

    assert result.success is True
    mock_submit.assert_called_once_with(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        call=fake_subtensor.compose_call.return_value,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )
    fake_subtensor.sign_and_send_extrinsic.assert_not_called()


def test_set_perpetual_lock_extrinsic(mocker):
    """Verify that set_perpetual_lock_extrinsic composes correct call and submits it."""
    fake_subtensor = mocker.Mock(
        **{
            "sign_and_send_extrinsic.return_value": ExtrinsicResponse(True, "Success"),
        }
    )
    fake_wallet = mocker.Mock()
    netuid = 3
    enabled = False

    result = lock.set_perpetual_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        enabled=enabled,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert result.success is True
    fake_subtensor.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="set_perpetual_lock",
        call_params={
            "netuid": netuid,
            "enabled": enabled,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_called_once_with(
        call=fake_subtensor.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )


def test_lock_stake_wallet_unlock_failure(mocker):
    """Verify that lock_stake_extrinsic returns early on wallet unlock failure."""
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    amount = Balance.from_tao(1)

    mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(False, "Wallet unlock failed"),
    )

    result = lock.lock_stake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58="hotkey",
        netuid=1,
        amount=amount,
    )

    assert result.success is False
    assert "unlock" in result.message.lower()
    fake_subtensor.compose_call.assert_not_called()
    fake_subtensor.sign_and_send_extrinsic.assert_not_called()


def test_move_lock_wallet_unlock_failure(mocker):
    """Verify that move_lock_extrinsic returns early on wallet unlock failure."""
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()

    mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(False, "Wallet unlock failed"),
    )

    result = lock.move_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        destination_hotkey_ss58="hotkey",
        netuid=1,
    )

    assert result.success is False
    assert "unlock" in result.message.lower()
    fake_subtensor.compose_call.assert_not_called()
    fake_subtensor.sign_and_send_extrinsic.assert_not_called()


def test_set_perpetual_lock_wallet_unlock_failure(mocker):
    """Verify that set_perpetual_lock_extrinsic returns early on wallet unlock failure."""
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()

    mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(False, "Wallet unlock failed"),
    )

    result = lock.set_perpetual_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=1,
        enabled=True,
    )

    assert result.success is False
    assert "unlock" in result.message.lower()
    fake_subtensor.compose_call.assert_not_called()
    fake_subtensor.sign_and_send_extrinsic.assert_not_called()


def test_lock_stake_extrinsic_exception(mocker):
    """Verify that lock_stake_extrinsic handles exceptions gracefully."""
    fake_subtensor = mocker.Mock(
        **{"sign_and_send_extrinsic.side_effect": RuntimeError("chain error")}
    )
    fake_wallet = mocker.Mock()

    result = lock.lock_stake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58="hotkey",
        netuid=1,
        amount=Balance.from_tao(1),
        mev_protection=False,
    )

    assert result.success is False


def test_move_lock_extrinsic_exception(mocker):
    """Verify that move_lock_extrinsic handles exceptions gracefully."""
    fake_subtensor = mocker.Mock(
        **{"sign_and_send_extrinsic.side_effect": RuntimeError("chain error")}
    )
    fake_wallet = mocker.Mock()

    result = lock.move_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        destination_hotkey_ss58="hotkey",
        netuid=1,
        mev_protection=False,
    )

    assert result.success is False


def test_set_perpetual_lock_extrinsic_exception(mocker):
    """Verify that set_perpetual_lock_extrinsic handles exceptions gracefully."""
    fake_subtensor = mocker.Mock(
        **{"sign_and_send_extrinsic.side_effect": RuntimeError("chain error")}
    )
    fake_wallet = mocker.Mock()

    result = lock.set_perpetual_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=1,
        enabled=True,
    )

    assert result.success is False
