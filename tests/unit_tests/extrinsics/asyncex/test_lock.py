import pytest

from bittensor.core.extrinsics.asyncex import lock
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance


@pytest.mark.asyncio
async def test_lock_stake_extrinsic(mocker):
    """Verify that async lock_stake_extrinsic composes correct call and submits it."""
    # Preps
    fake_subtensor = mocker.AsyncMock(
        **{
            "sign_and_send_extrinsic.return_value": ExtrinsicResponse(True, "Success"),
        }
    )
    fake_wallet = mocker.Mock()
    hotkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    netuid = 1
    amount = Balance.from_tao(5)

    result = await lock.lock_stake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        amount=amount,
        mev_protection=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    assert result.success is True
    fake_subtensor.compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="lock_stake",
        call_params={
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "amount": amount.rao,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_awaited_once_with(
        call=fake_subtensor.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )


@pytest.mark.asyncio
async def test_lock_stake_extrinsic_mev_protection(mocker):
    """Verify that async lock_stake_extrinsic uses submit_encrypted_extrinsic when mev_protection=True."""
    # Preps
    fake_subtensor = mocker.AsyncMock()
    fake_wallet = mocker.Mock()
    hotkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    netuid = 1
    amount = Balance.from_tao(5)

    mock_submit = mocker.patch(
        "bittensor.core.extrinsics.asyncex.lock.submit_encrypted_extrinsic",
        new_callable=mocker.AsyncMock,
        return_value=ExtrinsicResponse(True, "Success"),
    )

    result = await lock.lock_stake_extrinsic(
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

    # Asserts
    assert result.success is True
    mock_submit.assert_awaited_once_with(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        call=fake_subtensor.compose_call.return_value,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )
    fake_subtensor.sign_and_send_extrinsic.assert_not_awaited()


@pytest.mark.asyncio
async def test_move_lock_extrinsic(mocker):
    """Verify that async move_lock_extrinsic composes correct call and submits it."""
    # Preps
    fake_subtensor = mocker.AsyncMock(
        **{
            "sign_and_send_extrinsic.return_value": ExtrinsicResponse(True, "Success"),
        }
    )
    fake_wallet = mocker.Mock()
    destination_hotkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    netuid = 2

    result = await lock.move_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        destination_hotkey_ss58=destination_hotkey_ss58,
        netuid=netuid,
        mev_protection=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    assert result.success is True
    fake_subtensor.compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="move_lock",
        call_params={
            "destination_hotkey": destination_hotkey_ss58,
            "netuid": netuid,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_awaited_once_with(
        call=fake_subtensor.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )


@pytest.mark.asyncio
async def test_move_lock_extrinsic_mev_protection(mocker):
    """Verify that async move_lock_extrinsic uses submit_encrypted_extrinsic when mev_protection=True."""
    # Preps
    fake_subtensor = mocker.AsyncMock()
    fake_wallet = mocker.Mock()
    destination_hotkey_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    netuid = 2

    mock_submit = mocker.patch(
        "bittensor.core.extrinsics.asyncex.lock.submit_encrypted_extrinsic",
        new_callable=mocker.AsyncMock,
        return_value=ExtrinsicResponse(True, "Success"),
    )

    result = await lock.move_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        destination_hotkey_ss58=destination_hotkey_ss58,
        netuid=netuid,
        mev_protection=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )

    # Asserts
    assert result.success is True
    mock_submit.assert_awaited_once_with(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        call=fake_subtensor.compose_call.return_value,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )
    fake_subtensor.sign_and_send_extrinsic.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_perpetual_lock_extrinsic(mocker):
    """Verify that async set_perpetual_lock_extrinsic composes correct call and submits it."""
    # Preps
    fake_subtensor = mocker.AsyncMock(
        **{
            "sign_and_send_extrinsic.return_value": ExtrinsicResponse(True, "Success"),
        }
    )
    fake_wallet = mocker.Mock()
    netuid = 3
    enabled = False

    result = await lock.set_perpetual_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        enabled=enabled,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    assert result.success is True
    fake_subtensor.compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="set_perpetual_lock",
        call_params={
            "netuid": netuid,
            "enabled": enabled,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_awaited_once_with(
        call=fake_subtensor.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=None,
        raise_error=False,
    )


@pytest.mark.asyncio
async def test_lock_stake_wallet_unlock_failure(mocker):
    """Verify that async lock_stake_extrinsic returns early on wallet unlock failure."""
    # Preps
    fake_subtensor = mocker.AsyncMock()
    fake_wallet = mocker.Mock()

    mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(False, "Wallet unlock failed"),
    )

    result = await lock.lock_stake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58="hotkey",
        netuid=1,
        amount=Balance.from_tao(1),
    )

    # Asserts
    assert result.success is False
    assert "unlock" in result.message.lower()
    fake_subtensor.compose_call.assert_not_awaited()
    fake_subtensor.sign_and_send_extrinsic.assert_not_awaited()


@pytest.mark.asyncio
async def test_move_lock_wallet_unlock_failure(mocker):
    """Verify that async move_lock_extrinsic returns early on wallet unlock failure."""
    # Preps
    fake_subtensor = mocker.AsyncMock()
    fake_wallet = mocker.Mock()

    mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(False, "Wallet unlock failed"),
    )

    result = await lock.move_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        destination_hotkey_ss58="hotkey",
        netuid=1,
    )

    # Asserts
    assert result.success is False
    assert "unlock" in result.message.lower()
    fake_subtensor.compose_call.assert_not_awaited()
    fake_subtensor.sign_and_send_extrinsic.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_perpetual_lock_wallet_unlock_failure(mocker):
    """Verify that async set_perpetual_lock_extrinsic returns early on wallet unlock failure."""
    # Preps
    fake_subtensor = mocker.AsyncMock()
    fake_wallet = mocker.Mock()

    mocker.patch.object(
        ExtrinsicResponse,
        "unlock_wallet",
        return_value=ExtrinsicResponse(False, "Wallet unlock failed"),
    )

    result = await lock.set_perpetual_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=1,
        enabled=True,
    )

    # Asserts
    assert result.success is False
    assert "unlock" in result.message.lower()
    fake_subtensor.compose_call.assert_not_awaited()
    fake_subtensor.sign_and_send_extrinsic.assert_not_awaited()


@pytest.mark.asyncio
async def test_lock_stake_extrinsic_exception(mocker):
    """Verify that async lock_stake_extrinsic handles exceptions gracefully."""
    # Preps
    fake_subtensor = mocker.AsyncMock(
        **{"sign_and_send_extrinsic.side_effect": RuntimeError("chain error")}
    )
    fake_wallet = mocker.Mock()

    result = await lock.lock_stake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58="hotkey",
        netuid=1,
        amount=Balance.from_tao(1),
        mev_protection=False,
    )

    # Asserts
    assert result.success is False


@pytest.mark.asyncio
async def test_move_lock_extrinsic_exception(mocker):
    """Verify that async move_lock_extrinsic handles exceptions gracefully."""
    # Preps
    fake_subtensor = mocker.AsyncMock(
        **{"sign_and_send_extrinsic.side_effect": RuntimeError("chain error")}
    )
    fake_wallet = mocker.Mock()

    result = await lock.move_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        destination_hotkey_ss58="hotkey",
        netuid=1,
        mev_protection=False,
    )

    # Asserts
    assert result.success is False


@pytest.mark.asyncio
async def test_set_perpetual_lock_extrinsic_exception(mocker):
    """Verify that async set_perpetual_lock_extrinsic handles exceptions gracefully."""
    # Preps
    fake_subtensor = mocker.AsyncMock(
        **{"sign_and_send_extrinsic.side_effect": RuntimeError("chain error")}
    )
    fake_wallet = mocker.Mock()

    result = await lock.set_perpetual_lock_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=1,
        enabled=True,
    )

    # Asserts
    assert result.success is False
