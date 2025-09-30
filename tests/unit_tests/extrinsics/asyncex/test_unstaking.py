import pytest

from bittensor.core.extrinsics.asyncex import unstaking
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance


@pytest.mark.asyncio
async def test_unstake_extrinsic(fake_wallet, mocker):
    # Preps
    fake_substrate = mocker.AsyncMock(
        **{"get_payment_info.return_value": {"partial_fee": 10}}
    )
    fake_netuid = 14
    fake_subtensor = mocker.AsyncMock(
        **{
            "get_hotkey_owner.return_value": "hotkey_owner",
            "get_stake_for_coldkey_and_hotkey.return_value": Balance.from_tao(10.0, fake_netuid),
            "sign_and_send_extrinsic.return_value": ExtrinsicResponse(True, ""),
            "get_stake.return_value": Balance.from_tao(10.0, fake_netuid),
            "substrate": fake_substrate,
        }
    )

    fake_wallet.coldkeypub.ss58_address = "hotkey_owner"
    hotkey_ss58 = "hotkey"
    amount = Balance.from_tao(1.1)
    wait_for_inclusion = True
    wait_for_finalization = True

    # Call
    result = await unstaking.unstake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=fake_netuid,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    assert result.success is True

    fake_subtensor.substrate.compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="remove_stake",
        call_params={
            "hotkey": "hotkey",
            "amount_unstaked": 1100000000,
            "netuid": fake_netuid,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_awaited_once_with(
        call=fake_subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        nonce_key="coldkeypub",
        use_nonce=True,
        period=None,
        raise_error=False,
    )


@pytest.mark.asyncio
async def test_unstake_all_extrinsic(fake_wallet, mocker):
    # Preps
    fake_subtensor = mocker.AsyncMock(
        **{
            "subnet.return_value": mocker.Mock(price=100),
            "sign_and_send_extrinsic.return_value": (True, ""),
        }
    )
    fake_substrate = fake_subtensor.substrate.__aenter__.return_value
    hotkey = "hotkey"
    fake_netuid = 1

    # Call
    result = await unstaking.unstake_all_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey,
        netuid=fake_netuid,
    )

    # Asserts
    assert result[0] is True
    assert result[1] == ""

    fake_substrate.compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="remove_stake_full_limit",
        call_params={
            "hotkey": "hotkey",
            "netuid": fake_netuid,
            "limit_price": 100 * (1 - 0.005),
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_awaited_once_with(
        call=fake_substrate.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        nonce_key="coldkeypub",
        use_nonce=True,
        period=None,
        raise_error=False,
    )


@pytest.mark.asyncio
async def test_unstake_multiple_extrinsic_some_unstake_is_happy(fake_wallet, mocker):
    """Verify that sync `unstake_multiple_extrinsic` method calls proper async method."""
    # Preps
    sn_5 = 5
    sn_14 = 14
    fake_netuids = [sn_5, sn_14]
    fake_substrate = mocker.AsyncMock(
        **{"get_payment_info.return_value": {"partial_fee": 10}}
    )
    fake_subtensor = mocker.AsyncMock(
        **{
            "substrate": fake_substrate,
        }
    )
    mocked_unstake_extrinsic = mocker.patch.object(
        unstaking, "unstake_extrinsic", return_value=ExtrinsicResponse(True, "")
    )
    mocker.patch.object(
        unstaking, "get_old_stakes", return_value=[Balance.from_tao(1.1, sn_5), Balance.from_tao(0.3, sn_14)]
    )
    fake_wallet.coldkeypub.ss58_address = "hotkey_owner"
    hotkey_ss58s = ["hotkey1", "hotkey2"]
    amounts = [Balance.from_tao(1.1, sn_5), Balance.from_tao(1.2, sn_14)]
    wait_for_inclusion = True
    wait_for_finalization = True

    # Call
    response = await unstaking.unstake_multiple_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58s=hotkey_ss58s,
        netuids=fake_netuids,
        amounts=amounts,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    assert response.success is False
    assert response.message == "Some unstake were successful."
    assert len(response.data) == 2
    mocked_unstake_extrinsic.assert_awaited_once_with(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        amount=Balance.from_tao(1.1, sn_5),
        hotkey_ss58=hotkey_ss58s[0],
        netuid=fake_netuids[0],
        period=None,
        raise_error=False,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
