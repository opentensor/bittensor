import pytest

from bittensor.core.extrinsics.asyncex import unstaking
from bittensor.utils.balance import Balance


@pytest.mark.asyncio
async def test_unstake_extrinsic(fake_wallet, mocker):
    # Preps
    fake_subtensor = mocker.AsyncMock(
        **{
            "get_hotkey_owner.return_value": "hotkey_owner",
            "get_stake_for_coldkey_and_hotkey.return_value": Balance(10.0),
            "sign_and_send_extrinsic.return_value": (True, ""),
            "get_stake.return_value": Balance(10.0),
        }
    )

    fake_wallet.coldkeypub.ss58_address = "hotkey_owner"
    hotkey_ss58 = "hotkey"
    fake_netuid = 1
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
    assert result is True

    fake_subtensor.substrate.compose_call.assert_awaited_once_with(
        call_module="SubtensorModule",
        call_function="remove_stake",
        call_params={
            "hotkey": "hotkey",
            "amount_unstaked": 1100000000,
            "netuid": 1,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_awaited_once_with(
        call=fake_subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        sign_with="coldkey",
        nonce_key="coldkeypub",
        use_nonce=True,
        period=None,
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
        hotkey=hotkey,
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
        wait_for_finalization=False,
        sign_with="coldkey",
        nonce_key="coldkeypub",
        use_nonce=True,
        period=None,
    )


@pytest.mark.asyncio
async def test_unstake_multiple_extrinsic(fake_wallet, mocker):
    """Verify that sync `unstake_multiple_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.AsyncMock(
        **{
            "get_hotkey_owner.return_value": "hotkey_owner",
            "get_stake_for_coldkey_and_hotkey.return_value": [Balance(10.0)],
            "sign_and_send_extrinsic.return_value": (True, ""),
            "tx_rate_limit.return_value": 0,
        }
    )
    mocker.patch.object(
        unstaking, "get_old_stakes", return_value=[Balance(1.1), Balance(0.3)]
    )
    fake_wallet.coldkeypub.ss58_address = "hotkey_owner"
    hotkey_ss58s = ["hotkey1", "hotkey2"]
    fake_netuids = [1, 2]
    amounts = [Balance.from_tao(1.1), Balance.from_tao(1.2)]
    wait_for_inclusion = True
    wait_for_finalization = True

    # Call
    result = await unstaking.unstake_multiple_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58s=hotkey_ss58s,
        netuids=fake_netuids,
        amounts=amounts,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    assert result is True
    assert fake_subtensor.substrate.compose_call.call_count == 1
    assert fake_subtensor.sign_and_send_extrinsic.call_count == 1

    fake_subtensor.substrate.compose_call.assert_any_call(
        call_module="SubtensorModule",
        call_function="remove_stake",
        call_params={
            "hotkey": "hotkey1",
            "amount_unstaked": 1100000000,
            "netuid": 1,
        },
    )
    fake_subtensor.substrate.compose_call.assert_any_call(
        call_module="SubtensorModule",
        call_function="remove_stake",
        call_params={
            "hotkey": "hotkey1",
            "amount_unstaked": 1100000000,
            "netuid": 1,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_awaited_with(
        call=fake_subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        sign_with="coldkey",
        nonce_key="coldkeypub",
        use_nonce=True,
        period=None,
    )
