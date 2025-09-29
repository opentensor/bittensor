from bittensor.core.extrinsics import staking
from bittensor.utils.balance import Balance
from bittensor.core.types import ExtrinsicResponse


def test_add_stake_extrinsic(mocker):
    """Verify that sync `add_stake_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock(
        **{
            "get_balance.return_value": Balance(10),
            "get_existential_deposit.return_value": Balance(1),
            "get_hotkey_owner.return_value": "hotkey_owner",
            "sign_and_send_extrinsic.return_value": ExtrinsicResponse(True, "Success"),
        }
    )
    fake_wallet_ = mocker.Mock(
        **{
            "coldkeypub.ss58_address": "hotkey_owner",
        }
    )
    hotkey_ss58 = "hotkey"
    fake_netuid = 1
    amount = Balance.from_tao(1.1)
    wait_for_inclusion = True
    wait_for_finalization = True

    # Call
    result = staking.add_stake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet_,
        hotkey_ss58=hotkey_ss58,
        netuid=fake_netuid,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    assert result.success is True

    fake_subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="add_stake",
        call_params={"hotkey": "hotkey", "amount_staked": 9, "netuid": 1},
    )
    fake_subtensor.sign_and_send_extrinsic.assert_called_once_with(
        call=fake_subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet_,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        nonce_key="coldkeypub",
        use_nonce=True,
        period=None,
        raise_error=False,
    )


def test_add_stake_multiple_extrinsic(subtensor, mocker, fake_wallet):
    """Verify that sync `add_stake_multiple_extrinsic` method calls proper async method."""
    # Preps
    mocked_get_stake_for_coldkey = mocker.patch.object(
        subtensor, "get_stake_for_coldkey", return_value=[Balance(1.1), Balance(0.3)]
    )
    mocked_get_balance = mocker.patch.object(
        subtensor, "get_balance", return_value=Balance.from_tao(10)
    )
    mocker.patch.object(
        staking, "get_old_stakes", return_value=[Balance(1.1), Balance(0.3)]
    )
    mocked_add_stake_extrinsic = mocker.patch.object(
        staking,
        "add_stake_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    hotkey_ss58s = ["hotkey1", "hotkey2"]
    netuids = [1, 2]
    amounts = [Balance.from_tao(1.1), Balance.from_tao(2.2)]
    wait_for_inclusion = True
    wait_for_finalization = True

    # Call
    result = staking.add_stake_multiple_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuids=netuids,
        hotkey_ss58s=hotkey_ss58s,
        amounts=amounts,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        raise_error=True,
    )

    # Asserts
    mocked_get_stake_for_coldkey.assert_called_once_with(
        coldkey_ss58=fake_wallet.coldkeypub.ss58_address,
    )
    assert result.success is True
    assert mocked_add_stake_extrinsic.call_count == 2
