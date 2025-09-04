from bittensor.core.extrinsics import staking
from bittensor.utils.balance import Balance
import pytest


def test_add_stake_extrinsic(mocker):
    """Verify that sync `add_stake_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock(
        **{
            "get_balance.return_value": Balance(10),
            "get_existential_deposit.return_value": Balance(1),
            "get_hotkey_owner.return_value": "hotkey_owner",
            "sign_and_send_extrinsic.return_value": (True, ""),
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
    assert result is True

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
        sign_with="coldkey",
        use_nonce=True,
        period=None,
        raise_error=False,
    )


def test_add_stake_multiple_extrinsic(mocker):
    """Verify that sync `add_stake_multiple_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock(
        **{
            "get_balance.return_value": Balance(10.0),
            "sign_and_send_extrinsic.return_value": (True, ""),
            "substrate.query_multi.return_value": [
                (
                    mocker.Mock(
                        **{
                            "params": ["hotkey1"],
                        },
                    ),
                    0,
                ),
                (
                    mocker.Mock(
                        **{
                            "params": ["hotkey2"],
                        },
                    ),
                    0,
                ),
            ],
            "substrate.query.return_value": 0,
        }
    )
    mocker.patch.object(
        staking, "get_old_stakes", return_value=[Balance(1.1), Balance(0.3)]
    )
    fake_wallet_ = mocker.Mock(
        **{
            "coldkeypub.ss58_address": "hotkey_owner",
        }
    )
    hotkey_ss58s = ["hotkey1", "hotkey2"]
    netuids = [1, 2]
    amounts = [Balance.from_tao(1.1), Balance.from_tao(2.2)]
    wait_for_inclusion = True
    wait_for_finalization = True

    # Call
    result = staking.add_stake_multiple_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet_,
        hotkey_ss58s=hotkey_ss58s,
        netuids=netuids,
        amounts=amounts,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    assert result is True
    assert fake_subtensor.substrate.compose_call.call_count == 2
    assert fake_subtensor.sign_and_send_extrinsic.call_count == 2

    fake_subtensor.substrate.compose_call.assert_any_call(
        call_module="SubtensorModule",
        call_function="add_stake",
        call_params={
            "hotkey": "hotkey2",
            "amount_staked": 2199999333,
            "netuid": 2,
        },
    )
    fake_subtensor.substrate.compose_call.assert_any_call(
        call_module="SubtensorModule",
        call_function="add_stake",
        call_params={
            "hotkey": "hotkey2",
            "amount_staked": 2199999333,
            "netuid": 2,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_called_with(
        call=fake_subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet_,
        nonce_key="coldkeypub",
        sign_with="coldkey",
        use_nonce=True,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )


@pytest.mark.parametrize(
    "res_success, res_message",
    [
        (True, ""),
        (False, "Error"),
    ],
)
def test_set_auto_stake_extrinsic(
    subtensor, fake_wallet, mocker, res_success, res_message
):
    """Verify that `set_auto_stake_extrinsic` function calls proper methods."""
    # Preps
    netuid = mocker.Mock()
    hotkey_ss58 = mocker.Mock()
    mocked_unlock_key = mocker.patch.object(
        staking, "unlock_key", return_value=mocker.Mock(success=True, message="True")
    )

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")

    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=(res_success, res_message)
    )

    # Call
    success, message = staking.set_auto_stake_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
    )

    # Asserts
    mocked_unlock_key.assert_called_once_with(fake_wallet, raise_error=False)
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="set_coldkey_auto_stake_hotkey",
        call_params={"netuid": netuid, "hotkey": hotkey_ss58},
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is res_success
    assert message == res_message
