from bittensor.core.extrinsics import staking
from bittensor.utils.balance import Balance


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
    fake_wallet = mocker.Mock(
        **{
            "coldkeypub.ss58_address": "hotkey_owner",
        }
    )
    hotkey_ss58 = "hotkey"
    amount = 1.1
    wait_for_inclusion = True
    wait_for_finalization = True

    # Call
    result = staking.add_stake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    assert result is True

    fake_subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="add_stake",
        call_params={
            "hotkey": "hotkey",
            "amount_staked": 9,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_called_once_with(
        fake_subtensor.substrate.compose_call.return_value,
        fake_wallet,
        True,
        True,
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
    fake_wallet = mocker.Mock(
        **{
            "coldkeypub.ss58_address": "hotkey_owner",
        }
    )
    hotkey_ss58s = ["hotkey1", "hotkey2"]
    amounts = [1.1, 2.2]
    wait_for_inclusion = True
    wait_for_finalization = True

    # Call
    result = staking.add_stake_multiple_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58s=hotkey_ss58s,
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
            "hotkey": "hotkey1",
            "amount_staked": 1099999666,
        },
    )
    fake_subtensor.substrate.compose_call.assert_any_call(
        call_module="SubtensorModule",
        call_function="add_stake",
        call_params={
            "hotkey": "hotkey2",
            "amount_staked": 2199999333,
        },
    )
    fake_subtensor.sign_and_send_extrinsic.assert_called_with(
        fake_subtensor.substrate.compose_call.return_value,
        fake_wallet,
        True,
        True,
    )
