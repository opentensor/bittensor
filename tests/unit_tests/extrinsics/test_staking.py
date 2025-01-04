from bittensor.core.extrinsics import staking


def test_add_stake_extrinsic(mocker):
    """Verify that sync `add_stake_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    hotkey_ss58 = "hotkey"
    amount = 1.1
    wait_for_inclusion = True
    wait_for_finalization = True

    mocked_execute_coroutine = mocker.patch.object(staking, "execute_coroutine")
    mocked_add_stake_extrinsic = mocker.Mock()
    staking.async_add_stake_extrinsic = mocked_add_stake_extrinsic

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
    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_add_stake_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_add_stake_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    assert result == mocked_execute_coroutine.return_value


def test_add_stake_multiple_extrinsic(mocker):
    """Verify that sync `add_stake_multiple_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    hotkey_ss58s = ["hotkey1", "hotkey2"]
    amounts = [1.1, 2.2]
    wait_for_inclusion = True
    wait_for_finalization = True

    mocked_execute_coroutine = mocker.patch.object(staking, "execute_coroutine")
    mocked_add_stake_multiple_extrinsic = mocker.Mock()
    staking.async_add_stake_multiple_extrinsic = mocked_add_stake_multiple_extrinsic

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
    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_add_stake_multiple_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_add_stake_multiple_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        hotkey_ss58s=hotkey_ss58s,
        amounts=amounts,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    assert result == mocked_execute_coroutine.return_value
