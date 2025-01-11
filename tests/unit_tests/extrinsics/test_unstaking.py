from bittensor.core.extrinsics import unstaking


def test_unstake_extrinsic(mocker):
    """Verify that sync `unstake_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    hotkey_ss58 = "hotkey"
    amount = 1.1
    wait_for_inclusion = True
    wait_for_finalization = True

    mocked_execute_coroutine = mocker.patch.object(unstaking, "execute_coroutine")
    mocked_unstake_extrinsic = mocker.Mock()
    unstaking.async_unstake_extrinsic = mocked_unstake_extrinsic

    # Call
    result = unstaking.unstake_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_unstake_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_unstake_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        hotkey_ss58=hotkey_ss58,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    assert result == mocked_execute_coroutine.return_value


def test_unstake_multiple_extrinsic(mocker):
    """Verify that sync `unstake_multiple_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    hotkey_ss58s = ["hotkey1", "hotkey2"]
    amounts = [1.1, 1.2]
    wait_for_inclusion = True
    wait_for_finalization = True

    mocked_execute_coroutine = mocker.patch.object(unstaking, "execute_coroutine")
    mocked_unstake_multiple_extrinsic = mocker.Mock()
    unstaking.async_unstake_multiple_extrinsic = mocked_unstake_multiple_extrinsic

    # Call
    result = unstaking.unstake_multiple_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        hotkey_ss58s=hotkey_ss58s,
        amounts=amounts,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_unstake_multiple_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_unstake_multiple_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        hotkey_ss58s=hotkey_ss58s,
        amounts=amounts,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    assert result == mocked_execute_coroutine.return_value
