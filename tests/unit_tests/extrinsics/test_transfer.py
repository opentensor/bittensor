from bittensor.core.extrinsics import transfer


def test_transfer_extrinsic(mocker):
    """Verify that sync `transfer_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    dest = "hotkey"
    amount = 1.1
    transfer_all = True
    wait_for_inclusion = True
    wait_for_finalization = True
    keep_alive = False

    mocked_execute_coroutine = mocker.patch.object(transfer, "execute_coroutine")
    mocked_transfer_extrinsic = mocker.Mock()
    transfer.async_transfer_extrinsic = mocked_transfer_extrinsic

    # Call
    result = transfer.transfer_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        dest=dest,
        amount=amount,
        transfer_all=transfer_all,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        keep_alive=keep_alive,
    )

    # Asserts
    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_transfer_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_transfer_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        destination=dest,
        amount=amount,
        transfer_all=transfer_all,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        keep_alive=keep_alive,
    )
    assert result == mocked_execute_coroutine.return_value
