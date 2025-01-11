from bittensor.core.extrinsics import set_weights


def test_set_weights_extrinsic(mocker):
    """ "Verify that sync `set_weights_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    netuid = 2
    uids = [1, 2, 3, 4]
    weights = [0.1, 0.2, 0.3, 0.4]
    version_key = 2
    wait_for_inclusion = True
    wait_for_finalization = True

    mocked_execute_coroutine = mocker.patch.object(set_weights, "execute_coroutine")
    mocked_set_weights_extrinsic = mocker.Mock()
    set_weights.async_set_weights_extrinsic = mocked_set_weights_extrinsic

    # Call
    result = set_weights.set_weights_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        uids=uids,
        weights=weights,
        version_key=version_key,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_set_weights_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_set_weights_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        uids=uids,
        weights=weights,
        version_key=version_key,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    assert result == mocked_execute_coroutine.return_value
