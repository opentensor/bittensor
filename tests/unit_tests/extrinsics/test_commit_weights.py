from bittensor.core.extrinsics import commit_weights


def test_commit_weights_extrinsic(mocker):
    """ "Verify that sync `commit_weights_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    netuid = 1
    commit_hash = "0x1234567890abcdef"
    wait_for_inclusion = True
    wait_for_finalization = True

    mocked_execute_coroutine = mocker.patch.object(commit_weights, "execute_coroutine")
    mocked_commit_weights_extrinsic = mocker.Mock()
    commit_weights.async_commit_weights_extrinsic = mocked_commit_weights_extrinsic

    # Call
    result = commit_weights.commit_weights_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        commit_hash=commit_hash,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts

    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_commit_weights_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_commit_weights_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        commit_hash=commit_hash,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    assert result == mocked_execute_coroutine.return_value


def test_reveal_weights_extrinsic(mocker):
    """Verify that sync `reveal_weights_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    netuid = 1
    uids = [1, 2, 3, 4]
    weights = [5, 6, 7, 8]
    salt = [1, 2, 3, 4]
    version_key = 2
    wait_for_inclusion = True
    wait_for_finalization = True

    mocked_execute_coroutine = mocker.patch.object(commit_weights, "execute_coroutine")
    mocked_reveal_weights_extrinsic = mocker.Mock()
    commit_weights.async_reveal_weights_extrinsic = mocked_reveal_weights_extrinsic

    # Call
    result = commit_weights.reveal_weights_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        uids=uids,
        weights=weights,
        salt=salt,
        version_key=version_key,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts

    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_reveal_weights_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_reveal_weights_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        uids=uids,
        weights=weights,
        salt=salt,
        version_key=version_key,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    assert result == mocked_execute_coroutine.return_value
