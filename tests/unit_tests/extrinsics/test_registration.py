from bittensor.core.extrinsics import registration


def test_burned_register_extrinsic(mocker):
    """"Verify that sync `burned_register_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    netuid = 1
    wait_for_inclusion = True
    wait_for_finalization = True

    mocked_execute_coroutine = mocker.patch.object(registration, "execute_coroutine")
    mocked_burned_register_extrinsic = mocker.Mock()
    registration.async_burned_register_extrinsic = mocked_burned_register_extrinsic

    # Call
    result = registration.burned_register_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization
    )

    # Asserts

    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_burned_register_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop
    )
    mocked_burned_register_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization
    )
    assert result == mocked_execute_coroutine.return_value


def test_register_extrinsic(mocker):
    """"Verify that sync `register_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    netuid = 1
    wait_for_inclusion = True
    wait_for_finalization = True
    max_allowed_attempts = 7
    output_in_place = True
    cuda = True
    dev_id = 5
    tpb = 12
    num_processes = 8
    update_interval = 2
    log_verbose = True

    mocked_execute_coroutine = mocker.patch.object(registration, "execute_coroutine")
    mocked_register_extrinsic = mocker.Mock()
    registration.async_register_extrinsic = mocked_register_extrinsic

    # Call
    result = registration.register_extrinsic(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        max_allowed_attempts=max_allowed_attempts,
        output_in_place=output_in_place,
        cuda=cuda,
        dev_id=dev_id,
        tpb=tpb,
        num_processes=num_processes,
        update_interval=update_interval,
        log_verbose=log_verbose
    )

    # Asserts

    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_register_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop
    )
    mocked_register_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        max_allowed_attempts=max_allowed_attempts,
        output_in_place=output_in_place,
        cuda=cuda,
        dev_id=dev_id,
        tpb=tpb,
        num_processes=num_processes,
        update_interval=update_interval,
        log_verbose=log_verbose
    )
    assert result == mocked_execute_coroutine.return_value

