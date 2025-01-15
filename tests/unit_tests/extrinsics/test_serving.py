from bittensor.core.extrinsics import serving


def test_do_serve_axon(mocker):
    """Verify that sync `do_serve_axon` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    call_params = mocker.Mock()
    wait_for_inclusion = True
    wait_for_finalization = True

    mocked_execute_coroutine = mocker.patch.object(serving, "execute_coroutine")
    mocked_do_serve_axon = mocker.Mock()
    serving.async_do_serve_axon = mocked_do_serve_axon

    # Call
    result = serving.do_serve_axon(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        call_params=call_params,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts

    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_do_serve_axon.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_do_serve_axon.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        call_params=call_params,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    assert result == mocked_execute_coroutine.return_value


def test_serve_axon_extrinsic(mocker):
    """Verify that sync `serve_axon_extrinsic` method calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    netuid = 2
    axon = mocker.Mock()
    wait_for_inclusion = True
    wait_for_finalization = True
    certificate = mocker.Mock()

    mocked_execute_coroutine = mocker.patch.object(serving, "execute_coroutine")
    mocked_serve_axon_extrinsic = mocker.Mock()
    serving.async_serve_axon_extrinsic = mocked_serve_axon_extrinsic

    # Call
    result = serving.serve_axon_extrinsic(
        subtensor=fake_subtensor,
        netuid=netuid,
        axon=axon,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        certificate=certificate,
    )

    # Asserts

    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_serve_axon_extrinsic.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_serve_axon_extrinsic.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        netuid=netuid,
        axon=axon,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        certificate=certificate,
    )
    assert result == mocked_execute_coroutine.return_value


def test_publish_metadata(mocker):
    """Verify that `publish_metadata` calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    fake_wallet = mocker.Mock()
    netuid = 2
    data_type = "data_type"
    data = b"data"
    wait_for_inclusion = True
    wait_for_finalization = True

    mocked_execute_coroutine = mocker.patch.object(serving, "execute_coroutine")
    mocked_publish_metadata = mocker.Mock()
    serving.async_publish_metadata = mocked_publish_metadata

    # Call
    result = serving.publish_metadata(
        subtensor=fake_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        data_type=data_type,
        data=data,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts

    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_publish_metadata.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_publish_metadata.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        data_type=data_type,
        data=data,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    assert result == mocked_execute_coroutine.return_value


def test_get_metadata(mocker):
    """Verify that `get_metadata` calls proper async method."""
    # Preps
    fake_subtensor = mocker.Mock()
    netuid = 2
    hotkey = "hotkey"
    block = 123

    mocked_execute_coroutine = mocker.patch.object(serving, "execute_coroutine")
    mocked_get_metadata = mocker.Mock()
    serving.async_get_metadata = mocked_get_metadata

    # Call
    result = serving.get_metadata(
        subtensor=fake_subtensor,
        netuid=netuid,
        hotkey=hotkey,
        block=block,
    )

    # Asserts
    mocked_execute_coroutine.assert_called_once_with(
        coroutine=mocked_get_metadata.return_value,
        event_loop=fake_subtensor.event_loop,
    )
    mocked_get_metadata.assert_called_once_with(
        subtensor=fake_subtensor.async_subtensor,
        netuid=netuid,
        hotkey=hotkey,
        block=block,
    )
    assert result == mocked_execute_coroutine.return_value
