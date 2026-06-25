from bittensor.core.extrinsics import tempo_control
from bittensor.core.types import ExtrinsicResponse


def test_set_tempo_extrinsic(subtensor, mocker, fake_wallet):
    """Verify that set_tempo_extrinsic composes correct call and submits it."""
    # Preps
    netuid = 1
    tempo = 500
    mocked_set_tempo = mocker.patch.object(tempo_control.SubtensorModule, "set_tempo")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = tempo_control.set_tempo_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        tempo=tempo,
        mev_protection=False,
    )

    # Asserts
    mocked_set_tempo.assert_called_once_with(netuid=netuid, tempo=tempo)
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_set_tempo.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is True
    assert "Success" in message


def test_set_tempo_extrinsic_mev_protection(subtensor, mocker, fake_wallet):
    """Verify that set_tempo_extrinsic uses submit_encrypted_extrinsic when mev_protection=True."""
    # Preps
    netuid = 1
    tempo = 500
    mocked_set_tempo = mocker.patch.object(tempo_control.SubtensorModule, "set_tempo")
    mock_submit = mocker.patch(
        "bittensor.core.extrinsics.tempo_control.submit_encrypted_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    success, message = tempo_control.set_tempo_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        tempo=tempo,
        mev_protection=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )

    # Asserts
    assert success is True
    assert "Success" in message
    mock_submit.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=mocked_set_tempo.return_value,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )
    mocked_sign_and_send_extrinsic.assert_not_called()


def test_set_activity_cutoff_factor_extrinsic(subtensor, mocker, fake_wallet):
    """Verify that set_activity_cutoff_factor_extrinsic composes correct call and submits it."""
    # Preps
    netuid = 1
    factor_milli = 5000
    mocked_set_factor = mocker.patch.object(
        tempo_control.SubtensorModule, "set_activity_cutoff_factor"
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = tempo_control.set_activity_cutoff_factor_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        factor_milli=factor_milli,
        mev_protection=False,
    )

    # Asserts
    mocked_set_factor.assert_called_once_with(netuid=netuid, factor_milli=factor_milli)
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_set_factor.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is True
    assert "Success" in message


def test_set_activity_cutoff_factor_extrinsic_mev_protection(
    subtensor, mocker, fake_wallet
):
    """Verify that set_activity_cutoff_factor_extrinsic uses submit_encrypted_extrinsic when mev_protection=True."""
    # Preps
    netuid = 1
    factor_milli = 5000
    mocked_set_factor = mocker.patch.object(
        tempo_control.SubtensorModule, "set_activity_cutoff_factor"
    )
    mock_submit = mocker.patch(
        "bittensor.core.extrinsics.tempo_control.submit_encrypted_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    success, message = tempo_control.set_activity_cutoff_factor_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        factor_milli=factor_milli,
        mev_protection=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )

    # Asserts
    assert success is True
    assert "Success" in message
    mock_submit.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=mocked_set_factor.return_value,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )
    mocked_sign_and_send_extrinsic.assert_not_called()


def test_trigger_epoch_extrinsic(subtensor, mocker, fake_wallet):
    """Verify that trigger_epoch_extrinsic composes correct call and submits it."""
    # Preps
    netuid = 1
    mocked_trigger_epoch = mocker.patch.object(
        tempo_control.SubtensorModule, "trigger_epoch"
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = tempo_control.trigger_epoch_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        mev_protection=False,
    )

    # Asserts
    mocked_trigger_epoch.assert_called_once_with(netuid=netuid)
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_trigger_epoch.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is True
    assert "Success" in message


def test_trigger_epoch_extrinsic_mev_protection(subtensor, mocker, fake_wallet):
    """Verify that trigger_epoch_extrinsic uses submit_encrypted_extrinsic when mev_protection=True."""
    # Preps
    netuid = 1
    mocked_trigger_epoch = mocker.patch.object(
        tempo_control.SubtensorModule, "trigger_epoch"
    )
    mock_submit = mocker.patch(
        "bittensor.core.extrinsics.tempo_control.submit_encrypted_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    success, message = tempo_control.trigger_epoch_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        mev_protection=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )

    # Asserts
    assert success is True
    assert "Success" in message
    mock_submit.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=mocked_trigger_epoch.return_value,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )
    mocked_sign_and_send_extrinsic.assert_not_called()


def test_root_set_activity_cutoff_factor_extrinsic(subtensor, mocker, fake_wallet):
    """Verify root_set_activity_cutoff_factor_extrinsic extrinsic."""
    # Preps
    netuid = 1
    factor_milli = 5000
    mocked_set_factor = mocker.patch.object(
        tempo_control.SubtensorModule, "set_activity_cutoff_factor"
    )
    mocked_sudo = mocker.patch.object(tempo_control.Sudo, "sudo")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )

    # Call
    success, message = tempo_control.root_set_activity_cutoff_factor_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        factor_milli=factor_milli,
        mev_protection=False,
    )

    # Asserts
    mocked_set_factor.assert_called_once_with(netuid=netuid, factor_milli=factor_milli)
    mocked_sudo.assert_called_once_with(call=mocked_set_factor.return_value)
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        call=mocked_sudo.return_value,
        wallet=fake_wallet,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is True
    assert "Success" in message


def test_root_set_activity_cutoff_factor_extrinsic_mev_protection(
    subtensor, mocker, fake_wallet
):
    """Verify that root_set_activity_cutoff_factor_extrinsic uses submit_encrypted_extrinsic when mev_protection=True."""
    # Preps
    netuid = 1
    factor_milli = 5000
    mocked_set_factor = mocker.patch.object(
        tempo_control.SubtensorModule, "set_activity_cutoff_factor"
    )
    mocked_sudo = mocker.patch.object(tempo_control.Sudo, "sudo")
    mock_submit = mocker.patch(
        "bittensor.core.extrinsics.tempo_control.submit_encrypted_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic"
    )

    # Call
    success, message = tempo_control.root_set_activity_cutoff_factor_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        factor_milli=factor_milli,
        mev_protection=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )

    # Asserts
    assert success is True
    assert "Success" in message
    mocked_set_factor.assert_called_once_with(netuid=netuid, factor_milli=factor_milli)
    mocked_sudo.assert_called_once_with(call=mocked_set_factor.return_value)
    mock_submit.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        call=mocked_sudo.return_value,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        wait_for_revealed_execution=True,
    )
    mocked_sign_and_send_extrinsic.assert_not_called()
