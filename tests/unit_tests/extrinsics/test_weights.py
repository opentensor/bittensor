from bittensor.core.extrinsics import weights as weights_module
from bittensor.core.types import ExtrinsicResponse


def test_commit_timelocked_weights_extrinsic(mocker, subtensor, fake_wallet):
    """Test successful `commit_timelocked_weights_extrinsic` extrinsic."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey"

    netuid = mocker.Mock()
    mechid = mocker.Mock()
    uids = []
    weights = []
    block_time = mocker.Mock()

    mocked_convert_and_normalize_weights_and_uids = mocker.patch.object(
        weights_module,
        "convert_and_normalize_weights_and_uids",
        return_value=(uids, weights),
    )
    mocked_get_current_block = mocker.patch.object(subtensor, "get_current_block")
    mocked_get_subnet_hyperparameters = mocker.patch.object(
        subtensor, "get_subnet_hyperparameters"
    )
    fake_schedule = mocker.Mock()
    mocker.patch.object(
        subtensor, "get_epoch_schedule_state", return_value=fake_schedule
    )
    mocked_get_encrypted_commit_v2 = mocker.patch.object(
        weights_module,
        "get_encrypted_commit_v2",
        return_value=(mocker.Mock(), mocker.Mock()),
    )
    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(
            True,
            f"reveal_round:{mocked_get_encrypted_commit_v2.return_value[1]}",
        ),
    )

    # Call
    result = weights_module.commit_timelocked_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        mechid=mechid,
        uids=uids,
        weights=weights,
        block_time=block_time,
    )

    # Asserts
    mocked_convert_and_normalize_weights_and_uids.assert_called_once_with(uids, weights)
    mocked_get_encrypted_commit_v2.assert_called_once_with(
        uids=list(uids),
        weights=list(weights),
        version_key=weights_module.version_as_int,
        last_epoch_block=fake_schedule.last_epoch_block,
        pending_epoch_at=fake_schedule.pending_epoch_at,
        subnet_epoch_index=fake_schedule.subnet_epoch_index,
        tempo=fake_schedule.tempo,
        blocks_since_last_step=fake_schedule.blocks_since_last_step,
        current_block=fake_schedule.current_block,
        subnet_reveal_period_epochs=mocked_get_subnet_hyperparameters.return_value.commit_reveal_period,
        block_time=block_time,
        hotkey=fake_wallet.hotkey.public_key,
    )
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="commit_timelocked_mechanism_weights",
        call_params={
            "netuid": netuid,
            "mecid": mechid,
            "commit": mocked_get_encrypted_commit_v2.return_value[0],
            "reveal_round": mocked_get_encrypted_commit_v2.return_value[1],
            "commit_reveal_version": 4,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        call=mocked_compose_call.return_value,
        nonce_key="hotkey",
        sign_with="hotkey",
        use_nonce=True,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value


def test_commit_weights_extrinsic(mocker, subtensor, fake_wallet):
    """Test successful `commit_weights_extrinsic` extrinsic."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey"

    netuid = mocker.Mock()
    mechid = mocker.Mock()
    uids = []
    weights = []
    salt = []

    mocked_get_sub_subnet_storage_index = mocker.patch.object(
        weights_module, "get_mechid_storage_index"
    )
    mocked_generate_weight_hash = mocker.patch.object(
        weights_module, "generate_weight_hash"
    )
    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=ExtrinsicResponse(True, "")
    )

    # Call
    result = weights_module.commit_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        mechid=mechid,
        uids=uids,
        weights=weights,
        salt=salt,
    )

    # Asserts
    mocked_get_sub_subnet_storage_index.assert_called_once_with(
        netuid=netuid, mechid=mechid
    )
    mocked_generate_weight_hash.assert_called_once_with(
        address=fake_wallet.hotkey.ss58_address,
        netuid=mocked_get_sub_subnet_storage_index.return_value,
        uids=list(uids),
        values=list(weights),
        salt=salt,
        version_key=weights_module.version_as_int,
    )
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="commit_mechanism_weights",
        call_params={
            "netuid": netuid,
            "mecid": mechid,
            "commit_hash": mocked_generate_weight_hash.return_value,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        call=mocked_compose_call.return_value,
        nonce_key="hotkey",
        sign_with="hotkey",
        use_nonce=True,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value


def test_reveal_weights_extrinsic(mocker, subtensor, fake_wallet):
    """Test successful `reveal_weights_extrinsic` extrinsic."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey"

    netuid = mocker.Mock()
    mechid = mocker.Mock()
    uids = []
    weights = []
    salt = []

    mocked_convert_and_normalize_weights_and_uids = mocker.patch.object(
        weights_module,
        "convert_and_normalize_weights_and_uids",
        return_value=(uids, weights),
    )
    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=ExtrinsicResponse(True, "")
    )

    # Call
    result = weights_module.reveal_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        mechid=mechid,
        uids=uids,
        weights=weights,
        salt=salt,
        version_key=weights_module.version_as_int,
    )

    # Asserts
    mocked_convert_and_normalize_weights_and_uids.assert_called_once_with(uids, weights)
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="reveal_mechanism_weights",
        call_params={
            "netuid": netuid,
            "mecid": mechid,
            "uids": mocked_convert_and_normalize_weights_and_uids.return_value[0],
            "values": mocked_convert_and_normalize_weights_and_uids.return_value[0],
            "salt": salt,
            "version_key": weights_module.version_as_int,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        call=mocked_compose_call.return_value,
        nonce_key="hotkey",
        sign_with="hotkey",
        use_nonce=True,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value


def test_set_weights_extrinsic(mocker, subtensor, fake_wallet):
    """Verify that the `set_weights_extrinsic` function works as expected."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey"

    netuid = mocker.Mock()
    mechid = mocker.Mock()
    uids = []
    weights = []

    mocked_convert_and_normalize_weights_and_uids = mocker.patch.object(
        weights_module,
        "convert_and_normalize_weights_and_uids",
        return_value=(uids, weights),
    )
    mocked_compose_call = mocker.patch.object(subtensor, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(
            True,
            "",
        ),
    )

    # Call
    result = weights_module.set_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        mechid=mechid,
        uids=uids,
        weights=weights,
        version_key=weights_module.version_as_int,
    )

    # Asserts
    mocked_convert_and_normalize_weights_and_uids.assert_called_once_with(uids, weights)
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="set_mechanism_weights",
        call_params={
            "netuid": netuid,
            "mecid": mechid,
            "dests": uids,
            "weights": weights,
            "version_key": weights_module.version_as_int,
        },
    )
    mocked_sign_and_send_extrinsic.assert_called_once_with(
        wallet=fake_wallet,
        call=mocked_compose_call.return_value,
        nonce_key="hotkey",
        sign_with="hotkey",
        use_nonce=True,
        period=None,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result == mocked_sign_and_send_extrinsic.return_value
