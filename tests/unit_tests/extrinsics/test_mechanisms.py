import pytest
from bittensor.core.extrinsics import mechanism
from bittensor.core.types import ExtrinsicResponse


def test_commit_mechanism_weights_extrinsic(mocker, subtensor, fake_wallet):
    """Test successful `commit_mechanism_weights_extrinsic` extrinsic."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey"

    netuid = mocker.Mock()
    mechid = mocker.Mock()
    uids = []
    weights = []
    salt = []

    mocked_get_sub_subnet_storage_index = mocker.patch.object(
        mechanism, "get_mechid_storage_index"
    )
    mocked_generate_weight_hash = mocker.patch.object(mechanism, "generate_weight_hash")
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=ExtrinsicResponse(True, "")
    )

    # Call
    result = mechanism.commit_mechanism_weights_extrinsic(
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
        version_key=mechanism.version_as_int,
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


def test_commit_timelocked_mechanism_weights_extrinsic(mocker, subtensor, fake_wallet):
    """Test successful `commit_mechanism_weights_extrinsic` extrinsic."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey"

    netuid = mocker.Mock()
    mechid = mocker.Mock()
    uids = []
    weights = []
    block_time = mocker.Mock()

    mocked_convert_and_normalize_weights_and_uids = mocker.patch.object(
        mechanism,
        "convert_and_normalize_weights_and_uids",
        return_value=(uids, weights),
    )
    mocked_get_current_block = mocker.patch.object(subtensor, "get_current_block")
    mocked_get_subnet_hyperparameters = mocker.patch.object(
        subtensor, "get_subnet_hyperparameters"
    )
    mocked_get_sub_subnet_storage_index = mocker.patch.object(
        mechanism, "get_mechid_storage_index"
    )
    mocked_get_encrypted_commit = mocker.patch.object(
        mechanism,
        "get_encrypted_commit",
        return_value=(mocker.Mock(), mocker.Mock()),
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(
            True,
            f"reveal_round:{mocked_get_encrypted_commit.return_value[1]}",
        ),
    )

    # Call
    result = mechanism.commit_timelocked_mechanism_weights_extrinsic(
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
    mocked_get_sub_subnet_storage_index.assert_called_once_with(
        netuid=netuid, mechid=mechid
    )
    mocked_get_encrypted_commit.assert_called_once_with(
        uids=uids,
        weights=weights,
        subnet_reveal_period_epochs=mocked_get_subnet_hyperparameters.return_value.commit_reveal_period,
        version_key=mechanism.version_as_int,
        tempo=mocked_get_subnet_hyperparameters.return_value.tempo,
        netuid=mocked_get_sub_subnet_storage_index.return_value,
        current_block=mocked_get_current_block.return_value,
        block_time=block_time,
        hotkey=fake_wallet.hotkey.public_key,
    )
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="commit_timelocked_mechanism_weights",
        call_params={
            "netuid": netuid,
            "mecid": mechid,
            "commit": mocked_get_encrypted_commit.return_value[0],
            "reveal_round": mocked_get_encrypted_commit.return_value[1],
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


def test_reveal_mechanism_weights_extrinsic(mocker, subtensor, fake_wallet):
    """Test successful `reveal_mechanism_weights_extrinsic` extrinsic."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey"

    netuid = mocker.Mock()
    mechid = mocker.Mock()
    uids = []
    weights = []
    salt = []

    mocked_convert_and_normalize_weights_and_uids = mocker.patch.object(
        mechanism,
        "convert_and_normalize_weights_and_uids",
        return_value=(uids, weights),
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor, "sign_and_send_extrinsic", return_value=ExtrinsicResponse(True, "")
    )

    # Call
    result = mechanism.reveal_mechanism_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        mechid=mechid,
        uids=uids,
        weights=weights,
        salt=salt,
        version_key=mechanism.version_as_int,
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
            "version_key": mechanism.version_as_int,
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


def test_mechanism_sub_weights_extrinsic(mocker, subtensor, fake_wallet):
    """Verify that the `set_mechanism_weights_extrinsic` function works as expected."""
    # Preps
    fake_wallet.hotkey.ss58_address = "hotkey"

    netuid = mocker.Mock()
    mechid = mocker.Mock()
    uids = []
    weights = []

    mocked_convert_and_normalize_weights_and_uids = mocker.patch.object(
        mechanism,
        "convert_and_normalize_weights_and_uids",
        return_value=(uids, weights),
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(
            True,
            "",
        ),
    )

    # Call
    result = mechanism.set_mechanism_weights_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        mechid=mechid,
        uids=uids,
        weights=weights,
        version_key=mechanism.version_as_int,
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
            "version_key": mechanism.version_as_int,
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
