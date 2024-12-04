from bittensor.core import subtensor as subtensor_module
from bittensor.core.chain_data import SubnetHyperparameters
from bittensor.core.subtensor import Subtensor
from bittensor.core.extrinsics import commit_reveal
import pytest
import torch
import numpy as np


@pytest.fixture
def subtensor(mocker):
    fake_substrate = mocker.MagicMock()
    fake_substrate.websocket.sock.getsockopt.return_value = 0
    mocker.patch.object(
        subtensor_module, "SubstrateInterface", return_value=fake_substrate
    )
    yield Subtensor()


@pytest.fixture
def hyperparams():
    yield SubnetHyperparameters(
        rho=0,
        kappa=0,
        immunity_period=0,
        min_allowed_weights=0,
        max_weight_limit=0.0,
        tempo=0,
        min_difficulty=0,
        max_difficulty=0,
        weights_version=0,
        weights_rate_limit=0,
        adjustment_interval=0,
        activity_cutoff=0,
        registration_allowed=False,
        target_regs_per_interval=0,
        min_burn=0,
        max_burn=0,
        bonds_moving_avg=0,
        max_regs_per_block=0,
        serving_rate_limit=0,
        max_validators=0,
        adjustment_alpha=0,
        difficulty=0,
        commit_reveal_weights_interval=0,
        commit_reveal_weights_enabled=True,
        alpha_high=0,
        alpha_low=0,
        liquid_alpha_enabled=False,
    )


def test_do_commit_reveal_v3_success(mocker, subtensor):
    """Test successful commit-reveal with wait for finalization."""
    # Preps
    fake_wallet = mocker.Mock(autospec=subtensor_module.Wallet)
    fake_netuid = 1
    fake_commit = b"fake_commit"
    fake_reveal_round = 1

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_create_signed_extrinsic = mocker.patch.object(
        subtensor.substrate, "create_signed_extrinsic"
    )
    mocked_submit_extrinsic = mocker.patch.object(commit_reveal, "submit_extrinsic")

    # Call
    result = commit_reveal._do_commit_reveal_v3(
        self=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        commit=fake_commit,
        reveal_round=fake_reveal_round,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="commit_crv3_weights",
        call_params={
            "netuid": fake_netuid,
            "commit": fake_commit,
            "reveal_round": fake_reveal_round,
        },
    )
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value, keypair=fake_wallet.hotkey
    )
    mocked_submit_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        extrinsic=mocked_create_signed_extrinsic.return_value,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )
    assert result == (True, "Not waiting for finalization or inclusion.")


def test_do_commit_reveal_v3_failure_due_to_error(mocker, subtensor):
    """Test commit-reveal fails due to an error in submission."""
    # Preps
    fake_wallet = mocker.Mock(autospec=subtensor_module.Wallet)
    fake_netuid = 1
    fake_commit = b"fake_commit"
    fake_reveal_round = 1

    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_create_signed_extrinsic = mocker.patch.object(
        subtensor.substrate, "create_signed_extrinsic"
    )
    mocked_submit_extrinsic = mocker.patch.object(
        commit_reveal,
        "submit_extrinsic",
        return_value=mocker.Mock(is_success=False, error_message="Mocked error"),
    )
    mocked_format_error_message = mocker.patch.object(
        commit_reveal, "format_error_message", return_value="Formatted error"
    )

    # Call
    result = commit_reveal._do_commit_reveal_v3(
        self=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        commit=fake_commit,
        reveal_round=fake_reveal_round,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    mocked_compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="commit_crv3_weights",
        call_params={
            "netuid": fake_netuid,
            "commit": fake_commit,
            "reveal_round": fake_reveal_round,
        },
    )
    mocked_create_signed_extrinsic.assert_called_once_with(
        call=mocked_compose_call.return_value, keypair=fake_wallet.hotkey
    )
    mocked_submit_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        extrinsic=mocked_create_signed_extrinsic.return_value,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    mocked_format_error_message.assert_called_once_with(
        "Mocked error", substrate=subtensor.substrate
    )
    assert result == (False, "Formatted error")


def test_commit_reveal_v3_extrinsic_success_with_torch(mocker, subtensor, hyperparams):
    """Test successful commit-reveal with torch tensors."""
    # Preps
    fake_wallet = mocker.Mock(autospec=subtensor_module.Wallet)
    fake_netuid = 1
    fake_uids = torch.tensor([1, 2, 3], dtype=torch.int64)
    fake_weights = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
    fake_commit_for_reveal = b"mock_commit_for_reveal"
    fake_reveal_round = 1

    # Mocks
    mocker.patch.object(commit_reveal, "use_torch", return_value=True)

    mocked_uids = mocker.Mock()
    mocked_weights = mocker.Mock()
    mocked_convert_weights_and_uids_for_emit = mocker.patch.object(
        commit_reveal,
        "convert_weights_and_uids_for_emit",
        return_value=(mocked_uids, mocked_weights),
    )
    mocked_get_subnet_reveal_period_epochs = mocker.patch.object(
        subtensor, "get_subnet_reveal_period_epochs"
    )
    mocked_get_encrypted_commit = mocker.patch.object(
        commit_reveal,
        "get_encrypted_commit",
        return_value=(fake_commit_for_reveal, fake_reveal_round),
    )
    mock_do_commit_reveal_v3 = mocker.patch.object(
        commit_reveal, "_do_commit_reveal_v3", return_value=(True, "Success")
    )
    mock_block = mocker.patch.object(subtensor, "get_current_block", return_value=1)
    mock_hyperparams = mocker.patch.object(
        subtensor,
        "get_subnet_hyperparameters",
        return_value=hyperparams,
    )

    # Call
    success, message = commit_reveal.commit_reveal_v3_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    assert success is True
    assert message == "reveal_round:1"
    mocked_convert_weights_and_uids_for_emit.assert_called_once_with(
        fake_uids, fake_weights
    )
    mocked_get_encrypted_commit.assert_called_once_with(
        uids=mocked_uids,
        weights=mocked_weights,
        subnet_reveal_period_epochs=mock_hyperparams.return_value.commit_reveal_weights_interval,
        version_key=commit_reveal.version_as_int,
        tempo=mock_hyperparams.return_value.tempo,
        netuid=fake_netuid,
        current_block=mock_block.return_value,
    )
    mock_do_commit_reveal_v3.assert_called_once_with(
        self=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        commit=fake_commit_for_reveal,
        reveal_round=fake_reveal_round,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )


def test_commit_reveal_v3_extrinsic_success_with_numpy(mocker, subtensor, hyperparams):
    """Test successful commit-reveal with numpy arrays."""
    # Preps
    fake_wallet = mocker.Mock(autospec=subtensor_module.Wallet)
    fake_netuid = 1
    fake_uids = np.array([1, 2, 3], dtype=np.int64)
    fake_weights = np.array([0.1, 0.2, 0.7], dtype=np.float32)

    mocker.patch.object(commit_reveal, "use_torch", return_value=False)
    mock_convert = mocker.patch.object(
        commit_reveal,
        "convert_weights_and_uids_for_emit",
        return_value=(fake_uids, fake_weights),
    )
    mock_encode_drand = mocker.patch.object(
        commit_reveal, "get_encrypted_commit", return_value=(b"commit", 0)
    )
    mock_do_commit = mocker.patch.object(
        commit_reveal, "_do_commit_reveal_v3", return_value=(True, "Committed!")
    )
    mocker.patch.object(subtensor, "get_current_block", return_value=1)
    mocker.patch.object(
        subtensor,
        "get_subnet_hyperparameters",
        return_value=hyperparams,
    )

    # Call
    success, message = commit_reveal.commit_reveal_v3_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )

    # Asserts
    assert success is True
    assert message == "reveal_round:0"
    mock_convert.assert_called_once_with(fake_uids, fake_weights)
    mock_encode_drand.assert_called_once()
    mock_do_commit.assert_called_once()


def test_commit_reveal_v3_extrinsic_response_false(mocker, subtensor, hyperparams):
    """Test unsuccessful commit-reveal with torch."""
    # Preps
    fake_wallet = mocker.Mock(autospec=subtensor_module.Wallet)
    fake_netuid = 1
    fake_uids = torch.tensor([1, 2, 3], dtype=torch.int64)
    fake_weights = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
    fake_commit_for_reveal = b"mock_commit_for_reveal"
    fake_reveal_round = 1

    # Mocks
    mocker.patch.object(commit_reveal, "use_torch", return_value=True)
    mocker.patch.object(
        commit_reveal,
        "convert_weights_and_uids_for_emit",
        return_value=(fake_uids, fake_weights),
    )
    mocker.patch.object(
        commit_reveal,
        "get_encrypted_commit",
        return_value=(fake_commit_for_reveal, fake_reveal_round),
    )
    mock_do_commit_reveal_v3 = mocker.patch.object(
        commit_reveal, "_do_commit_reveal_v3", return_value=(False, "Failed")
    )
    mocker.patch.object(subtensor, "get_current_block", return_value=1)
    mocker.patch.object(
        subtensor,
        "get_subnet_hyperparameters",
        return_value=hyperparams,
    )

    # Call
    success, message = commit_reveal.commit_reveal_v3_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Asserts
    assert success is False
    assert message == "Failed"
    mock_do_commit_reveal_v3.assert_called_once_with(
        self=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        commit=fake_commit_for_reveal,
        reveal_round=fake_reveal_round,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )


def test_commit_reveal_v3_extrinsic_exception(mocker, subtensor):
    """Test exception handling in commit-reveal."""
    # Preps
    fake_wallet = mocker.Mock(autospec=subtensor_module.Wallet)
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocker.patch.object(
        commit_reveal,
        "convert_weights_and_uids_for_emit",
        side_effect=Exception("Test Error"),
    )

    # Call
    success, message = commit_reveal.commit_reveal_v3_extrinsic(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
    )

    # Asserts
    assert success is False
    assert "Test Error" in message
