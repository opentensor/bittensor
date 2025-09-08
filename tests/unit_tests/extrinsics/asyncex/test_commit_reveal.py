import numpy as np
import pytest
import torch

from bittensor.core.chain_data import SubnetHyperparameters
from bittensor.core.extrinsics.asyncex import commit_reveal as async_commit_reveal
from bittensor.core.types import ExtrinsicResponse


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
        commit_reveal_period=0,
        commit_reveal_weights_enabled=True,
        alpha_high=0,
        alpha_low=0,
        liquid_alpha_enabled=False,
        alpha_sigmoid_steepness=0,
        yuma_version=3,
        subnet_is_active=False,
        transfers_enabled=False,
        bonds_reset_enabled=False,
        user_liquidity_enabled=False,
    )


@pytest.mark.asyncio
async def test_commit_reveal_v3_extrinsic_success_with_torch(
    mocker, subtensor, hyperparams, fake_wallet
):
    """Test successful commit-reveal with torch tensors."""
    # Preps
    fake_netuid = 1
    fake_uids = torch.tensor([1, 2, 3], dtype=torch.int64)
    fake_weights = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
    fake_commit_for_reveal = b"mock_commit_for_reveal"
    fake_reveal_round = 1

    # Mocks

    mocked_uids = mocker.Mock()
    mocked_weights = mocker.Mock()
    mocked_convert_weights_and_uids_for_emit = mocker.patch.object(
        async_commit_reveal,
        "convert_and_normalize_weights_and_uids",
        return_value=(mocked_uids, mocked_weights),
    )
    mocked_get_encrypted_commit = mocker.patch.object(
        async_commit_reveal,
        "get_encrypted_commit",
        return_value=(fake_commit_for_reveal, fake_reveal_round),
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )
    mock_block = mocker.patch.object(
        subtensor.substrate,
        "get_block",
        return_value={"header": {"number": 1, "hash": "fakehash"}},
    )
    mock_hyperparams = mocker.patch.object(
        subtensor,
        "get_subnet_hyperparameters",
        return_value=hyperparams,
    )

    # Call
    success, message = await async_commit_reveal.commit_reveal_extrinsic(
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
    assert message == "Success"
    mocked_convert_weights_and_uids_for_emit.assert_called_once_with(
        fake_uids, fake_weights
    )
    mocked_get_encrypted_commit.assert_called_once_with(
        uids=mocked_uids,
        weights=mocked_weights,
        subnet_reveal_period_epochs=mock_hyperparams.return_value.commit_reveal_period,
        version_key=async_commit_reveal.version_as_int,
        tempo=mock_hyperparams.return_value.tempo,
        netuid=fake_netuid,
        current_block=mock_block.return_value["header"]["number"],
        block_time=12.0,
        hotkey=fake_wallet.hotkey.public_key,
    )
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        sign_with="hotkey",
        period=None,
        raise_error=False,
        calling_function="commit_reveal_extrinsic",
    )


@pytest.mark.asyncio
async def test_commit_reveal_v3_extrinsic_success_with_numpy(
    mocker, subtensor, hyperparams, fake_wallet
):
    """Test successful commit-reveal with numpy arrays."""
    # Preps
    fake_netuid = 1
    fake_uids = np.array([1, 2, 3], dtype=np.int64)
    fake_weights = np.array([0.1, 0.2, 0.7], dtype=np.float32)

    mock_convert = mocker.patch.object(
        async_commit_reveal,
        "convert_and_normalize_weights_and_uids",
        return_value=(fake_uids, fake_weights),
    )
    mock_encode_drand = mocker.patch.object(
        async_commit_reveal, "get_encrypted_commit", return_value=(b"commit", 0)
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(True, "Success"),
    )
    mocker.patch.object(subtensor.substrate, "get_block_number", return_value=1)
    mocker.patch.object(
        subtensor,
        "get_subnet_hyperparameters",
        return_value=hyperparams,
    )

    # Call
    success, message = await async_commit_reveal.commit_reveal_extrinsic(
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
    assert message == "Success"
    mock_convert.assert_called_once_with(fake_uids, fake_weights)
    mock_encode_drand.assert_called_once()
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=False,
        wait_for_finalization=False,
        sign_with="hotkey",
        period=None,
        raise_error=False,
        calling_function="commit_reveal_extrinsic",
    )


@pytest.mark.asyncio
async def test_commit_reveal_v3_extrinsic_response_false(
    mocker, subtensor, hyperparams, fake_wallet
):
    """Test unsuccessful commit-reveal with torch."""
    # Preps
    fake_netuid = 1
    fake_uids = torch.tensor([1, 2, 3], dtype=torch.int64)
    fake_weights = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
    fake_commit_for_reveal = b"mock_commit_for_reveal"
    fake_reveal_round = 1

    # Mocks
    mocker.patch.object(
        async_commit_reveal,
        "convert_and_normalize_weights_and_uids",
        return_value=(fake_uids, fake_weights),
    )
    mocker.patch.object(
        async_commit_reveal,
        "get_encrypted_commit",
        return_value=(fake_commit_for_reveal, fake_reveal_round),
    )
    mocked_compose_call = mocker.patch.object(subtensor.substrate, "compose_call")
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        subtensor,
        "sign_and_send_extrinsic",
        return_value=ExtrinsicResponse(False, "Failed"),
    )
    mocker.patch.object(subtensor.substrate, "get_block_number", return_value=1)
    mocker.patch.object(
        subtensor,
        "get_subnet_hyperparameters",
        return_value=hyperparams,
    )

    # Call
    success, message = await async_commit_reveal.commit_reveal_extrinsic(
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
    mocked_sign_and_send_extrinsic.assert_awaited_once_with(
        call=mocked_compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        sign_with="hotkey",
        period=None,
        raise_error=False,
        calling_function="commit_reveal_extrinsic",
    )


@pytest.mark.asyncio
async def test_commit_reveal_v3_extrinsic_exception(mocker, subtensor, fake_wallet):
    """Test exception handling in commit-reveal."""
    # Preps
    fake_netuid = 1
    fake_uids = [1, 2, 3]
    fake_weights = [0.1, 0.2, 0.7]

    mocker.patch.object(
        async_commit_reveal,
        "convert_and_normalize_weights_and_uids",
        side_effect=Exception("Test Error"),
    )

    # Call
    with pytest.raises(Exception):
        await async_commit_reveal.commit_reveal_extrinsic(
            subtensor=subtensor,
            wallet=fake_wallet,
            netuid=fake_netuid,
            uids=fake_uids,
            weights=fake_weights,
        )
