import pytest
from unittest.mock import MagicMock, patch
from bittensor.subtensor import subtensor as Subtensor
from bittensor.wallet import wallet as Wallet
from bittensor.utils.registration import POWSolution
from bittensor.extrinsics.registration import (
    MaxSuccessException,
    MaxAttemptsException,
)


# Mocking external dependencies
@pytest.fixture
def mock_subtensor():
    mock = MagicMock(spec=Subtensor)
    mock.network = "mock_network"
    mock.substrate = MagicMock()
    return mock


@pytest.fixture
def mock_wallet():
    mock = MagicMock(spec=Wallet)
    mock.coldkeypub.ss58_address = "mock_address"
    return mock


@pytest.fixture
def mock_pow_solution():
    mock = MagicMock(spec=POWSolution)
    mock.block_number = 123
    mock.nonce = 456
    mock.seal = [0, 1, 2, 3]
    mock.is_stale.return_value = False
    return mock


@pytest.mark.parametrize(
    "wait_for_inclusion,wait_for_finalization,prompt,cuda,dev_id,tpb,num_processes,update_interval,log_verbose,expected",
    [
        (
            False,
            True,
            False,
            False,
            0,
            256,
            None,
            None,
            False,
            True,
        ),
        (True, False, False, True, [0], 256, 1, 100, True, False),
        (False, False, False, True, 1, 512, 2, 200, False, False),
    ],
    ids=["happy-path-1", "happy-path-2", "happy-path-3"],
)
@pytest.mark.skip(reason="Waiting for fix to MaxAttemptedException")
def test_run_faucet_extrinsic_happy_path(
    mock_subtensor,
    mock_wallet,
    mock_pow_solution,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    cuda,
    dev_id,
    tpb,
    num_processes,
    update_interval,
    log_verbose,
    expected,
):
    with patch(
        "bittensor.utils.registration.create_pow", return_value=mock_pow_solution
    ) as mock_create_pow, patch("rich.prompt.Confirm.ask", return_value=True):
        from bittensor.extrinsics.registration import run_faucet_extrinsic

        # Arrange
        mock_subtensor.get_balance.return_value = 100
        mock_subtensor.substrate.submit_extrinsic.return_value.is_success = True

        # Act
        result = run_faucet_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
            cuda=cuda,
            dev_id=dev_id,
            tpb=tpb,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )

        # Assert
        assert result == expected
        mock_subtensor.get_balance.assert_called_with("mock_address")
        mock_subtensor.substrate.submit_extrinsic.assert_called()


@pytest.mark.parametrize(
    "cuda,torch_cuda_available,prompt_response,expected",
    [
        (
            True,
            False,
            False,
            False,
        ),  # ID: edge-case-1: CUDA required but not available, user declines prompt
        (
            True,
            False,
            True,
            False,
        ),  # ID: edge-case-2: CUDA required but not available, user accepts prompt but fails due to CUDA unavailability
    ],
    ids=["edge-case-1", "edge-case-2"],
)
def test_run_faucet_extrinsic_edge_cases(
    mock_subtensor, mock_wallet, cuda, torch_cuda_available, prompt_response, expected
):
    with patch("torch.cuda.is_available", return_value=torch_cuda_available), patch(
        "rich.prompt.Confirm.ask", return_value=prompt_response
    ):
        from bittensor.extrinsics.registration import run_faucet_extrinsic

        # Act
        result = run_faucet_extrinsic(
            subtensor=mock_subtensor, wallet=mock_wallet, cuda=cuda
        )

        # Assert
        assert result == expected


@pytest.mark.parametrize(
    "exception,expected",
    [
        (KeyboardInterrupt, (True, "Done")),  # ID: error-1: User interrupts the process
        (
            MaxSuccessException,
            (True, "Max successes reached: 3"),
        ),  # ID: error-2: Maximum successes reached
        (
            MaxAttemptsException,
            (False, "Max attempts reached: 3"),
        ),  # ID: error-3: Maximum attempts reached
    ],
    ids=["error-1", "error-2", "error-3"],
)
@pytest.mark.skip(reason="Waiting for fix to MaxAttemptedException")
def test_run_faucet_extrinsic_error_cases(
    mock_subtensor, mock_wallet, mock_pow_solution, exception, expected
):
    with patch(
        "bittensor.utils.registration.create_pow",
        side_effect=[mock_pow_solution, exception],
    ):
        from bittensor.extrinsics.registration import run_faucet_extrinsic

        # Act
        result = run_faucet_extrinsic(
            subtensor=mock_subtensor, wallet=mock_wallet, max_allowed_attempts=3
        )

        # Assert
        assert result == expected
