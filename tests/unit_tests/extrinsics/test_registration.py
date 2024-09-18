import pytest
from unittest.mock import MagicMock, patch
from bittensor.subtensor import Subtensor
from bittensor.wallet import wallet as Wallet
from bittensor.utils.registration import POWSolution
from bittensor.extrinsics.registration import (
    MaxSuccessException,
    MaxAttemptsException,
    swap_hotkey_extrinsic,
    burned_register_extrinsic,
    register_extrinsic,
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
    mock.coldkey = MagicMock()
    mock.hotkey = MagicMock()
    mock.hotkey.ss58_address = "fake_ss58_address"
    return mock


@pytest.fixture
def mock_pow_solution():
    mock = MagicMock(spec=POWSolution)
    mock.block_number = 123
    mock.nonce = 456
    mock.seal = [0, 1, 2, 3]
    mock.is_stale.return_value = False
    return mock


@pytest.fixture
def mock_new_wallet():
    mock = MagicMock(spec=Wallet)
    mock.coldkeypub.ss58_address = "mock_address"
    mock.coldkey = MagicMock()
    mock.hotkey = MagicMock()
    return mock


@pytest.mark.parametrize(
    "wait_for_inclusion,wait_for_finalization,prompt,cuda,dev_id,tpb,num_processes,update_interval,log_verbose,expected",
    [
        (False, True, False, False, 0, 256, None, None, False, True),
        (True, False, False, True, [0], 256, 1, 100, True, False),
        (False, False, False, True, 1, 512, 2, 200, False, False),
    ],
    ids=["happy-path-1", "happy-path-2", "happy-path-3"],
)
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
        "bittensor.utils.registration._solve_for_difficulty_fast",
        return_value=mock_pow_solution,
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
        if isinstance(result, tuple):
            assert result[0] == expected
            if result[0] is True:
                # Checks only if successful
                mock_subtensor.substrate.submit_extrinsic.assert_called()
        else:
            assert result == expected
        mock_subtensor.get_balance.assert_called_with("mock_address")


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
        assert result[0] == expected


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


@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, prompt, swap_success, prompt_response, expected_result, test_id",
    [
        # Happy paths
        (False, True, False, True, None, True, "happy-path-finalization-true"),
        (True, False, False, True, None, True, "happy-path-inclusion-true"),
        (True, True, False, True, None, True, "edge-both-waits-true"),
        # Error paths
        (False, True, False, False, None, False, "swap_failed"),
        (False, True, True, True, False, False, "error-prompt-declined"),
    ],
)
def test_swap_hotkey_extrinsic(
    mock_subtensor,
    mock_wallet,
    mock_new_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    swap_success,
    prompt_response,
    expected_result,
    test_id,
):
    # Arrange
    with patch.object(
        mock_subtensor,
        "_do_swap_hotkey",
        return_value=(swap_success, "Mock error message"),
    ):
        with patch(
            "rich.prompt.Confirm.ask", return_value=prompt_response
        ) as mock_confirm:
            # Act
            result = swap_hotkey_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                new_wallet=mock_new_wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                prompt=prompt,
            )

            # Assert
            assert result == expected_result, f"Test failed for test_id: {test_id}"

            if prompt:
                mock_confirm.assert_called_once()
            else:
                mock_confirm.assert_not_called()


@pytest.mark.parametrize(
    "subnet_exists, neuron_is_null, recycle_success, prompt, prompt_response, is_registered, expected_result, test_id",
    [
        # Happy paths
        (True, False, None, False, None, None, True, "neuron-not-null"),
        (True, True, True, True, True, True, True, "happy-path-wallet-registered"),
        # Error paths
        (False, True, False, False, None, None, False, "subnet-non-existence"),
        (True, True, True, True, False, None, False, "prompt-declined"),
        (True, True, False, True, True, False, False, "error-path-recycling-failed"),
        (True, True, True, True, True, False, False, "error-path-not-registered"),
    ],
)
def test_burned_register_extrinsic(
    mock_subtensor,
    mock_wallet,
    subnet_exists,
    neuron_is_null,
    recycle_success,
    prompt,
    prompt_response,
    is_registered,
    expected_result,
    test_id,
):
    # Arrange
    with patch.object(
        mock_subtensor, "subnet_exists", return_value=subnet_exists
    ), patch.object(
        mock_subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=MagicMock(is_null=neuron_is_null),
    ), patch.object(
        mock_subtensor,
        "_do_burned_register",
        return_value=(recycle_success, "Mock error message"),
    ), patch.object(
        mock_subtensor, "is_hotkey_registered", return_value=is_registered
    ), patch("rich.prompt.Confirm.ask", return_value=prompt_response) as mock_confirm:
        # Act
        result = burned_register_extrinsic(
            subtensor=mock_subtensor, wallet=mock_wallet, netuid=123, prompt=True
        )

        # Assert
        assert result == expected_result, f"Test failed for test_id: {test_id}"

        if prompt:
            mock_confirm.assert_called_once()
        else:
            mock_confirm.assert_not_called()


@pytest.mark.parametrize(
    "subnet_exists, neuron_is_null, prompt, prompt_response, cuda_available, expected_result, test_id",
    [
        (False, True, True, True, True, False, "subnet-does-not-exist"),
        (True, False, True, True, True, True, "neuron-already-registered"),
        (True, True, True, False, True, False, "user-declines-prompt"),
        (True, True, False, None, False, False, "cuda-unavailable"),
    ],
)
def test_register_extrinsic_without_pow(
    mock_subtensor,
    mock_wallet,
    subnet_exists,
    neuron_is_null,
    prompt,
    prompt_response,
    cuda_available,
    expected_result,
    test_id,
):
    # Arrange
    with patch.object(
        mock_subtensor, "subnet_exists", return_value=subnet_exists
    ), patch.object(
        mock_subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=MagicMock(is_null=neuron_is_null),
    ), patch("rich.prompt.Confirm.ask", return_value=prompt_response), patch(
        "torch.cuda.is_available", return_value=cuda_available
    ):
        # Act
        result = register_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            netuid=123,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            prompt=prompt,
            max_allowed_attempts=3,
            output_in_place=True,
            cuda=True,
            dev_id=0,
            tpb=256,
            num_processes=None,
            update_interval=None,
            log_verbose=False,
        )

        # Assert
        assert result == expected_result, f"Test failed for test_id: {test_id}"


@pytest.mark.parametrize(
    "pow_success, pow_stale, registration_success, cuda, hotkey_registered, expected_result, test_id",
    [
        (True, False, True, False, False, True, "successful-with-valid-pow"),
        (True, False, True, True, False, True, "successful-with-valid-cuda-pow"),
        # Pow failed but key was registered already
        (False, False, False, False, True, True, "hotkey-registered"),
        # Pow was a success but registration failed with error 'key already registered'
        (True, False, False, False, False, True, "registration-fail-key-registered"),
    ],
)
def test_register_extrinsic_with_pow(
    mock_subtensor,
    mock_wallet,
    mock_pow_solution,
    pow_success,
    pow_stale,
    registration_success,
    cuda,
    hotkey_registered,
    expected_result,
    test_id,
):
    # Arrange
    with patch(
        "bittensor.utils.registration._solve_for_difficulty_fast",
        return_value=mock_pow_solution if pow_success else None,
    ), patch(
        "bittensor.utils.registration._solve_for_difficulty_fast_cuda",
        return_value=mock_pow_solution if pow_success else None,
    ), patch.object(
        mock_subtensor,
        "_do_pow_register",
        return_value=(registration_success, "HotKeyAlreadyRegisteredInSubNet"),
    ), patch("torch.cuda.is_available", return_value=cuda):
        # Act
        if pow_success:
            mock_pow_solution.is_stale.return_value = pow_stale

        if not pow_success and hotkey_registered:
            mock_subtensor.is_hotkey_registered = MagicMock(
                return_value=hotkey_registered
            )

        result = register_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            netuid=123,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            prompt=False,
            max_allowed_attempts=3,
            output_in_place=True,
            cuda=cuda,
            dev_id=0,
            tpb=256,
            num_processes=None,
            update_interval=None,
            log_verbose=False,
        )

        # Assert
        assert result == expected_result, f"Test failed for test_id: {test_id}"
