import pytest
from unittest.mock import MagicMock, patch
from bittensor.utils.coldkey_swap_pow import (
    create_pow_for_coldkey_swap,
    _calculate_difficulty,
)
from bittensor.utils.registration import POWSolution


@pytest.fixture
def mock_subtensor():
    return MagicMock()


@pytest.fixture
def mock_wallet():
    return MagicMock()


@pytest.fixture
def pow_params():
    return {
        "base_difficulty": 1000000,
        "output_in_place": False,
        "cuda": False,
        "num_processes": 1,
        "update_interval": 1000,
        "log_verbose": False,
    }


@pytest.mark.parametrize(
    "swap_attempt, base_difficulty, test_coldkey_str",
    [
        (0, 1000000, "coldkey_address_ss58"),
        (1, 1000000, "coldkey_address_ss58"),
        (2, 1000000, "coldkey_address_ss58"),
        (3, 1000000, "coldkey_address_ss58"),
        (4, 1000000, "coldkey_address_ss58"),
    ],
)
def test_pow_calculation_across_swap_attempts(
    mock_subtensor,
    mock_wallet,
    pow_params,
    swap_attempt,
    base_difficulty,
    test_coldkey_str,
):
    expected_difficulty = _calculate_difficulty(base_difficulty, swap_attempt)

    with patch(
        "bittensor.utils.coldkey_swap_pow._solve_for_coldkey_swap_difficulty_cpu"
    ) as mock_solve:
        # Mock the POW solution
        mock_solution = POWSolution(
            nonce=12345,
            block_number=100,
            difficulty=expected_difficulty,
            seal=b"test_seal",
        )
        mock_solve.return_value = mock_solution

        # Call the create_pow function
        result = create_pow_for_coldkey_swap(
            mock_subtensor,
            mock_wallet,
            test_coldkey_str,
            pow_params["output_in_place"],
            pow_params["cuda"],
            num_processes=pow_params["num_processes"],
            update_interval=pow_params["update_interval"],
            log_verbose=pow_params["log_verbose"],
        )

        # Assert that _solve_for_coldkey_swap_difficulty_cpu was called with correct parameters
        mock_solve.assert_called_once_with(
            mock_subtensor,
            mock_wallet,
            test_coldkey_str,
            pow_params["output_in_place"],
            pow_params["num_processes"],
            pow_params["update_interval"],
            log_verbose=pow_params["log_verbose"],
        )

        # Assert that the result matches the expected POW solution
        assert result == mock_solution
        assert (
            result.difficulty == expected_difficulty
        ), f"Expected {expected_difficulty}, got {result.difficulty}"


def test_pow_calculation_failure(mock_subtensor, mock_wallet, pow_params):
    with patch(
        "bittensor.utils.coldkey_swap_pow._solve_for_coldkey_swap_difficulty_cpu"
    ) as mock_solve:
        # Mock the POW solution to return None (failure case)
        mock_solve.return_value = None

        # Call the create_pow function
        result = create_pow_for_coldkey_swap(
            mock_subtensor,
            mock_wallet,
            old_coldkey="coldkey_address_ss58",
            output_in_place=pow_params["output_in_place"],
            cuda=pow_params["cuda"],
            num_processes=pow_params["num_processes"],
            update_interval=pow_params["update_interval"],
            log_verbose=pow_params["log_verbose"],
        )

        # Assert that _solve_for_coldkey_swap_difficulty_cpu was called
        mock_solve.assert_called_once()

        # Assert that the result is None
        assert result is None
