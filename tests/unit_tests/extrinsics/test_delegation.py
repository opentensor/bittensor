import pytest
from unittest.mock import MagicMock, patch
from bittensor.subtensor import subtensor as Subtensor
from bittensor.wallet import wallet as Wallet
from bittensor.utils.balance import Balance
from bittensor.errors import (
    NominationError,
    NotDelegateError,
    NotRegisteredError,
    StakeError,
)


@pytest.fixture
def mock_subtensor():
    mock = MagicMock(spec=Subtensor)
    mock.network = "magic_mock"
    return mock


@pytest.fixture
def mock_wallet():
    mock = MagicMock(spec=Wallet)
    mock.hotkey.ss58_address = "fake_hotkey_address"
    mock.coldkey.ss58_address = "fake_coldkey_address"
    mock.coldkey = MagicMock()
    mock.hotkey = MagicMock()
    mock.name = "fake_wallet_name"
    mock.hotkey_str = "fake_hotkey_str"
    return mock


@pytest.mark.parametrize(
    "already_delegate, nomination_success, raises_exception, expected_result",
    [
        (False, True, None, True),  # Successful nomination
        (True, None, None, False),  # Already a delegate
        (False, None, NominationError, False),  # Failure - Nomination error
        (False, None, ValueError, False),  # Failure - ValueError
    ],
    ids=[
        "success-nomination-done",
        "failure-already-delegate",
        "failure-nomination-error",
        "failure-value-error",
    ],
)

@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, is_delegate, prompt_response, stake_amount, balance_sufficient, transaction_success, raises_error, expected_result, delegate_called",
    [
        (True, False, True, True, 100, True, True, None, True, True),  # Success case
        (
            False,
            False,
            True,
            True,
            100,
            True,
            True,
            None,
            True,
            True,
        ),  # Success case - no wait
        (
            True,
            False,
            True,
            True,
            None,
            True,
            True,
            None,
            True,
            True,
        ),  # Success case - all stake
        (
            True,
            False,
            True,
            True,
            0.000000100,
            True,
            True,
            None,
            True,
            True,
        ),  # Success case - below cutoff threshold
        (
            True,
            False,
            True,
            True,
            Balance.from_tao(1),
            True,
            True,
            None,
            True,
            True,
        ),  # Success case - from Tao
        (
            True,
            False,
            False,
            None,
            100,
            True,
            False,
            NotDelegateError,
            False,
            False,
        ),  # Not a delegate error
        (
            True,
            False,
            True,
            True,
            200,
            False,
            False,
            None,
            False,
            False,
        ),  # Insufficient balance
        (
            True,
            False,
            True,
            False,
            100,
            True,
            True,
            None,
            False,
            False,
        ),  # User declines prompt
        (
            True,
            False,
            True,
            True,
            100,
            True,
            False,
            None,
            False,
            True,
        ),  # Transaction fails
        (
            True,
            False,
            True,
            True,
            100,
            True,
            False,
            NotRegisteredError,
            False,
            True,
        ),  # Raises a NotRegisteredError
        (
            True,
            False,
            True,
            True,
            100,
            True,
            False,
            StakeError,
            False,
            True,
        ),  # Raises a StakeError
    ],
    ids=[
        "success-delegate",
        "success-no-wait",
        "success-all-stake",
        "success-below-existential-threshold",
        "success-from-tao",
        "failure-not-delegate",
        "failure-low-balance",
        "failure-prompt-declined",
        "failure-transaction-failed",
        "failure-NotRegisteredError",
        "failure-StakeError",
    ],
)

@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, is_delegate, prompt_response, unstake_amount, current_stake, transaction_success, raises_error, expected_result",
    [
        (True, False, True, True, 50, 100, True, None, True),  # Success case
        (False, False, True, True, 50, 100, True, None, True),  # Success case - no wait
        (
            False,
            False,
            True,
            True,
            Balance.from_tao(1),
            100,
            True,
            None,
            True,
        ),  # Success case - from tao
        (True, False, True, True, None, 100, True, None, True),  # Success - unstake all
        (
            True,
            False,
            True,
            True,
            1000,
            1000,
            False,
            None,
            False,
        ),  # Failure - transaction fails
        (
            True,
            False,
            False,
            None,
            100,
            120,
            True,
            NotDelegateError,
            False,
        ),  # Not a delegate
        (True, False, True, False, 100, 111, True, None, False),  # User declines prompt
        (
            True,
            False,
            True,
            True,
            100,
            90,
            True,
            None,
            False,
        ),  # Insufficient stake to unstake
        (
            True,
            False,
            True,
            True,
            100,
            100,
            False,
            StakeError,
            False,
        ),  # StakeError raised
        (
            True,
            False,
            True,
            True,
            100,
            100,
            False,
            NotRegisteredError,
            False,
        ),  # NotRegisteredError raised
    ],
    ids=[
        "success-undelegate",
        "success-undelegate-no-wait",
        "success-from-tao",
        "success-undelegate-all",
        "failure-transaction-failed",
        "failure-NotDelegateError",
        "failure-prompt-declined",
        "failure-insufficient-stake",
        "failure--StakeError",
        "failure-NotRegisteredError",
    ],
)
