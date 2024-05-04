import bittensor
import pytest

from unittest.mock import patch, MagicMock

from bittensor.utils.balance import Balance
from bittensor.extrinsics.unstaking import unstake_extrinsic


@pytest.fixture
def mock_subtensor():
    mock = MagicMock(spec=bittensor.subtensor)
    mock.network = "mock_network"
    return mock


@pytest.fixture
def mock_wallet():
    mock = MagicMock(spec=bittensor.wallet)
    mock.hotkey.ss58_address = "5FHneW46..."
    mock.coldkeypub.ss58_address = "5Gv8YYFu8..."
    mock.hotkey_str = "mock_hotkey_str"
    return mock


def mock_get_minimum_required_stake():
    # Valid minimum threshold as of 2024/05/01
    return Balance.from_rao(100_000_000)


@pytest.mark.parametrize(
    "hotkey_ss58, amount, wait_for_inclusion, wait_for_finalization, prompt, user_accepts, expected_success, unstake_attempted",
    [
        # Successful unstake without waiting for inclusion or finalization
        (None, 10.0, False, False, False, None, True, True),
        # Successful unstake with prompt accepted
        ("5FHneW46...", 10.0, True, True, True, True, True, True),
        # Prompt declined
        ("5FHneW46...", 10.0, True, True, True, False, False, False),
        # Not enough stake to unstake
        ("5FHneW46...", 1000.0, True, True, False, None, False, False),
        # Unsuccessful - unstake threshold not reached
        (None, 0.01, True, True, False, None, False, False),
        # Successful unstaking all
        (None, None, False, False, False, None, True, True),
        # Failure - unstaking failed
        (None, 10.0, False, False, False, None, False, True),
    ],
    ids=[
        "successful-no-wait",
        "successful-with-prompt",
        "failure-prompt-declined",
        "failure-not-enough-stake",
        "failure-threshold-not-reached",
        "success-unstake-all",
        "failure-unstake-failed",
    ],
)
def test_unstake_extrinsic(
    mock_subtensor,
    mock_wallet,
    hotkey_ss58,
    amount,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    user_accepts,
    expected_success,
    unstake_attempted,
):
    mock_current_stake = Balance.from_tao(50)
    mock_current_balance = Balance.from_tao(100)

    with patch.object(
        mock_subtensor, "_do_unstake", return_value=(expected_success)
    ), patch.object(
        mock_subtensor, "get_balance", return_value=mock_current_balance
    ), patch.object(
        mock_subtensor,
        "get_minimum_required_stake",
        side_effect=mock_get_minimum_required_stake,
    ), patch.object(
        mock_subtensor,
        "get_stake_for_coldkey_and_hotkey",
        return_value=mock_current_stake,
    ), patch(
        "rich.prompt.Confirm.ask", return_value=user_accepts
    ) as mock_confirm:
        result = unstake_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            hotkey_ss58=hotkey_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

        assert (
            result == expected_success
        ), f"Expected result {expected_success}, but got {result}"

        if prompt:
            mock_confirm.assert_called_once()

        if unstake_attempted:
            mock_subtensor._do_unstake.assert_called_once_with(
                wallet=mock_wallet,
                hotkey_ss58=hotkey_ss58 or mock_wallet.hotkey.ss58_address,
                amount=bittensor.Balance.from_tao(amount)
                if amount
                else mock_current_stake,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        else:
            mock_subtensor._do_unstake.assert_not_called()
