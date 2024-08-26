import bittensor
import pytest

from unittest.mock import patch, MagicMock

from bittensor.utils.balance import Balance
from bittensor.extrinsics.unstaking import unstake_extrinsic, unstake_multiple_extrinsic


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
        # Successful - unstake threshold not reached
        (None, 0.01, True, True, False, None, True, True),
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
        "success-threshold-not-reached",
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
    ), patch("rich.prompt.Confirm.ask", return_value=user_accepts) as mock_confirm:
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


@pytest.mark.parametrize(
    # TODO: Write dynamic test to test for amount = None with multiple hotkeys
    "hotkey_ss58s, amounts, wallet_balance, wait_for_inclusion, wait_for_finalization, prompt, prompt_response, unstake_responses, expected_success, unstake_attempted, exception, exception_msg",
    [
        # Successful unstake - no wait
        (
            ["5FHneW46...", "5FHneW47..."],
            [10.0, 20.0],
            100,
            False,
            False,
            True,
            True,
            [True, True],
            True,
            2,
            None,
            None,
        ),
        # Partial-success unstake - one unstake fails
        (
            ["5FHneW46...", "5FHneW47..."],
            [10.0, 20.0],
            100,
            True,
            False,
            True,
            True,
            [True, False],
            True,
            2,
            None,
            None,
        ),
        # Success, based on no hotkeys - func to be confirmed
        ([], [], 100, True, True, False, None, [None], True, 0, None, None),
        # Unsuccessful unstake - not enough stake
        (
            ["5FHneW46..."],
            [1000.0],
            100,
            True,
            True,
            False,
            True,
            [None],
            False,
            0,
            None,
            None,
        ),
        # Successful unstake - new stake below threshold
        (
            ["5FHneW46..."],
            [
                100 - mock_get_minimum_required_stake() + 0.01
            ],  # New stake just below threshold
            100,
            True,
            True,
            False,
            True,
            [True],
            True,  # Sucessful unstake
            1,
            None,
            None,
        ),
        # Unsuccessful unstake with prompt declined both times
        (
            ["5FHneW46...", "5FHneW48..."],
            [10.0, 10.0],
            100,
            True,
            True,
            True,
            False,
            [None, None],
            False,
            0,
            None,
            None,
        ),
        # Exception, TypeError for incorrect hotkey_ss58s
        (
            ["5FHneW46...", 123],
            [10.0, 20.0],
            100,
            True,
            False,
            False,
            None,
            [None, None],
            None,
            0,
            TypeError,
            "hotkey_ss58s must be a list of str",
        ),
        # Exception, ValueError for mismatch between hotkeys and amounts
        (
            ["5FHneW46...", "5FHneW48..."],
            [10.0],
            100,
            True,
            False,
            False,
            None,
            [None, None],
            None,
            0,
            ValueError,
            "amounts must be a list of the same length as hotkey_ss58s",
        ),
        # Exception, TypeError for incorrect amounts
        (
            ["5FHneW46...", "5FHneW48..."],
            [10.0, "tao"],
            100,
            True,
            False,
            False,
            None,
            [None, None],
            None,
            0,
            TypeError,
            "amounts must be a [list of bittensor.Balance or float] or None",
        ),
    ],
    ids=[
        "success-no-wait",
        "partial-success-one-fail",
        "success-no-hotkey",
        "failure-not-enough-stake",
        "success-threshold-not-reached",
        "failure-prompt-declined",
        "failure-type-error-hotkeys",
        "failure-value-error-amounts",
        "failure-type-error-amounts",
    ],
)
def test_unstake_multiple_extrinsic(
    mock_subtensor,
    mock_wallet,
    hotkey_ss58s,
    amounts,
    wallet_balance,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    prompt_response,
    unstake_responses,
    expected_success,
    unstake_attempted,
    exception,
    exception_msg,
):
    # Arrange
    mock_current_stake = Balance.from_tao(100)
    amounts_in_balances = [
        Balance.from_tao(amount) if isinstance(amount, float) else amount
        for amount in amounts
    ]

    def unstake_side_effect(hotkey_ss58, *args, **kwargs):
        index = hotkey_ss58s.index(hotkey_ss58)
        return unstake_responses[index]

    with patch.object(
        mock_subtensor, "_do_unstake", side_effect=unstake_side_effect
    ) as mock_unstake, patch.object(
        mock_subtensor,
        "get_minimum_required_stake",
        side_effect=mock_get_minimum_required_stake,
    ), patch.object(
        mock_subtensor, "get_balance", return_value=Balance.from_tao(wallet_balance)
    ), patch.object(mock_subtensor, "tx_rate_limit", return_value=0), patch.object(
        mock_subtensor,
        "get_stake_for_coldkey_and_hotkey",
        return_value=mock_current_stake,
    ), patch("rich.prompt.Confirm.ask", return_value=prompt_response) as mock_confirm:
        # Act
        if exception:
            with pytest.raises(exception) as exc_info:
                result = unstake_multiple_extrinsic(
                    subtensor=mock_subtensor,
                    wallet=mock_wallet,
                    hotkey_ss58s=hotkey_ss58s,
                    amounts=amounts,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    prompt=prompt,
                )
            # Assert
            assert str(exc_info.value) == exception_msg

        # Act
        else:
            result = unstake_multiple_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58s=hotkey_ss58s,
                amounts=amounts_in_balances,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                prompt=prompt,
            )

            # Assert
            assert (
                result == expected_success
            ), f"Expected {expected_success}, but got {result}"
            if prompt:
                assert mock_confirm.called
            assert mock_unstake.call_count == unstake_attempted
