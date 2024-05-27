import pytest
from unittest.mock import MagicMock, patch
from bittensor.subtensor import Subtensor
from bittensor.wallet import wallet as Wallet
from bittensor.utils.balance import Balance
from bittensor.extrinsics.delegation import (
    nominate_extrinsic,
    delegate_extrinsic,
    undelegate_extrinsic,
)
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
def test_nominate_extrinsic(
    mock_subtensor,
    mock_wallet,
    already_delegate,
    nomination_success,
    raises_exception,
    expected_result,
):
    # Arrange
    with patch.object(
        mock_subtensor, "is_hotkey_delegate", return_value=already_delegate
    ), patch.object(
        mock_subtensor, "_do_nominate", return_value=nomination_success
    ) as mock_nominate:
        if raises_exception:
            mock_subtensor._do_nominate.side_effect = raises_exception

        # Act
        result = nominate_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            wait_for_finalization=False,
            wait_for_inclusion=True,
        )
        # Assert
        assert result == expected_result

        if not already_delegate and nomination_success is not None:
            mock_nominate.assert_called_once_with(
                wallet=mock_wallet, wait_for_inclusion=True, wait_for_finalization=False
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
def test_delegate_extrinsic(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    is_delegate,
    prompt_response,
    stake_amount,
    balance_sufficient,
    transaction_success,
    raises_error,
    expected_result,
    delegate_called,
):
    # Arrange
    wallet_balance = Balance.from_tao(500)
    wallet_insufficient_balance = Balance.from_tao(0.002)

    with patch("rich.prompt.Confirm.ask", return_value=prompt_response), patch.object(
        mock_subtensor,
        "get_balance",
        return_value=wallet_balance
        if balance_sufficient
        else wallet_insufficient_balance,
    ), patch.object(
        mock_subtensor, "is_hotkey_delegate", return_value=is_delegate
    ), patch.object(
        mock_subtensor, "_do_delegation", return_value=transaction_success
    ) as mock_delegate:
        if raises_error:
            mock_delegate.side_effect = raises_error

        # Act
        if raises_error == NotDelegateError:
            with pytest.raises(raises_error):
                result = delegate_extrinsic(
                    subtensor=mock_subtensor,
                    wallet=mock_wallet,
                    delegate_ss58=mock_wallet.hotkey.ss58_address,
                    amount=stake_amount,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    prompt=True,
                )
        else:
            result = delegate_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                delegate_ss58=mock_wallet.hotkey.ss58_address,
                amount=stake_amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                prompt=True,
            )
            # Assert
            assert result == expected_result

        if delegate_called:
            if stake_amount is None:
                called_stake_amount = wallet_balance
            elif isinstance(stake_amount, Balance):
                called_stake_amount = stake_amount
            else:
                called_stake_amount = Balance.from_tao(stake_amount)

            if called_stake_amount > Balance.from_rao(1000):
                called_stake_amount -= Balance.from_rao(1000)

            mock_delegate.assert_called_once_with(
                wallet=mock_wallet,
                delegate_ss58=mock_wallet.hotkey.ss58_address,
                amount=called_stake_amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
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
def test_undelegate_extrinsic(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    is_delegate,
    prompt_response,
    unstake_amount,
    current_stake,
    transaction_success,
    raises_error,
    expected_result,
):
    # Arrange
    wallet_balance = Balance.from_tao(500)

    with patch("rich.prompt.Confirm.ask", return_value=prompt_response), patch.object(
        mock_subtensor, "is_hotkey_delegate", return_value=is_delegate
    ), patch.object(
        mock_subtensor, "get_balance", return_value=wallet_balance
    ), patch.object(
        mock_subtensor,
        "get_stake_for_coldkey_and_hotkey",
        return_value=Balance.from_tao(current_stake),
    ), patch.object(
        mock_subtensor, "_do_undelegation", return_value=transaction_success
    ) as mock_undelegate:
        if raises_error:
            mock_undelegate.side_effect = raises_error

        # Act
        if raises_error == NotDelegateError:
            with pytest.raises(raises_error):
                result = undelegate_extrinsic(
                    subtensor=mock_subtensor,
                    wallet=mock_wallet,
                    delegate_ss58=mock_wallet.hotkey.ss58_address,
                    amount=unstake_amount,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    prompt=True,
                )
        else:
            result = undelegate_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                delegate_ss58=mock_wallet.hotkey.ss58_address,
                amount=unstake_amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                prompt=True,
            )

            # Assert
            assert result == expected_result

        if expected_result and prompt_response:
            if unstake_amount is None:
                called_unstake_amount = Balance.from_tao(current_stake)
            elif isinstance(unstake_amount, Balance):
                called_unstake_amount = unstake_amount
            else:
                called_unstake_amount = Balance.from_tao(unstake_amount)

            mock_undelegate.assert_called_once_with(
                wallet=mock_wallet,
                delegate_ss58=mock_wallet.hotkey.ss58_address,
                amount=called_unstake_amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
