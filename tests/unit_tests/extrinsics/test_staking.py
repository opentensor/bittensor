import pytest
from unittest.mock import patch, MagicMock
import bittensor
from bittensor.utils.balance import Balance
from bittensor.extrinsics.staking import (
    add_stake_extrinsic,
    add_stake_multiple_extrinsic,
)
from bittensor.errors import NotDelegateError


# Mocking external dependencies
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
    mock.name = "mock_wallet"
    return mock


@pytest.fixture
def mock_other_owner_wallet():
    mock = MagicMock(spec=bittensor.wallet)
    mock.hotkey.ss58_address = "11HneC46..."
    mock.coldkeypub.ss58_address = "6Gv9ZZFu8..."
    mock.hotkey_str = "mock_hotkey_str_other_owner"
    mock.name = "mock_wallet_other_owner"
    return mock


# Parametrized test cases
@pytest.mark.parametrize(
    "hotkey_ss58, hotkey_owner, hotkey_delegate, amount, wait_for_inclusion, wait_for_finalization, prompt, user_accepts, expected_success, exception",
    [
        # Simple staking to own hotkey, float
        (None, True, None, 10.0, True, False, False, None, True, None),
        # Simple staking to own hotkey, int
        (None, True, None, 10, True, False, False, None, True, None),
        # Not waiting for inclusion & finalization, own hotkey
        ("5FHneW46...", True, None, 10.0, False, False, False, None, True, None),
        # Prompt refused
        (None, True, None, 10.0, True, False, True, False, False, None),
        # Stake all
        (None, True, None, None, True, False, False, None, True, None),
        # Insufficient balance
        (None, True, None, 110, True, False, False, None, False, None),
        # No deduction scenario
        (None, True, None, 0.000000100, True, False, False, None, True, None),
        # Not owner but Delegate
        ("5FHneW46...", False, True, 10.0, True, False, False, None, True, None),
        # Not owner but Delegate and prompt refused
        ("5FHneW46...", False, True, 10.0, True, False, True, False, False, None),
        # Not owner and not delegate
        (
            "5FHneW46...",
            False,
            False,
            10.0,
            True,
            False,
            False,
            None,
            False,
            NotDelegateError,
        ),
        # Staking failed
        (None, True, None, 10.0, True, False, False, None, False, None),
    ],
    ids=[
        "success-own-hotkey-float",
        "success-own-hotkey-int",
        "success-own-hotkey-no-wait",
        "prompt-refused",
        "success-staking-all",
        "failure-insufficient-balance",
        "success-no-deduction",
        "success-delegate",
        "failure-delegate-prompt-refused",
        "failure-not-delegate",
        "failure-staking",
    ],
)
def test_add_stake_extrinsic(
    mock_subtensor,
    mock_wallet,
    mock_other_owner_wallet,
    hotkey_ss58,
    hotkey_owner,
    hotkey_delegate,
    amount,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    user_accepts,
    expected_success,
    exception,
):
    # Arrange
    if not amount:
        staking_balance = amount if amount else Balance.from_tao(100)
    else:
        staking_balance = (
            Balance.from_tao(amount)
            if not isinstance(amount, bittensor.Balance)
            else amount
        )

    with patch.object(
        mock_subtensor, "_do_stake", return_value=expected_success
    ) as mock_add_stake, patch.object(
        mock_subtensor, "get_balance", return_value=Balance.from_tao(100)
    ), patch.object(
        mock_subtensor,
        "get_stake_for_coldkey_and_hotkey",
        return_value=Balance.from_tao(50),
    ), patch.object(
        mock_subtensor,
        "get_hotkey_owner",
        return_value=mock_wallet.coldkeypub.ss58_address
        if hotkey_owner
        else mock_other_owner_wallet.coldkeypub.ss58_address,
    ), patch.object(
        mock_subtensor, "is_hotkey_delegate", return_value=hotkey_delegate
    ), patch.object(mock_subtensor, "get_delegate_take", return_value=0.01), patch(
        "rich.prompt.Confirm.ask", return_value=user_accepts
    ) as mock_confirm, patch.object(
        mock_subtensor,
        "get_minimum_required_stake",
        return_value=bittensor.Balance.from_tao(0.01),
    ), patch.object(
        mock_subtensor,
        "get_existential_deposit",
        return_value=bittensor.Balance.from_rao(100_000),
    ):
        mock_balance = mock_subtensor.get_balance()
        existential_deposit = mock_subtensor.get_existential_deposit()
        if staking_balance > mock_balance - existential_deposit:
            staking_balance = mock_balance - existential_deposit

        # Act
        if not hotkey_owner and not hotkey_delegate:
            with pytest.raises(exception):
                result = add_stake_extrinsic(
                    subtensor=mock_subtensor,
                    wallet=mock_wallet,
                    hotkey_ss58=hotkey_ss58,
                    amount=amount,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    prompt=prompt,
                )
        else:
            result = add_stake_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58=hotkey_ss58,
                amount=amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                prompt=prompt,
            )

            # Assert
            assert (
                result == expected_success
            ), f"Expected {expected_success}, but got {result}"

            if prompt:
                mock_confirm.assert_called_once()

            if expected_success:
                if not hotkey_ss58:
                    hotkey_ss58 = mock_wallet.hotkey.ss58_address

                mock_add_stake.assert_called_once_with(
                    wallet=mock_wallet,
                    hotkey_ss58=hotkey_ss58,
                    amount=staking_balance,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )


# Parametrized test cases
@pytest.mark.parametrize(
    "hotkey_ss58s, amounts, hotkey_owner, hotkey_delegates ,wallet_balance, wait_for_inclusion, wait_for_finalization, prompt, prompt_response, stake_responses, expected_success, stake_attempted, exception, exception_msg",
    [
        # Successful stake
        (
            ["5FHneW46...", "11HneC46..."],
            [10.0, 20.0],
            [True, True],
            [None, None],
            100.0,
            True,
            False,
            False,
            None,
            [True, True],
            True,
            2,
            None,
            None,
        ),
        # Successful stake with prompt
        (
            ["5FHneW46...", "11HneC46..."],
            [10.0, 20.0],
            [True, True],
            [None, None],
            100.0,
            True,
            True,
            True,
            True,
            [True, True],
            True,
            2,
            None,
            None,
        ),
        # Successful stake, no deduction scenario
        (
            ["5FHneW46...", "11HneC46..."],
            [0.000000100, 0.000000100],
            [True, True],
            [None, None],
            100.0,
            True,
            False,
            False,
            None,
            [True, True],
            True,
            2,
            None,
            None,
        ),
        # Successful stake, not waiting for finalization & inclusion
        (
            ["5FHneW46...", "11HneC46..."],
            [10.0, 20.0],
            [True, True],
            [None, None],
            100.0,
            False,
            False,
            False,
            None,
            [True, True],
            True,
            2,
            None,
            None,
        ),
        # Successful stake, one key is a delegate
        (
            ["5FHneW46...", "11HneC46..."],
            [10.0, 20.0],
            [True, False],
            [True, True],
            100.0,
            True,
            False,
            False,
            None,
            [True, True],
            True,
            2,
            None,
            None,
        ),
        # Partial successful stake, one key is not a delegate
        (
            ["5FHneW46...", "11HneC46..."],
            [10.0, 20.0],
            [True, False],
            [True, False],
            100.0,
            True,
            False,
            False,
            None,
            [True, False],
            True,
            1,
            None,
            None,
        ),
        # Successful, staking all tao to first wallet, not waiting for finalization + inclusion
        (
            ["5FHneW46...", "11HneC46..."],
            None,
            [True, True],
            [None, None],
            100.0,
            False,
            False,
            False,
            None,
            [True, False],
            True,
            1,
            None,
            None,
        ),
        # Successful, staking all tao to first wallet
        (
            ["5FHneW46...", "11HneC46..."],
            None,
            [True, True],
            [None, None],
            100.0,
            True,
            False,
            False,
            None,
            [True, False],
            True,
            1,
            None,
            None,
        ),
        # Success, staking 0 tao
        (
            ["5FHneW46...", "11HneC46..."],
            [0.0, 0.0],
            [True, True],
            [None, None],
            100.0,
            True,
            False,
            False,
            None,
            [None, None],
            True,
            0,
            None,
            None,
        ),
        # Complete failure to stake for both keys
        (
            ["5FHneW46...", "11HneC46..."],
            [10.0, 20.0],
            [True, True],
            [None, None],
            100.0,
            True,
            False,
            False,
            None,
            [False, False],
            False,
            2,
            None,
            None,
        ),
        # Complete failure, both keys are not delegates
        (
            ["5FHneW46...", "11HneC46..."],
            [10.0, 20.0],
            [False, False],
            [False, False],
            100.0,
            True,
            False,
            False,
            None,
            [False, False],
            False,
            0,
            None,
            None,
        ),
        # Unsuccessful stake with prompt declined both times
        (
            ["5FHneW46...", "11HneC46..."],
            [10.0, 20.0],
            [True, True],
            [None, None],
            100.0,
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
            [123, "11HneC46..."],
            [10.0, 20.0],
            [False, False],
            [False, False],
            100.0,
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
            ["5FHneW46...", "11HneC46..."],
            [10.0],
            [False, False],
            [False, False],
            100.0,
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
            ["5FHneW46...", "11HneC46..."],
            ["abc", 12],
            [False, False],
            [False, False],
            100.0,
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
        "success-basic-path",
        "success-with-prompt",
        "success-no-deduction",
        "success-no-wait",
        "success-one-delegate",
        "partial-success-one-not-delegate",
        "success-all-tao-no-wait",
        "success-all-tao",
        "success-0-tao",
        "failure-both-keys",
        "failure-both-not-delegates",
        "failure-prompt-declined",
        "failure-type-error-hotkeys",
        "failure-value-error-amount",
        "failure-type-error-amount",
    ],
)
def test_add_stake_multiple_extrinsic(
    mock_subtensor,
    mock_wallet,
    mock_other_owner_wallet,
    hotkey_ss58s,
    amounts,
    hotkey_owner,
    hotkey_delegates,
    wallet_balance,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    prompt_response,
    stake_responses,
    expected_success,
    stake_attempted,
    exception,
    exception_msg,
):
    # Arrange
    def hotkey_delegate_side_effect(hotkey_ss58):
        index = hotkey_ss58s.index(hotkey_ss58)
        return hotkey_delegates[index]

    def owner_side_effect(hotkey_ss58):
        index = hotkey_ss58s.index(hotkey_ss58)
        return (
            mock_wallet.coldkeypub.ss58_address
            if hotkey_owner[index]
            else mock_other_owner_wallet.coldkeypub.ss58_address
        )

    def stake_side_effect(hotkey_ss58, *args, **kwargs):
        index = hotkey_ss58s.index(hotkey_ss58)
        return stake_responses[index]

    with patch.object(
        mock_subtensor, "get_balance", return_value=Balance.from_tao(wallet_balance)
    ), patch.object(
        mock_subtensor, "is_hotkey_delegate", side_effect=hotkey_delegate_side_effect
    ), patch.object(
        mock_subtensor, "get_hotkey_owner", side_effect=owner_side_effect
    ), patch.object(
        mock_subtensor, "_do_stake", side_effect=stake_side_effect
    ) as mock_do_stake, patch.object(
        mock_subtensor, "tx_rate_limit", return_value=0
    ), patch("rich.prompt.Confirm.ask", return_value=prompt_response) as mock_confirm:
        # Act
        if exception:
            with pytest.raises(exception) as exc_info:
                result = add_stake_multiple_extrinsic(
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
            result = add_stake_multiple_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58s=hotkey_ss58s,
                amounts=amounts,
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
            assert mock_do_stake.call_count == stake_attempted
