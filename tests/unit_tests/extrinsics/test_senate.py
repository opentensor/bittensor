import pytest
from unittest.mock import MagicMock, patch
from bittensor import subtensor, wallet
from bittensor.extrinsics.senate import (
    leave_senate_extrinsic,
    register_senate_extrinsic,
    vote_senate_extrinsic,
)


# Mocking external dependencies
@pytest.fixture
def mock_subtensor():
    mock = MagicMock(spec=subtensor)
    mock.substrate = MagicMock()
    return mock


@pytest.fixture
def mock_wallet():
    mock = MagicMock(spec=wallet)
    mock.coldkey = MagicMock()
    mock.hotkey = MagicMock()
    mock.hotkey.ss58_address = "fake_hotkey_address"
    mock.is_senate_member = None
    return mock


# Parametrized test cases
@pytest.mark.parametrize(
    "wait_for_inclusion,wait_for_finalization,prompt,response_success,is_registered,expected_result, test_id",
    [
        # Happy path tests
        (False, True, False, True, True, True, "happy-path-finalization-true"),
        (True, False, False, True, True, True, "happy-path-inclusion-true"),
        (False, False, False, True, True, True, "happy-path-no_wait"),
        # Edge cases
        (True, True, False, True, True, True, "edge-both-waits-true"),
        # Error cases
        (False, True, False, False, False, None, "error-finalization-failed"),
        (True, False, False, False, False, None, "error-inclusion-failed"),
        (False, True, True, True, False, False, "error-prompt-declined"),
    ],
)
def test_register_senate_extrinsic(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    response_success,
    is_registered,
    expected_result,
    test_id,
):
    # Arrange
    with patch(
        "bittensor.extrinsics.senate.Confirm.ask", return_value=not prompt
    ), patch("bittensor.extrinsics.senate.time.sleep"), patch.object(
        mock_subtensor.substrate, "compose_call"
    ), patch.object(mock_subtensor.substrate, "create_signed_extrinsic"), patch.object(
        mock_subtensor.substrate,
        "submit_extrinsic",
        return_value=MagicMock(
            is_success=response_success,
            process_events=MagicMock(),
            error_message="error",
        ),
    ) as mock_submit_extrinsic, patch.object(
        mock_wallet, "is_senate_member", return_value=is_registered
    ):
        # Act
        result = register_senate_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

        # Assert
        assert result == expected_result, f"Test ID: {test_id}"


@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, prompt, response_success, \
        vote, vote_in_ayes, vote_in_nays, expected_result, test_id",
    [
        # Happy path tests
        (False, True, False, True, True, True, False, True, "happy-finalization-aye"),
        (True, False, False, True, False, False, True, True, "happy-inclusion-nay"),
        (False, False, False, True, True, True, False, True, "happy-no-wait-aye"),
        # Edge cases
        (True, True, False, True, True, True, False, True, "edge-both-waits-true-aye"),
        # Error cases
        (True, False, False, False, True, False, False, None, "error-inclusion-failed"),
        (True, False, True, True, True, True, False, False, "error-prompt-declined"),
        (
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            None,
            "error-no-vote-registered-aye",
        ),
        (
            False,
            True,
            False,
            True,
            False,
            False,
            False,
            None,
            "error-no-vote-registered-nay",
        ),
        (
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            None,
            "error-finalization-failed",
        ),
    ],
)
def test_vote_senate_extrinsic(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    vote,
    response_success,
    vote_in_ayes,
    vote_in_nays,
    expected_result,
    test_id,
):
    # Arrange
    proposal_hash = "mock_hash"
    proposal_idx = 123

    with patch(
        "bittensor.extrinsics.senate.Confirm.ask", return_value=not prompt
    ), patch("bittensor.extrinsics.senate.time.sleep"), patch.object(
        mock_subtensor.substrate, "compose_call"
    ), patch.object(mock_subtensor.substrate, "create_signed_extrinsic"), patch.object(
        mock_subtensor.substrate,
        "submit_extrinsic",
        return_value=MagicMock(
            is_success=response_success,
            process_events=MagicMock(),
            error_message="error",
        ),
    ), patch.object(
        mock_subtensor,
        "get_vote_data",
        return_value={
            "ayes": [mock_wallet.hotkey.ss58_address] if vote_in_ayes else [],
            "nays": [mock_wallet.hotkey.ss58_address] if vote_in_nays else [],
        },
    ):
        # Act
        result = vote_senate_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            proposal_hash=proposal_hash,
            proposal_idx=proposal_idx,
            vote=vote,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

        # Assert
        assert result == expected_result, f"Test ID: {test_id}"


# Parametrized test cases
@pytest.mark.parametrize(
    "wait_for_inclusion,wait_for_finalization,prompt,response_success,is_registered,expected_result, test_id",
    [
        # Happy path tests
        (False, True, False, True, False, True, "happy-path-finalization-true"),
        (True, False, False, True, False, True, "happy-path-inclusion-true"),
        (False, False, False, True, False, True, "happy-path-no_wait"),
        # Edge cases
        (True, True, False, True, False, True, "edge-both-waits-true"),
        # Error cases
        (False, True, False, False, True, None, "error-finalization-failed"),
        (True, False, False, False, True, None, "error-inclusion-failed"),
        (False, True, True, True, False, False, "error-prompt-declined"),
    ],
)
def test_leave_senate_extrinsic(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    response_success,
    is_registered,
    expected_result,
    test_id,
):
    # Arrange
    with patch(
        "bittensor.extrinsics.senate.Confirm.ask", return_value=not prompt
    ), patch("bittensor.extrinsics.senate.time.sleep"), patch.object(
        mock_subtensor.substrate, "compose_call"
    ), patch.object(mock_subtensor.substrate, "create_signed_extrinsic"), patch.object(
        mock_subtensor.substrate,
        "submit_extrinsic",
        return_value=MagicMock(
            is_success=response_success,
            process_events=MagicMock(),
            error_message="error",
        ),
    ), patch.object(mock_wallet, "is_senate_member", return_value=is_registered):
        # Act
        result = leave_senate_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

        # Assert
        assert result == expected_result, f"Test ID: {test_id}"
