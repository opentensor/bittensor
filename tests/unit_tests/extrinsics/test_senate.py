import pytest
from unittest.mock import MagicMock, patch
from bittensor import subtensor, wallet
from bittensor.extrinsics.senate import register_senate_extrinsic


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
    ), patch.object(
        mock_subtensor.substrate, "create_signed_extrinsic"
    ), patch.object(
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
