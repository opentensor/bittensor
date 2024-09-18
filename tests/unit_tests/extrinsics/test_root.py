import pytest
from unittest.mock import MagicMock, patch
from bittensor.subtensor import Subtensor
from bittensor.extrinsics.root import (
    root_register_extrinsic,
    set_root_weights_extrinsic,
)


@pytest.fixture
def mock_subtensor():
    mock = MagicMock(spec=Subtensor)
    mock.network = "magic_mock"
    return mock


@pytest.fixture
def mock_wallet():
    mock = MagicMock()
    mock.hotkey.ss58_address = "fake_hotkey_address"
    return mock


@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, hotkey_registered, registration_success, prompt, user_response, expected_result",
    [
        (
            False,
            True,
            [True, None],
            True,
            True,
            True,
            True,
        ),  # Already registered after attempt
        (
            False,
            True,
            [False, True],
            True,
            True,
            True,
            True,
        ),  # Registration succeeds with user confirmation
        (False, True, [False, False], False, False, None, None),  # Registration fails
        (
            False,
            True,
            [False, False],
            True,
            False,
            None,
            None,
        ),  # Registration succeeds but neuron not found
        (
            False,
            True,
            [False, False],
            True,
            True,
            False,
            False,
        ),  # User declines registration
    ],
    ids=[
        "success-already-registered",
        "success-registration-succeeds",
        "failure-registration-failed",
        "failure-neuron-not-found",
        "failure-prompt-declined",
    ],
)
def test_root_register_extrinsic(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    hotkey_registered,
    registration_success,
    prompt,
    user_response,
    expected_result,
):
    # Arrange
    mock_subtensor.is_hotkey_registered.side_effect = hotkey_registered

    with patch.object(
        mock_subtensor,
        "_do_root_register",
        return_value=(registration_success, "Error registering"),
    ) as mock_register, patch("rich.prompt.Confirm.ask", return_value=user_response):
        # Act
        result = root_register_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )
        # Assert
        assert result == expected_result

        if not hotkey_registered[0] and user_response:
            mock_register.assert_called_once()


@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, netuids, weights, prompt, user_response, expected_success",
    [
        (True, False, [1, 2], [0.5, 0.5], True, True, True),  # Success - weights set
        (
            False,
            False,
            [1, 2],
            [0.5, 0.5],
            False,
            None,
            True,
        ),  # Success - weights set no wait
        (
            True,
            False,
            [1, 2],
            [2000, 20],
            True,
            True,
            True,
        ),  # Success - large value to be normalized
        (
            True,
            False,
            [1, 2],
            [2000, 0],
            True,
            True,
            True,
        ),  # Success - single large value
        (
            True,
            False,
            [1, 2],
            [0.5, 0.5],
            True,
            False,
            False,
        ),  # Failure - prompt declined
        (
            True,
            False,
            [1, 2],
            [0.5, 0.5],
            False,
            None,
            False,
        ),  # Failure - setting weights failed
        (
            True,
            False,
            [],
            [],
            None,
            False,
            False,
        ),  # Exception catched - ValueError 'min() arg is an empty sequence'
    ],
    ids=[
        "success-weights-set",
        "success-not-wait",
        "success-large-value",
        "success-single-value",
        "failure-user-declines",
        "failure-setting-weights",
        "failure-value-error-exception",
    ],
)
def test_set_root_weights_extrinsic(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    netuids,
    weights,
    prompt,
    user_response,
    expected_success,
):
    # Arrange
    with patch.object(
        mock_subtensor,
        "_do_set_root_weights",
        return_value=(expected_success, "Mock error"),
    ), patch.object(
        mock_subtensor, "min_allowed_weights", return_value=0
    ), patch.object(mock_subtensor, "max_weight_limit", return_value=1), patch(
        "rich.prompt.Confirm.ask", return_value=user_response
    ) as mock_confirm:
        # Act
        result = set_root_weights_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            netuids=netuids,
            weights=weights,
            version_key=0,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

        # Assert
        assert result == expected_success
        if prompt:
            mock_confirm.assert_called_once()
        else:
            mock_confirm.assert_not_called()


@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, netuids, weights, prompt, user_response, expected_success",
    [
        (True, False, [1, 2], [0.5, 0.5], True, True, True),  # Success - weights set
        (
            False,
            False,
            [1, 2],
            [0.5, 0.5],
            False,
            None,
            True,
        ),  # Success - weights set no wait
        (
            True,
            False,
            [1, 2],
            [2000, 20],
            True,
            True,
            True,
        ),  # Success - large value to be normalized
        (
            True,
            False,
            [1, 2],
            [2000, 0],
            True,
            True,
            True,
        ),  # Success - single large value
        (
            True,
            False,
            [1, 2],
            [0.5, 0.5],
            True,
            False,
            False,
        ),  # Failure - prompt declined
        (
            True,
            False,
            [1, 2],
            [0.5, 0.5],
            False,
            None,
            False,
        ),  # Failure - setting weights failed
        (
            True,
            False,
            [],
            [],
            None,
            False,
            False,
        ),  # Exception catched - ValueError 'min() arg is an empty sequence'
    ],
    ids=[
        "success-weights-set",
        "success-not-wait",
        "success-large-value",
        "success-single-value",
        "failure-user-declines",
        "failure-setting-weights",
        "failure-value-error-exception",
    ],
)
def test_set_root_weights_extrinsic_torch(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    netuids,
    weights,
    prompt,
    user_response,
    expected_success,
    force_legacy_torch_compat_api,
):
    test_set_root_weights_extrinsic(
        mock_subtensor,
        mock_wallet,
        wait_for_inclusion,
        wait_for_finalization,
        netuids,
        weights,
        prompt,
        user_response,
        expected_success,
    )
