import pytest
from bittensor.core.subtensor import Subtensor
from bittensor.core.extrinsics import root


@pytest.fixture
def mock_subtensor(mocker):
    mock = mocker.MagicMock(spec=Subtensor)
    mock.network = "magic_mock"
    return mock


@pytest.fixture
def mock_wallet(mocker):
    mock = mocker.MagicMock()
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
    mocker,
):
    # Arrange
    mock_subtensor.is_hotkey_registered.side_effect = hotkey_registered

    with mocker.patch("rich.prompt.Confirm.ask", return_value=user_response):
        # Preps
        mock_register = mocker.Mock(
            return_value=(registration_success, "Error registering")
        )
        root._do_root_register = mock_register

        # Act
        result = root.root_register_extrinsic(
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
    mocker,
):
    # Preps
    root._do_set_root_weights = mocker.Mock(
        return_value=(expected_success, "Mock error")
    )
    mock_subtensor.min_allowed_weights = mocker.Mock(return_value=0)
    mock_subtensor.max_weight_limit = mocker.Mock(return_value=1)
    mock_confirm = mocker.Mock(return_value=(expected_success, "Mock error"))
    root.Confirm.ask = mock_confirm

    # Call
    result = root.set_root_weights_extrinsic(
        subtensor=mock_subtensor,
        wallet=mock_wallet,
        netuids=netuids,
        weights=weights,
        version_key=0,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        prompt=prompt,
    )

    # Asserts
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
    force_legacy_torch_compatible_api,
    mocker,
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
        mocker,
    )
