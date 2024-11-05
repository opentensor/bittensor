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
    "wait_for_inclusion, wait_for_finalization, hotkey_registered, registration_success, expected_result",
    [
        (
            False,
            True,
            [True, None],
            True,
            True,
        ),  # Already registered after attempt
        (
            False,
            True,
            [False, True],
            True,
            True,
        ),  # Registration succeeds with user confirmation
        (False, True, [False, False], False, None),  # Registration fails
        (
            False,
            True,
            [False, False],
            True,
            None,
        ),  # Registration succeeds but neuron not found
    ],
    ids=[
        "success-already-registered",
        "success-registration-succeeds",
        "failure-registration-failed",
        "failure-neuron-not-found",
    ],
)
def test_root_register_extrinsic(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    hotkey_registered,
    registration_success,
    expected_result,
    mocker,
):
    # Arrange
    mock_subtensor.is_hotkey_registered.side_effect = hotkey_registered

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
    )
    # Assert
    assert result == expected_result

    if not hotkey_registered[0]:
        mock_register.assert_called_once()


@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, netuids, weights, user_response, expected_success",
    [
        (True, False, [1, 2], [0.5, 0.5], True, True),  # Success - weights set
        (
            False,
            False,
            [1, 2],
            [0.5, 0.5],
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
        ),  # Success - large value to be normalized
        (
            True,
            False,
            [1, 2],
            [2000, 0],
            True,
            True,
        ),  # Success - single large value
        (
            True,
            False,
            [1, 2],
            [0.5, 0.5],
            False,
            False,
        ),  # Failure - prompt declined
        (
            True,
            False,
            [1, 2],
            [0.5, 0.5],
            None,
            False,
        ),  # Failure - setting weights failed
        (
            True,
            False,
            [],
            [],
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

    # Call
    result = root.set_root_weights_extrinsic(
        subtensor=mock_subtensor,
        wallet=mock_wallet,
        netuids=netuids,
        weights=weights,
        version_key=0,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Asserts
    assert result == expected_success


@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, netuids, weights, user_response, expected_success",
    [
        (True, False, [1, 2], [0.5, 0.5], True, True),  # Success - weights set
        (
            False,
            False,
            [1, 2],
            [0.5, 0.5],
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
        ),  # Success - large value to be normalized
        (
            True,
            False,
            [1, 2],
            [2000, 0],
            True,
            True,
        ),  # Success - single large value
        (
            True,
            False,
            [1, 2],
            [0.5, 0.5],
            False,
            False,
        ),  # Failure - prompt declined
        (
            True,
            False,
            [1, 2],
            [0.5, 0.5],
            None,
            False,
        ),  # Failure - setting weights failed
        (
            True,
            False,
            [],
            [],
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
        user_response,
        expected_success,
        mocker,
    )
