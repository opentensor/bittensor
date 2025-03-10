import pytest
from bittensor.core.subtensor import Subtensor
from bittensor.core.extrinsics import root
from bittensor.utils.balance import Balance


@pytest.fixture
def mock_subtensor(mocker):
    mock = mocker.MagicMock(spec=Subtensor)
    mock.network = "magic_mock"
    mock.substrate = mocker.Mock()
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
            [False, 1],
            True,
            True,
        ),  # Registration succeeds with user confirmation
        (False, True, [False, None], False, False),  # Registration fails
        (
            False,
            True,
            [False, None],
            True,
            False,
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
    mock_subtensor.is_hotkey_registered.return_value = hotkey_registered[0]

    # Preps
    mocked_sign_and_send_extrinsic = mocker.patch.object(
        mock_subtensor,
        "sign_and_send_extrinsic",
        return_value=(registration_success, "Error registering"),
    )
    mocker.patch.object(
        mock_subtensor.substrate,
        "query",
        return_value=hotkey_registered[1],
    )
    mocker.patch.object(
        mock_subtensor,
        "get_balance",
        return_value=Balance(1),
    )

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
        mock_subtensor.substrate.compose_call.assert_called_once_with(
            call_module="SubtensorModule",
            call_function="root_register",
            call_params={"hotkey": "fake_hotkey_address"},
        )
        mocked_sign_and_send_extrinsic.assert_called_once_with(
            mock_subtensor.substrate.compose_call.return_value,
            wallet=mock_wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )


def test_root_register_extrinsic_insufficient_balance(
    mock_subtensor,
    mock_wallet,
    mocker,
):
    mocker.patch.object(
        mock_subtensor,
        "get_balance",
        return_value=Balance(0),
    )

    success = root.root_register_extrinsic(
        subtensor=mock_subtensor,
        wallet=mock_wallet,
    )

    assert success is False

    mock_subtensor.get_balance.assert_called_once_with(
        mock_wallet.coldkeypub.ss58_address,
        block=mock_subtensor.get_current_block.return_value,
    )
    mock_subtensor.substrate.submit_extrinsic.assert_not_called()


@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, netuids, weights, expected_success",
    [
        (True, False, [1, 2], [0.5, 0.5], True),  # Success - weights set
        (
            False,
            False,
            [1, 2],
            [0.5, 0.5],
            True,
        ),  # Success - weights set no wait
        (
            True,
            False,
            [1, 2],
            [2000, 20],
            True,
        ),  # Success - large value to be normalized
        (
            True,
            False,
            [1, 2],
            [2000, 0],
            True,
        ),  # Success - single large value
        (
            True,
            False,
            [1, 2],
            [0.5, 0.5],
            False,
        ),  # Failure - setting weights failed
    ],
    ids=[
        "success-weights-set",
        "success-not-wait",
        "success-large-value",
        "success-single-value",
        "failure-setting-weights",
    ],
)
def test_set_root_weights_extrinsic(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    netuids,
    weights,
    expected_success,
    mocker,
):
    # Preps
    mocker.patch.object(
        root, "_do_set_root_weights", return_value=(expected_success, "Mock error")
    )
    mocker.patch.object(
        root,
        "_get_limits",
        return_value=(0, 1),
    )

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
            None,
            False,
        ),  # Failure - setting weights failed
    ],
    ids=[
        "success-weights-set",
        "success-not-wait",
        "success-large-value",
        "success-single-value",
        "failure-setting-weights",
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
        expected_success,
        mocker,
    )
