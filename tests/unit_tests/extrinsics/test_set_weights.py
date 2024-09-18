from unittest.mock import MagicMock, patch

import pytest
import torch
from bittensor_wallet import Wallet

from bittensor.core import subtensor as subtensor_module
from bittensor.core.extrinsics.set_weights import (
    do_set_weights,
    set_weights_extrinsic,
)
from bittensor.core.settings import version_as_int
from bittensor.core.subtensor import Subtensor


@pytest.fixture
def mock_subtensor():
    mock = MagicMock(spec=Subtensor)
    mock.network = "mock_network"
    mock.substrate = MagicMock()
    return mock


@pytest.fixture
def mock_wallet():
    mock = MagicMock(spec=Wallet)
    return mock


@pytest.mark.parametrize(
    "uids, weights, version_key, wait_for_inclusion, wait_for_finalization, prompt, user_accepts, expected_success, expected_message",
    [
        (
            [1, 2],
            [0.5, 0.5],
            0,
            True,
            False,
            True,
            True,
            True,
            "Successfully set weights and Finalized.",
        ),
        (
            [1, 2],
            [0.5, 0.4],
            0,
            False,
            False,
            False,
            True,
            True,
            "Not waiting for finalization or inclusion.",
        ),
        (
            [1, 2],
            [0.5, 0.5],
            0,
            True,
            False,
            True,
            True,
            False,
            "Subtensor returned `UnknownError(UnknownType)` error. This means: `Unknown Description`.",
        ),
        ([1, 2], [0.5, 0.5], 0, True, True, True, False, False, "Prompt refused."),
    ],
    ids=[
        "happy-flow",
        "not-waiting-finalization-inclusion",
        "error-flow",
        "prompt-refused",
    ],
)
def test_set_weights_extrinsic(
    mock_subtensor,
    mock_wallet,
    uids,
    weights,
    version_key,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    user_accepts,
    expected_success,
    expected_message,
):
    uids_tensor = torch.tensor(uids, dtype=torch.int64)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    with patch(
        "bittensor.utils.weight_utils.convert_weights_and_uids_for_emit",
        return_value=(uids_tensor, weights_tensor),
    ), patch("rich.prompt.Confirm.ask", return_value=user_accepts), patch(
        "bittensor.core.extrinsics.set_weights.do_set_weights",
        return_value=(expected_success, "Mock error message"),
    ) as mock_do_set_weights:
        result, message = set_weights_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            netuid=123,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

        assert result == expected_success, f"Test {expected_message} failed."
        assert message == expected_message, f"Test {expected_message} failed."
        if user_accepts is not False:
            mock_do_set_weights.assert_called_once_with(
                self=mock_subtensor,
                wallet=mock_wallet,
                netuid=123,
                uids=uids_tensor,
                vals=weights_tensor,
                version_key=version_key,
                wait_for_finalization=wait_for_finalization,
                wait_for_inclusion=wait_for_inclusion,
            )


def test_do_set_weights_is_success(mock_subtensor, mocker):
    """Successful _do_set_weights call."""
    # Prep
    fake_wallet = mocker.MagicMock()
    fake_uids = [1, 2, 3]
    fake_vals = [4, 5, 6]
    fake_netuid = 1
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = True

    mock_subtensor.substrate.submit_extrinsic.return_value.is_success = True

    # Call
    result = do_set_weights(
        self=mock_subtensor,
        wallet=fake_wallet,
        uids=fake_uids,
        vals=fake_vals,
        netuid=fake_netuid,
        version_key=version_as_int,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    # Asserts
    mock_subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="set_weights",
        call_params={
            "dests": fake_uids,
            "weights": fake_vals,
            "netuid": fake_netuid,
            "version_key": version_as_int,
        },
    )

    mock_subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=mock_subtensor.substrate.compose_call.return_value,
        keypair=fake_wallet.hotkey,
        era={"period": 5},
    )

    mock_subtensor.substrate.submit_extrinsic.assert_called_once_with(
        mock_subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    mock_subtensor.substrate.submit_extrinsic.return_value.process_events.assert_called_once()
    assert result == (True, "Successfully set weights.")


def test_do_set_weights_is_not_success(mock_subtensor, mocker):
    """Unsuccessful _do_set_weights call."""
    # Prep
    fake_wallet = mocker.MagicMock()
    fake_uids = [1, 2, 3]
    fake_vals = [4, 5, 6]
    fake_netuid = 1
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = True

    mock_subtensor.substrate.submit_extrinsic.return_value.is_success = False
    mocked_format_error_message = mocker.MagicMock()
    subtensor_module.format_error_message = mocked_format_error_message

    # Call
    result = do_set_weights(
        self=mock_subtensor,
        wallet=fake_wallet,
        uids=fake_uids,
        vals=fake_vals,
        netuid=fake_netuid,
        version_key=version_as_int,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    # Asserts
    mock_subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="set_weights",
        call_params={
            "dests": fake_uids,
            "weights": fake_vals,
            "netuid": fake_netuid,
            "version_key": version_as_int,
        },
    )

    mock_subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=mock_subtensor.substrate.compose_call.return_value,
        keypair=fake_wallet.hotkey,
        era={"period": 5},
    )

    mock_subtensor.substrate.submit_extrinsic.assert_called_once_with(
        mock_subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    mock_subtensor.substrate.submit_extrinsic.return_value.process_events.assert_called_once()
    assert result == (
        False,
        mock_subtensor.substrate.submit_extrinsic.return_value.error_message,
    )


def test_do_set_weights_no_waits(mock_subtensor, mocker):
    """Successful _do_set_weights call without wait flags for fake_wait_for_inclusion and fake_wait_for_finalization."""
    # Prep
    fake_wallet = mocker.MagicMock()
    fake_uids = [1, 2, 3]
    fake_vals = [4, 5, 6]
    fake_netuid = 1
    fake_wait_for_inclusion = False
    fake_wait_for_finalization = False

    # Call
    result = do_set_weights(
        self=mock_subtensor,
        wallet=fake_wallet,
        uids=fake_uids,
        vals=fake_vals,
        netuid=fake_netuid,
        version_key=version_as_int,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    # Asserts
    mock_subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="set_weights",
        call_params={
            "dests": fake_uids,
            "weights": fake_vals,
            "netuid": fake_netuid,
            "version_key": version_as_int,
        },
    )

    mock_subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=mock_subtensor.substrate.compose_call.return_value,
        keypair=fake_wallet.hotkey,
        era={"period": 5},
    )

    mock_subtensor.substrate.submit_extrinsic.assert_called_once_with(
        mock_subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )
    assert result == (True, "Not waiting for finalization or inclusion.")
