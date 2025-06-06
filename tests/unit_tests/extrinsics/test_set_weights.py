from unittest.mock import MagicMock, patch

import pytest
import torch

from bittensor.core import subtensor as subtensor_module
from bittensor.core.extrinsics.set_weights import (
    _do_set_weights,
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


@pytest.mark.parametrize(
    "uids, weights, version_key, wait_for_inclusion, wait_for_finalization, expected_success, expected_message",
    [
        (
            [1, 2],
            [0.5, 0.5],
            0,
            True,
            False,
            True,
            "Successfully set weights and Finalized.",
        ),
        (
            [1, 2],
            [0.5, 0.4],
            0,
            False,
            False,
            True,
            "Not waiting for finalization or inclusion.",
        ),
        (
            [1, 2],
            [0.5, 0.5],
            0,
            True,
            False,
            False,
            "Mock error message",
        ),
    ],
    ids=[
        "happy-flow",
        "not-waiting-finalization-inclusion",
        "error-flow",
    ],
)
def test_set_weights_extrinsic(
    mock_subtensor,
    fake_wallet,
    uids,
    weights,
    version_key,
    wait_for_inclusion,
    wait_for_finalization,
    expected_success,
    expected_message,
):
    # uids_tensor = torch.tensor(uids, dtype=torch.int64)
    # weights_tensor = torch.tensor(weights, dtype=torch.float32)
    with (
        patch(
            "bittensor.utils.weight_utils.convert_weights_and_uids_for_emit",
            return_value=(uids, weights),
        ),
        patch(
            "bittensor.core.extrinsics.set_weights._do_set_weights",
            return_value=(expected_success, expected_message),
        ),
    ):
        result, message = set_weights_extrinsic(
            subtensor=mock_subtensor,
            wallet=fake_wallet,
            netuid=123,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        assert result == expected_success, f"Test {expected_message} failed."
        assert message == expected_message, f"Test {expected_message} failed."


def test_do_set_weights_is_success(mock_subtensor, fake_wallet, mocker):
    """Successful _do_set_weights call."""
    # Prep
    fake_uids = [1, 2, 3]
    fake_vals = [4, 5, 6]
    fake_netuid = 1
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = True

    mock_subtensor.substrate.submit_extrinsic.return_value.is_success = True
    mocker.patch.object(
        mock_subtensor, "sign_and_send_extrinsic", return_value=(True, "")
    )

    # Call
    result = _do_set_weights(
        subtensor=mock_subtensor,
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

    mock_subtensor.sign_and_send_extrinsic.assert_called_once_with(
        call=mock_subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
        nonce_key="hotkey",
        sign_with="hotkey",
        use_nonce=True,
        period=None,
    )
    assert result == (True, "")


def test_do_set_weights_no_waits(mock_subtensor, fake_wallet, mocker):
    """Successful _do_set_weights call without wait flags for fake_wait_for_inclusion and fake_wait_for_finalization."""
    # Prep
    fake_uids = [1, 2, 3]
    fake_vals = [4, 5, 6]
    fake_netuid = 1
    fake_wait_for_inclusion = False
    fake_wait_for_finalization = False

    mocker.patch.object(
        mock_subtensor,
        "sign_and_send_extrinsic",
        return_value=(True, "Not waiting for finalization or inclusion."),
    )

    # Call
    result = _do_set_weights(
        subtensor=mock_subtensor,
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

    mock_subtensor.sign_and_send_extrinsic(
        call=mock_subtensor.substrate.compose_call.return_value,
        wallet=fake_wallet,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
        period=None,
    )
    assert result == (True, "Not waiting for finalization or inclusion.")
