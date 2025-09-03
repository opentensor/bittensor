from unittest.mock import MagicMock, patch

import pytest

from bittensor.core.extrinsics.set_weights import (
    set_weights_extrinsic,
)
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
        patch.object(
            mock_subtensor,
            "sign_and_send_extrinsic",
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
