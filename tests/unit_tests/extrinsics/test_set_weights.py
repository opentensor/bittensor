import torch
import pytest
from unittest.mock import MagicMock, patch
from bittensor import subtensor, wallet
from bittensor.extrinsics.set_weights import set_weights_extrinsic


@pytest.fixture
def mock_subtensor():
    mock = MagicMock(spec=subtensor)
    mock.network = "mock_network"
    return mock


@pytest.fixture
def mock_wallet():
    mock = MagicMock(spec=wallet)
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
        ([1, 2], [0.5, 0.5], 0, True, False, True, True, False, "Mock error message"),
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
    ), patch("rich.prompt.Confirm.ask", return_value=user_accepts), patch.object(
        mock_subtensor,
        "_do_set_weights",
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
                wallet=mock_wallet,
                netuid=123,
                uids=uids_tensor,
                vals=weights_tensor,
                version_key=version_key,
                wait_for_finalization=wait_for_finalization,
                wait_for_inclusion=wait_for_inclusion,
            )
