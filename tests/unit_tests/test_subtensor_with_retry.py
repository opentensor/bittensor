import pytest

from bittensor.core import subtensor_with_retry
from bittensor.core.subtensor_with_retry import (
    SubtensorWithRetry,
    SubtensorWithRetryError,
)


def create_subtensor_with_retry():
    """Creates a SubtensorWithRetry instance helper."""
    return SubtensorWithRetry(
        endpoints=["endpoint1", "endpoint2"],
        retry_seconds=1,
        retry_attempts=3,
    )


def test_initialization():
    """Tests initialization of SubtensorWithRetry."""
    subtensor = create_subtensor_with_retry()
    assert subtensor._retry_seconds == 1
    assert subtensor._retry_attempts == 3
    assert subtensor._endpoints == ["endpoint1", "endpoint2"]
    assert subtensor._subtensor is None


def test_invalid_retry_args():
    """Tests invalid retry arguments."""
    with pytest.raises(ValueError):
        SubtensorWithRetry(endpoints=["endpoint1"], retry_seconds=1, retry_epoch=2)
    with pytest.raises(ValueError):
        SubtensorWithRetry(endpoints=["endpoint1"])


def test_get_retry_seconds_fixed():
    """Tests get_retry_seconds() with fixed retry seconds."""
    subtensor = create_subtensor_with_retry()
    assert subtensor.get_retry_seconds() == 1


def test_call_with_retry_failure():
    """Tests call_with_retry() with a permanent failure."""
    subtensor = create_subtensor_with_retry()

    @subtensor_with_retry.call_with_retry
    def mock_method(self):
        raise RuntimeError("Permanent failure")

    with pytest.raises(SubtensorWithRetryError):
        mock_method(subtensor)
