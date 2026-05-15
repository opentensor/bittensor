import pytest

import bittensor.core.async_subtensor


@pytest.fixture
def mock_substrate(mocker):
    mocked = mocker.patch(
        "bittensor.core.async_subtensor.AsyncSubstrateInterface",
        autospec=True,
    )
    instance = mocked.return_value
    # `autospec=True` doesn't pick up instance attributes set in __init__,
    # so callers that touch `substrate._nonces` (the per-account nonce cache)
    # would AttributeError. Provide a real dict.
    instance._nonces = {}

    return instance


@pytest.fixture
def subtensor(mock_substrate):
    return bittensor.core.async_subtensor.AsyncSubtensor()
