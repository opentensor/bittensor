import pytest

import bittensor.core.async_subtensor


@pytest.fixture
def mock_substrate(mocker):
    mocked = mocker.patch(
        "bittensor.core.async_subtensor.AsyncSubstrateInterface",
        autospec=True,
    )

    return mocked.return_value


@pytest.fixture
def subtensor(mock_substrate):
    return bittensor.core.async_subtensor.AsyncSubtensor()
