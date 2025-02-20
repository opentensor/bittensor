import pytest
from aioresponses import aioresponses
from async_substrate_interface.sync_substrate import SubstrateInterface

import bittensor.core.subtensor


@pytest.fixture
def force_legacy_torch_compatible_api(monkeypatch):
    monkeypatch.setenv("USE_TORCH", "1")


@pytest.fixture
def mock_aio_response():
    with aioresponses() as m:
        yield m


@pytest.fixture
def mock_substrate_interface(mocker):
    mocked = mocker.MagicMock(
        autospec=SubstrateInterface,
    )

    mocker.patch("bittensor.core.subtensor.SubstrateInterface", return_value=mocked)

    return mocked


@pytest.fixture
def subtensor(mock_substrate_interface):
    return bittensor.core.subtensor.Subtensor()


@pytest.fixture
def mock_get_external_ip(mocker):
    mocked = mocker.Mock(
        return_value="192.168.1.1",
    )

    mocker.patch("bittensor.utils.networking.get_external_ip", mocked)

    return mocked
