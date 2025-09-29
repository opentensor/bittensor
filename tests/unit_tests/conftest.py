import pytest
from aioresponses import aioresponses
from bittensor_wallet import Wallet

import bittensor.core.subtensor


@pytest.fixture
def force_legacy_torch_compatible_api(monkeypatch):
    monkeypatch.setenv("USE_TORCH", "1")


@pytest.fixture
def mock_aio_response():
    with aioresponses() as m:
        yield m


@pytest.fixture
def mock_substrate(mocker):
    mocked = mocker.patch(
        "bittensor.core.subtensor.SubstrateInterface",
        autospec=True,
    )

    return mocked.return_value


@pytest.fixture
def subtensor(mock_substrate):
    return bittensor.core.subtensor.Subtensor(_mock=True)


@pytest.fixture
def fake_wallet(mocker):
    return mocker.Mock(spec_set=Wallet)


@pytest.fixture
def mock_get_external_ip(mocker):
    mocked = mocker.Mock(
        return_value="192.168.1.1",
    )

    mocker.patch("bittensor.utils.networking.get_external_ip", mocked)

    return mocked
