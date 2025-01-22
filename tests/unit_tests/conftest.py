import pytest
from aioresponses import aioresponses
from substrateinterface.base import SubstrateInterface
from websockets.sync.client import ClientConnection

import bittensor.core.subtensor


@pytest.fixture
def force_legacy_torch_compatible_api(monkeypatch):
    monkeypatch.setenv("USE_TORCH", "1")


@pytest.fixture
def mock_aio_response():
    with aioresponses() as m:
        yield m


@pytest.fixture
def websockets_client_connection(mocker):
    return mocker.Mock(
        autospec=ClientConnection,
        **{
            "close_code": None,
            "socket.getsockopt.return_value": 0,
        },
    )


@pytest.fixture
def subtensor(websockets_client_connection, mock_substrate_interface):
    return bittensor.core.subtensor.Subtensor(
        websocket=websockets_client_connection,
    )


@pytest.fixture
def mock_substrate_interface(websockets_client_connection, mocker):
    mocked = mocker.MagicMock(
        autospec=SubstrateInterface,
        **{
            "websocket": websockets_client_connection,
        },
    )

    mocker.patch("bittensor.core.subtensor.SubstrateInterface", return_value=mocked)

    return mocked


@pytest.fixture
def mock_get_external_ip(mocker):
    mocked = mocker.Mock(
        return_value="192.168.1.1",
    )

    mocker.patch("bittensor.utils.networking.get_external_ip", mocked)

    return mocked
