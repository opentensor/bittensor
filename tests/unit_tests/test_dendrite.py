import asyncio
import typing
from unittest.mock import MagicMock, Mock

import aiohttp
from bittensor_wallet.mock import get_mock_wallet
import pytest

from bittensor.core.axon import Axon
from bittensor.core.dendrite import (
    DENDRITE_ERROR_MAPPING,
    DENDRITE_DEFAULT_ERROR,
    Dendrite,
)
from bittensor.core.synapse import TerminalInfo
from tests.helpers import get_mock_wallet
from bittensor.core.synapse import Synapse
from bittensor.core.chain_data import AxonInfo


class SynapseDummy(Synapse):
    input: int
    output: typing.Optional[int] = None


def dummy(synapse: SynapseDummy) -> SynapseDummy:
    synapse.output = synapse.input + 1
    return synapse


@pytest.fixture
def setup_dendrite(mock_get_external_ip):
    # Assuming bittensor.Wallet() returns a wallet object
    user_wallet = get_mock_wallet()
    dendrite_obj = Dendrite(user_wallet)
    yield dendrite_obj


@pytest.fixture
def axon_info():
    return AxonInfo(
        version=1,
        ip="127.0.0.1",
        port=666,
        ip_type=4,
        hotkey="hot",
        coldkey="cold",
    )


@pytest.fixture(scope="session")
def setup_axon():
    wallet = get_mock_wallet()
    axon = Axon(
        wallet,
        external_ip="192.168.1.1",
    )
    axon.attach(forward_fn=dummy)
    axon.start()
    yield axon
    del axon


def test_init(setup_dendrite):
    assert isinstance(setup_dendrite, Dendrite)


def test_str(setup_dendrite):
    expected_string = f"dendrite({setup_dendrite.keypair.ss58_address})"
    assert str(setup_dendrite) == expected_string


def test_repr(setup_dendrite):
    expected_string = f"dendrite({setup_dendrite.keypair.ss58_address})"
    assert repr(setup_dendrite) == expected_string


def test_close(setup_dendrite, setup_axon):
    axon = setup_axon
    # Query the axon to open a session
    setup_dendrite.query(axon, SynapseDummy(input=1))
    # Session should be automatically closed after query
    assert setup_dendrite._session is None


@pytest.mark.asyncio
async def test_aclose(setup_dendrite, setup_axon):
    axon = setup_axon
    # Use context manager to open an async session
    async with setup_dendrite:
        await setup_dendrite([axon], SynapseDummy(input=1), deserialize=False)
    # Close should automatically be called on the session after context manager scope
    assert setup_dendrite._session is None


class AsyncMock(Mock):
    def __call__(self, *args, **kwargs):
        sup = super(AsyncMock, self)

        async def coro():
            return sup.__call__(*args, **kwargs)

        return coro()

    def __await__(self):
        return self().__await__()


def test_dendrite_create_wallet(mock_get_external_ip):
    d = Dendrite(get_mock_wallet())
    d = Dendrite(get_mock_wallet().hotkey)
    d = Dendrite(get_mock_wallet().coldkeypub)
    assert d.__str__() == d.__repr__()


@pytest.mark.asyncio
async def test_forward_many(mock_get_external_ip):
    n = 10
    d = Dendrite(wallet=get_mock_wallet())
    d.call = AsyncMock()
    axons = [MagicMock() for _ in range(n)]

    resps = await d(axons)
    assert len(resps) == n
    resp = await d(axons[0])
    assert len([resp]) == 1

    resps = await d.forward(axons)
    assert len(resps) == n
    resp = await d.forward(axons[0])
    assert len([resp]) == 1


def test_pre_process_synapse(mock_get_external_ip):
    d = Dendrite(wallet=get_mock_wallet())
    s = Synapse()
    synapse = d.preprocess_synapse_for_request(
        target_axon_info=Axon(wallet=get_mock_wallet()).info(),
        synapse=s,
        timeout=12,
    )
    assert synapse.timeout == 12
    assert synapse.dendrite
    assert synapse.axon
    assert synapse.dendrite.ip
    assert synapse.dendrite.version
    assert synapse.dendrite.nonce
    assert synapse.dendrite.uuid
    assert synapse.dendrite.hotkey
    assert synapse.axon.ip
    assert synapse.axon.port
    assert synapse.axon.hotkey
    assert synapse.dendrite.signature


# Helper functions for casting, assuming they exist and work correctly.
def cast_int(value: typing.Any) -> int:
    return int(value)


def cast_float(value: typing.Any) -> float:
    return float(value)


# Happy path tests
@pytest.mark.parametrize(
    "status_code, status_message, process_time, ip, port, version, nonce, uuid, hotkey, signature, expected",
    [
        (
            200,
            "Success",
            0.1,
            "198.123.23.1",
            9282,
            111,
            111111,
            "5ecbd69c-1cec-11ee-b0dc-e29ce36fec1a",
            "5EnjDGNqqWnuL2HCAdxeEtN2oqtXZw6BMBe936Kfy2PFz1J1",
            "0x0813029319030129u4120u10841824y0182u091u230912u",
            True,
        ),
        # Add more test cases with different combinations of realistic values
    ],
    ids=["basic-success"],
)
def test_terminal_info_happy_path(
    status_code,
    status_message,
    process_time,
    ip,
    port,
    version,
    nonce,
    uuid,
    hotkey,
    signature,
    expected,
):
    # Act
    terminal_info = TerminalInfo(
        status_code=status_code,
        status_message=status_message,
        process_time=process_time,
        ip=ip,
        port=port,
        version=version,
        nonce=nonce,
        uuid=uuid,
        hotkey=hotkey,
        signature=signature,
    )

    # Assert
    assert isinstance(terminal_info, TerminalInfo) == expected
    assert terminal_info.status_code == status_code
    assert terminal_info.status_message == status_message
    assert terminal_info.process_time == process_time
    assert terminal_info.ip == ip
    assert terminal_info.port == port
    assert terminal_info.version == version
    assert terminal_info.nonce == nonce
    assert terminal_info.uuid == uuid
    assert terminal_info.hotkey == hotkey
    assert terminal_info.signature == signature


# Edge cases
@pytest.mark.parametrize(
    "status_code, process_time, port, version, nonce, expected_exception",
    [
        ("not-an-int", 0.1, 9282, 111, 111111, ValueError),  # status_code not an int
        (200, "not-a-float", 9282, 111, 111111, ValueError),  # process_time not a float
        (200, 0.1, "not-an-int", 111, 111111, ValueError),  # port not an int
        # Add more edge cases as needed
    ],
    ids=["status_code-not-int", "process_time-not-float", "port-not-int"],
)
def test_terminal_info_edge_cases(
    status_code, process_time, port, version, nonce, expected_exception
):
    # Act & Assert
    with pytest.raises(expected_exception):
        TerminalInfo(
            status_code=status_code,
            process_time=process_time,
            port=port,
            version=version,
            nonce=nonce,
        )


# Error case
@pytest.mark.parametrize(
    "status_code, process_time, port, ip, version, nonce, expected_exception",
    [
        (None, 0.1, 9282, 111, TerminalInfo(), 111111, TypeError),
    ],
    ids=[
        "int() argument must be a string, a bytes-like object or a real number, not 'TerminalInfo'"
    ],
)
def test_terminal_info_error_cases(
    status_code, process_time, port, ip, version, nonce, expected_exception
):
    # Act & Assert
    with pytest.raises(expected_exception):
        TerminalInfo(
            status_code=status_code,
            process_time=process_time,
            port=port,
            ip=ip,
            version=version,
            nonce=nonce,
        )


@pytest.mark.asyncio
async def test_dendrite__call__success_response(
    axon_info, setup_dendrite, mock_aio_response
):
    input_synapse = SynapseDummy(input=1)
    expected_synapse = SynapseDummy(
        **(
            input_synapse.model_dump()
            | dict(
                output=2,
                axon=TerminalInfo(
                    status_code=200,
                    status_message="Success",
                    process_time=0.1,
                ),
            )
        )
    )
    mock_aio_response.post(
        f"http://127.0.0.1:666/SynapseDummy",
        body=expected_synapse.json(),
    )
    synapse = await setup_dendrite.call(axon_info, synapse=input_synapse)

    assert synapse.input == 1
    assert synapse.output == 2
    assert synapse.dendrite.status_code == 200
    assert synapse.dendrite.status_message == "Success"
    assert synapse.dendrite.process_time >= 0


@pytest.mark.asyncio
async def test_dendrite__call__handles_http_error_response(
    axon_info, setup_dendrite, mock_aio_response
):
    status_code = 414
    message = "Custom Error"

    mock_aio_response.post(
        "http://127.0.0.1:666/SynapseDummy",
        status=status_code,
        payload={"message": message},
    )
    synapse = await setup_dendrite.call(axon_info, synapse=SynapseDummy(input=1))

    assert synapse.axon.status_code == synapse.dendrite.status_code == status_code
    assert synapse.axon.status_message == synapse.dendrite.status_message == message


@pytest.mark.parametrize(
    "exception, expected_status_code, expected_message, synapse_timeout, synapse_ip, synapse_port, request_name",
    [
        (
            aiohttp.ClientConnectorError(Mock(), Mock()),
            DENDRITE_ERROR_MAPPING[aiohttp.ClientConnectorError][0],
            f"{DENDRITE_ERROR_MAPPING[aiohttp.ClientConnectorError][1]} at 127.0.0.1:8080/test_request",
            None,
            "127.0.0.1",
            "8080",
            "test_request_client_connector_error",
        ),
        (
            asyncio.TimeoutError(),
            DENDRITE_ERROR_MAPPING[asyncio.TimeoutError][0],
            f"{DENDRITE_ERROR_MAPPING[asyncio.TimeoutError][1]} after 5 seconds",
            5,
            None,
            None,
            "test_request_timeout",
        ),
        (
            aiohttp.ClientResponseError(Mock(), Mock(), status=404),
            "404",
            f"{DENDRITE_ERROR_MAPPING[aiohttp.ClientResponseError][1]}: 404, message=''",
            None,
            None,
            None,
            "test_request_client_response_error",
        ),
        (
            Exception("Unknown error"),
            DENDRITE_DEFAULT_ERROR[0],
            f"{DENDRITE_DEFAULT_ERROR[1]}: Unknown error",
            None,
            None,
            None,
            "test_request_unknown_error",
        ),
    ],
    ids=[
        "ClientConnectorError",
        "TimeoutError",
        "ClientResponseError",
        "GenericException",
    ],
)
def test_process_error_message(
    exception,
    expected_status_code,
    expected_message,
    synapse_timeout,
    synapse_ip,
    synapse_port,
    request_name,
    setup_dendrite,
):
    # Arrange
    synapse = Mock()

    synapse.timeout = synapse_timeout
    synapse.axon.ip = synapse_ip
    synapse.axon.port = synapse_port

    # Act
    result = setup_dendrite.process_error_message(synapse, request_name, exception)

    # Assert
    assert result.dendrite.status_code == expected_status_code
    assert expected_message in result.dendrite.status_message
