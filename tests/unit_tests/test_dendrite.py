# The MIT License (MIT)
# Copyright © 2022 Yuma Rao
# Copyright © 2022-2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from pydantic import ValidationError
import pytest
import typing
import bittensor
from unittest.mock import MagicMock, Mock
from tests.helpers import _get_mock_wallet

from bittensor.synapse import TerminalInfo


class SynapseDummy(bittensor.Synapse):
    input: int
    output: typing.Optional[int] = None


def dummy(synapse: SynapseDummy) -> SynapseDummy:
    synapse.output = synapse.input + 1
    return synapse


@pytest.fixture
def setup_dendrite():
    user_wallet = (
        _get_mock_wallet()
    )  # assuming bittensor.wallet() returns a wallet object
    dendrite_obj = bittensor.dendrite(user_wallet)
    return dendrite_obj


@pytest.fixture(scope="session")
def setup_axon():
    axon = bittensor.axon()
    axon.attach(forward_fn=dummy)
    axon.start()
    yield axon
    del axon


def test_init(setup_dendrite):
    dendrite_obj = setup_dendrite
    assert isinstance(dendrite_obj, bittensor.dendrite)
    assert dendrite_obj.keypair == setup_dendrite.keypair


def test_str(setup_dendrite):
    dendrite_obj = setup_dendrite
    expected_string = "dendrite({})".format(setup_dendrite.keypair.ss58_address)
    assert str(dendrite_obj) == expected_string


def test_repr(setup_dendrite):
    dendrite_obj = setup_dendrite
    expected_string = "dendrite({})".format(setup_dendrite.keypair.ss58_address)
    assert repr(dendrite_obj) == expected_string


def test_close(setup_dendrite, setup_axon):
    axon = setup_axon
    dendrite_obj = setup_dendrite
    # Query the axon to open a session
    dendrite_obj.query(axon, SynapseDummy(input=1))
    # Session should be automatically closed after query
    assert dendrite_obj._session == None


@pytest.mark.asyncio
async def test_aclose(setup_dendrite, setup_axon):
    axon = setup_axon
    dendrite_obj = setup_dendrite
    # Use context manager to open an async session
    async with dendrite_obj:
        resp = await dendrite_obj([axon], SynapseDummy(input=1), deserialize=False)
    # Close should automatically be called on the session after context manager scope
    assert dendrite_obj._session == None


class AsyncMock(Mock):
    def __call__(self, *args, **kwargs):
        sup = super(AsyncMock, self)

        async def coro():
            return sup.__call__(*args, **kwargs)

        return coro()

    def __await__(self):
        return self().__await__()


def test_dendrite_create_wallet():
    d = bittensor.dendrite(_get_mock_wallet())
    d = bittensor.dendrite(_get_mock_wallet().hotkey)
    d = bittensor.dendrite(_get_mock_wallet().coldkeypub)
    assert d.__str__() == d.__repr__()


@pytest.mark.asyncio
async def test_forward_many():
    n = 10
    d = bittensor.dendrite(wallet=_get_mock_wallet())
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


def test_pre_process_synapse():
    d = bittensor.dendrite(wallet=_get_mock_wallet())
    s = bittensor.Synapse()
    synapse = d.preprocess_synapse_for_request(
        target_axon_info=bittensor.axon(wallet=_get_mock_wallet()).info(),
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
