# The MIT License (MIT)
# Copyright © 2022 Opentensor Foundation

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

# Standard Lib
import argparse
import unittest.mock as mock
from unittest.mock import MagicMock

# 3rd Party
import pytest

# Application
import bittensor
from bittensor import subtensor_module
from bittensor.subtensor import subtensor as Subtensor, _logger


def test_serve_axon_with_external_ip_set():
    internal_ip: str = "192.0.2.146"
    external_ip: str = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

    mock_serve_axon = MagicMock(return_value=True)

    mock_subtensor = MagicMock(spec=bittensor.subtensor, serve_axon=mock_serve_axon)

    mock_add_insecure_port = mock.MagicMock(return_value=None)
    mock_wallet = MagicMock(
        spec=bittensor.wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )

    mock_config = bittensor.axon.config()
    mock_axon_with_external_ip_set = bittensor.axon(
        wallet=mock_wallet,
        ip=internal_ip,
        external_ip=external_ip,
        config=mock_config,
    )

    mock_subtensor.serve_axon(
        netuid=-1,
        axon=mock_axon_with_external_ip_set,
    )

    mock_serve_axon.assert_called_once()

    # verify that the axon is served to the network with the external ip
    _, kwargs = mock_serve_axon.call_args
    axon_info = kwargs["axon"].info()
    assert axon_info.ip == external_ip


def test_serve_axon_with_external_port_set():
    external_ip: str = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

    internal_port: int = 1234
    external_port: int = 5678

    mock_serve = MagicMock(return_value=True)

    mock_serve_axon = MagicMock(return_value=True)

    mock_subtensor = MagicMock(
        spec=bittensor.subtensor,
        serve=mock_serve,
        serve_axon=mock_serve_axon,
    )

    mock_wallet = MagicMock(
        spec=bittensor.wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )

    mock_config = bittensor.axon.config()

    mock_axon_with_external_port_set = bittensor.axon(
        wallet=mock_wallet,
        port=internal_port,
        external_port=external_port,
        config=mock_config,
    )

    with mock.patch(
        "bittensor.utils.networking.get_external_ip", return_value=external_ip
    ):
        # mock the get_external_ip function to return the external ip
        mock_subtensor.serve_axon(
            netuid=-1,
            axon=mock_axon_with_external_port_set,
        )

    mock_serve_axon.assert_called_once()
    # verify that the axon is served to the network with the external port
    _, kwargs = mock_serve_axon.call_args
    axon_info = kwargs["axon"].info()
    assert axon_info.port == external_port


class ExitEarly(Exception):
    """Mock exception to exit early from the called code"""

    pass


def test_stake_multiple():
    mock_amount: bittensor.Balance = bittensor.Balance.from_tao(1.0)

    mock_wallet = MagicMock(
        spec=bittensor.wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )

    mock_hotkey_ss58s = ["5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"]

    mock_amounts = [mock_amount]  # more than 1000 RAO

    mock_neuron = MagicMock(
        is_null=False,
    )

    mock_do_stake = MagicMock(side_effect=ExitEarly)

    mock_subtensor = MagicMock(
        spec=bittensor.subtensor,
        network="mock_net",
        get_balance=MagicMock(
            return_value=bittensor.Balance.from_tao(mock_amount.tao + 20.0)
        ),  # enough balance to stake
        get_neuron_for_pubkey_and_subnet=MagicMock(return_value=mock_neuron),
        _do_stake=mock_do_stake,
    )

    with pytest.raises(ExitEarly):
        bittensor.subtensor.add_stake_multiple(
            mock_subtensor,
            wallet=mock_wallet,
            hotkey_ss58s=mock_hotkey_ss58s,
            amounts=mock_amounts,
        )

        mock_do_stake.assert_called_once()
        # args, kwargs
        _, kwargs = mock_do_stake.call_args

        assert kwargs["amount"] == pytest.approx(
            mock_amount.rao, rel=1e9
        )  # delta of 1.0 TAO


@pytest.mark.parametrize(
    "test_id, expected_output",
    [
        # Happy path test
        (
            "happy_path_default",
            "Create and return a new object.  See help(type) for accurate signature.",
        ),
    ],
)
def test_help(test_id, expected_output, capsys):
    # Act
    Subtensor.help()

    # Assert
    captured = capsys.readouterr()
    assert expected_output in captured.out, f"Test case {test_id} failed"


@pytest.fixture
def parser():
    return argparse.ArgumentParser()


# Mocking argparse.ArgumentParser.add_argument method to simulate ArgumentError
def test_argument_error_handling(monkeypatch, parser):
    def mock_add_argument(*args, **kwargs):
        raise argparse.ArgumentError(None, "message")

    monkeypatch.setattr(argparse.ArgumentParser, "add_argument", mock_add_argument)
    # No exception should be raised
    Subtensor.add_args(parser)


@pytest.mark.parametrize(
    "network, expected_network, expected_endpoint",
    [
        # Happy path tests
        ("finney", "finney", bittensor.__finney_entrypoint__),
        ("local", "local", bittensor.__local_entrypoint__),
        ("test", "test", bittensor.__finney_test_entrypoint__),
        ("archive", "archive", bittensor.__archive_entrypoint__),
        # Endpoint override tests
        (
            bittensor.__finney_entrypoint__,
            "finney",
            bittensor.__finney_entrypoint__,
        ),
        (
            "entrypoint-finney.opentensor.ai",
            "finney",
            bittensor.__finney_entrypoint__,
        ),
        (
            bittensor.__finney_test_entrypoint__,
            "test",
            bittensor.__finney_test_entrypoint__,
        ),
        (
            "test.finney.opentensor.ai",
            "test",
            bittensor.__finney_test_entrypoint__,
        ),
        (
            bittensor.__archive_entrypoint__,
            "archive",
            bittensor.__archive_entrypoint__,
        ),
        (
            "archive.chain.opentensor.ai",
            "archive",
            bittensor.__archive_entrypoint__,
        ),
        ("127.0.0.1", "local", "127.0.0.1"),
        ("localhost", "local", "localhost"),
        # Edge cases
        (None, None, None),
        ("unknown", "unknown", "unknown"),
    ],
)
def test_determine_chain_endpoint_and_network(
    network, expected_network, expected_endpoint
):
    # Act
    result_network, result_endpoint = Subtensor.determine_chain_endpoint_and_network(
        network
    )

    # Assert
    assert result_network == expected_network
    assert result_endpoint == expected_endpoint


@pytest.fixture
def substrate():
    class MockSubstrate:
        pass

    return MockSubstrate()


@pytest.fixture
def subtensor(substrate):
    mock.patch.object(
        subtensor_module,
        "get_subtensor_errors",
        return_value={
            "1": ("ErrorOne", "Description one"),
            "2": ("ErrorTwo", "Description two"),
        },
    ).start()
    return Subtensor()


def test_get_error_info_by_index_known_error(subtensor):
    name, description = subtensor.get_error_info_by_index(1)
    assert name == "ErrorOne"
    assert description == "Description one"


@pytest.fixture
def mock_logger():
    with mock.patch.object(_logger, "warning") as mock_warning:
        yield mock_warning


def test_get_error_info_by_index_unknown_error(subtensor, mock_logger):
    fake_index = 999
    name, description = subtensor.get_error_info_by_index(fake_index)
    assert name == "Unknown Error"
    assert description == ""
    mock_logger.assert_called_once_with(
        f"Subtensor returned an error with an unknown index: {fake_index}"
    )
