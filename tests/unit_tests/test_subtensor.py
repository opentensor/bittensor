# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import unittest.mock as mock
from unittest.mock import MagicMock

import pytest
from bittensor_wallet import Wallet

from bittensor.core import subtensor as subtensor_module, settings
from bittensor.core.axon import Axon
from bittensor.core.chain_data import SubnetHyperparameters
from bittensor.core.settings import version_as_int
from bittensor.core.subtensor import Subtensor, logging
from bittensor.utils import u16_normalized_float, u64_normalized_float, Certificate
from bittensor.utils.balance import Balance

U16_MAX = 65535
U64_MAX = 18446744073709551615


@pytest.fixture
def fake_call_params():
    return call_params()


def call_params():
    return {
        "version": "1.0",
        "ip": "0.0.0.0",
        "port": 9090,
        "ip_type": 4,
        "netuid": 1,
        "certificate": None,
    }


def call_params_with_certificate():
    params = call_params()
    params["certificate"] = Certificate("fake_cert")
    return params


def test_serve_axon_with_external_ip_set():
    internal_ip: str = "192.0.2.146"
    external_ip: str = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

    mock_serve_axon = MagicMock(return_value=True)

    mock_subtensor = MagicMock(spec=Subtensor, serve_axon=mock_serve_axon)

    mock_wallet = MagicMock(
        spec=Wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )

    mock_config = Axon.config()
    mock_axon_with_external_ip_set = Axon(
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
        spec=Subtensor,
        serve=mock_serve,
        serve_axon=mock_serve_axon,
    )

    mock_wallet = MagicMock(
        spec=Wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )

    mock_config = Axon.config()

    mock_axon_with_external_port_set = Axon(
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
        ("finney", "finney", settings.FINNEY_ENTRYPOINT),
        ("local", "local", settings.LOCAL_ENTRYPOINT),
        ("test", "test", settings.FINNEY_TEST_ENTRYPOINT),
        ("archive", "archive", settings.ARCHIVE_ENTRYPOINT),
        # Endpoint override tests
        (
            settings.FINNEY_ENTRYPOINT,
            "finney",
            settings.FINNEY_ENTRYPOINT,
        ),
        (
            "entrypoint-finney.opentensor.ai",
            "finney",
            settings.FINNEY_ENTRYPOINT,
        ),
        (
            settings.FINNEY_TEST_ENTRYPOINT,
            "test",
            settings.FINNEY_TEST_ENTRYPOINT,
        ),
        (
            "test.finney.opentensor.ai",
            "test",
            settings.FINNEY_TEST_ENTRYPOINT,
        ),
        (
            settings.ARCHIVE_ENTRYPOINT,
            "archive",
            settings.ARCHIVE_ENTRYPOINT,
        ),
        (
            "archive.chain.opentensor.ai",
            "archive",
            settings.ARCHIVE_ENTRYPOINT,
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
def subtensor(mocker):
    fake_substrate = mocker.MagicMock()
    fake_substrate.websocket.sock.getsockopt.return_value = 0
    mocker.patch.object(
        subtensor_module, "SubstrateInterface", return_value=fake_substrate
    )
    return Subtensor()


@pytest.fixture
def mock_logger():
    with mock.patch.object(logging, "warning") as mock_warning:
        yield mock_warning


def test_hyperparameter_subnet_does_not_exist(subtensor, mocker):
    """Tests when the subnet does not exist."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=False)
    assert subtensor._get_hyperparameter("Difficulty", 1, None) is None
    subtensor.subnet_exists.assert_called_once_with(1, None)


def test_hyperparameter_result_is_none(subtensor, mocker):
    """Tests when query_subtensor returns None."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.query_subtensor = mocker.MagicMock(return_value=None)
    assert subtensor._get_hyperparameter("Difficulty", 1, None) is None
    subtensor.subnet_exists.assert_called_once_with(1, None)
    subtensor.query_subtensor.assert_called_once_with("Difficulty", None, [1])


def test_hyperparameter_result_has_no_value(subtensor, mocker):
    """Test when the result has no 'value' attribute."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.query_subtensor = mocker.MagicMock(return_value=None)
    assert subtensor._get_hyperparameter("Difficulty", 1, None) is None
    subtensor.subnet_exists.assert_called_once_with(1, None)
    subtensor.query_subtensor.assert_called_once_with("Difficulty", None, [1])


def test_hyperparameter_success_int(subtensor, mocker):
    """Test when query_subtensor returns an integer value."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.query_subtensor = mocker.MagicMock(
        return_value=mocker.MagicMock(value=100)
    )
    assert subtensor._get_hyperparameter("Difficulty", 1, None) == 100
    subtensor.subnet_exists.assert_called_once_with(1, None)
    subtensor.query_subtensor.assert_called_once_with("Difficulty", None, [1])


def test_hyperparameter_success_float(subtensor, mocker):
    """Test when query_subtensor returns a float value."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.query_subtensor = mocker.MagicMock(
        return_value=mocker.MagicMock(value=0.5)
    )
    assert subtensor._get_hyperparameter("Difficulty", 1, None) == 0.5
    subtensor.subnet_exists.assert_called_once_with(1, None)
    subtensor.query_subtensor.assert_called_once_with("Difficulty", None, [1])


def test_blocks_since_last_update_success_calls(subtensor, mocker):
    """Tests the weights_rate_limit method to ensure it correctly fetches the LastUpdate hyperparameter."""
    # Prep
    uid = 7
    mocked_current_block = 2
    mocked_result = {uid: 1}
    subtensor._get_hyperparameter = mocker.MagicMock(return_value=mocked_result)
    subtensor.get_current_block = mocker.MagicMock(return_value=mocked_current_block)

    # Call
    result = subtensor.blocks_since_last_update(netuid=7, uid=uid)

    # Assertions
    subtensor.get_current_block.assert_called_once()
    subtensor._get_hyperparameter.assert_called_once_with(
        param_name="LastUpdate", netuid=7
    )
    assert result == 1
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(result, int)


def test_weights_rate_limit_success_calls(subtensor, mocker):
    """Tests the weights_rate_limit method to ensure it correctly fetches the WeightsSetRateLimit hyperparameter."""
    # Prep
    subtensor._get_hyperparameter = mocker.MagicMock(return_value=5)

    # Call
    result = subtensor.weights_rate_limit(netuid=7)

    # Assertions
    subtensor._get_hyperparameter.assert_called_once_with(
        param_name="WeightsSetRateLimit", netuid=7
    )
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(result, int)


@pytest.fixture
def sample_hyperparameters():
    return MagicMock(spec=SubnetHyperparameters)


def normalize_hyperparameters(
    subnet: "SubnetHyperparameters",
) -> list[tuple[str, str, str]]:
    """
    Normalizes the hyperparameters of a subnet.

    Args:
        subnet: The subnet hyperparameters object.

    Returns:
        A list of tuples containing the parameter name, value, and normalized value.
    """
    param_mappings = {
        "adjustment_alpha": u64_normalized_float,
        "min_difficulty": u64_normalized_float,
        "max_difficulty": u64_normalized_float,
        "difficulty": u64_normalized_float,
        "bonds_moving_avg": u64_normalized_float,
        "max_weight_limit": u16_normalized_float,
        "kappa": u16_normalized_float,
        "alpha_high": u16_normalized_float,
        "alpha_low": u16_normalized_float,
        "min_burn": Balance.from_rao,
        "max_burn": Balance.from_rao,
    }

    normalized_values: list[tuple[str, str, str]] = []
    subnet_dict = subnet.__dict__

    for param, value in subnet_dict.items():
        try:
            if param in param_mappings:
                norm_value = param_mappings[param](value)
                if isinstance(norm_value, float):
                    norm_value = f"{norm_value:.{10}g}"
            else:
                norm_value = value
        except Exception as e:
            logging.warning(f"Error normalizing parameter '{param}': {e}")
            norm_value = "-"

        normalized_values.append((param, str(value), str(norm_value)))

    return normalized_values


def get_normalized_value(normalized_data, param_name):
    return next(
        (
            norm_value
            for p_name, _, norm_value in normalized_data
            if p_name == param_name
        ),
        None,
    )


@pytest.mark.parametrize(
    "param_name, max_value, mid_value, zero_value, is_balance",
    [
        ("adjustment_alpha", U64_MAX, U64_MAX / 2, 0, False),
        ("max_weight_limit", U16_MAX, U16_MAX / 2, 0, False),
        ("difficulty", U64_MAX, U64_MAX / 2, 0, False),
        ("min_difficulty", U64_MAX, U64_MAX / 2, 0, False),
        ("max_difficulty", U64_MAX, U64_MAX / 2, 0, False),
        ("bonds_moving_avg", U64_MAX, U64_MAX / 2, 0, False),
        ("min_burn", 10000000000, 5000000000, 0, True),  # These are in rao
        ("max_burn", 20000000000, 10000000000, 0, True),
    ],
    ids=[
        "adjustment-alpha",
        "max_weight_limit",
        "difficulty",
        "min_difficulty",
        "max_difficulty",
        "bonds_moving_avg",
        "min_burn",
        "max_burn",
    ],
)
def test_hyperparameter_normalization(
    sample_hyperparameters, param_name, max_value, mid_value, zero_value, is_balance
):
    setattr(sample_hyperparameters, param_name, mid_value)
    normalized = normalize_hyperparameters(sample_hyperparameters)
    norm_value = get_normalized_value(normalized, param_name)

    # Mid-value test
    if is_balance:
        numeric_value = float(str(norm_value).lstrip(settings.TAO_SYMBOL))
        expected_tao = mid_value / 1e9
        assert (
            numeric_value == expected_tao
        ), f"Mismatch in tao value for {param_name} at mid value"
    else:
        assert float(norm_value) == 0.5, f"Failed mid-point test for {param_name}"

    # Max-value test
    setattr(sample_hyperparameters, param_name, max_value)
    normalized = normalize_hyperparameters(sample_hyperparameters)
    norm_value = get_normalized_value(normalized, param_name)

    if is_balance:
        numeric_value = float(str(norm_value).lstrip(settings.TAO_SYMBOL))
        expected_tao = max_value / 1e9
        assert (
            numeric_value == expected_tao
        ), f"Mismatch in tao value for {param_name} at max value"
    else:
        assert float(norm_value) == 1.0, f"Failed max value test for {param_name}"

    # Zero-value test
    setattr(sample_hyperparameters, param_name, zero_value)
    normalized = normalize_hyperparameters(sample_hyperparameters)
    norm_value = get_normalized_value(normalized, param_name)

    if is_balance:
        numeric_value = float(str(norm_value).lstrip(settings.TAO_SYMBOL))
        expected_tao = zero_value / 1e9
        assert (
            numeric_value == expected_tao
        ), f"Mismatch in tao value for {param_name} at zero value"
    else:
        assert float(norm_value) == 0.0, f"Failed zero value test for {param_name}"


###########################
# Account functions tests #
###########################


# get_prometheus_info tests
def test_get_prometheus_info_success(mocker, subtensor):
    """Test get_prometheus_info returns correct data when information is found."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    block = 123
    mock_result = mocker.MagicMock(
        value={
            "ip": 3232235777,  # 192.168.1.1
            "ip_type": 4,
            "port": 9090,
            "version": "1.0",
            "block": 1000,
        }
    )
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_prometheus_info(netuid, hotkey_ss58, block)

    # Asserts
    assert result is not None
    assert result.ip == "192.168.1.1"
    assert result.ip_type == 4
    assert result.port == 9090
    assert result.version == "1.0"
    assert result.block == 1000
    subtensor.query_subtensor.assert_called_once_with(
        "Prometheus", block, [netuid, hotkey_ss58]
    )


def test_get_prometheus_info_no_data(mocker, subtensor):
    """Test get_prometheus_info returns None when no information is found."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.get_prometheus_info(netuid, hotkey_ss58, block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with(
        "Prometheus", block, [netuid, hotkey_ss58]
    )


def test_get_prometheus_info_no_value_attribute(mocker, subtensor):
    """Test get_prometheus_info returns None when result has no value attribute."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_prometheus_info(netuid, hotkey_ss58, block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with(
        "Prometheus", block, [netuid, hotkey_ss58]
    )


def test_get_prometheus_info_no_block(mocker, subtensor):
    """Test get_prometheus_info with no block specified."""
    # Prep
    netuid = 1
    hotkey_ss58 = "test_hotkey"
    mock_result = MagicMock(
        value={
            "ip": "192.168.1.1",
            "ip_type": 4,
            "port": 9090,
            "version": "1.0",
            "block": 1000,
        }
    )
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_prometheus_info(netuid, hotkey_ss58)

    # Asserts
    assert result is not None
    assert result.ip == "192.168.1.1"
    assert result.ip_type == 4
    assert result.port == 9090
    assert result.version == "1.0"
    assert result.block == 1000
    subtensor.query_subtensor.assert_called_once_with(
        "Prometheus", None, [netuid, hotkey_ss58]
    )


###########################
# Global Parameters tests #
###########################


# `block` property test
def test_block_property(mocker, subtensor):
    """Test block property returns the correct block number."""
    expected_block = 123
    mocker.patch.object(subtensor, "get_current_block", return_value=expected_block)

    result = subtensor.block

    assert result == expected_block
    subtensor.get_current_block.assert_called_once()


# `subnet_exists` tests
def test_subnet_exists_success(mocker, subtensor):
    """Test subnet_exists returns True when subnet exists."""
    # Prep
    netuid = 1
    block = 123
    mock_result = mocker.MagicMock(value=True)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.subnet_exists(netuid, block)

    # Asserts
    assert result is True
    subtensor.query_subtensor.assert_called_once_with("NetworksAdded", block, [netuid])


def test_subnet_exists_no_data(mocker, subtensor):
    """Test subnet_exists returns False when no subnet information is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.subnet_exists(netuid, block)

    # Asserts
    assert result is False
    subtensor.query_subtensor.assert_called_once_with("NetworksAdded", block, [netuid])


def test_subnet_exists_no_value_attribute(mocker, subtensor):
    """Test subnet_exists returns False when result has no value attribute."""
    # Prep
    netuid = 1
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.subnet_exists(netuid, block)

    # Asserts
    assert result is False
    subtensor.query_subtensor.assert_called_once_with("NetworksAdded", block, [netuid])


def test_subnet_exists_no_block(mocker, subtensor):
    """Test subnet_exists with no block specified."""
    # Prep
    netuid = 1
    mock_result = mocker.MagicMock(value=True)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.subnet_exists(netuid)

    # Asserts
    assert result is True
    subtensor.query_subtensor.assert_called_once_with("NetworksAdded", None, [netuid])


# `get_total_subnets` tests
def test_get_total_subnets_success(mocker, subtensor):
    """Test get_total_subnets returns correct data when total subnet information is found."""
    # Prep
    block = 123
    total_subnets_value = 10
    mock_result = mocker.MagicMock(value=total_subnets_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_total_subnets(block)

    # Asserts
    assert result is not None
    assert result == total_subnets_value
    subtensor.query_subtensor.assert_called_once_with("TotalNetworks", block)


def test_get_total_subnets_no_data(mocker, subtensor):
    """Test get_total_subnets returns None when no total subnet information is found."""
    # Prep
    block = 123
    mocker.patch.object(subtensor, "query_subtensor", return_value=None)

    # Call
    result = subtensor.get_total_subnets(block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("TotalNetworks", block)


def test_get_total_subnets_no_value_attribute(mocker, subtensor):
    """Test get_total_subnets returns None when result has no value attribute."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value  # Simulating a missing value attribute
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_total_subnets(block)

    # Asserts
    assert result is None
    subtensor.query_subtensor.assert_called_once_with("TotalNetworks", block)


def test_get_total_subnets_no_block(mocker, subtensor):
    """Test get_total_subnets with no block specified."""
    # Prep
    total_subnets_value = 10
    mock_result = mocker.MagicMock(value=total_subnets_value)
    mocker.patch.object(subtensor, "query_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_total_subnets()

    # Asserts
    assert result is not None
    assert result == total_subnets_value
    subtensor.query_subtensor.assert_called_once_with("TotalNetworks", None)


# `get_subnets` tests
def test_get_subnets_success(mocker, subtensor):
    """Test get_subnets returns correct list when subnet information is found."""
    # Prep
    block = 123
    mock_netuid1 = mocker.MagicMock(value=1)
    mock_netuid2 = mocker.MagicMock(value=2)
    mock_result = mocker.MagicMock()
    mock_result.records = [(mock_netuid1, True), (mock_netuid2, True)]
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnets(block)

    # Asserts
    assert result == [1, 2]
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", block)


def test_get_subnets_no_data(mocker, subtensor):
    """Test get_subnets returns empty list when no subnet information is found."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    mock_result.records = []
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnets(block)

    # Asserts
    assert result == []
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", block)


def test_get_subnets_no_records_attribute(mocker, subtensor):
    """Test get_subnets returns empty list when result has no records attribute."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.records  # Simulating a missing records attribute
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnets(block)

    # Asserts
    assert result == []
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", block)


def test_get_subnets_no_block_specified(mocker, subtensor):
    """Test get_subnets with no block specified."""
    # Prep
    mock_netuid1 = mocker.MagicMock(value=1)
    mock_netuid2 = mocker.MagicMock(value=2)
    mock_result = mocker.MagicMock()
    mock_result.records = [(mock_netuid1, True), (mock_netuid2, True)]
    mocker.patch.object(subtensor, "query_map_subtensor", return_value=mock_result)

    # Call
    result = subtensor.get_subnets()

    # Asserts
    assert result == [1, 2]
    subtensor.query_map_subtensor.assert_called_once_with("NetworksAdded", None)


# `get_subnet_hyperparameters` tests
def test_get_subnet_hyperparameters_success(mocker, subtensor):
    """Test get_subnet_hyperparameters returns correct data when hyperparameters are found."""
    # Prep
    netuid = 1
    block = 123
    hex_bytes_result = "0x010203"
    bytes_result = bytes.fromhex(hex_bytes_result[2:])
    mocker.patch.object(subtensor, "query_runtime_api", return_value=hex_bytes_result)
    mocker.patch.object(
        subtensor_module.SubnetHyperparameters,
        "from_vec_u8",
        return_value=["from_vec_u8"],
    )

    # Call
    result = subtensor.get_subnet_hyperparameters(netuid, block)

    # Asserts
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams",
        params=[netuid],
        block=block,
    )
    subtensor_module.SubnetHyperparameters.from_vec_u8.assert_called_once_with(
        bytes_result
    )


def test_get_subnet_hyperparameters_hex_without_prefix(subtensor, mocker):
    """Test get_subnet_hyperparameters correctly processes hex string without '0x' prefix."""
    # Prep
    netuid = 1
    block = 123
    hex_bytes_result = "010203"
    bytes_result = bytes.fromhex(hex_bytes_result)
    mocker.patch.object(subtensor, "query_runtime_api", return_value=hex_bytes_result)
    mocker.patch.object(subtensor_module.SubnetHyperparameters, "from_vec_u8")

    # Call
    result = subtensor.get_subnet_hyperparameters(netuid, block)

    # Asserts
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams",
        params=[netuid],
        block=block,
    )
    subtensor_module.SubnetHyperparameters.from_vec_u8.assert_called_once_with(
        bytes_result
    )


def test_get_subnet_hyperparameters_no_data(mocker, subtensor):
    """Test get_subnet_hyperparameters returns empty list when no data is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(subtensor, "query_runtime_api", return_value=None)
    mocker.patch.object(subtensor_module.SubnetHyperparameters, "from_vec_u8")

    # Call
    result = subtensor.get_subnet_hyperparameters(netuid, block)

    # Asserts
    assert result == []
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams",
        params=[netuid],
        block=block,
    )
    subtensor_module.SubnetHyperparameters.from_vec_u8.assert_not_called()


def test_query_subtensor(subtensor, mocker):
    """Tests query_subtensor call."""
    # Prep
    fake_name = "module_name"

    # Call
    result = subtensor.query_subtensor(fake_name)

    # Asserts
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function=fake_name,
        params=None,
        block_hash=None,
    )
    assert result == subtensor.substrate.query.return_value


def test_query_runtime_api(subtensor, mocker):
    """Tests query_runtime_api call."""
    # Prep
    fake_runtime_api = "NeuronInfoRuntimeApi"
    fake_method = "get_neuron_lite"

    mocked_state_call = mocker.MagicMock()
    subtensor.state_call = mocked_state_call

    mocked_runtime_configuration = mocker.patch.object(
        subtensor_module, "RuntimeConfiguration"
    )
    mocked_scalecodec = mocker.patch.object(subtensor_module.scalecodec, "ScaleBytes")

    # Call
    result = subtensor.query_runtime_api(fake_runtime_api, fake_method, None)

    # Asserts
    subtensor.state_call.assert_called_once_with(
        method=f"{fake_runtime_api}_{fake_method}", data="0x", block=None
    )
    mocked_scalecodec.assert_called_once_with(
        subtensor.state_call.return_value.__getitem__.return_value
    )
    mocked_runtime_configuration.assert_called_once()
    mocked_runtime_configuration.return_value.update_type_registry.assert_called()
    mocked_runtime_configuration.return_value.create_scale_object.assert_called()
    assert (
        result
        == mocked_runtime_configuration.return_value.create_scale_object.return_value.decode.return_value
    )


def test_query_map_subtensor(subtensor, mocker):
    """Tests query_map_subtensor call."""
    # Prep
    fake_name = "module_name"

    # Call
    result = subtensor.query_map_subtensor(fake_name)

    # Asserts
    subtensor.substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function=fake_name,
        params=None,
        block_hash=None,
    )
    assert result == subtensor.substrate.query_map.return_value


def test_state_call(subtensor, mocker):
    """Tests state_call call."""
    # Prep
    fake_method = "method"
    fake_data = "data"

    # Call
    result = subtensor.state_call(fake_method, fake_data)

    # Asserts
    subtensor.substrate.rpc_request.assert_called_once_with(
        method="state_call",
        params=[fake_method, fake_data],
    )
    assert result == subtensor.substrate.rpc_request.return_value


def test_query_map(subtensor, mocker):
    """Tests query_map call."""
    # Prep
    fake_module_name = "module_name"
    fake_name = "constant_name"

    # Call
    result = subtensor.query_map(fake_module_name, fake_name)

    # Asserts
    subtensor.substrate.query_map.assert_called_once_with(
        module=fake_module_name,
        storage_function=fake_name,
        params=None,
        block_hash=None,
    )
    assert result == subtensor.substrate.query_map.return_value


def test_query_constant(subtensor, mocker):
    """Tests query_constant call."""
    # Prep
    fake_module_name = "module_name"
    fake_constant_name = "constant_name"

    # Call
    result = subtensor.query_constant(fake_module_name, fake_constant_name)

    # Asserts
    subtensor.substrate.get_constant.assert_called_once_with(
        module_name=fake_module_name,
        constant_name=fake_constant_name,
        block_hash=None,
    )
    assert result == subtensor.substrate.get_constant.return_value


def test_query_module(subtensor):
    # Prep
    fake_module = "module"
    fake_name = "function_name"

    # Call
    result = subtensor.query_module(fake_module, fake_name)

    # Asserts
    subtensor.substrate.query.assert_called_once_with(
        module=fake_module,
        storage_function=fake_name,
        params=None,
        block_hash=None,
    )
    assert result == subtensor.substrate.query.return_value


def test_metagraph(subtensor, mocker):
    """Tests subtensor.metagraph call."""
    # Prep
    fake_netuid = 1
    fake_lite = True
    mocked_metagraph = mocker.patch.object(subtensor_module, "Metagraph")

    # Call
    result = subtensor.metagraph(fake_netuid, fake_lite)

    # Asserts
    mocked_metagraph.assert_called_once_with(
        network=subtensor.network, netuid=fake_netuid, lite=fake_lite, sync=False
    )
    mocked_metagraph.return_value.sync.assert_called_once_with(
        block=None, lite=fake_lite, subtensor=subtensor
    )
    assert result == mocked_metagraph.return_value


def test_get_netuids_for_hotkey(subtensor, mocker):
    """Tests get_netuids_for_hotkey call."""
    # Prep
    fake_hotkey_ss58 = "hotkey_ss58"
    fake_block = 123

    mocked_query_map_subtensor = mocker.MagicMock()
    subtensor.query_map_subtensor = mocked_query_map_subtensor

    # Call
    result = subtensor.get_netuids_for_hotkey(fake_hotkey_ss58, fake_block)

    # Asserts
    mocked_query_map_subtensor.assert_called_once_with(
        "IsNetworkMember", fake_block, [fake_hotkey_ss58]
    )
    assert result == []


def test_get_current_block(subtensor):
    """Tests get_current_block call."""
    # Call
    result = subtensor.get_current_block()

    # Asserts
    subtensor.substrate.get_block_number.assert_called_once_with(None)
    assert result == subtensor.substrate.get_block_number.return_value


def test_is_hotkey_registered_any(subtensor, mocker):
    """Tests is_hotkey_registered_any call"""
    # Prep
    fake_hotkey_ss58 = "hotkey_ss58"
    fake_block = 123
    return_value = [1, 2]

    mocked_get_netuids_for_hotkey = mocker.MagicMock(return_value=return_value)
    subtensor.get_netuids_for_hotkey = mocked_get_netuids_for_hotkey

    # Call
    result = subtensor.is_hotkey_registered_any(fake_hotkey_ss58, fake_block)

    # Asserts
    mocked_get_netuids_for_hotkey.assert_called_once_with(fake_hotkey_ss58, fake_block)
    assert result is (len(return_value) > 0)


def test_is_hotkey_registered_on_subnet(subtensor, mocker):
    """Tests is_hotkey_registered_on_subnet call."""
    # Prep
    fake_hotkey_ss58 = "hotkey_ss58"
    fake_netuid = 1
    fake_block = 123

    mocked_get_uid_for_hotkey_on_subnet = mocker.MagicMock()
    subtensor.get_uid_for_hotkey_on_subnet = mocked_get_uid_for_hotkey_on_subnet

    # Call
    result = subtensor.is_hotkey_registered_on_subnet(
        fake_hotkey_ss58, fake_netuid, fake_block
    )

    # Asserts
    mocked_get_uid_for_hotkey_on_subnet.assert_called_once_with(
        fake_hotkey_ss58, fake_netuid, fake_block
    )
    assert result is (mocked_get_uid_for_hotkey_on_subnet.return_value is not None)


def test_is_hotkey_registered_without_netuid(subtensor, mocker):
    """Tests is_hotkey_registered call with no netuid specified."""
    # Prep
    fake_hotkey_ss58 = "hotkey_ss58"

    mocked_is_hotkey_registered_any = mocker.MagicMock()
    subtensor.is_hotkey_registered_any = mocked_is_hotkey_registered_any

    # Call

    result = subtensor.is_hotkey_registered(fake_hotkey_ss58)

    # Asserts
    mocked_is_hotkey_registered_any.assert_called_once_with(fake_hotkey_ss58, None)
    assert result == mocked_is_hotkey_registered_any.return_value


def test_is_hotkey_registered_with_netuid(subtensor, mocker):
    """Tests is_hotkey_registered call with netuid specified."""
    # Prep
    fake_hotkey_ss58 = "hotkey_ss58"
    fake_netuid = 123

    mocked_is_hotkey_registered_on_subnet = mocker.MagicMock()
    subtensor.is_hotkey_registered_on_subnet = mocked_is_hotkey_registered_on_subnet

    # Call

    result = subtensor.is_hotkey_registered(fake_hotkey_ss58, fake_netuid)

    # Asserts
    mocked_is_hotkey_registered_on_subnet.assert_called_once_with(
        fake_hotkey_ss58, fake_netuid, None
    )
    assert result == mocked_is_hotkey_registered_on_subnet.return_value


def test_set_weights(subtensor, mocker):
    """Successful set_weights call."""
    # Preps
    fake_wallet = mocker.MagicMock()
    fake_netuid = 1
    fake_uids = [2, 4]
    fake_weights = [0.4, 0.6]
    fake_wait_for_inclusion = False
    fake_wait_for_finalization = False
    fake_max_retries = 5

    expected_result = (True, None)

    mocked_get_uid_for_hotkey_on_subnet = mocker.MagicMock()
    subtensor.get_uid_for_hotkey_on_subnet = mocked_get_uid_for_hotkey_on_subnet

    mocked_blocks_since_last_update = mocker.MagicMock(return_value=2)
    subtensor.blocks_since_last_update = mocked_blocks_since_last_update

    mocked_weights_rate_limit = mocker.MagicMock(return_value=1)
    subtensor.weights_rate_limit = mocked_weights_rate_limit

    mocked_set_weights_extrinsic = mocker.patch.object(
        subtensor_module, "set_weights_extrinsic", return_value=expected_result
    )

    # Call
    result = subtensor.set_weights(
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        version_key=settings.version_as_int,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
        max_retries=fake_max_retries,
    )

    # Asserts
    mocked_get_uid_for_hotkey_on_subnet.assert_called_once_with(
        fake_wallet.hotkey.ss58_address, fake_netuid
    )
    mocked_blocks_since_last_update.assert_called_with(
        fake_netuid, mocked_get_uid_for_hotkey_on_subnet.return_value
    )
    mocked_weights_rate_limit.assert_called_with(fake_netuid)
    mocked_set_weights_extrinsic.assert_called_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        version_key=settings.version_as_int,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )
    assert result == expected_result


def test_serve_axon(subtensor, mocker):
    """Tests successful serve_axon call."""
    # Prep
    fake_netuid = 123
    fake_axon = mocker.MagicMock()
    fake_wait_for_inclusion = False
    fake_wait_for_finalization = True
    fake_certificate = None

    mocked_serve_axon_extrinsic = mocker.patch.object(
        subtensor_module, "serve_axon_extrinsic"
    )

    # Call
    result = subtensor.serve_axon(
        fake_netuid, fake_axon, fake_wait_for_inclusion, fake_wait_for_finalization
    )

    # Asserts
    mocked_serve_axon_extrinsic.assert_called_once_with(
        subtensor,
        fake_netuid,
        fake_axon,
        fake_wait_for_inclusion,
        fake_wait_for_finalization,
        fake_certificate,
    )
    assert result == mocked_serve_axon_extrinsic.return_value


def test_get_block_hash(subtensor, mocker):
    """Tests successful get_block_hash call."""
    # Prep
    fake_block_id = 123

    # Call
    result = subtensor.get_block_hash(fake_block_id)

    # Asserts
    subtensor.substrate.get_block_hash.assert_called_once_with(block_id=fake_block_id)
    assert result == subtensor.substrate.get_block_hash.return_value


def test_commit(subtensor, mocker):
    """Test successful commit call."""
    # Preps
    fake_wallet = mocker.MagicMock()
    fake_netuid = 1
    fake_data = "some data to network"
    mocked_publish_metadata = mocker.patch.object(subtensor_module, "publish_metadata")

    # Call
    result = subtensor.commit(fake_wallet, fake_netuid, fake_data)

    # Asserts
    mocked_publish_metadata.assert_called_once_with(
        subtensor, fake_wallet, fake_netuid, f"Raw{len(fake_data)}", fake_data.encode()
    )
    assert result is None


def test_subnetwork_n(subtensor, mocker):
    """Test successful subnetwork_n call."""
    # Prep
    fake_netuid = 1
    fake_block = 123
    fake_result = 2

    mocked_get_hyperparameter = mocker.MagicMock()
    mocked_get_hyperparameter.return_value = fake_result
    subtensor._get_hyperparameter = mocked_get_hyperparameter

    # Call
    result = subtensor.subnetwork_n(fake_netuid, fake_block)

    # Asserts
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="SubnetworkN", netuid=fake_netuid, block=fake_block
    )
    assert result == mocked_get_hyperparameter.return_value


def test_transfer(subtensor, mocker):
    """Tests successful transfer call."""
    # Prep
    fake_wallet = mocker.MagicMock()
    fake_dest = "SS58PUBLICKEY"
    fake_amount = 1.1
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = True
    mocked_transfer_extrinsic = mocker.patch.object(
        subtensor_module, "transfer_extrinsic"
    )

    # Call
    result = subtensor.transfer(
        fake_wallet,
        fake_dest,
        fake_amount,
        fake_wait_for_inclusion,
        fake_wait_for_finalization,
    )

    # Asserts
    mocked_transfer_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        dest=fake_dest,
        amount=fake_amount,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )
    assert result == mocked_transfer_extrinsic.return_value


def test_get_neuron_for_pubkey_and_subnet(subtensor, mocker):
    """Successful call to get_neuron_for_pubkey_and_subnet."""
    # Prep
    fake_hotkey_ss58 = "fake_hotkey"
    fake_netuid = 1
    fake_block = 123

    mocked_neuron_for_uid = mocker.MagicMock()
    subtensor.neuron_for_uid = mocked_neuron_for_uid

    mocked_get_uid_for_hotkey_on_subnet = mocker.MagicMock()
    subtensor.get_uid_for_hotkey_on_subnet = mocked_get_uid_for_hotkey_on_subnet

    # Call
    result = subtensor.get_neuron_for_pubkey_and_subnet(
        hotkey_ss58=fake_hotkey_ss58,
        netuid=fake_netuid,
        block=fake_block,
    )

    # Asserts
    mocked_neuron_for_uid.assert_called_once_with(
        mocked_get_uid_for_hotkey_on_subnet.return_value,
        fake_netuid,
        block=fake_block,
    )
    assert result == mocked_neuron_for_uid.return_value


def test_neuron_for_uid_none(subtensor, mocker):
    """Test neuron_for_uid successful call."""
    # Prep
    fake_uid = None
    fake_netuid = 2
    fake_block = 123
    mocked_neuron_info = mocker.patch.object(
        subtensor_module.NeuronInfo, "get_null_neuron"
    )

    # Call
    result = subtensor.neuron_for_uid(
        uid=fake_uid, netuid=fake_netuid, block=fake_block
    )

    # Asserts
    mocked_neuron_info.assert_called_once()
    assert result == mocked_neuron_info.return_value


def test_neuron_for_uid_response_none(subtensor, mocker):
    """Test neuron_for_uid successful call."""
    # Prep
    fake_uid = 1
    fake_netuid = 2
    fake_block = 123
    mocked_neuron_info = mocker.patch.object(
        subtensor_module.NeuronInfo, "get_null_neuron"
    )

    subtensor.substrate.rpc_request.return_value.get.return_value = None

    # Call
    result = subtensor.neuron_for_uid(
        uid=fake_uid, netuid=fake_netuid, block=fake_block
    )

    # Asserts
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    subtensor.substrate.rpc_request.assert_called_once_with(
        method="neuronInfo_getNeuron",
        params=[fake_netuid, fake_uid, subtensor.substrate.get_block_hash.return_value],
    )

    mocked_neuron_info.assert_called_once()
    assert result == mocked_neuron_info.return_value


def test_neuron_for_uid_success(subtensor, mocker):
    """Test neuron_for_uid successful call."""
    # Prep
    fake_uid = 1
    fake_netuid = 2
    fake_block = 123
    mocked_neuron_from_vec_u8 = mocker.patch.object(
        subtensor_module.NeuronInfo, "from_vec_u8"
    )

    # Call
    result = subtensor.neuron_for_uid(
        uid=fake_uid, netuid=fake_netuid, block=fake_block
    )

    # Asserts
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    subtensor.substrate.rpc_request.assert_called_once_with(
        method="neuronInfo_getNeuron",
        params=[fake_netuid, fake_uid, subtensor.substrate.get_block_hash.return_value],
    )

    mocked_neuron_from_vec_u8.assert_called_once_with(
        subtensor.substrate.rpc_request.return_value.get.return_value
    )
    assert result == mocked_neuron_from_vec_u8.return_value


@pytest.mark.parametrize(
    ["fake_call_params", "expected_call_function"],
    [
        (call_params(), "serve_axon"),
        (call_params_with_certificate(), "serve_axon_tls"),
    ],
)
def test_do_serve_axon_is_success(
    subtensor, mocker, fake_call_params, expected_call_function
):
    """Successful do_serve_axon call."""
    # Prep
    fake_wallet = mocker.MagicMock()
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = True

    subtensor.substrate.submit_extrinsic.return_value.is_success = True

    # Call
    result = subtensor._do_serve_axon(
        wallet=fake_wallet,
        call_params=fake_call_params,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function=expected_call_function,
        call_params=fake_call_params,
    )

    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value,
        keypair=fake_wallet.hotkey,
    )

    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    subtensor.substrate.submit_extrinsic.return_value.process_events.assert_called_once()
    assert result[0] is True
    assert result[1] is None


def test_do_serve_axon_is_not_success(subtensor, mocker, fake_call_params):
    """Unsuccessful do_serve_axon call."""
    # Prep
    fake_wallet = mocker.MagicMock()
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = True

    subtensor.substrate.submit_extrinsic.return_value.is_success = None

    # Call
    result = subtensor._do_serve_axon(
        wallet=fake_wallet,
        call_params=fake_call_params,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="serve_axon",
        call_params=fake_call_params,
    )

    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value,
        keypair=fake_wallet.hotkey,
    )

    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    subtensor.substrate.submit_extrinsic.return_value.process_events.assert_called_once()
    assert result == (
        False,
        subtensor.substrate.submit_extrinsic.return_value.error_message,
    )


def test_do_serve_axon_no_waits(subtensor, mocker, fake_call_params):
    """Unsuccessful do_serve_axon call."""
    # Prep
    fake_wallet = mocker.MagicMock()
    fake_wait_for_inclusion = False
    fake_wait_for_finalization = False

    # Call
    result = subtensor._do_serve_axon(
        wallet=fake_wallet,
        call_params=fake_call_params,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="SubtensorModule",
        call_function="serve_axon",
        call_params=fake_call_params,
    )

    subtensor.substrate.create_signed_extrinsic.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value,
        keypair=fake_wallet.hotkey,
    )

    subtensor.substrate.submit_extrinsic.assert_called_once_with(
        subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )
    assert result == (True, None)


def test_immunity_period(subtensor, mocker):
    """Successful immunity_period call."""
    # Preps
    fake_netuid = 1
    fake_block = 123
    fare_result = 101

    mocked_get_hyperparameter = mocker.MagicMock()
    mocked_get_hyperparameter.return_value = fare_result
    subtensor._get_hyperparameter = mocked_get_hyperparameter

    # Call
    result = subtensor.immunity_period(netuid=fake_netuid, block=fake_block)

    # Assertions
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="ImmunityPeriod",
        netuid=fake_netuid,
        block=fake_block,
    )
    assert result == mocked_get_hyperparameter.return_value


def test_get_uid_for_hotkey_on_subnet(subtensor, mocker):
    """Successful get_uid_for_hotkey_on_subnet call."""
    # Prep
    fake_hotkey_ss58 = "fake_hotkey_ss58"
    fake_netuid = 1
    fake_block = 123
    mocked_query_subtensor = mocker.MagicMock()
    subtensor.query_subtensor = mocked_query_subtensor

    # Call
    result = subtensor.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=fake_hotkey_ss58, netuid=fake_netuid, block=fake_block
    )

    # Assertions
    mocked_query_subtensor.assert_called_once_with(
        "Uids", fake_block, [fake_netuid, fake_hotkey_ss58]
    )

    assert result == mocked_query_subtensor.return_value.value


def test_tempo(subtensor, mocker):
    """Successful tempo call."""
    # Preps
    fake_netuid = 1
    fake_block = 123
    fare_result = 101

    mocked_get_hyperparameter = mocker.MagicMock()
    mocked_get_hyperparameter.return_value = fare_result
    subtensor._get_hyperparameter = mocked_get_hyperparameter

    # Call
    result = subtensor.tempo(netuid=fake_netuid, block=fake_block)

    # Assertions
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="Tempo",
        netuid=fake_netuid,
        block=fake_block,
    )
    assert result == mocked_get_hyperparameter.return_value


def test_get_commitment(subtensor, mocker):
    """Successful get_commitment call."""
    # Preps
    fake_netuid = 1
    fake_uid = 2
    fake_block = 3
    fake_hotkey = "hotkey"
    fake_hex_data = "0x010203"
    expected_result = bytes.fromhex(fake_hex_data[2:]).decode()

    mocked_metagraph = mocker.MagicMock()
    subtensor.metagraph = mocked_metagraph
    mocked_metagraph.return_value.hotkeys = {fake_uid: fake_hotkey}

    mocked_get_metadata = mocker.patch.object(subtensor_module, "get_metadata")
    mocked_get_metadata.return_value = {
        "info": {"fields": [{fake_hex_data: fake_hex_data}]}
    }

    # Call
    result = subtensor.get_commitment(
        netuid=fake_netuid, uid=fake_uid, block=fake_block
    )

    # Assertions
    mocked_metagraph.assert_called_once_with(fake_netuid)
    assert result == expected_result


def test_min_allowed_weights(subtensor, mocker):
    """Successful min_allowed_weights call."""
    fake_netuid = 1
    fake_block = 123
    return_value = 10

    mocked_get_hyperparameter = mocker.MagicMock(return_value=return_value)
    subtensor._get_hyperparameter = mocked_get_hyperparameter

    # Call
    result = subtensor.min_allowed_weights(netuid=fake_netuid, block=fake_block)

    # Assertion
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="MinAllowedWeights", block=fake_block, netuid=fake_netuid
    )
    assert result == return_value


def test_max_weight_limit(subtensor, mocker):
    """Successful max_weight_limit call."""
    fake_netuid = 1
    fake_block = 123
    return_value = 100

    mocked_get_hyperparameter = mocker.MagicMock(return_value=return_value)
    subtensor._get_hyperparameter = mocked_get_hyperparameter

    mocked_u16_normalized_float = mocker.MagicMock()
    subtensor_module.u16_normalized_float = mocked_u16_normalized_float

    # Call
    result = subtensor.max_weight_limit(netuid=fake_netuid, block=fake_block)

    # Assertion
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="MaxWeightsLimit", block=fake_block, netuid=fake_netuid
    )
    assert result == mocked_u16_normalized_float.return_value


def test_get_transfer_fee(subtensor, mocker):
    """Successful get_transfer_fee call."""
    # Preps
    fake_wallet = mocker.MagicMock()
    fake_dest = "SS58ADDRESS"
    value = 1

    fake_payment_info = {"partialFee": int(2e10)}
    subtensor.substrate.get_payment_info.return_value = fake_payment_info

    # Call
    result = subtensor.get_transfer_fee(wallet=fake_wallet, dest=fake_dest, value=value)

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_allow_death",
        call_params={"dest": fake_dest, "value": value},
    )

    subtensor.substrate.get_payment_info.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value,
        keypair=fake_wallet.coldkeypub,
    )

    assert result == 2e10


def test_get_transfer_fee_incorrect_value(subtensor, mocker):
    """Successful get_transfer_fee call."""
    # Preps
    fake_wallet = mocker.MagicMock()
    fake_dest = mocker.MagicMock()
    value = "no_int_no_float_no_Balance"

    mocked_substrate = mocker.MagicMock()
    subtensor.substrate = mocked_substrate
    spy_balance_from_rao = mocker.spy(Balance, "from_rao")

    # Call
    result = subtensor.get_transfer_fee(wallet=fake_wallet, dest=fake_dest, value=value)

    # Asserts
    spy_balance_from_rao.assert_called_once_with(2e7)

    assert result == Balance.from_rao(int(2e7))


def test_get_existential_deposit(subtensor, mocker):
    """Successful get_existential_deposit call."""
    # Prep
    block = 123

    mocked_query_constant = mocker.MagicMock()
    value = 10
    mocked_query_constant.return_value.value = value
    subtensor.query_constant = mocked_query_constant

    # Call
    result = subtensor.get_existential_deposit(block=block)

    # Assertions
    mocked_query_constant.assert_called_once_with(
        module_name="Balances", constant_name="ExistentialDeposit", block=block
    )

    assert isinstance(result, Balance)
    assert result == Balance.from_rao(value)


def test_commit_weights(subtensor, mocker):
    """Successful commit_weights call."""
    # Preps
    fake_wallet = mocker.MagicMock()
    netuid = 1
    salt = [1, 3]
    uids = [2, 4]
    weights = [0.4, 0.6]
    wait_for_inclusion = False
    wait_for_finalization = False
    max_retries = 5

    expected_result = (True, None)
    mocked_generate_weight_hash = mocker.patch.object(
        subtensor_module, "generate_weight_hash", return_value=expected_result
    )
    mocked_commit_weights_extrinsic = mocker.patch.object(
        subtensor_module, "commit_weights_extrinsic", return_value=expected_result
    )

    # Call
    result = subtensor.commit_weights(
        wallet=fake_wallet,
        netuid=netuid,
        salt=salt,
        uids=uids,
        weights=weights,
        version_key=settings.version_as_int,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        max_retries=max_retries,
    )

    # Asserts
    mocked_generate_weight_hash.assert_called_once_with(
        address=fake_wallet.hotkey.ss58_address,
        netuid=netuid,
        uids=list(uids),
        values=list(weights),
        salt=list(salt),
        version_key=settings.version_as_int,
    )

    mocked_commit_weights_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        commit_hash=mocked_generate_weight_hash.return_value,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    assert result == expected_result


def test_reveal_weights(subtensor, mocker):
    """Successful test_reveal_weights call."""
    # Preps
    fake_wallet = mocker.MagicMock()
    netuid = 1
    uids = [1, 2, 3, 4]
    weights = [0.1, 0.2, 0.3, 0.4]
    salt = [4, 2, 2, 1]
    expected_result = (True, None)
    mocked_extrinsic = mocker.patch.object(
        subtensor_module, "reveal_weights_extrinsic", return_value=expected_result
    )

    # Call
    result = subtensor.reveal_weights(
        wallet=fake_wallet,
        netuid=netuid,
        uids=uids,
        weights=weights,
        salt=salt,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )

    # Assertions
    assert result == (True, None)
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        uids=uids,
        version_key=version_as_int,
        weights=weights,
        salt=salt,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )


def test_reveal_weights_false(subtensor, mocker):
    """Failed test_reveal_weights call."""
    # Preps
    fake_wallet = mocker.MagicMock()
    netuid = 1
    uids = [1, 2, 3, 4]
    weights = [0.1, 0.2, 0.3, 0.4]
    salt = [4, 2, 2, 1]

    expected_result = (
        False,
        "No attempt made. Perhaps it is too soon to reveal weights!",
    )
    mocked_extrinsic = mocker.patch.object(subtensor_module, "reveal_weights_extrinsic")

    # Call
    result = subtensor.reveal_weights(
        wallet=fake_wallet,
        netuid=netuid,
        uids=uids,
        weights=weights,
        salt=salt,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )

    # Assertion
    assert result == expected_result
    assert mocked_extrinsic.call_count == 5


def test_connect_without_substrate(mocker):
    """Ensure re-connection is called when using an alive substrate."""
    # Prep
    fake_substrate = mocker.MagicMock()
    fake_substrate.websocket.sock.getsockopt.return_value = 1
    mocker.patch.object(
        subtensor_module, "SubstrateInterface", return_value=fake_substrate
    )
    fake_subtensor = Subtensor()
    spy_get_substrate = mocker.spy(Subtensor, "_get_substrate")

    # Call
    _ = fake_subtensor.block

    # Assertions
    assert spy_get_substrate.call_count == 1


def test_connect_with_substrate(mocker):
    """Ensure re-connection is non called when using an alive substrate."""
    # Prep
    fake_substrate = mocker.MagicMock()
    fake_substrate.websocket.socket.getsockopt.return_value = 0
    mocker.patch.object(
        subtensor_module, "SubstrateInterface", return_value=fake_substrate
    )
    fake_subtensor = Subtensor()
    spy_get_substrate = mocker.spy(Subtensor, "_get_substrate")

    # Call
    _ = fake_subtensor.block

    # Assertions
    assert spy_get_substrate.call_count == 0


def test_get_subnet_burn_cost_success(subtensor, mocker):
    """Tests get_subnet_burn_cost method with successfully result."""
    # Preps
    mocked_query_runtime_api = mocker.patch.object(subtensor, "query_runtime_api")
    fake_block = 123

    # Call
    result = subtensor.get_subnet_burn_cost(fake_block)

    # Asserts
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="SubnetRegistrationRuntimeApi",
        method="get_network_registration_cost",
        params=[],
        block=fake_block,
    )

    assert result == mocked_query_runtime_api.return_value


def test_get_subnet_burn_cost_none(subtensor, mocker):
    """Tests get_subnet_burn_cost method with None result."""
    # Preps
    mocked_query_runtime_api = mocker.patch.object(
        subtensor, "query_runtime_api", return_value=None
    )
    fake_block = 123

    # Call
    result = subtensor.get_subnet_burn_cost(fake_block)

    # Asserts
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="SubnetRegistrationRuntimeApi",
        method="get_network_registration_cost",
        params=[],
        block=fake_block,
    )

    assert result is None


def test_difficulty_success(subtensor, mocker):
    """Tests difficulty method with successfully result."""
    # Preps
    mocked_get_hyperparameter = mocker.patch.object(subtensor, "_get_hyperparameter")
    fake_netuid = 1
    fake_block = 2

    # Call
    result = subtensor.difficulty(fake_netuid, fake_block)

    # Asserts
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="Difficulty",
        netuid=fake_netuid,
        block=fake_block,
    )

    assert result == int(mocked_get_hyperparameter.return_value)


def test_difficulty_none(subtensor, mocker):
    """Tests difficulty method with None result."""
    # Preps
    mocked_get_hyperparameter = mocker.patch.object(
        subtensor, "_get_hyperparameter", return_value=None
    )
    fake_netuid = 1
    fake_block = 2

    # Call
    result = subtensor.difficulty(fake_netuid, fake_block)

    # Asserts
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="Difficulty",
        netuid=fake_netuid,
        block=fake_block,
    )

    assert result is None


def test_recycle_success(subtensor, mocker):
    """Tests recycle method with successfully result."""
    # Preps
    mocked_get_hyperparameter = mocker.patch.object(
        subtensor, "_get_hyperparameter", return_value=0.1
    )
    fake_netuid = 1
    fake_block = 2
    mocked_balance = mocker.patch("bittensor.utils.balance.Balance")

    # Call
    result = subtensor.recycle(fake_netuid, fake_block)

    # Asserts
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="Burn",
        netuid=fake_netuid,
        block=fake_block,
    )

    mocked_balance.assert_called_once_with(int(mocked_get_hyperparameter.return_value))
    assert result == mocked_balance.return_value


def test_recycle_none(subtensor, mocker):
    """Tests recycle method with None result."""
    # Preps
    mocked_get_hyperparameter = mocker.patch.object(
        subtensor, "_get_hyperparameter", return_value=None
    )
    fake_netuid = 1
    fake_block = 2

    # Call
    result = subtensor.recycle(fake_netuid, fake_block)

    # Asserts
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="Burn",
        netuid=fake_netuid,
        block=fake_block,
    )

    assert result is None


# `get_all_subnets_info` tests
def test_get_all_subnets_info_success(mocker, subtensor):
    """Test get_all_subnets_info returns correct data when subnet information is found."""
    # Prep
    block = 123
    mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="mock_block_hash"
    )
    hex_bytes_result = "0x010203"
    bytes_result = bytes.fromhex(hex_bytes_result[2:])
    mocker.patch.object(subtensor, "query_runtime_api", return_value=hex_bytes_result)
    mocker.patch.object(
        subtensor_module.SubnetInfo,
        "list_from_vec_u8",
        return_value="list_from_vec_u80",
    )

    # Call
    subtensor.get_all_subnets_info(block)

    # Asserts
    subtensor.query_runtime_api.assert_called_once_with(
        "SubnetInfoRuntimeApi", "get_subnets_info", params=[], block=block
    )
    subtensor_module.SubnetInfo.list_from_vec_u8.assert_called_once_with(bytes_result)


@pytest.mark.parametrize("result_", [[], None])
def test_get_all_subnets_info_no_data(mocker, subtensor, result_):
    """Test get_all_subnets_info returns empty list when no subnet information is found."""
    # Prep
    block = 123
    mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="mock_block_hash"
    )
    mocker.patch.object(subtensor_module.SubnetInfo, "list_from_vec_u8")

    mocker.patch.object(subtensor, "query_runtime_api", return_value=result_)

    # Call
    result = subtensor.get_all_subnets_info(block)

    # Asserts
    assert result == []
    subtensor.query_runtime_api.assert_called_once_with(
        "SubnetInfoRuntimeApi", "get_subnets_info", params=[], block=block
    )
    subtensor_module.SubnetInfo.list_from_vec_u8.assert_not_called()


def test_get_delegate_take_success(subtensor, mocker):
    """Verify `get_delegate_take` method successful path."""
    # Preps
    fake_hotkey_ss58 = "FAKE_SS58"
    fake_block = 123

    subtensor_module.u16_normalized_float = mocker.Mock()
    subtensor.query_subtensor = mocker.Mock(return_value=mocker.Mock(value="value"))

    # Call
    result = subtensor.get_delegate_take(hotkey_ss58=fake_hotkey_ss58, block=fake_block)

    # Asserts
    subtensor.query_subtensor.assert_called_once_with(
        "Delegates", fake_block, [fake_hotkey_ss58]
    )
    subtensor_module.u16_normalized_float.assert_called_once_with(
        subtensor.query_subtensor.return_value.value
    )
    assert result == subtensor_module.u16_normalized_float.return_value


def test_get_delegate_take_none(subtensor, mocker):
    """Verify `get_delegate_take` method returns None."""
    # Preps
    fake_hotkey_ss58 = "FAKE_SS58"
    fake_block = 123

    subtensor.query_subtensor = mocker.Mock(return_value=mocker.Mock(value=None))
    subtensor_module.u16_normalized_float = mocker.Mock()

    # Call
    result = subtensor.get_delegate_take(hotkey_ss58=fake_hotkey_ss58, block=fake_block)

    # Asserts
    subtensor.query_subtensor.assert_called_once_with(
        "Delegates", fake_block, [fake_hotkey_ss58]
    )

    subtensor_module.u16_normalized_float.assert_not_called()
    assert result is None


def test_networks_during_connection(mocker):
    """Test networks during_connection."""
    # Preps
    subtensor_module.SubstrateInterface = mocker.Mock()
    mocker.patch("websockets.sync.client.connect")
    # Call
    for network in list(settings.NETWORK_MAP.keys()) + ["undefined"]:
        sub = Subtensor(network)

        # Assertions
        sub.network = network
        sub.chain_endpoint = settings.NETWORK_MAP.get(network)


@pytest.mark.parametrize(
    "fake_value_result",
    [1, None],
    ids=["result has value attr", "result has not value attr"],
)
def test_get_stake_for_coldkey_and_hotkey(subtensor, mocker, fake_value_result):
    """Test get_stake_for_coldkey_and_hotkey calls right method with correct arguments."""
    # Preps
    fake_hotkey_ss58 = "FAKE_H_SS58"
    fake_coldkey_ss58 = "FAKE_C_SS58"
    fake_block = 123

    return_value = (
        mocker.Mock(value=fake_value_result)
        if fake_value_result is not None
        else fake_value_result
    )

    subtensor.query_subtensor = mocker.patch.object(
        subtensor, "query_subtensor", return_value=return_value
    )
    spy_balance_from_rao = mocker.spy(subtensor_module.Balance, "from_rao")

    # Call
    result = subtensor.get_stake_for_coldkey_and_hotkey(
        hotkey_ss58=fake_hotkey_ss58,
        coldkey_ss58=fake_coldkey_ss58,
        block=fake_block,
    )

    # Asserts
    subtensor.query_subtensor.assert_called_once_with(
        "Stake", fake_block, [fake_hotkey_ss58, fake_coldkey_ss58]
    )
    if fake_value_result is not None:
        spy_balance_from_rao.assert_called_once_with(fake_value_result)
    else:
        spy_balance_from_rao.assert_not_called()
    assert result == fake_value_result


def test_does_hotkey_exist_true(mocker, subtensor):
    """Test when the hotkey exists."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_owner = "valid_owner"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor,
        "query_subtensor",
        return_value=mocker.Mock(value=fake_owner),
    )

    # Call
    result = subtensor.does_hotkey_exist(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        "Owner", fake_block, [fake_hotkey_ss58]
    )
    assert result is True


def test_does_hotkey_exist_no_value(mocker, subtensor):
    """Test when query_subtensor returns no value."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor,
        "query_subtensor",
        return_value=None,
    )

    # Call
    result = subtensor.does_hotkey_exist(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        "Owner", fake_block, [fake_hotkey_ss58]
    )
    assert result is False


def test_does_hotkey_exist_special_id(mocker, subtensor):
    """Test when query_subtensor returns the special invalid owner identifier."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_owner = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor,
        "query_subtensor",
        return_value=mocker.Mock(value=fake_owner),
    )

    # Call
    result = subtensor.does_hotkey_exist(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        "Owner", fake_block, [fake_hotkey_ss58]
    )
    assert result is False


def test_does_hotkey_exist_latest_block(mocker, subtensor):
    """Test when no block is provided (latest block)."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_owner = "valid_owner"

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor,
        "query_subtensor",
        return_value=mocker.Mock(value=fake_owner),
    )

    # Call
    result = subtensor.does_hotkey_exist(fake_hotkey_ss58)

    # Assertions
    mock_query_subtensor.assert_called_once_with("Owner", None, [fake_hotkey_ss58])
    assert result is True


def test_get_hotkey_owner_success(mocker, subtensor):
    """Test when hotkey exists and owner is found."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_coldkey_ss58 = "fake_coldkey"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor,
        "query_subtensor",
        return_value=mocker.Mock(value=fake_coldkey_ss58),
    )
    mock_does_hotkey_exist = mocker.patch.object(
        subtensor, "does_hotkey_exist", return_value=True
    )

    # Call
    result = subtensor.get_hotkey_owner(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        "Owner", fake_block, [fake_hotkey_ss58]
    )
    mock_does_hotkey_exist.assert_called_once_with(fake_hotkey_ss58, fake_block)
    assert result == fake_coldkey_ss58


def test_get_hotkey_owner_no_value(mocker, subtensor):
    """Test when query_subtensor returns no value."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor,
        "query_subtensor",
        return_value=None,
    )
    mock_does_hotkey_exist = mocker.patch.object(
        subtensor, "does_hotkey_exist", return_value=True
    )

    # Call
    result = subtensor.get_hotkey_owner(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        "Owner", fake_block, [fake_hotkey_ss58]
    )
    mock_does_hotkey_exist.assert_not_called()
    assert result is None


def test_get_hotkey_owner_does_not_exist(mocker, subtensor):
    """Test when hotkey does not exist."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor,
        "query_subtensor",
        return_value=mocker.Mock(value="fake_coldkey"),
    )
    mock_does_hotkey_exist = mocker.patch.object(
        subtensor, "does_hotkey_exist", return_value=False
    )

    # Call
    result = subtensor.get_hotkey_owner(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        "Owner", fake_block, [fake_hotkey_ss58]
    )
    mock_does_hotkey_exist.assert_called_once_with(fake_hotkey_ss58, fake_block)
    assert result is None


def test_get_hotkey_owner_latest_block(mocker, subtensor):
    """Test when no block is provided (latest block)."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_coldkey_ss58 = "fake_coldkey"

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor,
        "query_subtensor",
        return_value=mocker.Mock(value=fake_coldkey_ss58),
    )
    mock_does_hotkey_exist = mocker.patch.object(
        subtensor, "does_hotkey_exist", return_value=True
    )

    # Call
    result = subtensor.get_hotkey_owner(fake_hotkey_ss58)

    # Assertions
    mock_query_subtensor.assert_called_once_with("Owner", None, [fake_hotkey_ss58])
    mock_does_hotkey_exist.assert_called_once_with(fake_hotkey_ss58, None)
    assert result == fake_coldkey_ss58


def test_get_minimum_required_stake_success(mocker, subtensor):
    """Test successful call to get_minimum_required_stake."""
    # Mock data
    fake_min_stake = "1000000000"  # Example value in rao

    # Mocking
    mock_query = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=mocker.Mock(decode=mocker.Mock(return_value=fake_min_stake)),
    )
    mock_balance_from_rao = mocker.patch("bittensor.utils.balance.Balance.from_rao")

    # Call
    result = subtensor.get_minimum_required_stake()

    # Assertions
    mock_query.assert_called_once_with(
        module="SubtensorModule", storage_function="NominatorMinRequiredStake"
    )
    mock_balance_from_rao.assert_called_once_with(fake_min_stake)
    assert result == mock_balance_from_rao.return_value


def test_get_minimum_required_stake_query_failure(mocker, subtensor):
    """Test query failure in get_minimum_required_stake."""
    # Mocking
    mock_query = mocker.patch.object(
        subtensor.substrate,
        "query",
        side_effect=Exception("Query failed"),
    )

    # Call and Assertions
    with pytest.raises(Exception, match="Query failed"):
        subtensor.get_minimum_required_stake()
    mock_query.assert_called_once_with(
        module="SubtensorModule", storage_function="NominatorMinRequiredStake"
    )


def test_get_minimum_required_stake_invalid_result(mocker, subtensor):
    """Test when the result cannot be decoded."""
    # Mock data
    fake_invalid_stake = None  # Simulate a failure in decoding

    # Mocking
    mock_query = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=mocker.Mock(decode=mocker.Mock(return_value=fake_invalid_stake)),
    )
    mock_balance_from_rao = mocker.patch("bittensor.utils.balance.Balance.from_rao")

    # Call
    result = subtensor.get_minimum_required_stake()

    # Assertions
    mock_query.assert_called_once_with(
        module="SubtensorModule", storage_function="NominatorMinRequiredStake"
    )
    mock_balance_from_rao.assert_called_once_with(fake_invalid_stake)
    assert result == mock_balance_from_rao.return_value


def test_tx_rate_limit_success(mocker, subtensor):
    """Test when tx_rate_limit is successfully retrieved."""
    # Mock data
    fake_rate_limit = 100
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor,
        "query_subtensor",
        return_value=mocker.Mock(value=fake_rate_limit),
    )

    # Call
    result = subtensor.tx_rate_limit(block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with("TxRateLimit", fake_block)
    assert result == fake_rate_limit


def test_tx_rate_limit_no_value(mocker, subtensor):
    """Test when query_subtensor returns None."""
    # Mock data
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor,
        "query_subtensor",
        return_value=None,
    )

    # Call
    result = subtensor.tx_rate_limit(block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with("TxRateLimit", fake_block)
    assert result is None


def test_get_delegates_success(mocker, subtensor):
    """Test when delegates are successfully retrieved."""
    # Mock data
    fake_block = 123
    fake_block_hash = "0xabc123"
    fake_json_body = {
        "result": "mock_encoded_delegates",
    }

    # Mocks
    mock_get_block_hash = mocker.patch.object(
        subtensor.substrate,
        "get_block_hash",
        return_value=fake_block_hash,
    )
    mock_rpc_request = mocker.patch.object(
        subtensor.substrate,
        "rpc_request",
        return_value=fake_json_body,
    )
    mock_list_from_vec_u8 = mocker.patch.object(
        subtensor_module.DelegateInfo,
        "list_from_vec_u8",
        return_value=["delegate1", "delegate2"],
    )

    # Call
    result = subtensor.get_delegates(block=fake_block)

    # Assertions
    mock_get_block_hash.assert_called_once_with(fake_block)
    mock_rpc_request.assert_called_once_with(
        method="delegateInfo_getDelegates",
        params=[fake_block_hash],
    )
    mock_list_from_vec_u8.assert_called_once_with(fake_json_body["result"])
    assert result == ["delegate1", "delegate2"]


def test_get_delegates_no_result(mocker, subtensor):
    """Test when rpc_request returns no result."""
    # Mock data
    fake_block = 123
    fake_block_hash = "0xabc123"
    fake_json_body = {}

    # Mocks
    mock_get_block_hash = mocker.patch.object(
        subtensor.substrate,
        "get_block_hash",
        return_value=fake_block_hash,
    )
    mock_rpc_request = mocker.patch.object(
        subtensor.substrate,
        "rpc_request",
        return_value=fake_json_body,
    )

    # Call
    result = subtensor.get_delegates(block=fake_block)

    # Assertions
    mock_get_block_hash.assert_called_once_with(fake_block)
    mock_rpc_request.assert_called_once_with(
        method="delegateInfo_getDelegates",
        params=[fake_block_hash],
    )
    assert result == []


def test_get_delegates_latest_block(mocker, subtensor):
    """Test when no block is provided (latest block)."""
    # Mock data
    fake_json_body = {
        "result": "mock_encoded_delegates",
    }

    # Mocks
    mock_rpc_request = mocker.patch.object(
        subtensor.substrate,
        "rpc_request",
        return_value=fake_json_body,
    )
    mock_list_from_vec_u8 = mocker.patch.object(
        subtensor_module.DelegateInfo,
        "list_from_vec_u8",
        return_value=["delegate1", "delegate2"],
    )

    # Call
    result = subtensor.get_delegates()

    # Assertions
    mock_rpc_request.assert_called_once_with(
        method="delegateInfo_getDelegates",
        params=[],
    )
    mock_list_from_vec_u8.assert_called_once_with(fake_json_body["result"])
    assert result == ["delegate1", "delegate2"]


def test_is_hotkey_delegate_true(mocker, subtensor):
    """Test when hotkey is a delegate."""
    # Mock data
    fake_hotkey_ss58 = "hotkey_1"
    fake_block = 123
    fake_delegates = [
        mocker.Mock(hotkey_ss58="hotkey_1"),
        mocker.Mock(hotkey_ss58="hotkey_2"),
    ]

    # Mocks
    mock_get_delegates = mocker.patch.object(
        subtensor, "get_delegates", return_value=fake_delegates
    )

    # Call
    result = subtensor.is_hotkey_delegate(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_get_delegates.assert_called_once_with(block=fake_block)
    assert result is True


def test_is_hotkey_delegate_false(mocker, subtensor):
    """Test when hotkey is not a delegate."""
    # Mock data
    fake_hotkey_ss58 = "hotkey_3"
    fake_block = 123
    fake_delegates = [
        mocker.Mock(hotkey_ss58="hotkey_1"),
        mocker.Mock(hotkey_ss58="hotkey_2"),
    ]

    # Mocks
    mock_get_delegates = mocker.patch.object(
        subtensor, "get_delegates", return_value=fake_delegates
    )

    # Call
    result = subtensor.is_hotkey_delegate(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_get_delegates.assert_called_once_with(block=fake_block)
    assert result is False


def test_is_hotkey_delegate_empty_list(mocker, subtensor):
    """Test when delegate list is empty."""
    # Mock data
    fake_hotkey_ss58 = "hotkey_1"
    fake_block = 123

    # Mocks
    mock_get_delegates = mocker.patch.object(
        subtensor, "get_delegates", return_value=[]
    )

    # Call
    result = subtensor.is_hotkey_delegate(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_get_delegates.assert_called_once_with(block=fake_block)
    assert result is False


def test_add_stake_success(mocker, subtensor):
    """Test add_stake returns True on successful staking."""
    # Prep
    fake_wallet = mocker.Mock()
    fake_hotkey_ss58 = "fake_hotkey"
    fake_amount = 10.0

    mock_add_stake_extrinsic = mocker.patch.object(
        subtensor_module, "add_stake_extrinsic"
    )

    # Call
    result = subtensor.add_stake(
        wallet=fake_wallet,
        hotkey_ss58=fake_hotkey_ss58,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    # Assertions
    mock_add_stake_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=fake_hotkey_ss58,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    assert result == mock_add_stake_extrinsic.return_value


def test_add_stake_multiple_success(mocker, subtensor):
    """Test add_stake_multiple successfully stakes for all hotkeys."""
    # Prep
    fake_wallet = mocker.Mock()
    fake_hotkey_ss58 = ["fake_hotkey"]
    fake_amount = [10.0]

    mock_add_stake_multiple_extrinsic = mocker.patch.object(
        subtensor_module, "add_stake_multiple_extrinsic"
    )

    # Call
    result = subtensor.add_stake_multiple(
        wallet=fake_wallet,
        hotkey_ss58s=fake_hotkey_ss58,
        amounts=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    # Assertions
    mock_add_stake_multiple_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58s=fake_hotkey_ss58,
        amounts=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    assert result == mock_add_stake_multiple_extrinsic.return_value


def test_unstake_success(mocker, subtensor):
    """Test unstake operation is successful."""
    # Preps
    fake_wallet = mocker.Mock()
    fake_hotkey_ss58 = "hotkey_1"
    fake_amount = 10.0

    mock_unstake_extrinsic = mocker.patch.object(subtensor_module, "unstake_extrinsic")

    # Call
    result = subtensor.unstake(
        wallet=fake_wallet,
        hotkey_ss58=fake_hotkey_ss58,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    # Assertions
    mock_unstake_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=fake_hotkey_ss58,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    assert result == mock_unstake_extrinsic.return_value


def test_unstake_multiple_success(mocker, subtensor):
    """Test unstake_multiple succeeds for all hotkeys."""
    # Preps
    fake_wallet = mocker.Mock()
    fake_hotkeys = ["hotkey_1", "hotkey_2"]
    fake_amounts = [10.0, 20.0]

    mock_unstake_multiple_extrinsic = mocker.patch(
        "bittensor.core.subtensor.unstake_multiple_extrinsic", return_value=True
    )

    # Call
    result = subtensor.unstake_multiple(
        wallet=fake_wallet,
        hotkey_ss58s=fake_hotkeys,
        amounts=fake_amounts,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    # Assertions
    mock_unstake_multiple_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58s=fake_hotkeys,
        amounts=fake_amounts,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    assert result == mock_unstake_multiple_extrinsic.return_value
