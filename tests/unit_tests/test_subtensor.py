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
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
from bittensor_wallet import Wallet

from bittensor.core import subtensor as subtensor_module, settings
from bittensor.core.axon import Axon
from bittensor.core.chain_data import SubnetHyperparameters
from bittensor.core.subtensor import Subtensor, logging
from bittensor.utils import u16_normalized_float, u64_normalized_float
from bittensor.utils.balance import Balance

U16_MAX = 65535
U64_MAX = 18446744073709551615


def test_serve_axon_with_external_ip_set():
    internal_ip: str = "192.0.2.146"
    external_ip: str = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

    mock_serve_axon = MagicMock(return_value=True)

    mock_subtensor = MagicMock(spec=Subtensor, serve_axon=mock_serve_axon)

    mock_add_insecure_port = mock.MagicMock(return_value=None)
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


# Subtensor().get_error_info_by_index tests
@pytest.fixture
def substrate():
    class MockSubstrate:
        pass

    return MockSubstrate()


@pytest.fixture
def subtensor(substrate):
    return Subtensor()


@pytest.fixture
def mock_logger():
    with mock.patch.object(logging, "warning") as mock_warning:
        yield mock_warning


# Subtensor()._get_hyperparameter tests
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
) -> List[Tuple[str, str, str]]:
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

    normalized_values: List[Tuple[str, str, str]] = []
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


def test_get_subnet_hyperparameters_hex_without_prefix(mocker, subtensor):
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
