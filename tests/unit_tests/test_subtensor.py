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
import datetime
from unittest.mock import MagicMock

import pytest
from bittensor_wallet import Wallet
from async_substrate_interface import sync_substrate
from async_substrate_interface.types import ScaleObj
import websockets

from bittensor import StakeInfo
from bittensor.core import settings
from bittensor.core import subtensor as subtensor_module
from bittensor.core.async_subtensor import AsyncSubtensor, logging
from bittensor.core.axon import Axon
from bittensor.core.chain_data import SubnetHyperparameters
from bittensor.core.extrinsics.serving import do_serve_axon
from bittensor.core.settings import version_as_int
from bittensor.core.subtensor import Subtensor
from bittensor.core.types import AxonServeCallParams
from bittensor.utils import (
    Certificate,
    u16_normalized_float,
    u64_normalized_float,
    determine_chain_endpoint_and_network,
)
from bittensor.utils.balance import Balance

U16_MAX = 65535
U64_MAX = 18446744073709551615


@pytest.fixture
def fake_call_params():
    return call_params()


def call_params():
    return AxonServeCallParams(
        version=settings.version_as_int,
        ip=0,
        port=9090,
        ip_type=4,
        netuid=1,
        hotkey="str",
        coldkey="str",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        certificate=None,
    )


def call_params_with_certificate():
    params = call_params()
    params.certificate = Certificate("fake_cert")
    return params


def test_methods_comparable(mock_substrate_interface):
    """Verifies that methods in sync and async Subtensors are comparable."""
    # Preps
    subtensor = Subtensor(_mock=True)
    async_subtensor = AsyncSubtensor(_mock=True)

    # methods which lives in async subtensor only
    excluded_async_subtensor_methods = ["initialize"]
    subtensor_methods = [m for m in dir(subtensor) if not m.startswith("_")]

    async_subtensor_methods = [
        m
        for m in dir(async_subtensor)
        if not m.startswith("_") and m not in excluded_async_subtensor_methods
    ]

    # Assertions
    for method in subtensor_methods:
        assert (
            method in async_subtensor_methods
        ), f"`Subtensor.{method}` not in `AsyncSubtensor` class."

    for method in async_subtensor_methods:
        assert (
            method in subtensor_methods
        ), f"`AsyncSubtensor.{method}` not in `Subtensor` class."


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


def test_serve_axon_with_external_port_set(mock_get_external_ip):
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
        ("ws://127.0.0.1:9945", "local", "ws://127.0.0.1:9945"),
        ("ws://localhost:9945", "local", "ws://localhost:9945"),
        # Edge cases
        (None, None, None),
        ("unknown", "unknown", "unknown"),
    ],
)
def test_determine_chain_endpoint_and_network(
    network, expected_network, expected_endpoint
):
    # Act
    result_network, result_endpoint = determine_chain_endpoint_and_network(network)

    # Assert
    assert result_network == expected_network
    assert result_endpoint == expected_endpoint


@pytest.fixture
def mock_logger():
    with mock.patch.object(logging, "warning") as mock_warning:
        yield mock_warning


def test_hyperparameter_subnet_does_not_exist(subtensor, mocker):
    """Tests when the subnet does not exist."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=False)
    assert subtensor.get_hyperparameter("Difficulty", 1, None) is None
    subtensor.subnet_exists.assert_called_once_with(1, block=None)


def test_hyperparameter_result_is_none(subtensor, mocker):
    """Tests when query_subtensor returns None."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.substrate.query = mocker.MagicMock(return_value=None)
    assert subtensor.get_hyperparameter("Difficulty", 1, None) is None
    subtensor.subnet_exists.assert_called_once_with(1, block=None)
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Difficulty",
        params=[1],
        block_hash=None,
    )


def test_hyperparameter_result_has_no_value(subtensor, mocker):
    """Test when the result has no 'value' attribute."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.substrate.query = mocker.MagicMock(return_value=None)
    assert subtensor.get_hyperparameter("Difficulty", 1, None) is None
    subtensor.subnet_exists.assert_called_once_with(1, block=None)
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Difficulty",
        params=[1],
        block_hash=None,
    )


def test_hyperparameter_success_int(subtensor, mocker):
    """Test when query_subtensor returns an integer value."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.substrate.query = mocker.MagicMock(
        return_value=mocker.MagicMock(value=100)
    )
    assert subtensor.get_hyperparameter("Difficulty", 1, None) == 100
    subtensor.subnet_exists.assert_called_once_with(1, block=None)
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Difficulty",
        params=[1],
        block_hash=None,
    )


def test_hyperparameter_success_float(subtensor, mocker):
    """Test when query_subtensor returns a float value."""
    subtensor.subnet_exists = mocker.MagicMock(return_value=True)
    subtensor.substrate.query = mocker.MagicMock(
        return_value=mocker.MagicMock(value=0.5)
    )
    assert subtensor.get_hyperparameter("Difficulty", 1, None) == 0.5
    subtensor.subnet_exists.assert_called_once_with(1, block=None)
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Difficulty",
        params=[1],
        block_hash=None,
    )


def test_blocks_since_last_update_success_calls(subtensor, mocker):
    """Tests the weights_rate_limit method to ensure it correctly fetches the LastUpdate hyperparameter."""
    # Prep
    uid = 7
    mocked_current_block = 2
    mocked_result = {uid: 1}
    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=mocked_result,
    )
    mocked_get_current_block = mocker.patch.object(
        subtensor,
        "get_current_block",
        return_value=mocked_current_block,
    )

    # Call
    result = subtensor.blocks_since_last_update(netuid=7, uid=uid)

    # Assertions
    mocked_get_current_block.assert_called_once()
    mocked_get_hyperparameter.assert_called_once_with(param_name="LastUpdate", netuid=7)
    assert result == 1
    # if we change the methods logic in the future we have to be make sure the returned type is correct
    assert isinstance(result, int)


def test_weights_rate_limit_success_calls(subtensor, mocker):
    """Tests the weights_rate_limit method to ensure it correctly fetches the WeightsSetRateLimit hyperparameter."""
    # Prep
    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=5,
    )

    # Call
    result = subtensor.weights_rate_limit(netuid=7)

    # Assertions
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="WeightsSetRateLimit",
        netuid=7,
        block=None,
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
            logging.console.error(f"❌ Error normalizing parameter '{param}': {e}")
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


def test_commit_reveal_enabled(subtensor, mocker):
    """Test commit_reveal_enabled."""
    # Preps
    netuid = 1
    block = 123
    mocked_get_hyperparameter = mocker.patch.object(subtensor, "get_hyperparameter")

    # Call
    result = subtensor.commit_reveal_enabled(netuid, block)

    # Assertions
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="CommitRevealWeightsEnabled", block=block, netuid=netuid
    )
    assert result is False


def test_get_subnet_reveal_period_epochs(subtensor, mocker):
    """Test get_subnet_reveal_period_epochs."""
    # Preps
    netuid = 1
    block = 123
    mocked_get_hyperparameter = mocker.patch.object(subtensor, "get_hyperparameter")

    # Call
    result = subtensor.get_subnet_reveal_period_epochs(netuid, block)

    # Assertions
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="RevealPeriodEpochs", block=block, netuid=netuid
    )
    assert result == mocked_get_hyperparameter.return_value


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
    mocker.patch.object(subtensor.substrate, "query", return_value=mock_result)

    # Call
    result = subtensor.subnet_exists(netuid, block)

    # Asserts
    assert result is True
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        params=[netuid],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)


def test_subnet_exists_no_data(mocker, subtensor):
    """Test subnet_exists returns False when no subnet information is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(subtensor.substrate, "query", return_value=None)

    # Call
    result = subtensor.subnet_exists(netuid, block)

    # Asserts
    assert result is False
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        params=[netuid],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)


def test_subnet_exists_no_value_attribute(mocker, subtensor):
    """Test subnet_exists returns False when result has no value attribute."""
    # Prep
    netuid = 1
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value
    mocker.patch.object(subtensor.substrate, "query", return_value=mock_result)

    # Call
    result = subtensor.subnet_exists(netuid, block)

    # Asserts
    assert result is False
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        params=[netuid],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)


def test_subnet_exists_no_block(mocker, subtensor):
    """Test subnet_exists with no block specified."""
    # Prep
    netuid = 1
    mock_result = mocker.MagicMock(value=True)
    mocker.patch.object(subtensor.substrate, "query", return_value=mock_result)

    # Call
    result = subtensor.subnet_exists(netuid)

    # Asserts
    assert result is True
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        params=[netuid],
        block_hash=None,
    )
    subtensor.substrate.get_block_hash.assert_not_called()


# `get_total_subnets` tests
def test_get_total_subnets_success(mocker, subtensor):
    """Test get_total_subnets returns correct data when total subnet information is found."""
    # Prep
    block = 123
    total_subnets_value = 10
    mock_result = mocker.MagicMock(value=total_subnets_value)
    mocker.patch.object(subtensor.substrate, "query", return_value=mock_result)

    # Call
    result = subtensor.get_total_subnets(block)

    # Asserts
    assert result is not None
    assert result == total_subnets_value
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="TotalNetworks",
        params=[],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)


def test_get_total_subnets_no_data(mocker, subtensor):
    """Test get_total_subnets returns None when no total subnet information is found."""
    # Prep
    block = 123
    mocker.patch.object(subtensor.substrate, "query", return_value=None)

    # Call
    result = subtensor.get_total_subnets(block)

    # Asserts
    assert result is None
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="TotalNetworks",
        params=[],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)


def test_get_total_subnets_no_value_attribute(mocker, subtensor):
    """Test get_total_subnets returns None when result has no value attribute."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    del mock_result.value  # Simulating a missing value attribute
    mocker.patch.object(subtensor.substrate, "query", return_value=mock_result)

    # Call
    result = subtensor.get_total_subnets(block)

    # Asserts
    assert result is None
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="TotalNetworks",
        params=[],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)


def test_get_total_subnets_no_block(mocker, subtensor):
    """Test get_total_subnets with no block specified."""
    # Prep
    total_subnets_value = 10
    mock_result = mocker.MagicMock(value=total_subnets_value)
    mocker.patch.object(subtensor.substrate, "query", return_value=mock_result)

    # Call
    result = subtensor.get_total_subnets()

    # Asserts
    assert result is not None
    assert result == total_subnets_value
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="TotalNetworks",
        params=[],
        block_hash=None,
    )
    subtensor.substrate.get_block_hash.assert_not_called()


# `get_subnets` tests
def test_get_subnets_success(mocker, subtensor):
    """Test get_subnets returns correct list when subnet information is found."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    mock_result.records = [(1, True), (2, True)]
    mock_result.__iter__.return_value = iter(mock_result.records)
    mocker.patch.object(subtensor.substrate, "query_map", return_value=mock_result)

    # Call
    result = subtensor.get_subnets(block)

    # Asserts
    assert result == [1, 2]
    subtensor.substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)


def test_get_subnets_no_data(mocker, subtensor):
    """Test get_subnets returns empty list when no subnet information is found."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    mock_result.records = []
    mocker.patch.object(subtensor.substrate, "query_map", return_value=mock_result)

    # Call
    result = subtensor.get_subnets(block)

    # Asserts
    assert result == []
    subtensor.substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)


def test_get_subnets_no_block_specified(mocker, subtensor):
    """Test get_subnets with no block specified."""
    # Prep
    mock_result = mocker.MagicMock()
    mock_result.records = [(1, True), (2, True)]
    mock_result.__iter__.return_value = iter(mock_result.records)
    mocker.patch.object(subtensor.substrate, "query_map", return_value=mock_result)

    # Call
    result = subtensor.get_subnets()

    # Asserts
    assert result == [1, 2]
    subtensor.substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        block_hash=None,
    )
    subtensor.substrate.get_block_hash.assert_not_called


# `get_subnet_hyperparameters` tests
def test_get_subnet_hyperparameters_success(mocker, subtensor):
    """Test get_subnet_hyperparameters returns correct data when hyperparameters are found."""
    # Prep
    netuid = 1
    block = 123

    mocker.patch.object(
        subtensor,
        "query_runtime_api",
    )
    mocker.patch.object(
        subtensor_module.SubnetHyperparameters,
        "from_dict",
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
    subtensor_module.SubnetHyperparameters.from_dict.assert_called_once_with(
        subtensor.query_runtime_api.return_value,
    )


def test_get_subnet_hyperparameters_no_data(mocker, subtensor):
    """Test get_subnet_hyperparameters returns empty list when no data is found."""
    # Prep
    netuid = 1
    block = 123
    mocker.patch.object(subtensor, "query_runtime_api", return_value=None)
    mocker.patch.object(subtensor_module.SubnetHyperparameters, "from_dict")

    # Call
    result = subtensor.get_subnet_hyperparameters(netuid, block)

    # Asserts
    assert result is None
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams",
        params=[netuid],
        block=block,
    )
    subtensor_module.SubnetHyperparameters.from_dict.assert_not_called()


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

    mock_determine_block_hash = mocker.patch.object(
        subtensor,
        "determine_block_hash",
    )
    # mock_runtime_call = mocker.patch.object(
    #     subtensor.substrate,
    #     "runtime_call",
    # )

    # Call
    result = subtensor.query_runtime_api(fake_runtime_api, fake_method, None)

    # Asserts
    subtensor.substrate.runtime_call.assert_called_once_with(
        fake_runtime_api,
        fake_method,
        None,
        mock_determine_block_hash.return_value,
    )
    mock_determine_block_hash.assert_called_once_with(None)

    assert result == subtensor.substrate.runtime_call.return_value.value


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
        network=subtensor.chain_endpoint,
        netuid=fake_netuid,
        lite=fake_lite,
        sync=False,
        subtensor=subtensor,
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
    mocker.patch.object(subtensor.substrate, "query_map", mocked_query_map_subtensor)

    # Call
    result = subtensor.get_netuids_for_hotkey(fake_hotkey_ss58, fake_block)

    # Asserts
    mocked_query_map_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="IsNetworkMember",
        params=[fake_hotkey_ss58],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
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
        fake_hotkey_ss58, fake_netuid, block=fake_block
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
        subtensor=subtensor,
        netuid=fake_netuid,
        axon=fake_axon,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
        certificate=fake_certificate,
    )
    assert result == mocked_serve_axon_extrinsic.return_value


def test_get_block_hash(subtensor, mocker):
    """Tests successful get_block_hash call."""
    # Prep
    fake_block_id = 123

    # Call
    result = subtensor.get_block_hash(fake_block_id)

    # Asserts
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block_id)
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
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        data_type=f"Raw{len(fake_data)}",
        data=fake_data.encode(),
    )
    assert result is mocked_publish_metadata.return_value


def test_subnetwork_n(subtensor, mocker):
    """Test successful subnetwork_n call."""
    # Prep
    fake_netuid = 1
    fake_block = 123
    fake_result = 2

    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=fake_result,
    )

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
        amount=Balance(fake_amount),
        transfer_all=False,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
        keep_alive=True,
    )
    assert result == mocked_transfer_extrinsic.return_value


def test_get_neuron_for_pubkey_and_subnet(subtensor, mocker):
    """Successful call to get_neuron_for_pubkey_and_subnet."""
    # Prep
    fake_hotkey_ss58 = "fake_hotkey"
    fake_netuid = 1
    fake_block = 123

    mocker.patch.object(
        subtensor.substrate,
        "rpc_request",
        return_value=mocker.MagicMock(
            **{
                "get.return_value": "0x32",
            },
        ),
    )
    mock_neuron_from_dict = mocker.patch.object(
        subtensor_module.NeuronInfo,
        "from_dict",
        return_value=["delegate1", "delegate2"],
    )

    # Call
    result = subtensor.get_neuron_for_pubkey_and_subnet(
        hotkey_ss58=fake_hotkey_ss58,
        netuid=fake_netuid,
        block=fake_block,
    )

    # Asserts
    subtensor.substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[fake_netuid, fake_hotkey_ss58],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.runtime_call.assert_called_once_with(
        "NeuronInfoRuntimeApi",
        "get_neuron",
        [fake_netuid, subtensor.substrate.query.return_value.value],
        subtensor.substrate.get_block_hash.return_value,
    )
    assert result == mock_neuron_from_dict.return_value


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

    subtensor.substrate.runtime_call.return_value.value = None

    # Call
    result = subtensor.neuron_for_uid(
        uid=fake_uid, netuid=fake_netuid, block=fake_block
    )

    # Asserts
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    subtensor.substrate.runtime_call.assert_called_once_with(
        "NeuronInfoRuntimeApi",
        "get_neuron",
        [fake_netuid, fake_uid],
        subtensor.substrate.get_block_hash.return_value,
    )

    mocked_neuron_info.assert_called_once()
    assert result == mocked_neuron_info.return_value


def test_neuron_for_uid_success(subtensor, mocker):
    """Test neuron_for_uid successful call."""
    # Prep
    fake_uid = 1
    fake_netuid = 2
    fake_block = 123
    mocked_neuron_from_dict = mocker.patch.object(
        subtensor_module.NeuronInfo, "from_dict"
    )

    # Call
    result = subtensor.neuron_for_uid(
        uid=fake_uid, netuid=fake_netuid, block=fake_block
    )

    # Asserts
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    subtensor.substrate.runtime_call.assert_called_once_with(
        "NeuronInfoRuntimeApi",
        "get_neuron",
        [fake_netuid, fake_uid],
        subtensor.substrate.get_block_hash.return_value,
    )

    assert result == mocked_neuron_from_dict.return_value


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
    result = do_serve_axon(
        subtensor=subtensor,
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
        extrinsic=subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    # subtensor.substrate.submit_extrinsic.return_value.process_events.assert_called_once()
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
    result = do_serve_axon(
        subtensor=subtensor,
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
        extrinsic=subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

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
    result = do_serve_axon(
        subtensor=subtensor,
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
        extrinsic=subtensor.substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )
    assert result == (True, None)


def test_immunity_period(subtensor, mocker):
    """Successful immunity_period call."""
    # Preps
    fake_netuid = 1
    fake_block = 123
    fake_result = 101

    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=fake_result,
    )

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
    subtensor.substrate.query = mocked_query_subtensor

    # Call
    result = subtensor.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=fake_hotkey_ss58, netuid=fake_netuid, block=fake_block
    )

    # Assertions
    mocked_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Uids",
        params=[fake_netuid, fake_hotkey_ss58],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)

    assert result == mocked_query_subtensor.return_value.value


def test_tempo(subtensor, mocker):
    """Successful tempo call."""
    # Preps
    fake_netuid = 1
    fake_block = 123
    fake_result = 101

    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=fake_result,
    )

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
    expected_result = (
        "{'peer_id': '12D3KooWFWnHBmUFxvfL6PfZ5eGHdhgsEqNnsxuN1HE9EtfW8THi', "
        "'model_huggingface_id': 'kmfoda/gpt2-1b-miner-3'}"
    )

    mocked_metagraph = mocker.MagicMock()
    subtensor.metagraph = mocked_metagraph
    mocked_metagraph.return_value.hotkeys = {fake_uid: fake_hotkey}

    mocked_get_metadata = mocker.patch.object(subtensor_module, "get_metadata")
    mocked_get_metadata.return_value = {
        "deposit": 0,
        "block": 3843930,
        "info": {
            "fields": (
                (
                    {
                        "Raw117": (
                            (
                                123,
                                39,
                                112,
                                101,
                                101,
                                114,
                                95,
                                105,
                                100,
                                39,
                                58,
                                32,
                                39,
                                49,
                                50,
                                68,
                                51,
                                75,
                                111,
                                111,
                                87,
                                70,
                                87,
                                110,
                                72,
                                66,
                                109,
                                85,
                                70,
                                120,
                                118,
                                102,
                                76,
                                54,
                                80,
                                102,
                                90,
                                53,
                                101,
                                71,
                                72,
                                100,
                                104,
                                103,
                                115,
                                69,
                                113,
                                78,
                                110,
                                115,
                                120,
                                117,
                                78,
                                49,
                                72,
                                69,
                                57,
                                69,
                                116,
                                102,
                                87,
                                56,
                                84,
                                72,
                                105,
                                39,
                                44,
                                32,
                                39,
                                109,
                                111,
                                100,
                                101,
                                108,
                                95,
                                104,
                                117,
                                103,
                                103,
                                105,
                                110,
                                103,
                                102,
                                97,
                                99,
                                101,
                                95,
                                105,
                                100,
                                39,
                                58,
                                32,
                                39,
                                107,
                                109,
                                102,
                                111,
                                100,
                                97,
                                47,
                                103,
                                112,
                                116,
                                50,
                                45,
                                49,
                                98,
                                45,
                                109,
                                105,
                                110,
                                101,
                                114,
                                45,
                                51,
                                39,
                                125,
                            ),
                        )
                    },
                ),
            )
        },
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

    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=return_value,
    )

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

    mocked_get_hyperparameter = mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        return_value=return_value,
    )

    mocked_u16_normalized_float = mocker.patch.object(
        subtensor_module,
        "u16_normalized_float",
    )

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
    value = Balance(1)

    fake_payment_info = {"partial_fee": int(2e10)}
    subtensor.substrate.get_payment_info.return_value = fake_payment_info

    # Call
    result = subtensor.get_transfer_fee(wallet=fake_wallet, dest=fake_dest, value=value)

    # Asserts
    subtensor.substrate.compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_allow_death",
        call_params={"dest": fake_dest, "value": value.rao},
    )

    subtensor.substrate.get_payment_info.assert_called_once_with(
        call=subtensor.substrate.compose_call.return_value,
        keypair=fake_wallet.coldkeypub,
    )

    assert result == 2e10


def test_get_existential_deposit(subtensor, mocker):
    """Successful get_existential_deposit call."""
    # Prep
    block = 123

    mocked_query_constant = mocker.MagicMock()
    value = 10
    mocked_query_constant.return_value.value = value
    subtensor.substrate.get_constant = mocked_query_constant

    # Call
    result = subtensor.get_existential_deposit(block=block)

    # Assertions
    mocked_query_constant.assert_called_once_with(
        module_name="Balances",
        constant_name="ExistentialDeposit",
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)

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


def test_get_subnet_burn_cost_success(subtensor, mocker):
    """Tests get_subnet_burn_cost method with successfully result."""
    # Preps
    mocked_query_runtime_api = mocker.patch.object(
        subtensor, "query_runtime_api", return_value=1000
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
    mocked_get_hyperparameter = mocker.patch.object(subtensor, "get_hyperparameter")
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
        subtensor, "get_hyperparameter", return_value=None
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
        subtensor, "get_hyperparameter", return_value=0.1
    )
    fake_netuid = 1
    fake_block = 2
    mocked_balance = mocker.patch("bittensor.core.subtensor.Balance")

    # Call
    result = subtensor.recycle(fake_netuid, fake_block)

    # Asserts
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="Burn",
        netuid=fake_netuid,
        block=fake_block,
    )

    mocked_balance.from_rao.assert_called_once_with(
        int(mocked_get_hyperparameter.return_value)
    )
    assert result == mocked_balance.from_rao.return_value


def test_recycle_none(subtensor, mocker):
    """Tests recycle method with None result."""
    # Preps
    mocked_get_hyperparameter = mocker.patch.object(
        subtensor, "get_hyperparameter", return_value=None
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

    mocker.patch.object(subtensor, "query_runtime_api")
    mocker.patch.object(
        subtensor_module.SubnetInfo,
        "list_from_dicts",
    )

    # Call
    subtensor.get_all_subnets_info(block)

    # Asserts
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnets_info_v2",
        params=[],
        block=block,
    )
    subtensor_module.SubnetInfo.list_from_dicts.assert_called_once_with(
        subtensor.query_runtime_api.return_value,
    )


@pytest.mark.parametrize("result_", [[], None])
def test_get_all_subnets_info_no_data(mocker, subtensor, result_):
    """Test get_all_subnets_info returns empty list when no subnet information is found."""
    # Prep
    block = 123
    mocker.patch.object(
        subtensor.substrate, "get_block_hash", return_value="mock_block_hash"
    )
    mocker.patch.object(subtensor_module.SubnetInfo, "list_from_dicts")

    mocker.patch.object(subtensor, "query_runtime_api", return_value=result_)

    # Call
    result = subtensor.get_all_subnets_info(block)

    # Asserts
    assert result == []
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnets_info_v2",
        params=[],
        block=block,
    )
    subtensor_module.SubnetInfo.list_from_dicts.assert_not_called()


def test_get_delegate_take_success(subtensor, mocker):
    """Verify `get_delegate_take` method successful path."""
    # Preps
    fake_hotkey_ss58 = "FAKE_SS58"
    fake_block = 123

    mocker.patch.object(subtensor_module, "u16_normalized_float")
    subtensor.query_subtensor = mocker.Mock(return_value=mocker.Mock(value="value"))

    # Call
    result = subtensor.get_delegate_take(hotkey_ss58=fake_hotkey_ss58, block=fake_block)

    # Asserts
    subtensor.query_subtensor.assert_called_once_with(
        name="Delegates",
        block=fake_block,
        params=[fake_hotkey_ss58],
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

    subtensor.query_subtensor = mocker.Mock(return_value=None)
    mocker.patch.object(subtensor_module, "u16_normalized_float")

    # Call
    result = subtensor.get_delegate_take(hotkey_ss58=fake_hotkey_ss58, block=fake_block)

    # Asserts
    subtensor.query_subtensor.assert_called_once_with(
        name="Delegates",
        block=fake_block,
        params=[fake_hotkey_ss58],
    )

    subtensor_module.u16_normalized_float.assert_not_called()
    assert result is None


def test_networks_during_connection(mocker):
    """Test networks during_connection."""
    # Preps
    mocker.patch.object(subtensor_module, "SubstrateInterface")
    mocker.patch("websockets.sync.client.connect")
    # Call
    for network in list(settings.NETWORK_MAP.keys()) + ["undefined"]:
        sub = Subtensor(network)

        # Assertions
        sub.network = network
        sub.chain_endpoint = settings.NETWORK_MAP.get(network)


@pytest.mark.asyncio
def test_get_stake_for_coldkey_and_hotkey(subtensor, mocker):
    netuids = [1, 2, 3]
    stake_info_dict = {
        "netuid": 1,
        "hotkey": b"\x16:\xech\r\xde,g\x03R1\xb9\x88q\xe79\xb8\x88\x93\xae\xd2)?*\rp\xb2\xe62\xads\x1c",
        "coldkey": b"\x16:\xech\r\xde,g\x03R1\xb9\x88q\xe79\xb8\x88\x93\xae\xd2)?*\rp\xb2\xe62\xads\x1c",
        "stake": 1,
        "locked": False,
        "emission": 1,
        "drain": 1,
        "is_registered": True,
    }
    query_result = stake_info_dict
    expected_result = {
        netuid: StakeInfo.from_dict(stake_info_dict) for netuid in netuids
    }

    query_fetcher = mocker.Mock(return_value=query_result)

    mocked_query_runtime_api = mocker.patch.object(
        subtensor, "query_runtime_api", side_effect=query_fetcher
    )
    mocked_get_subnets = mocker.patch.object(
        subtensor, "get_subnets", return_value=netuids
    )

    result = subtensor.get_stake_for_coldkey_and_hotkey(
        hotkey_ss58="hotkey", coldkey_ss58="coldkey", block=None, netuids=None
    )

    assert result == expected_result

    # validate that mocked functions were called with the right arguments
    mocked_query_runtime_api.assert_has_calls(
        [
            mock.call(
                "StakeInfoRuntimeApi",
                "get_stake_info_for_hotkey_coldkey_netuid",
                params=["hotkey", "coldkey", netuid],
                block=None,
            )
            for netuid in netuids
        ]
    )
    mocked_get_subnets.assert_called_once_with(block=None)


def test_does_hotkey_exist_true(mocker, subtensor):
    """Test when the hotkey exists."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_owner = "valid_owner"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=mocker.Mock(value=[fake_owner]),
    )
    mocker.patch.object(
        subtensor_module,
        "decode_account_id",
        return_value=fake_owner,
    )

    # Call
    result = subtensor.does_hotkey_exist(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    assert result is True


def test_does_hotkey_exist_no_value(mocker, subtensor):
    """Test when query_subtensor returns no value."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor.substrate, "query", return_value=None
    )

    # Call
    result = subtensor.does_hotkey_exist(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    assert result is False


def test_does_hotkey_exist_special_id(mocker, subtensor):
    """Test when query_subtensor returns the special invalid owner identifier."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_owner = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=fake_owner,
    )
    mocker.patch.object(
        subtensor_module,
        "decode_account_id",
        return_value=fake_owner,
    )

    # Call
    result = subtensor.does_hotkey_exist(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    assert result is False


def test_does_hotkey_exist_latest_block(mocker, subtensor):
    """Test when no block is provided (latest block)."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_owner = "valid_owner"

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=mocker.Mock(value=[fake_owner]),
    )
    mocker.patch.object(
        subtensor_module,
        "decode_account_id",
        return_value=fake_owner,
    )

    # Call
    result = subtensor.does_hotkey_exist(fake_hotkey_ss58)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=None,
    )
    assert result is True


def test_get_hotkey_owner_success(mocker, subtensor):
    """Test when hotkey exists and owner is found."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_coldkey_ss58 = "fake_coldkey"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor.substrate, "query", return_value=fake_coldkey_ss58
    )
    mock_does_hotkey_exist = mocker.patch.object(
        subtensor, "does_hotkey_exist", return_value=True
    )

    # Call
    result = subtensor.get_hotkey_owner(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    mock_does_hotkey_exist.assert_called_once_with(fake_hotkey_ss58, block=fake_block)
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    assert result == fake_coldkey_ss58


def test_get_hotkey_owner_no_value(mocker, subtensor):
    """Test when query_subtensor returns no value."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=None,
    )
    mock_does_hotkey_exist = mocker.patch.object(
        subtensor, "does_hotkey_exist", return_value=True
    )

    # Call
    result = subtensor.get_hotkey_owner(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    mock_does_hotkey_exist.assert_not_called()
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    assert result is None


def test_get_hotkey_owner_does_not_exist(mocker, subtensor):
    """Test when hotkey does not exist."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=mocker.Mock(value=[fake_hotkey_ss58]),
    )
    mock_does_hotkey_exist = mocker.patch.object(
        subtensor, "does_hotkey_exist", return_value=False
    )
    mocker.patch.object(
        subtensor_module,
        "decode_account_id",
        return_value=fake_hotkey_ss58,
    )

    # Call
    result = subtensor.get_hotkey_owner(fake_hotkey_ss58, block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    mock_does_hotkey_exist.assert_called_once_with(fake_hotkey_ss58, block=fake_block)
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    assert result is None


def test_get_hotkey_owner_latest_block(mocker, subtensor):
    """Test when no block is provided (latest block)."""
    # Mock data
    fake_hotkey_ss58 = "fake_hotkey"
    fake_coldkey_ss58 = "fake_coldkey"

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor.substrate, "query", return_value=fake_coldkey_ss58
    )
    mock_does_hotkey_exist = mocker.patch.object(
        subtensor, "does_hotkey_exist", return_value=True
    )

    # Call
    result = subtensor.get_hotkey_owner(fake_hotkey_ss58)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Owner",
        params=[fake_hotkey_ss58],
        block_hash=None,
    )
    mock_does_hotkey_exist.assert_called_once_with(fake_hotkey_ss58, block=None)
    assert result == fake_coldkey_ss58


def test_get_minimum_required_stake_success(mocker, subtensor):
    """Test successful call to get_minimum_required_stake."""
    # Mock data
    fake_min_stake = "1000000000"  # Example value in rao

    # Mocking
    mock_query = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=mocker.Mock(value=fake_min_stake),
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
        return_value=mocker.Mock(value=fake_invalid_stake),
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
        subtensor.substrate,
        "query",
        return_value=mocker.Mock(value=fake_rate_limit),
    )

    # Call
    result = subtensor.tx_rate_limit(block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="TxRateLimit",
        params=None,
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    assert result == fake_rate_limit


def test_tx_rate_limit_no_value(mocker, subtensor):
    """Test when query_subtensor returns None."""
    # Mock data
    fake_block = 123

    # Mocks
    mock_query_subtensor = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=None,
    )

    # Call
    result = subtensor.tx_rate_limit(block=fake_block)

    # Assertions
    mock_query_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="TxRateLimit",
        params=None,
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(fake_block)
    assert result is None


def test_get_delegates_success(mocker, subtensor):
    """Test when delegates are successfully retrieved."""
    # Mock data
    fake_block = 123

    # Mocks
    mock_query_runtime_api = mocker.patch.object(
        subtensor,
        "query_runtime_api",
    )
    mock_list_from_dicts = mocker.patch.object(
        subtensor_module.DelegateInfo,
        "list_from_dicts",
    )

    # Call
    result = subtensor.get_delegates(block=fake_block)

    # Assertions
    mock_query_runtime_api.assert_called_once_with(
        runtime_api="DelegateInfoRuntimeApi",
        method="get_delegates",
        params=[],
        block=123,
    )
    mock_list_from_dicts.assert_called_once_with(mock_query_runtime_api.return_value)

    assert result == mock_list_from_dicts.return_value


def test_get_delegates_no_result(mocker, subtensor):
    """Test when rpc_request returns no result."""
    # Mock data
    fake_block = 123

    # Mocks
    mock_query_runtime_api = mocker.patch.object(
        subtensor,
        "query_runtime_api",
        return_value=None,
    )
    mock_list_from_dicts = mocker.patch.object(
        subtensor_module.DelegateInfo,
        "list_from_dicts",
    )

    # Call
    result = subtensor.get_delegates(block=fake_block)

    # Assertions
    mock_query_runtime_api.assert_called_once_with(
        runtime_api="DelegateInfoRuntimeApi",
        method="get_delegates",
        params=[],
        block=123,
    )
    mock_list_from_dicts.assert_not_called()

    assert result == []


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
    mock_get_delegates.assert_called_once_with(fake_block)
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
    mock_get_delegates.assert_called_once_with(fake_block)
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
    mock_get_delegates.assert_called_once_with(fake_block)
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
        netuid=None,
        amount=Balance.from_rao(fake_amount),
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
        netuids=[1],
        amounts=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    # Assertions
    mock_add_stake_multiple_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58s=fake_hotkey_ss58,
        netuids=[1],
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
        netuid=None,
        amount=Balance.from_rao(fake_amount),
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
        netuids=[1, 2],
        amounts=fake_amounts,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    # Assertions
    mock_unstake_multiple_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58s=fake_hotkeys,
        netuids=[1, 2],
        amounts=fake_amounts,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    assert result == mock_unstake_multiple_extrinsic.return_value


def test_set_weights_with_commit_reveal_enabled(subtensor, mocker):
    """Test set_weights with commit_reveal_enabled is True."""
    # Preps
    fake_wallet = mocker.Mock()
    fake_netuid = 1
    fake_uids = [1, 5]
    fake_weights = [0.1, 0.9]
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = False

    mocked_commit_reveal_enabled = mocker.patch.object(
        subtensor, "commit_reveal_enabled", return_value=True
    )
    mocked_commit_reveal_v3_extrinsic = mocker.patch.object(
        subtensor_module, "commit_reveal_v3_extrinsic"
    )
    mocked_commit_reveal_v3_extrinsic.return_value = (
        True,
        "Weights committed successfully",
    )
    mocker.patch.object(subtensor, "blocks_since_last_update", return_value=181)
    mocker.patch.object(subtensor, "weights_rate_limit", return_value=180)

    # Call
    result = subtensor.set_weights(
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    # Asserts
    mocked_commit_reveal_enabled.assert_called_once_with(netuid=fake_netuid)
    mocked_commit_reveal_v3_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        version_key=subtensor_module.version_as_int,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )
    assert result == mocked_commit_reveal_v3_extrinsic.return_value


def test_connection_limit(mocker):
    """Test connection limit is not exceeded."""
    # Technically speaking, this test should exist in integration tests. But to reduce server costs we will leave this
    # test here.

    # Preps
    mocker.patch.object(
        sync_substrate,
        "connect",
        side_effect=websockets.InvalidStatus(
            response=mocker.Mock(
                response=mocker.Mock(
                    status_code=429, message="test connection limit error"
                )
            )
        ),
    )

    # Call with assertions
    with pytest.raises(websockets.InvalidStatus):
        for i in range(2):
            Subtensor("test")


def test_set_subnet_identity(mocker, subtensor):
    """Verify that subtensor method `set_subnet_identity` calls proper function with proper arguments."""
    # Preps
    fake_wallet = mocker.Mock()
    fake_netuid = 123
    fake_subnet_identity = mocker.MagicMock()

    mocked_extrinsic = mocker.patch.object(
        subtensor_module, "set_subnet_identity_extrinsic"
    )

    # Call
    result = subtensor.set_subnet_identity(
        wallet=fake_wallet, netuid=fake_netuid, subnet_identity=fake_subnet_identity
    )

    # Asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        subnet_name=fake_subnet_identity.subnet_name,
        github_repo=fake_subnet_identity.github_repo,
        subnet_contact=fake_subnet_identity.subnet_contact,
        subnet_url=fake_subnet_identity.subnet_url,
        discord=fake_subnet_identity.discord,
        description=fake_subnet_identity.description,
        additional=fake_subnet_identity.additional,
        wait_for_finalization=True,
        wait_for_inclusion=False,
    )
    assert result == mocked_extrinsic.return_value


def test_get_all_neuron_certificates(mocker, subtensor):
    fake_netuid = 12
    mocked_query_map_subtensor = mocker.MagicMock()
    mocker.patch.object(subtensor.substrate, "query_map", mocked_query_map_subtensor)
    subtensor.get_all_neuron_certificates(fake_netuid)
    mocked_query_map_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NeuronCertificates",
        params=[fake_netuid],
        block_hash=None,
    )


def test_get_timestamp(mocker, subtensor):
    fake_block = 1000
    mocked_query = mocker.MagicMock(return_value=ScaleObj(1740586018 * 1000))
    mocker.patch.object(subtensor.substrate, "query", mocked_query)
    expected_result = datetime.datetime(
        2025, 2, 26, 16, 6, 58, tzinfo=datetime.timezone.utc
    )
    actual_result = subtensor.get_timestamp(block=fake_block)
    assert expected_result == actual_result
