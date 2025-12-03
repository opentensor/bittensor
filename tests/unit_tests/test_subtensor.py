import argparse
import datetime
import unittest.mock as mock
from unittest.mock import MagicMock, ANY

import pytest
import websockets
from async_substrate_interface import sync_substrate
from async_substrate_interface.types import ScaleObj, Runtime
from bittensor_wallet import Wallet
from scalecodec import GenericCall

from bittensor import StakeInfo
from bittensor.core import settings
from bittensor.core import subtensor as subtensor_module
from bittensor.core.async_subtensor import AsyncSubtensor, logging
from bittensor.core.axon import Axon
from bittensor.core.chain_data import SubnetHyperparameters, SelectiveMetagraphIndex
from bittensor.core.settings import (
    DEFAULT_MEV_PROTECTION,
    DEFAULT_PERIOD,
    version_as_int,
)
from bittensor.core.subtensor import Subtensor
from bittensor.core.types import AxonServeCallParams
from bittensor.core.types import ExtrinsicResponse
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


def test_methods_comparable(mock_substrate):
    """Verifies that methods in sync and async Subtensors are comparable."""
    # Preps
    subtensor = Subtensor(mock=True)
    async_subtensor = AsyncSubtensor(mock=True)

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
        assert method in async_subtensor_methods, (
            f"`Subtensor.{method}` not in `AsyncSubtensor` class."
        )

    for method in async_subtensor_methods:
        assert method in subtensor_methods, (
            f"`AsyncSubtensor.{method}` not in `Subtensor` class."
        )


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
    mocked_get_hyperparameter.assert_called_once_with(
        param_name="LastUpdate", netuid=7, block=mocked_current_block
    )
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

    Parameters:
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
        assert numeric_value == expected_tao, (
            f"Mismatch in tao value for {param_name} at mid value"
        )
    else:
        assert float(norm_value) == 0.5, f"Failed mid-point test for {param_name}"

    # Max-value test
    setattr(sample_hyperparameters, param_name, max_value)
    normalized = normalize_hyperparameters(sample_hyperparameters)
    norm_value = get_normalized_value(normalized, param_name)

    if is_balance:
        numeric_value = float(str(norm_value).lstrip(settings.TAO_SYMBOL))
        expected_tao = max_value / 1e9
        assert numeric_value == expected_tao, (
            f"Mismatch in tao value for {param_name} at max value"
        )
    else:
        assert float(norm_value) == 1.0, f"Failed max value test for {param_name}"

    # Zero-value test
    setattr(sample_hyperparameters, param_name, zero_value)
    normalized = normalize_hyperparameters(sample_hyperparameters)
    norm_value = get_normalized_value(normalized, param_name)

    if is_balance:
        numeric_value = float(str(norm_value).lstrip(settings.TAO_SYMBOL))
        expected_tao = zero_value / 1e9
        assert numeric_value == expected_tao, (
            f"Mismatch in tao value for {param_name} at zero value"
        )
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


# `get_all_subnets_netuid` tests
def test_get_subnets_success(mocker, subtensor):
    """Test get_all_subnets_netuid returns correct list when subnet information is found."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    mock_result.records = [(1, True), (2, True)]
    mock_result.__iter__.return_value = iter(mock_result.records)
    mocker.patch.object(subtensor.substrate, "query_map", return_value=mock_result)

    # Call
    result = subtensor.get_all_subnets_netuid(block)

    # Asserts
    assert result == [1, 2]
    subtensor.substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)


def test_get_subnets_no_data(mocker, subtensor):
    """Test get_all_subnets_netuid returns empty list when no subnet information is found."""
    # Prep
    block = 123
    mock_result = mocker.MagicMock()
    mock_result.records = []
    mocker.patch.object(subtensor.substrate, "query_map", return_value=mock_result)

    # Call
    result = subtensor.get_all_subnets_netuid(block)

    # Asserts
    assert result == []
    subtensor.substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        block_hash=subtensor.substrate.get_block_hash.return_value,
    )
    subtensor.substrate.get_block_hash.assert_called_once_with(block)


def test_get_subnets_no_block_specified(mocker, subtensor):
    """Test get_all_subnets_netuid with no block specified."""
    # Prep
    mock_result = mocker.MagicMock()
    mock_result.records = [(1, True), (2, True)]
    mock_result.__iter__.return_value = iter(mock_result.records)
    mocker.patch.object(subtensor.substrate, "query_map", return_value=mock_result)

    # Call
    result = subtensor.get_all_subnets_netuid()

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
    subtensor.get_subnet_hyperparameters(netuid, block)

    # Asserts
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_hyperparams_v2",
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
        method="get_subnet_hyperparams_v2",
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
        method="state_call", params=[fake_method, fake_data], block_hash=None
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
    default_mechid = 0
    fake_lite = True
    mocked_metagraph = mocker.patch.object(subtensor_module, "Metagraph")

    # Call
    result = subtensor.metagraph(fake_netuid, lite=fake_lite)

    # Asserts
    mocked_metagraph.assert_called_once_with(
        network=subtensor.chain_endpoint,
        netuid=fake_netuid,
        mechid=default_mechid,
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
        netuid=fake_netuid,
        axon=fake_axon,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    # Asserts
    mocked_serve_axon_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        netuid=fake_netuid,
        axon=fake_axon,
        certificate=fake_certificate,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
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


def test_commit(subtensor, fake_wallet, mocker):
    """Test a successful commit call."""
    # Preps
    fake_netuid = 1
    fake_data = "some data to network"
    mocked_publish_metadata = mocker.patch.object(
        subtensor_module, "publish_metadata_extrinsic"
    )

    # Call
    result = subtensor.set_commitment(fake_wallet, fake_netuid, fake_data)

    # Asserts
    mocked_publish_metadata.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        data_type=f"Raw{len(fake_data)}",
        data=fake_data.encode(),
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
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


def test_transfer(subtensor, fake_wallet, mocker):
    """Tests successful transfer call."""
    # Prep
    fake_dest = "SS58PUBLICKEY"
    fake_amount = Balance.from_tao(1.1)
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = True
    mocked_transfer_extrinsic = mocker.patch.object(
        subtensor_module, "transfer_extrinsic"
    )

    # Call
    result = subtensor.transfer(
        wallet=fake_wallet,
        destination_ss58=fake_dest,
        amount=fake_amount,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
    )

    # Asserts
    mocked_transfer_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        destination_ss58=fake_dest,
        amount=fake_amount,
        transfer_all=False,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
        keep_alive=True,
        period=DEFAULT_PERIOD,
        raise_error=False,
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

    mocked_get_metadata = mocker.patch.object(subtensor, "get_commitment_metadata")
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


def test_get_last_commitment_bonds_reset_block(subtensor, mocker):
    """Successful get_last_commitment_bonds_reset_block call."""
    # Preps
    fake_netuid = 1
    fake_uid = 2
    fake_hotkey = "hotkey"

    mocked_get_last_bonds_reset = mocker.patch.object(subtensor, "get_last_bonds_reset")
    mocked_decode_block = mocker.patch.object(subtensor_module, "decode_block")

    mocked_metagraph = mocker.MagicMock()
    subtensor.metagraph = mocked_metagraph
    mocked_metagraph.return_value.hotkeys = {fake_uid: fake_hotkey}

    # Call
    result = subtensor.get_last_commitment_bonds_reset_block(
        netuid=fake_netuid, uid=fake_uid
    )

    # Assertions
    mocked_metagraph.assert_called_once_with(fake_netuid, block=None)
    mocked_get_last_bonds_reset.assert_called_once_with(fake_netuid, fake_hotkey, None)
    mocked_decode_block.assert_called_once_with(
        mocked_get_last_bonds_reset.return_value
    )
    assert result == mocked_decode_block.return_value


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


def test_get_transfer_fee(subtensor, fake_wallet, mocker):
    """Successful get_transfer_fee call."""
    # Preps
    fake_dest = "SS58ADDRESS"
    value = Balance(1)

    fake_payment_info = {"partial_fee": int(2e10)}
    subtensor.substrate.get_payment_info.return_value = fake_payment_info
    mocker_compose_call = mocker.patch.object(subtensor, "compose_call")

    # Call
    result = subtensor.get_transfer_fee(
        wallet=fake_wallet, destination_ss58=fake_dest, amount=value
    )

    # Asserts
    mocker_compose_call.assert_called_once_with(
        call_module="Balances",
        call_function="transfer_keep_alive",
        call_params={"dest": fake_dest, "value": value.rao},
    )

    subtensor.substrate.get_payment_info.assert_called_once_with(
        call=mocker_compose_call.return_value,
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


def test_reveal_weights(subtensor, fake_wallet, mocker):
    """Successful test_reveal_weights call."""
    # Preps
    netuid = 1
    uids = [1, 2, 3, 4]
    weights = [0.1, 0.2, 0.3, 0.4]
    salt = [4, 2, 2, 1]
    expected_result = ExtrinsicResponse(True, None)
    mocked_extrinsic = mocker.patch.object(
        subtensor_module,
        "reveal_weights_extrinsic",
        return_value=expected_result,
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
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=16,
        raise_error=False,
        wait_for_inclusion=False,
        wait_for_finalization=False,
        mechid=0,
    )


def test_reveal_weights_false(subtensor, fake_wallet, mocker):
    """Failed test_reveal_weights call."""
    # Preps
    netuid = 1
    uids = [1, 2, 3, 4]
    weights = [0.1, 0.2, 0.3, 0.4]
    salt = [4, 2, 2, 1]

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
    assert result == mocked_extrinsic.return_value
    assert mocked_extrinsic.call_count == 1


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


def test_networks_during_connection(mock_substrate, mocker):
    """Test networks during_connection."""
    # Preps
    mocker.patch("websockets.sync.client.connect")
    # Call
    for network in list(settings.NETWORK_MAP.keys()) + ["undefined"]:
        sub = Subtensor(network)

        # Assertions
        sub.network = network
        sub.chain_endpoint = settings.NETWORK_MAP.get(network)


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
        subtensor, "get_all_subnets_netuid", return_value=netuids
    )

    result = subtensor.get_stake_for_coldkey_and_hotkey(
        hotkey_ss58="hotkey", coldkey_ss58="coldkey", block=None, netuids=None
    )

    assert result == expected_result

    # validate that mocked functions were called with the right arguments
    mocked_query_runtime_api.assert_has_calls(
        [
            mock.call(
                runtime_api="StakeInfoRuntimeApi",
                method="get_stake_info_for_hotkey_coldkey_netuid",
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


def test_add_stake_success(mocker, fake_wallet, subtensor):
    """Test add_stake returns True on successful staking."""
    # Prep
    fake_hotkey_ss58 = "fake_hotkey"
    fake_amount = Balance.from_tao(10.0)
    fake_netuid = 14

    mock_add_stake_extrinsic = mocker.patch.object(
        subtensor_module, "add_stake_extrinsic"
    )

    # Call
    result = subtensor.add_stake(
        wallet=fake_wallet,
        netuid=fake_netuid,
        hotkey_ss58=fake_hotkey_ss58,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        safe_staking=False,
        allow_partial_stake=False,
        rate_tolerance=0.005,
    )

    # Assertions
    mock_add_stake_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=fake_hotkey_ss58,
        netuid=14,
        amount=fake_amount.rao,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        safe_staking=False,
        allow_partial_stake=False,
        rate_tolerance=0.005,
        period=DEFAULT_PERIOD,
        raise_error=False,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )
    assert result == mock_add_stake_extrinsic.return_value


def test_add_stake_with_safe_staking(mocker, fake_wallet, subtensor):
    """Test add_stake with safe staking parameters enabled."""
    # Prep
    fake_netuid = 14
    fake_hotkey_ss58 = "fake_hotkey"
    fake_amount = Balance.from_tao(10.0)
    fake_rate_tolerance = 0.01  # 1% threshold

    mock_add_stake_extrinsic = mocker.patch.object(
        subtensor_module, "add_stake_extrinsic"
    )

    # Call
    result = subtensor.add_stake(
        wallet=fake_wallet,
        netuid=fake_netuid,
        hotkey_ss58=fake_hotkey_ss58,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        safe_staking=True,
        allow_partial_stake=False,
        rate_tolerance=fake_rate_tolerance,
    )

    # Assertions
    mock_add_stake_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=fake_hotkey_ss58,
        netuid=14,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        safe_staking=True,
        allow_partial_stake=False,
        rate_tolerance=fake_rate_tolerance,
        period=DEFAULT_PERIOD,
        raise_error=False,
        mev_protection=DEFAULT_MEV_PROTECTION,
    )
    assert result == mock_add_stake_extrinsic.return_value


def test_add_stake_multiple_success(mocker, fake_wallet, subtensor):
    """Test add_stake_multiple successfully stakes for all hotkeys."""
    # Prep
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
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        period=DEFAULT_PERIOD,
        raise_error=False,
    )
    assert result == mock_add_stake_multiple_extrinsic.return_value


def test_unstake_success(mocker, subtensor, fake_wallet):
    """Test unstake operation is successful."""
    # Preps
    fake_hotkey_ss58 = "hotkey_1"
    fake_netuid = 1
    fake_amount = Balance.from_tao(10.0)

    mock_unstake_extrinsic = mocker.patch.object(subtensor_module, "unstake_extrinsic")

    # Call
    result = subtensor.unstake(
        wallet=fake_wallet,
        netuid=fake_netuid,
        hotkey_ss58=fake_hotkey_ss58,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        safe_unstaking=False,
        allow_partial_stake=False,
        rate_tolerance=0.005,
    )

    # Assertions
    mock_unstake_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        hotkey_ss58=fake_hotkey_ss58,
        amount=fake_amount,
        safe_unstaking=False,
        allow_partial_stake=False,
        rate_tolerance=0.005,
        period=DEFAULT_PERIOD,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        raise_error=False,
    )
    assert result == mock_unstake_extrinsic.return_value


def test_unstake_with_safe_unstaking(mocker, subtensor, fake_wallet):
    """Test unstake with `safe_unstaking` parameters enabled."""
    fake_hotkey_ss58 = "hotkey_1"
    fake_amount = Balance.from_tao(10.0)
    fake_netuid = 14
    fake_rate_tolerance = 0.01  # 1% threshold

    mock_unstake_extrinsic = mocker.patch.object(subtensor_module, "unstake_extrinsic")

    # Call
    result = subtensor.unstake(
        wallet=fake_wallet,
        netuid=fake_netuid,
        hotkey_ss58=fake_hotkey_ss58,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        safe_unstaking=True,
        allow_partial_stake=True,
        rate_tolerance=fake_rate_tolerance,
    )

    # Assertions
    mock_unstake_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        hotkey_ss58=fake_hotkey_ss58,
        amount=fake_amount,
        safe_unstaking=True,
        allow_partial_stake=True,
        rate_tolerance=fake_rate_tolerance,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    assert result == mock_unstake_extrinsic.return_value


def test_swap_stake_success(mocker, subtensor, fake_wallet):
    """Test swap_stake operation is successful."""
    # Preps
    fake_hotkey_ss58 = "hotkey_1"
    fake_origin_netuid = 1
    fake_destination_netuid = 2
    fake_amount = Balance.from_tao(10.0)

    mock_swap_stake_extrinsic = mocker.patch.object(
        subtensor_module, "swap_stake_extrinsic"
    )

    # Call
    result = subtensor.swap_stake(
        wallet=fake_wallet,
        hotkey_ss58=fake_hotkey_ss58,
        origin_netuid=fake_origin_netuid,
        destination_netuid=fake_destination_netuid,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        safe_swapping=False,
        allow_partial_stake=False,
        rate_tolerance=0.005,
    )

    # Assertions
    mock_swap_stake_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=fake_hotkey_ss58,
        origin_netuid=fake_origin_netuid,
        destination_netuid=fake_destination_netuid,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        safe_swapping=False,
        allow_partial_stake=False,
        rate_tolerance=0.005,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
    )
    assert result == mock_swap_stake_extrinsic.return_value


def test_swap_stake_with_safe_staking(mocker, subtensor, fake_wallet):
    """Test swap_stake with safe staking parameters enabled."""
    # Preps
    fake_hotkey_ss58 = "hotkey_1"
    fake_origin_netuid = 1
    fake_destination_netuid = 2
    fake_amount = Balance.from_tao(10.0)
    fake_rate_tolerance = 0.01  # 1% threshold

    mock_swap_stake_extrinsic = mocker.patch.object(
        subtensor_module, "swap_stake_extrinsic"
    )

    # Call
    result = subtensor.swap_stake(
        wallet=fake_wallet,
        hotkey_ss58=fake_hotkey_ss58,
        origin_netuid=fake_origin_netuid,
        destination_netuid=fake_destination_netuid,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        safe_swapping=True,
        allow_partial_stake=True,
        rate_tolerance=fake_rate_tolerance,
    )

    # Assertions
    mock_swap_stake_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=fake_hotkey_ss58,
        origin_netuid=fake_origin_netuid,
        destination_netuid=fake_destination_netuid,
        amount=fake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        safe_swapping=True,
        allow_partial_stake=True,
        rate_tolerance=fake_rate_tolerance,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
    )
    assert result == mock_swap_stake_extrinsic.return_value


def test_unstake_multiple_success(mocker, subtensor, fake_wallet):
    """Test unstake_multiple succeeds for all hotkeys."""
    # Preps
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
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        unstake_all=False,
        raise_error=False,
    )
    assert result == mock_unstake_multiple_extrinsic.return_value


def test_set_weights_with_commit_reveal_enabled(subtensor, fake_wallet, mocker):
    """Test set_weights with commit_reveal_enabled is True."""
    # Preps
    fake_netuid = 1
    fake_uids = [1, 5]
    fake_weights = [0.1, 0.9]
    fake_wait_for_inclusion = True
    fake_wait_for_finalization = False

    mocked_commit_reveal_enabled = mocker.patch.object(
        subtensor, "commit_reveal_enabled", return_value=True
    )
    mocked_commit_timelocked_mechanism_weights_extrinsic = mocker.patch.object(
        subtensor_module, "commit_timelocked_weights_extrinsic"
    )
    mocked_commit_timelocked_mechanism_weights_extrinsic.return_value = (
        ExtrinsicResponse(
            True,
            "Weights committed successfully",
        )
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
    mocked_commit_timelocked_mechanism_weights_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=fake_netuid,
        uids=fake_uids,
        weights=fake_weights,
        commit_reveal_version=4,
        version_key=subtensor_module.version_as_int,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=fake_wait_for_inclusion,
        wait_for_finalization=fake_wait_for_finalization,
        block_time=12.0,
        period=8,
        raise_error=False,
        mechid=0,
    )
    assert result == mocked_commit_timelocked_mechanism_weights_extrinsic.return_value


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


def test_set_subnet_identity(mocker, subtensor, fake_wallet):
    """Verify that subtensor method `set_subnet_identity` calls proper function with proper arguments."""
    # Preps
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
        logo_url=fake_subnet_identity.logo_url,
        discord=fake_subnet_identity.discord,
        description=fake_subnet_identity.description,
        additional=fake_subnet_identity.additional,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_finalization=True,
        wait_for_inclusion=True,
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


def test_get_owned_hotkeys_happy_path(subtensor, mocker):
    """Tests that the output of get_owned_hotkeys."""
    # Prep
    fake_coldkey = "fake_hotkey"
    fake_hotkey = "fake_hotkey"
    fake_hotkeys = [
        [
            fake_hotkey,
        ]
    ]
    mocked_subtensor = mocker.Mock(return_value=fake_hotkeys)
    mocker.patch.object(subtensor.substrate, "query", new=mocked_subtensor)

    mocked_decode_account_id = mocker.Mock()
    mocker.patch.object(
        subtensor_module, "decode_account_id", new=mocked_decode_account_id
    )

    # Call
    result = subtensor.get_owned_hotkeys(fake_coldkey)

    # Asserts
    mocked_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="OwnedHotkeys",
        params=[fake_coldkey],
        block_hash=None,
    )
    assert result == [mocked_decode_account_id.return_value]
    mocked_decode_account_id.assert_called_once_with(fake_hotkey)


def test_get_owned_hotkeys_return_empty(subtensor, mocker):
    """Tests that the output of get_owned_hotkeys is empty."""
    # Prep
    fake_coldkey = "fake_hotkey"
    mocked_subtensor = mocker.Mock(return_value=[])
    mocker.patch.object(subtensor.substrate, "query", new=mocked_subtensor)

    # Call
    result = subtensor.get_owned_hotkeys(fake_coldkey)

    # Asserts
    mocked_subtensor.assert_called_once_with(
        module="SubtensorModule",
        storage_function="OwnedHotkeys",
        params=[fake_coldkey],
        block_hash=None,
    )
    assert result == []


def test_start_call(subtensor, mocker):
    """Test start_call extrinsic calls properly."""
    # preps
    wallet_name = mocker.Mock(spec=Wallet)
    netuid = 123
    mocked_extrinsic = mocker.patch.object(subtensor_module, "start_call_extrinsic")

    # Call
    result = subtensor.start_call(wallet_name, netuid)

    # Asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet_name,
        netuid=netuid,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=True,
        wait_for_finalization=False,
        period=DEFAULT_PERIOD,
        raise_error=False,
    )
    assert result == mocked_extrinsic.return_value


def test_get_metagraph_info_all_fields(subtensor, mocker):
    """Test get_metagraph_info with all fields (default behavior)."""
    # Preps
    netuid = 1
    default_mechid = 0
    mock_value = {"mock": "data"}

    mock_runtime_call = mocker.patch.object(
        subtensor.substrate,
        "runtime_call",
        return_value=mocker.Mock(value=mock_value),
    )
    mock_chain_head = mocker.patch.object(
        subtensor.substrate,
        "get_chain_head",
        return_value="0xfakechainhead",
    )
    mock_from_dict = mocker.patch.object(
        subtensor_module.MetagraphInfo, "from_dict", return_value="parsed_metagraph"
    )
    mocked_runtime_metadata_v15 = {
        "apis": [
            {
                "name": "SubnetInfoRuntimeApi",
                "methods": [
                    {"name": "get_selective_metagraph"},
                    {"name": "get_metagraph"},
                    {"name": "get_selective_mechagraph"},
                ],
            },
        ]
    }
    mocked_runtime = mocker.Mock(spec=Runtime)
    mocked_metadata = mocker.Mock()
    mocked_metadata.value.return_value = mocked_runtime_metadata_v15
    mocked_runtime.metadata_v15 = mocked_metadata
    mocker.patch.object(
        subtensor.substrate,
        "init_runtime",
        return_value=mocked_runtime,
    )

    # Call
    result = subtensor.get_metagraph_info(
        netuid=netuid, selected_indices=[f for f in range(len(SelectiveMetagraphIndex))]
    )

    # Asserts
    assert result == "parsed_metagraph"
    mock_runtime_call.assert_called_once_with(
        api="SubnetInfoRuntimeApi",
        method="get_selective_mechagraph",
        params=[netuid, default_mechid, SelectiveMetagraphIndex.all_indices()],
        block_hash=mock_chain_head.return_value,
    )
    mock_from_dict.assert_called_once_with(mock_value)


def test_get_metagraph_info_specific_fields(subtensor, mocker):
    """Test get_metagraph_info with specific fields."""
    # Preps
    netuid = 1
    default_mechid = 0
    mock_value = {"mock": "data"}
    fields = [SelectiveMetagraphIndex.Name, 5]

    mock_runtime_call = mocker.patch.object(
        subtensor.substrate,
        "runtime_call",
        return_value=mocker.Mock(value=mock_value),
    )
    mock_chain_head = mocker.patch.object(
        subtensor.substrate,
        "get_chain_head",
        return_value="0xfakechainhead",
    )
    mocked_runtime_metadata_v15 = {
        "apis": [
            {
                "name": "SubnetInfoRuntimeApi",
                "methods": [
                    {"name": "get_selective_metagraph"},
                    {"name": "get_metagraph"},
                    {"name": "get_selective_mechagraph"},
                ],
            },
        ]
    }
    mocked_runtime = mocker.Mock(spec=Runtime)
    mocked_metadata = mocker.Mock()
    mocked_metadata.value.return_value = mocked_runtime_metadata_v15
    mocked_runtime.metadata_v15 = mocked_metadata
    mocker.patch.object(
        subtensor.substrate,
        "init_runtime",
        return_value=mocked_runtime,
    )
    mock_from_dict = mocker.patch.object(
        subtensor_module.MetagraphInfo, "from_dict", return_value="parsed_metagraph"
    )

    # Call
    result = subtensor.get_metagraph_info(netuid=netuid, selected_indices=fields)

    # Asserts
    assert result == "parsed_metagraph"
    mock_runtime_call.assert_called_once_with(
        api="SubnetInfoRuntimeApi",
        method="get_selective_mechagraph",
        params=[
            netuid,
            default_mechid,
            [0]
            + [
                f.value if isinstance(f, SelectiveMetagraphIndex) else f for f in fields
            ],
        ],
        block_hash=mock_chain_head.return_value,
    )
    mock_from_dict.assert_called_once_with(mock_value)


def test_get_metagraph_info_subnet_not_exist(subtensor, mocker):
    """Test get_metagraph_info returns None when subnet doesn't exist."""
    netuid = 1
    default_mechid = 0
    mocker.patch.object(
        subtensor.substrate,
        "runtime_call",
        return_value=None,
    )
    mocked_runtime_metadata_v15 = {
        "apis": [
            {
                "name": "SubnetInfoRuntimeApi",
                "methods": [
                    {"name": "get_selective_metagraph"},
                    {"name": "get_metagraph"},
                    {"name": "get_selective_mechagraph"},
                ],
            },
        ]
    }
    mocked_runtime = mocker.Mock(spec=Runtime)
    mocked_metadata = mocker.Mock()
    mocked_metadata.value.return_value = mocked_runtime_metadata_v15
    mocked_runtime.metadata_v15 = mocked_metadata
    mocker.patch.object(
        subtensor.substrate,
        "init_runtime",
        return_value=mocked_runtime,
    )

    mocked_logger = mocker.Mock()
    mocker.patch("bittensor.core.subtensor.logging.error", new=mocked_logger)

    result = subtensor.get_metagraph_info(netuid=netuid)

    assert result is None
    mocked_logger.assert_called_once_with(
        f"Subnet mechanism {netuid}.{default_mechid} does not exist."
    )


@pytest.mark.parametrize(
    "block,selected_indices,expected",
    [
        (5_500_000, [1, 2], "get_selective_metagraph"),
        (5_500_000, None, "get_metagraph"),
        (6_500_000, [1, 2], "get_selective_metagraph"),
        (6_500_000, None, "get_metagraph"),
        (6_800_000, [1, 2], "get_selective_mechagraph"),
        (6_800_000, None, "get_selective_mechagraph"),
    ],
)
def test_get_metagraph_info_older_runtime_version(
    subtensor, mocker, block, selected_indices, expected
):
    """Test get_metagraph_info with older runtime version."""
    netuid = 0
    mock_chain_head = mocker.patch.object(
        subtensor,
        "determine_block_hash",
        return_value=str(block),
    )
    mocked_runtime_call = mocker.patch.object(
        subtensor.substrate,
        "runtime_call",
    )
    mocked_runtime_metadata_v15 = {
        "apis": [
            {
                "name": "SubnetInfoRuntimeApi",
                "methods": [
                    {"name": "get_selective_metagraph"},
                    {"name": "get_metagraph"},
                ],
            },
        ]
    }
    if block == 6_800_000:
        # only the newer block should have 'mechagraph' runtime
        mocked_runtime_metadata_v15["apis"][0]["methods"].append(
            {"name": "get_selective_mechagraph"}
        )
    mocked_runtime = mocker.Mock(spec=Runtime)
    mocked_metadata = mocker.Mock()
    mocked_metadata.value.return_value = mocked_runtime_metadata_v15
    mocked_runtime.metadata_v15 = mocked_metadata
    mocker.patch.object(
        subtensor.substrate,
        "init_runtime",
        return_value=mocked_runtime,
    )
    mocker.patch.object(
        subtensor_module.MetagraphInfo, "from_dict", return_value="parsed_metagraph"
    )
    subtensor.get_metagraph_info(netuid=netuid, selected_indices=selected_indices)
    mocked_runtime_call.assert_called_once_with(
        api="SubnetInfoRuntimeApi",
        method=expected,
        params=ANY,
        block_hash=mock_chain_head.return_value,
    )


def test_blocks_since_last_step_with_value(subtensor, mocker):
    """Test blocks_since_last_step returns correct value."""
    # preps
    netuid = 1
    block = 123
    mocked_query_subtensor = mocker.MagicMock()
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = subtensor.blocks_since_last_step(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_called_once_with(
        name="BlocksSinceLastStep",
        block=block,
        params=[netuid],
    )

    assert result == mocked_query_subtensor.return_value.value


def test_blocks_since_last_step_is_none(subtensor, mocker):
    """Test blocks_since_last_step returns None correctly."""
    # preps
    netuid = 1
    block = 123
    mocked_query_subtensor = mocker.MagicMock(return_value=None)
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = subtensor.blocks_since_last_step(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_called_once_with(
        name="BlocksSinceLastStep",
        block=block,
        params=[netuid],
    )

    assert result is None


def test_get_subnet_owner_hotkey_has_return(subtensor, mocker):
    """Test get_subnet_owner_hotkey returns correct value."""
    # preps
    netuid = 14
    block = 123
    expected_owner_hotkey = "owner_hotkey"
    mocked_query_subtensor = mocker.MagicMock(return_value=expected_owner_hotkey)
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = subtensor.get_subnet_owner_hotkey(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_called_once_with(
        name="SubnetOwnerHotkey",
        block=block,
        params=[netuid],
    )

    assert result == expected_owner_hotkey


def test_get_subnet_owner_hotkey_is_none(subtensor, mocker):
    """Test get_subnet_owner_hotkey returns None correctly."""
    # preps
    netuid = 14
    block = 123
    mocked_query_subtensor = mocker.MagicMock(return_value=None)
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = subtensor.get_subnet_owner_hotkey(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_called_once_with(
        name="SubnetOwnerHotkey",
        block=block,
        params=[netuid],
    )

    assert result is None


def test_get_subnet_validator_permits_has_values(subtensor, mocker):
    """Test get_subnet_validator_permits returns correct value."""
    # preps
    netuid = 14
    block = 123
    expected_validator_permits = [False, True, False]
    mocked_query_subtensor = mocker.MagicMock(return_value=expected_validator_permits)
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = subtensor.get_subnet_validator_permits(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_called_once_with(
        name="ValidatorPermit",
        block=block,
        params=[netuid],
    )

    assert result == expected_validator_permits


def test_get_subnet_validator_permits_is_none(subtensor, mocker):
    """Test get_subnet_validator_permits returns correct value."""
    # preps
    netuid = 14
    block = 123

    mocked_query_subtensor = mocker.MagicMock(return_value=None)
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = subtensor.get_subnet_validator_permits(netuid=netuid, block=block)

    # asserts
    mocked_query_subtensor.assert_called_once_with(
        name="ValidatorPermit",
        block=block,
        params=[netuid],
    )

    assert result is None


@pytest.mark.parametrize(
    "query_return, expected",
    [
        [111, True],
        [0, False],
    ],
)
def test_is_subnet_active(subtensor, mocker, query_return, expected):
    # preps
    netuid = mocker.Mock()
    block = mocker.Mock()
    mocked_query_subtensor = mocker.MagicMock(
        return_value=mocker.Mock(value=query_return)
    )
    subtensor.query_subtensor = mocked_query_subtensor

    # call
    result = subtensor.is_subnet_active(netuid=netuid, block=block)

    # Asserts
    mocked_query_subtensor.assert_called_once_with(
        name="FirstEmissionBlockNumber",
        block=block,
        params=[netuid],
    )

    assert result == expected


# `geg_l_subnet_info` tests
def test_get_subnet_info_success(mocker, subtensor):
    """Test get_subnet_info returns correct data when subnet information is found."""
    # Prep
    netuid = mocker.Mock()
    block = mocker.Mock()

    mocker.patch.object(subtensor, "query_runtime_api")
    mocker.patch.object(
        subtensor_module.SubnetInfo,
        "from_dict",
    )

    # Call
    result = subtensor.get_subnet_info(netuid=netuid, block=block)

    # Asserts
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_info_v2",
        params=[netuid],
        block=block,
    )
    subtensor_module.SubnetInfo.from_dict.assert_called_once_with(
        subtensor.query_runtime_api.return_value,
    )
    assert result == subtensor_module.SubnetInfo.from_dict.return_value


def test_get_subnet_info_no_data(mocker, subtensor):
    """Test get_subnet_info returns None."""
    # Prep
    netuid = mocker.Mock()
    block = mocker.Mock()
    mocker.patch.object(subtensor_module.SubnetInfo, "from_dict")
    mocker.patch.object(subtensor, "query_runtime_api", return_value=None)

    # Call
    result = subtensor.get_subnet_info(netuid=netuid, block=block)

    # Asserts
    subtensor.query_runtime_api.assert_called_once_with(
        runtime_api="SubnetInfoRuntimeApi",
        method="get_subnet_info_v2",
        params=[netuid],
        block=block,
    )
    subtensor_module.SubnetInfo.from_dict.assert_not_called()
    assert result is None


def test_get_next_epoch_start_block(mocker, subtensor):
    """Check that get_next_epoch_start_block returns the correct value."""
    # Prep
    netuid = 14
    block = 20

    mocked_tempo = mocker.patch.object(subtensor, "tempo", return_value=100)
    mocked_blocks_until_next_epoch = mocker.patch.object(
        subtensor,
        "blocks_until_next_epoch",
    )

    # Call
    result = subtensor.get_next_epoch_start_block(netuid=netuid, block=block)

    # Asserts
    mocked_tempo.assert_called_once_with(
        netuid=netuid,
        block=block,
    )
    assert result == mocked_blocks_until_next_epoch.return_value.__radd__().__add__()


def test_get_parents_success(subtensor, mocker):
    """Tests get_parents when parents are successfully retrieved and formatted."""
    # Preps
    fake_hotkey = "valid_hotkey"
    fake_netuid = 1
    fake_parents = mocker.Mock(
        value=[
            (1000, ["parent_key_1"]),
            (2000, ["parent_key_2"]),
        ]
    )

    mocked_query = mocker.MagicMock(return_value=fake_parents)
    subtensor.substrate.query = mocked_query

    mocked_decode_account_id = mocker.Mock(
        side_effect=["decoded_parent_key_1", "decoded_parent_key_2"]
    )
    mocker.patch.object(subtensor_module, "decode_account_id", mocked_decode_account_id)

    expected_formatted_parents = [
        (u64_normalized_float(1000), "decoded_parent_key_1"),
        (u64_normalized_float(2000), "decoded_parent_key_2"),
    ]

    # Call
    result = subtensor.get_parents(hotkey_ss58=fake_hotkey, netuid=fake_netuid)

    # Asserts
    mocked_query.assert_called_once_with(
        block_hash=None,
        module="SubtensorModule",
        storage_function="ParentKeys",
        params=[fake_hotkey, fake_netuid],
    )
    mocked_decode_account_id.assert_has_calls(
        [mocker.call("parent_key_1"), mocker.call("parent_key_2")]
    )
    assert result == expected_formatted_parents


def test_get_parents_no_parents(subtensor, mocker):
    """Tests get_parents when there are no parents to retrieve."""
    # Preps
    fake_hotkey = "valid_hotkey"
    fake_netuid = 1
    fake_parents = []

    mocked_query = mocker.MagicMock(return_value=fake_parents)
    subtensor.substrate.query = mocked_query

    # Call
    result = subtensor.get_parents(hotkey_ss58=fake_hotkey, netuid=fake_netuid)

    # Asserts
    mocked_query.assert_called_once_with(
        block_hash=None,
        module="SubtensorModule",
        storage_function="ParentKeys",
        params=[fake_hotkey, fake_netuid],
    )
    assert result == []


def test_set_children(subtensor, fake_wallet, mocker):
    """Tests set_children extrinsic calls properly."""
    # Preps
    fake_netuid = mocker.Mock()
    mocked_set_children_extrinsic = mocker.Mock()
    mocker.patch.object(
        subtensor_module, "set_children_extrinsic", mocked_set_children_extrinsic
    )
    fake_children = [
        (
            1.0,
            "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        ),
    ]

    # Call
    result = subtensor.set_children(
        wallet=fake_wallet,
        netuid=fake_netuid,
        hotkey_ss58=fake_wallet.hotkey.ss58_address,
        children=fake_children,
    )

    # Asserts
    mocked_set_children_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=fake_wallet.hotkey.ss58_address,
        netuid=fake_netuid,
        children=fake_children,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_finalization=True,
        wait_for_inclusion=True,
        raise_error=False,
        period=DEFAULT_PERIOD,
    )
    assert result == mocked_set_children_extrinsic.return_value


def test_unstake_all(subtensor, fake_wallet, mocker):
    """Verifies unstake_all calls properly."""
    # Preps
    fake_unstake_all_extrinsic = mocker.Mock()
    mocker.patch.object(
        subtensor_module, "unstake_all_extrinsic", fake_unstake_all_extrinsic
    )
    # Call
    result = subtensor.unstake_all(
        wallet=fake_wallet,
        hotkey_ss58=fake_wallet.hotkey.ss58_address,
        netuid=1,
    )
    # Asserts
    fake_unstake_all_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        hotkey_ss58=fake_wallet.hotkey.ss58_address,
        netuid=1,
        rate_tolerance=0.005,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=DEFAULT_PERIOD,
        raise_error=False,
    )
    assert result == fake_unstake_all_extrinsic.return_value


def test_get_liquidity_list_subnet_does_not_exits(subtensor, mocker):
    """Test get_liquidity_list returns None when subnet doesn't exist."""
    # Preps
    mocker.patch.object(subtensor, "subnet_exists", return_value=False)

    # Call
    result = subtensor.get_liquidity_list(wallet=mocker.Mock(), netuid=1)

    # Asserts
    subtensor.subnet_exists.assert_called_once_with(netuid=1)
    assert result is None


def test_get_liquidity_list_subnet_is_not_active(subtensor, mocker):
    """Test get_liquidity_list returns None when subnet is not active."""
    # Preps
    mocker.patch.object(subtensor, "subnet_exists", return_value=True)
    mocker.patch.object(subtensor, "is_subnet_active", return_value=False)

    # Call
    result = subtensor.get_liquidity_list(wallet=mocker.Mock(), netuid=1)

    # Asserts
    subtensor.subnet_exists.assert_called_once_with(netuid=1)
    subtensor.is_subnet_active.assert_called_once_with(netuid=1)
    assert result is None


def test_get_liquidity_list_happy_path(subtensor, fake_wallet, mocker):
    """Tests `get_liquidity_list` returns the correct value."""
    netuid = 2

    # Mock network state
    mocker.patch.object(subtensor, "subnet_exists", return_value=True)
    mocker.patch.object(subtensor, "is_subnet_active", return_value=True)
    mocker.patch.object(subtensor, "determine_block_hash", return_value="0x1234")

    # Mock price and fee calculation
    mocker.patch.object(subtensor_module, "price_to_tick", return_value=100)
    mocker.patch.object(
        subtensor_module,
        "calculate_fees",
        return_value=(Balance.from_tao(0.0), Balance.from_tao(0.0, netuid)),
    )

    # Fake positions to return from query_map
    fake_positions = [
        [
            (2,),
            mocker.Mock(
                value={
                    "id": (2,),
                    "netuid": 2,
                    "tick_low": (206189,),
                    "tick_high": (208196,),
                    "liquidity": 1000000000000,
                    "fees_tao": {"bits": 0},
                    "fees_alpha": {"bits": 0},
                }
            ),
        ],
    ]
    fake_result = mocker.MagicMock(records=fake_positions, autospec=list)
    fake_result.__iter__.return_value = iter(fake_positions)

    mocked_query_map = mocker.Mock(return_value=fake_result)
    mocker.patch.object(subtensor, "query_map", new=mocked_query_map)

    # Mock storage key creation
    mocker.patch.object(
        subtensor.substrate,
        "create_storage_key",
        side_effect=lambda pallet,
        storage_function,
        params,
        block_hash=None: f"{pallet}:{storage_function}:{params}",
    )

    # Mock query_multi for fee + sqrt_price + tick data
    mock_query_multi = mocker.MagicMock(
        side_effect=[
            [
                ("key1", {"bits": 0}),  # fee_global_tao
                ("key2", {"bits": 0}),  # fee_global_alpha
                ("key3", {"bits": 1072693248}),
            ],
            [
                (
                    "tick_low",
                    {"fees_out_tao": {"bits": 0}, "fees_out_alpha": {"bits": 0}},
                ),
                (
                    "tick_high",
                    {"fees_out_tao": {"bits": 0}, "fees_out_alpha": {"bits": 0}},
                ),
            ],
        ]
    )
    mocker.patch.object(subtensor.substrate, "query_multi", new=mock_query_multi)

    # Call
    result = subtensor.get_liquidity_list(wallet=fake_wallet, netuid=netuid)

    # Asserts
    assert subtensor.determine_block_hash.call_count == 1
    assert subtensor_module.price_to_tick.call_count == 1
    assert subtensor_module.calculate_fees.call_count == len(fake_positions)
    mocked_query_map.assert_called_once_with(
        module="Swap",
        name="Positions",
        params=[netuid, fake_wallet.coldkeypub.ss58_address],
        block=None,
    )
    assert mock_query_multi.call_count == 2  # one for fees, one for ticks
    assert len(result) == len(fake_positions)
    assert all(isinstance(p, subtensor_module.LiquidityPosition) for p in result)


def test_add_liquidity(subtensor, fake_wallet, mocker):
    """Test add_liquidity extrinsic calls properly."""
    # preps
    netuid = 123
    mocked_extrinsic = mocker.patch.object(subtensor_module, "add_liquidity_extrinsic")

    # Call
    result = subtensor.add_liquidity(
        wallet=fake_wallet,
        netuid=netuid,
        liquidity=Balance.from_tao(150),
        price_low=Balance.from_tao(180).rao,
        price_high=Balance.from_tao(130).rao,
    )

    # Asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        liquidity=Balance.from_tao(150),
        price_low=Balance.from_tao(180).rao,
        price_high=Balance.from_tao(130).rao,
        hotkey_ss58=None,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=DEFAULT_PERIOD,
        raise_error=False,
    )
    assert result == mocked_extrinsic.return_value


def test_modify_liquidity(subtensor, fake_wallet, mocker):
    """Test modify_liquidity extrinsic calls properly."""
    # preps
    netuid = 123
    mocked_extrinsic = mocker.patch.object(
        subtensor_module, "modify_liquidity_extrinsic"
    )
    position_id = 2

    # Call
    result = subtensor.modify_liquidity(
        wallet=fake_wallet,
        netuid=netuid,
        position_id=position_id,
        liquidity_delta=Balance.from_tao(150),
    )

    # Asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        position_id=position_id,
        liquidity_delta=Balance.from_tao(150),
        hotkey_ss58=None,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=DEFAULT_PERIOD,
        raise_error=False,
    )
    assert result == mocked_extrinsic.return_value


def test_remove_liquidity(subtensor, fake_wallet, mocker):
    """Test remove_liquidity extrinsic calls properly."""
    # preps
    netuid = 123
    mocked_extrinsic = mocker.patch.object(
        subtensor_module, "remove_liquidity_extrinsic"
    )
    position_id = 2

    # Call
    result = subtensor.remove_liquidity(
        wallet=fake_wallet,
        netuid=netuid,
        position_id=position_id,
    )

    # Asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        position_id=position_id,
        hotkey_ss58=None,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=DEFAULT_PERIOD,
        raise_error=False,
    )
    assert result == mocked_extrinsic.return_value


def test_toggle_user_liquidity(subtensor, fake_wallet, mocker):
    """Test toggle_user_liquidity extrinsic calls properly."""
    # preps
    netuid = 123
    mocked_extrinsic = mocker.patch.object(
        subtensor_module, "toggle_user_liquidity_extrinsic"
    )
    enable = mocker.Mock()

    # Call
    result = subtensor.toggle_user_liquidity(
        wallet=fake_wallet,
        netuid=netuid,
        enable=enable,
    )

    # Asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=fake_wallet,
        netuid=netuid,
        enable=enable,
        mev_protection=DEFAULT_MEV_PROTECTION,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=DEFAULT_PERIOD,
        raise_error=False,
    )
    assert result == mocked_extrinsic.return_value


def test_get_subnet_price(subtensor, mocker):
    """Test get_subnet_price returns the correct value."""
    # preps
    netuid = 123
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    fake_price = 29258617
    expected_price = Balance.from_tao(0.029258617)
    mocked_query = mocker.patch.object(
        subtensor.substrate, "runtime_call", return_value=mocker.Mock(value=fake_price)
    )

    # Call
    result = subtensor.get_subnet_price(
        netuid=netuid,
    )

    # Asserts
    mocked_determine_block_hash.assert_called_once_with(block=None)
    mocked_query.assert_called_once_with(
        api="SwapRuntimeApi",
        method="current_alpha_price",
        params=[netuid],
        block_hash=mocked_determine_block_hash.return_value,
    )

    assert result == expected_price


def test_get_subnet_prices(subtensor, mocker):
    """Test get_subnet_prices returns the correct value."""
    # preps
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    fake_prices = [
        [0, {"bits": 0}],
        [1, {"bits": 3155343338053956962}],
    ]
    expected_prices = {0: Balance.from_tao(1), 1: Balance.from_tao(0.029258617)}
    mocked_query_map = mocker.patch.object(
        subtensor.substrate, "query_map", return_value=fake_prices
    )

    # Call
    result = subtensor.get_subnet_prices()

    # Asserts
    mocked_determine_block_hash.assert_called_once_with(block=None)
    mocked_query_map.assert_called_once_with(
        module="Swap",
        storage_function="AlphaSqrtPrice",
        block_hash=mocked_determine_block_hash.return_value,
        page_size=129,  # total number of subnets
    )
    assert result == expected_prices


def test_all_subnets(subtensor, mocker):
    """Verify that `all_subnets` calls proper methods and returns the correct value."""
    # Preps
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_di_list_from_dicts = mocker.patch.object(
        subtensor_module.DynamicInfo, "list_from_dicts"
    )
    mocked_get_subnet_prices = mocker.patch.object(
        subtensor,
        "get_subnet_prices",
        return_value={0: Balance.from_tao(1), 1: Balance.from_tao(0.029258617)},
    )
    mocked_decode = mocker.Mock(return_value=[{"netuid": 0}, {"netuid": 1}])
    mocked_runtime_call = mocker.Mock(decode=mocked_decode)
    mocker.patch.object(
        subtensor.substrate, "runtime_call", return_value=mocked_runtime_call
    )

    # Call
    result = subtensor.all_subnets()

    # Asserts
    mocked_determine_block_hash.assert_called_once_with(block=None)
    subtensor.substrate.runtime_call.assert_called_once_with(
        api="SubnetInfoRuntimeApi",
        method="get_all_dynamic_info",
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_get_subnet_prices.assert_called_once()
    mocked_di_list_from_dicts.assert_called_once_with(
        [
            {"netuid": 0, "price": Balance.from_tao(1)},
            {"netuid": 1, "price": Balance.from_tao(0.029258617)},
        ]
    )
    assert result == mocked_di_list_from_dicts.return_value


def test_subnet(subtensor, mocker):
    """Verify that `subnet` calls proper methods and returns the correct value."""
    # Preps
    netuid = 14
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_di_from_dict = mocker.patch.object(subtensor_module.DynamicInfo, "from_dict")
    mocked_get_subnet_price = mocker.patch.object(
        subtensor, "get_subnet_price", return_value=Balance.from_tao(100.0)
    )
    mocked_decode = mocker.Mock(return_value={"netuid": netuid})
    mocked_runtime_call = mocker.Mock(decode=mocked_decode)
    mocker.patch.object(
        subtensor.substrate, "runtime_call", return_value=mocked_runtime_call
    )

    # Call
    result = subtensor.subnet(netuid=netuid)

    # Asserts
    subtensor.substrate.runtime_call.assert_called_once_with(
        api="SubnetInfoRuntimeApi",
        method="get_dynamic_info",
        params=[netuid],
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_determine_block_hash.assert_called_once_with(block=None)
    mocked_get_subnet_price.assert_called_once_with(netuid=netuid, block=None)
    mocked_di_from_dict.assert_called_once_with(
        {"netuid": netuid, "price": Balance.from_tao(100.0)}
    )
    assert result == mocked_di_from_dict.return_value


def test_get_stake_add_fee(subtensor, mocker):
    """Verify that `get_stake_add_fee` calls proper methods and returns the correct value."""
    # Preps
    netuid = mocker.Mock()
    amount = mocker.Mock(spec=Balance)
    mocked_sim_swap = mocker.patch.object(subtensor, "sim_swap")

    # Call
    result = subtensor.get_stake_add_fee(
        amount=amount,
        netuid=netuid,
    )

    # Asserts
    mocked_sim_swap.assert_called_once_with(
        origin_netuid=0,
        destination_netuid=netuid,
        amount=amount,
        block=None,
    )
    assert result == mocked_sim_swap.return_value.tao_fee


def test_get_unstake_fee(subtensor, mocker):
    """Verify that `get_unstake_fee` calls proper methods and returns the correct value."""
    # Preps
    netuid = mocker.Mock()
    amount = mocker.Mock(spec=Balance)
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_sim_swap = mocker.patch.object(
        subtensor,
        "sim_swap",
        return_value=mocker.MagicMock(alpha_fee=mocker.MagicMock()),
    )

    # Call
    result = subtensor.get_unstake_fee(
        amount=amount,
        netuid=netuid,
    )

    # Asserts
    mocked_sim_swap.assert_called_once_with(
        origin_netuid=netuid,
        destination_netuid=0,
        amount=amount,
        block=None,
    )
    assert result == mocked_sim_swap.return_value.alpha_fee.set_unit.return_value


def test_get_stake_movement_fee(subtensor, mocker):
    """Verify that `get_stake_movement_fee` calls proper methods and returns the correct value."""
    # Preps
    origin_netuid = mocker.Mock()
    destination_netuid = mocker.Mock()
    amount = mocker.Mock(spec=Balance)

    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_sim_swap = mocker.patch.object(
        subtensor,
        "sim_swap",
        return_value=mocker.MagicMock(alpha_fee=mocker.MagicMock()),
    )

    # Call
    result = subtensor.get_stake_movement_fee(
        origin_netuid=origin_netuid,
        destination_netuid=destination_netuid,
        amount=amount,
    )

    # Asserts
    mocked_sim_swap.assert_called_once_with(
        origin_netuid=origin_netuid,
        destination_netuid=destination_netuid,
        amount=amount,
        block=None,
    )
    assert result == mocked_sim_swap.return_value.tao_fee


def test_get_stake_weight(subtensor, mocker):
    """Verify that `get_stake_weight` method calls proper methods and returns the correct value."""
    # Preps
    netuid = mocker.Mock()
    fake_weights = [0, 100, 15000]
    expected_result = [0.0, 0.0015259021896696422, 0.22888532845044632]

    mock_determine_block_hash = mocker.patch.object(
        subtensor,
        "determine_block_hash",
    )
    mocked_query = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=fake_weights,
    )

    # Call
    result = subtensor.get_stake_weight(netuid=netuid)

    # Asserts
    mock_determine_block_hash.assert_called_once()
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="StakeWeight",
        params=[netuid],
        block_hash=mock_determine_block_hash.return_value,
    )
    assert result == expected_result


def test_get_timelocked_weight_commits(subtensor, mocker):
    """Verify that `get_timelocked_weight_commits` method calls proper methods and returns the correct value."""
    # Preps
    netuid = 14

    mock_determine_block_hash = mocker.patch.object(
        subtensor,
        "determine_block_hash",
    )
    mocked_query_map = mocker.patch.object(
        subtensor.substrate,
        "query_map",
    )

    # Call
    result = subtensor.get_timelocked_weight_commits(netuid=netuid)

    # Asserts
    mock_determine_block_hash.assert_called_once_with(block=None)
    mocked_query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="TimelockedWeightCommits",
        params=[netuid],
        block_hash=mock_determine_block_hash.return_value,
    )
    assert result == []


@pytest.mark.parametrize(
    "query_return, expected_result",
    (
        ["value", [10, 90]],
        [None, None],
    ),
)
def test_get_mechanism_emission_split(subtensor, mocker, query_return, expected_result):
    """Verify that get_mechanism_emission_split calls the correct methods."""
    # Preps
    netuid = mocker.Mock()
    query_return = (
        mocker.Mock(value=[6553, 58982]) if query_return == "value" else query_return
    )
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query = mocker.patch.object(
        subtensor.substrate, "query", return_value=query_return
    )

    # Call

    result = subtensor.get_mechanism_emission_split(netuid)

    # Asserts
    mocked_determine_block_hash.assert_called_once()
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="MechanismEmissionSplit",
        params=[netuid],
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result == expected_result


def test_get_mechanism_count(subtensor, mocker):
    """Verify that `get_mechanism_count` method processed the data correctly."""
    # Preps
    netuid = 14

    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_result = mocker.MagicMock()
    mocker.patch.object(subtensor.substrate, "runtime_call", return_value=mocked_result)
    mocked_query = mocker.patch.object(subtensor.substrate, "query")

    # Call
    result = subtensor.get_mechanism_count(netuid=netuid)

    # Asserts
    mocked_determine_block_hash.assert_called_once()
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="MechanismCountCurrent",
        params=[netuid],
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result is mocked_query.return_value.value


def test_is_in_admin_freeze_window_root_net(subtensor, mocker):
    """Verify that root net has no admin freeze window."""
    # Preps
    netuid = 0
    mocked_get_next_epoch_start_block = mocker.patch.object(
        subtensor, "get_next_epoch_start_block"
    )

    # Call
    result = subtensor.is_in_admin_freeze_window(netuid=netuid)

    # Asserts
    mocked_get_next_epoch_start_block.assert_not_called()
    assert result is False


@pytest.mark.parametrize(
    "block, next_esb, expected_result",
    (
        [89, 100, False],
        [90, 100, False],
        [91, 100, True],
    ),
)
def test_is_in_admin_freeze_window(subtensor, mocker, block, next_esb, expected_result):
    """Verify that `is_in_admin_freeze_window` method processed the data correctly."""
    # Preps
    netuid = 14
    mocker.patch.object(subtensor, "get_current_block", return_value=block)
    mocker.patch.object(subtensor, "get_next_epoch_start_block", return_value=next_esb)
    mocker.patch.object(subtensor, "get_admin_freeze_window", return_value=10)

    # Call

    result = subtensor.is_in_admin_freeze_window(netuid=netuid)

    # Asserts
    assert result is expected_result


def test_get_admin_freeze_window(subtensor, mocker):
    """Verify that `get_admin_freeze_window` calls proper methods."""
    # Preps
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query = mocker.patch.object(subtensor.substrate, "query")

    # Call
    result = subtensor.get_admin_freeze_window()

    # Asserts
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="AdminFreezeWindow",
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result == mocked_query.return_value.value


def test_get_auto_stakes(subtensor, mocker):
    """Tests that `get_auto_stakes` calls proper methods and returns the correct value."""
    # Preps
    fake_coldkey = mocker.Mock()
    mock_determine_block_hash = mocker.patch.object(
        subtensor,
        "determine_block_hash",
    )
    fake_hk_1 = mocker.Mock()
    fake_hk_2 = mocker.Mock()

    dest_value_1 = mocker.Mock(value=[fake_hk_1])
    dest_value_2 = mocker.Mock(value=[fake_hk_2])

    mock_result = mocker.MagicMock()
    mock_result.__iter__.return_value = iter([(0, dest_value_1), (1, dest_value_2)])
    mocked_query_map = mocker.patch.object(
        subtensor.substrate, "query_map", return_value=mock_result
    )

    mocked_decode_account_id = mocker.patch.object(
        subtensor_module,
        "decode_account_id",
        side_effect=[fake_hk_1, fake_hk_2],
    )

    # Call
    result = subtensor.get_auto_stakes(coldkey_ss58=fake_coldkey)

    # Asserts
    mock_determine_block_hash.assert_called_once()
    mocked_query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="AutoStakeDestination",
        params=[fake_coldkey],
        block_hash=mock_determine_block_hash.return_value,
    )
    mocked_decode_account_id.assert_has_calls(
        [mocker.call(dest_value_1.value[0]), mocker.call(dest_value_2.value[0])]
    )
    assert result == {0: fake_hk_1, 1: fake_hk_2}


def test_set_auto_stake(subtensor, mocker):
    """Tests that `set_auto_stake` calls proper methods and returns the correct value."""
    # Preps
    wallet = mocker.Mock()
    netuid = mocker.Mock()
    hotkey = mocker.Mock()
    mocked_extrinsic = mocker.patch.object(subtensor_module, "set_auto_stake_extrinsic")

    # Call
    result = subtensor.set_auto_stake(
        wallet=wallet,
        netuid=netuid,
        hotkey_ss58=hotkey,
    )

    # Asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        hotkey_ss58=hotkey,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert result == mocked_extrinsic.return_value


def test_get_block_info(subtensor, mocker):
    """Tests that `get_block_info` calls proper methods and returns the correct value."""
    # Preps
    fake_block = mocker.Mock(spec=int)
    fake_hash = mocker.Mock(spec=str)
    fake_timestamp = mocker.Mock(spec=int)
    fake_decoded = mocker.Mock(
        value_serialized={
            "call": {
                "call_module": "Timestamp",
                "call_args": [{"value": fake_timestamp}],
            }
        }
    )
    fake_substrate_block = {
        "header": {
            "number": fake_block,
            "hash": fake_hash,
        },
        "extrinsics": [
            fake_decoded,
        ],
    }
    mocked_get_block = mocker.patch.object(
        subtensor.substrate, "get_block", return_value=fake_substrate_block
    )
    mocked_BlockInfo = mocker.patch.object(subtensor_module, "BlockInfo")

    # Call
    result = subtensor.get_block_info()

    # Asserts
    mocked_get_block.assert_called_once_with(
        block_hash=None,
        block_number=None,
        ignore_decoding_errors=True,
    )
    mocked_BlockInfo.assert_called_once_with(
        number=fake_block,
        hash=fake_hash,
        timestamp=fake_timestamp,
        header=fake_substrate_block.get("header"),
        extrinsics=fake_substrate_block.get("extrinsics"),
        explorer=f"{settings.TAO_APP_BLOCK_EXPLORER}{fake_block}",
    )
    assert result == mocked_BlockInfo.return_value


def test_contribute_crowdloan(mocker, subtensor):
    """Tests subtensor `contribute_crowdloan` method."""
    # Preps
    wallet = mocker.Mock()
    crowdloan_id = mocker.Mock()
    amount = mocker.Mock(spec=Balance)

    mocked_extrinsic = mocker.patch.object(
        subtensor_module, "contribute_crowdloan_extrinsic"
    )

    # Call
    response = subtensor.contribute_crowdloan(
        wallet=wallet,
        crowdloan_id=crowdloan_id,
        amount=amount,
    )

    # asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        crowdloan_id=crowdloan_id,
        amount=amount,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_extrinsic.return_value


def test_create_crowdloan(mocker, subtensor):
    """Tests subtensor `create_crowdloan` method."""
    # Preps
    wallet = mocker.Mock(spec=Wallet)
    deposit = mocker.Mock(spec=Balance)
    min_contribution = mocker.Mock(spec=Balance)
    cap = mocker.Mock(spec=Balance)
    end = mocker.Mock(spec=int)
    call = mocker.Mock(spec=GenericCall)
    target_address = mocker.Mock(spec=str)

    mocked_extrinsic = mocker.patch.object(
        subtensor_module, "create_crowdloan_extrinsic"
    )

    # Call
    response = subtensor.create_crowdloan(
        wallet=wallet,
        deposit=deposit,
        min_contribution=min_contribution,
        cap=cap,
        end=end,
        call=call,
        target_address=target_address,
    )

    # asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        deposit=deposit,
        min_contribution=min_contribution,
        cap=cap,
        end=end,
        call=call,
        target_address=target_address,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_extrinsic.return_value


@pytest.mark.parametrize(
    "method, extrinsic",
    [
        ("dissolve_crowdloan", "dissolve_crowdloan_extrinsic"),
        ("finalize_crowdloan", "finalize_crowdloan_extrinsic"),
        ("refund_crowdloan", "refund_crowdloan_extrinsic"),
        ("withdraw_crowdloan", "withdraw_crowdloan_extrinsic"),
    ],
)
def test_crowdloan_methods_with_crowdloan_id_parameter(
    mocker, subtensor, method, extrinsic
):
    """Tests subtensor methods with the same list of parameters."""
    # Preps
    wallet = mocker.Mock()
    crowdloan_id = mocker.Mock()

    mocked_extrinsic = mocker.patch.object(subtensor_module, extrinsic)

    # Call
    response = getattr(subtensor, method)(
        wallet=wallet,
        crowdloan_id=crowdloan_id,
    )

    # asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        crowdloan_id=crowdloan_id,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_extrinsic.return_value


def test_update_cap_crowdloan(mocker, subtensor):
    """Tests subtensor `update_cap_crowdloan` method."""
    # Preps
    wallet = mocker.Mock()
    crowdloan_id = mocker.Mock()
    new_cap = mocker.Mock(spec=Balance)

    mocked_extrinsic = mocker.patch.object(
        subtensor_module, "update_cap_crowdloan_extrinsic"
    )

    # Call
    response = subtensor.update_cap_crowdloan(
        wallet=wallet,
        crowdloan_id=crowdloan_id,
        new_cap=new_cap,
    )

    # asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        crowdloan_id=crowdloan_id,
        new_cap=new_cap,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_extrinsic.return_value


def test_update_end_crowdloan(mocker, subtensor):
    """Tests subtensor `update_end_crowdloan` method."""
    # Preps
    wallet = mocker.Mock()
    crowdloan_id = mocker.Mock()
    new_end = mocker.Mock(spec=int)

    mocked_extrinsic = mocker.patch.object(
        subtensor_module, "update_end_crowdloan_extrinsic"
    )

    # Call
    response = subtensor.update_end_crowdloan(
        wallet=wallet,
        crowdloan_id=crowdloan_id,
        new_end=new_end,
    )

    # asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        crowdloan_id=crowdloan_id,
        new_end=new_end,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_extrinsic.return_value


def test_update_min_contribution_crowdloan(mocker, subtensor):
    """Tests subtensor `update_min_contribution_crowdloan` method."""
    # Preps
    wallet = mocker.Mock()
    crowdloan_id = mocker.Mock()
    new_min_contribution = mocker.Mock(spec=Balance)

    mocked_extrinsic = mocker.patch.object(
        subtensor_module, "update_min_contribution_crowdloan_extrinsic"
    )

    # Call
    response = subtensor.update_min_contribution_crowdloan(
        wallet=wallet,
        crowdloan_id=crowdloan_id,
        new_min_contribution=new_min_contribution,
    )

    # asserts
    mocked_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        crowdloan_id=crowdloan_id,
        new_min_contribution=new_min_contribution,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_extrinsic.return_value


def test_get_crowdloan_constants(mocker, subtensor):
    """Test subtensor `get_crowdloan_constants` method."""
    # Preps
    fake_constant_name = mocker.Mock(spec=str)
    mocked_crowdloan_constants = mocker.patch.object(
        subtensor_module.CrowdloanConstants,
        "constants_names",
        return_value=[fake_constant_name],
    )
    mocked_query_constant = mocker.patch.object(subtensor, "query_constant")
    mocked_from_dict = mocker.patch.object(
        subtensor_module.CrowdloanConstants, "from_dict"
    )

    # Call
    result = subtensor.get_crowdloan_constants()

    # Asserts
    mocked_crowdloan_constants.assert_called_once()
    mocked_query_constant.assert_called_once_with(
        module_name="Crowdloan",
        constant_name=fake_constant_name,
        block=None,
    )
    mocked_from_dict.assert_called_once_with(
        {fake_constant_name: mocked_query_constant.return_value.value}
    )
    assert result == mocked_from_dict.return_value


def test_get_crowdloan_contributions(mocker, subtensor):
    """Tests subtensor `get_crowdloan_contributions` method."""
    # Preps
    fake_hk_array = mocker.Mock(spec=list)
    fake_contribution = mocker.Mock(value=mocker.Mock(spec=Balance))

    fake_crowdloan_id = mocker.Mock(spec=int)
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query_map = mocker.patch.object(subtensor.substrate, "query_map")
    mocked_query_map.return_value.records = [(fake_hk_array, fake_contribution)]
    mocked_decode_account_id = mocker.patch.object(
        subtensor_module, "decode_account_id"
    )
    mocked_from_rao = mocker.patch.object(subtensor_module.Balance, "from_rao")

    # Call
    result = subtensor.get_crowdloan_contributions(fake_crowdloan_id)

    # Asserts
    mocked_determine_block_hash.assert_called_once()
    assert result == {
        mocked_decode_account_id.return_value: mocked_from_rao.return_value
    }


@pytest.mark.parametrize(
    "query_return, expected_result", [(None, None), ("Some", "decode_crowdloan_entry")]
)
def test_get_crowdloan_by_id(mocker, subtensor, query_return, expected_result):
    """Tests subtensor `get_crowdloan_by_id` method."""
    # Preps
    fake_crowdloan_id = mocker.Mock(spec=int)
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")

    mocked_query_return = (
        None if query_return is None else mocker.Mock(value=query_return)
    )
    mocked_query = mocker.patch.object(
        subtensor.substrate, "query", return_value=mocked_query_return
    )

    mocked_decode_crowdloan_entry = mocker.patch.object(
        subtensor, "_decode_crowdloan_entry"
    )

    # Call
    result = subtensor.get_crowdloan_by_id(fake_crowdloan_id)

    # Asserts
    mocked_determine_block_hash.assert_called_once()
    mocked_query.assert_called_once_with(
        module="Crowdloan",
        storage_function="Crowdloans",
        params=[fake_crowdloan_id],
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert (
        result == expected_result
        if query_return is None
        else mocked_decode_crowdloan_entry.return_value
    )


def test_get_crowdloan_next_id(mocker, subtensor):
    """Tests subtensor `get_crowdloan_next_id` method."""
    # Preps
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query = mocker.patch.object(
        subtensor.substrate, "query", return_value=mocker.Mock(value=3)
    )

    # Call
    result = subtensor.get_crowdloan_next_id()

    # Asserts
    mocked_determine_block_hash.assert_called_once()
    mocked_query.assert_called_once_with(
        module="Crowdloan",
        storage_function="NextCrowdloanId",
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result == int(mocked_query.return_value.value)


def test_get_crowdloans(mocker, subtensor):
    """Tests subtensor `get_crowdloans` method."""
    # Preps
    fake_id = mocker.Mock(spec=int)
    fake_crowdloan = mocker.Mock(value=mocker.Mock(spec=dict))

    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query_map = mocker.patch.object(
        subtensor.substrate,
        "query_map",
        return_value=mocker.Mock(records=[(fake_id, fake_crowdloan)]),
    )
    mocked_decode_crowdloan_entry = mocker.patch.object(
        subtensor, "_decode_crowdloan_entry"
    )

    # Call
    result = subtensor.get_crowdloans()

    # Asserts
    mocked_determine_block_hash.assert_called_once()
    mocked_query_map.assert_called_once_with(
        module="Crowdloan",
        storage_function="Crowdloans",
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_decode_crowdloan_entry.assert_called_once_with(
        crowdloan_id=fake_id,
        data=fake_crowdloan.value,
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result == [mocked_decode_crowdloan_entry.return_value]


@pytest.mark.parametrize(
    "method, add_salt",
    [
        ("commit_weights", True),
        ("reveal_weights", True),
        ("set_weights", False),
    ],
    ids=["commit_weights", "reveal_weights", "set_weights"],
)
def test_commit_weights_with_zero_max_attempts(
    mocker, subtensor, caplog, method, add_salt
):
    """Verify that commit_weights returns response with proper error message."""
    # Preps
    wallet = mocker.Mock(spec=Wallet)
    netuid = mocker.Mock(spec=int)
    salt = mocker.Mock(spec=list)
    uids = mocker.Mock(spec=list)
    weights = mocker.Mock(spec=list)
    max_attempts = 0
    expected_message = (
        f"`max_attempts` parameter must be greater than 0, not {max_attempts}."
    )

    params = {
        "wallet": wallet,
        "netuid": netuid,
        "uids": uids,
        "weights": weights,
        "max_attempts": max_attempts,
    }
    if add_salt:
        params["salt"] = salt

    # Call
    # with caplog.at_level(logging.WARNING):
    response = getattr(subtensor, method)(**params)

    # Asserts
    assert response.success is False
    assert response.message == expected_message
    assert isinstance(response.error, ValueError)
    assert expected_message in str(response.error)
    assert expected_message in caplog.text


@pytest.mark.parametrize(
    "fake_result, expected_result",
    [
        ({"Swap": ()}, "Swap"),
        ({"Keep": ()}, "Keep"),
        (
            {
                "KeepSubnets": {
                    "subnets": (
                        (
                            2,
                            3,
                        ),
                    )
                }
            },
            {"KeepSubnets": {"subnets": [2, 3]}},
        ),
        (
            {"KeepSubnets": {"subnets": ((2,),)}},
            {
                "KeepSubnets": {
                    "subnets": [
                        2,
                    ]
                }
            },
        ),
    ],
)
def test_get_root_claim_type(mocker, subtensor, fake_result, expected_result):
    """Tests that `get_root_claim_type` calls proper methods and returns the correct value."""
    # Preps
    fake_coldkey_ss58 = mocker.Mock(spec=str)
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_map = mocker.patch.object(
        subtensor.substrate, "query", return_value=fake_result
    )

    # call
    result = subtensor.get_root_claim_type(fake_coldkey_ss58)

    # asserts
    mocked_determine_block_hash.assert_called_once()
    mocked_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="RootClaimType",
        params=[fake_coldkey_ss58],
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result == expected_result


def test_get_root_claimable_rate(mocker, subtensor):
    """Tests `get_root_claimable_rate` method."""
    # Preps
    hotkey_ss58 = mocker.Mock(spec=str)
    netuid = mocker.Mock(spec=int)

    mocked_get_root_claimable_all_rates = mocker.patch.object(
        subtensor, "get_root_claimable_all_rates"
    )

    # Call
    result = subtensor.get_root_claimable_rate(
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
    )

    # Asserts
    mocked_get_root_claimable_all_rates.assert_called_once_with(
        hotkey_ss58=hotkey_ss58,
        block=None,
    )
    mocked_get_root_claimable_all_rates.return_value.get.assert_called_once_with(
        netuid, 0.0
    )
    assert result == mocked_get_root_claimable_all_rates.return_value.get.return_value


def test_get_root_claimable_all_rates(mocker, subtensor):
    """Tests `get_root_claimable_all_rates` method."""
    # Preps
    hotkey_ss58 = mocker.Mock(spec=str)
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    fake_value = [((14, {"bits": 6520190}),)]
    fake_result = mocker.MagicMock(value=fake_value)
    fake_result.__iter__ = fake_value
    mocked_query = mocker.patch.object(
        subtensor.substrate, "query", return_value=fake_result
    )
    mocked_fixed_to_float = mocker.patch.object(subtensor_module, "fixed_to_float")

    # Call
    result = subtensor.get_root_claimable_all_rates(
        hotkey_ss58=hotkey_ss58,
    )

    # Asserts
    mocked_determine_block_hash.assert_called_once()
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="RootClaimable",
        params=[hotkey_ss58],
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_fixed_to_float.assert_called_once_with({"bits": 6520190}, frac_bits=32)
    assert result == {14: mocked_fixed_to_float.return_value}


def test_get_root_claimable_stake(mocker, subtensor):
    """Tests `get_root_claimable_stake` method."""
    # Preps
    coldkey_ss58 = mocker.Mock(spec=str)
    hotkey_ss58 = mocker.Mock(spec=str)
    netuid = 14

    mocked_get_stake = mocker.patch.object(
        subtensor, "get_stake", return_value=Balance.from_tao(1)
    )
    mocked_get_root_claimable_rate = mocker.patch.object(
        subtensor, "get_root_claimable_rate", return_value=0.5
    )
    mocked_get_root_claimed = mocker.patch.object(
        subtensor, "get_root_claimed", spec=int
    )

    # Call
    result = subtensor.get_root_claimable_stake(
        coldkey_ss58=coldkey_ss58,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
    )

    # Asserts
    mocked_get_stake.assert_called_once_with(
        coldkey_ss58=coldkey_ss58,
        hotkey_ss58=hotkey_ss58,
        netuid=0,
        block=None,
    )
    mocked_get_root_claimable_rate.assert_called_once_with(
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        block=None,
    )
    mocked_get_root_claimed.assert_called_once_with(
        coldkey_ss58=coldkey_ss58,
        hotkey_ss58=hotkey_ss58,
        block=None,
        netuid=netuid,
    )
    assert result == Balance.from_rao(1).set_unit(netuid)


def test_get_root_claimed(mocker, subtensor):
    """Tests `get_root_claimed` method."""
    # Preps
    coldkey_ss58 = mocker.Mock(spec=str)
    hotkey_ss58 = mocker.Mock(spec=str)
    netuid = 14
    fake_value = mocker.Mock(value=1)
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query = mocker.patch.object(
        subtensor.substrate, "query", return_value=fake_value
    )

    # Call
    result = subtensor.get_root_claimed(
        coldkey_ss58=coldkey_ss58,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
    )

    # Asserts
    mocked_determine_block_hash.assert_called_once()
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="RootClaimed",
        params=[netuid, hotkey_ss58, coldkey_ss58],
        block_hash=mocked_determine_block_hash.return_value,
    )
    assert result == Balance.from_rao(1).set_unit(netuid)


def test_claim_root(mocker, subtensor):
    """Tests `claim_root` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    netuids = mocker.Mock(spec=list)
    mocked_claim_root_extrinsic = mocker.patch.object(
        subtensor_module, "claim_root_extrinsic"
    )

    # call
    response = subtensor.claim_root(
        wallet=wallet,
        netuids=netuids,
    )

    # asserts
    mocked_claim_root_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        netuids=netuids,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_claim_root_extrinsic.return_value


def test_set_root_claim_type(mocker, subtensor):
    """Tests that `set_root_claim_type` calls proper methods and returns the correct value."""
    # Preps
    faked_wallet = mocker.Mock(spec=Wallet)
    fake_new_root_claim_type = mocker.Mock(spec=str)
    mocked_set_root_claim_type_extrinsic = mocker.patch.object(
        subtensor_module, "set_root_claim_type_extrinsic"
    )

    # call
    response = subtensor.set_root_claim_type(
        wallet=faked_wallet, new_root_claim_type=fake_new_root_claim_type
    )

    # asserts
    mocked_set_root_claim_type_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=faked_wallet,
        new_root_claim_type=fake_new_root_claim_type,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_set_root_claim_type_extrinsic.return_value


def test_get_all_ema_tao_inflow(subtensor, mocker):
    """Test get_all_ema_tao_inflow returns correct values."""
    # Preps
    fake_block = 123
    fake_netuid = 1
    fake_block_updated = 100
    fake_tao_bits = {"bits": 6520190}

    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    fake_query_result = [(fake_netuid, (fake_block_updated, fake_tao_bits))]
    mock_query_map = mocker.patch.object(
        subtensor.substrate, "query_map", return_value=fake_query_result
    )
    mocked_fixed_to_float = mocker.patch.object(
        subtensor_module, "fixed_to_float", return_value=1000000
    )

    # Call
    result = subtensor.get_all_ema_tao_inflow(block=fake_block)

    # Asserts
    mocked_determine_block_hash.assert_called_once_with(fake_block)
    mock_query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="SubnetEmaTaoFlow",
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_fixed_to_float.assert_called_once_with(fake_tao_bits)
    assert result == {fake_netuid: (fake_block_updated, Balance.from_rao(1000000))}


def test_get_ema_tao_inflow(subtensor, mocker):
    """Test get_ema_tao_inflow returns correct values."""
    # Preps
    fake_block = 123
    fake_netuid = 1
    fake_block_updated = 100
    fake_tao_bits = {"bits": 6520190}

    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query = mocker.patch.object(
        subtensor.substrate,
        "query",
        return_value=mocker.Mock(value=(fake_block_updated, fake_tao_bits)),
    )
    mocked_fixed_to_float = mocker.patch.object(
        subtensor_module, "fixed_to_float", return_value=1000000
    )

    # Call
    result = subtensor.get_ema_tao_inflow(netuid=fake_netuid, block=fake_block)

    # Asserts
    mocked_determine_block_hash.assert_called_once_with(fake_block)
    mocked_query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="SubnetEmaTaoFlow",
        params=[fake_netuid],
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_fixed_to_float.assert_called_once_with(fake_tao_bits)
    assert result == (fake_block_updated, Balance.from_rao(1000000))


def test_get_proxies(subtensor, mocker):
    """Test get_proxies returns correct data when proxy information is found."""
    # Prep
    block = 123
    fake_real_account1 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    fake_real_account2 = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    fake_proxy_data1 = [
        {
            "delegate": {"Id": b"\x00" * 32},
            "proxy_type": {"Any": None},
            "delay": 0,
        }
    ]
    fake_proxy_data2 = [
        {
            "delegate": {"Id": b"\x01" * 32},
            "proxy_type": {"Transfer": None},
            "delay": 100,
        }
    ]
    fake_query_map_records = [
        (fake_real_account1.encode(), mocker.Mock(value=([fake_proxy_data1], 1000000))),
        (fake_real_account2.encode(), mocker.Mock(value=([fake_proxy_data2], 2000000))),
    ]

    mocked_determine_block_hash = mocker.patch.object(
        subtensor, "determine_block_hash", return_value="mock_block_hash"
    )
    mocked_query_map = mocker.patch.object(
        subtensor.substrate,
        "query_map",
        return_value=fake_query_map_records,
    )
    mocked_from_query_map_record = mocker.patch.object(
        subtensor_module.ProxyInfo,
        "from_query_map_record",
        side_effect=[
            (fake_real_account1, [mocker.Mock()]),
            (fake_real_account2, [mocker.Mock()]),
        ],
    )

    # Call
    result = subtensor.get_proxies(block=block)

    # Asserts
    mocked_determine_block_hash.assert_called_once_with(block)
    mocked_query_map.assert_called_once_with(
        module="Proxy",
        storage_function="Proxies",
        block_hash="mock_block_hash",
    )
    assert mocked_from_query_map_record.call_count == 2
    assert isinstance(result, dict)
    assert fake_real_account1 in result
    assert fake_real_account2 in result


def test_get_proxies_for_real_account(subtensor, mocker):
    """Test get_proxies_for_real_account returns correct data when proxy information is found."""
    # Prep
    fake_real_account_ss58 = mocker.Mock(spec=str)

    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query = mocker.patch.object(
        subtensor.substrate,
        "query",
    )
    mocked_from_query = mocker.patch.object(
        subtensor_module.ProxyInfo,
        "from_query",
    )

    # Call
    result = subtensor.get_proxies_for_real_account(
        real_account_ss58=fake_real_account_ss58
    )

    # Asserts
    mocked_determine_block_hash.assert_called_once_with(None)
    mocked_query.assert_called_once_with(
        module="Proxy",
        storage_function="Proxies",
        params=[fake_real_account_ss58],
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_from_query.assert_called_once_with(mocked_query.return_value)
    assert result == mocked_from_query.return_value


def test_get_proxy_announcement(subtensor, mocker):
    """Test get_proxy_announcement returns correct data when announcement information is found."""
    # Prep
    fake_delegate_account_ss58 = mocker.Mock(spec=str)
    mocked_determine_block_hash = mocker.patch.object(subtensor, "determine_block_hash")
    mocked_query = mocker.patch.object(
        subtensor.substrate,
        "query",
    )
    mocked_from_dict = mocker.patch.object(
        subtensor_module.ProxyAnnouncementInfo,
        "from_dict",
    )

    # Call
    result = subtensor.get_proxy_announcement(
        delegate_account_ss58=fake_delegate_account_ss58
    )

    # Asserts
    mocked_determine_block_hash.assert_called_once_with(None)
    mocked_query.assert_called_once_with(
        module="Proxy",
        storage_function="Announcements",
        params=[fake_delegate_account_ss58],
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_from_dict.assert_called_once_with(mocked_query.return_value.value[0])
    assert result == mocked_from_dict.return_value


def test_get_proxy_announcements(subtensor, mocker):
    """Test get_proxy_announcements returns correct data when announcement information is found."""
    # Prep
    fake_delegate = mocker.Mock(spec=str)
    fake_proxies_list = mocker.Mock(spec=list)
    mocked_determine_block_hash = mocker.patch.object(
        subtensor, "determine_block_hash", return_value="mock_block_hash"
    )

    fake_record = (fake_delegate, fake_proxies_list)
    fake_query_map_records = [fake_record]

    mocked_query_map = mocker.patch.object(
        subtensor.substrate,
        "query_map",
        return_value=fake_query_map_records,
    )
    mocked_from_query_map_record = mocker.patch.object(
        subtensor_module.ProxyAnnouncementInfo,
        "from_query_map_record",
        side_effect=fake_query_map_records,
    )

    # Call
    result = subtensor.get_proxy_announcements()

    # Asserts
    mocked_determine_block_hash.assert_called_once_with(None)
    mocked_query_map.assert_called_once_with(
        module="Proxy",
        storage_function="Announcements",
        block_hash=mocked_determine_block_hash.return_value,
    )
    mocked_from_query_map_record.assert_called_once_with(fake_record)
    assert result == {fake_delegate: fake_proxies_list}


def test_get_proxy_constants(subtensor, mocker):
    """Test get_proxy_constants returns correct data when constants are found."""
    # Prep
    fake_constants = {
        "AnnouncementDepositBase": 1000000,
        "AnnouncementDepositFactor": 500000,
        "MaxProxies": 32,
        "MaxPending": 32,
        "ProxyDepositBase": 2000000,
        "ProxyDepositFactor": 1000000,
    }

    mocked_query_constant = mocker.patch.object(
        subtensor,
        "query_constant",
        side_effect=[mocker.Mock(value=value) for value in fake_constants.values()],
    )
    mocked_from_dict = mocker.patch.object(subtensor_module.ProxyConstants, "from_dict")

    # Call
    result = subtensor.get_proxy_constants()

    # Asserts
    assert mocked_query_constant.call_count == len(fake_constants)
    mocked_from_dict.assert_called_once_with(fake_constants)
    assert result == mocked_from_dict.return_value


def test_get_proxy_constants_as_dict(subtensor, mocker):
    """Test get_proxy_constants returns dict when as_dict=True."""
    # Prep
    fake_constants = {
        "AnnouncementDepositBase": 1000000,
        "AnnouncementDepositFactor": 500000,
        "MaxProxies": 32,
        "MaxPending": 32,
        "ProxyDepositBase": 2000000,
        "ProxyDepositFactor": 1000000,
    }

    mocked_query_constant = mocker.patch.object(
        subtensor,
        "query_constant",
        side_effect=[mocker.Mock(value=value) for value in fake_constants.values()],
    )
    mocked_proxy_constants = mocker.Mock()
    mocked_from_dict = mocker.patch.object(
        subtensor_module.ProxyConstants,
        "from_dict",
        return_value=mocked_proxy_constants,
    )
    mocked_to_dict = mocker.patch.object(
        mocked_proxy_constants,
        "to_dict",
        return_value=fake_constants,
    )

    # Call
    result = subtensor.get_proxy_constants(as_dict=True)

    # Asserts
    assert mocked_query_constant.call_count == len(fake_constants)
    mocked_from_dict.assert_called_once_with(fake_constants)
    mocked_to_dict.assert_called_once()
    assert result == fake_constants


def test_add_proxy(mocker, subtensor):
    """Tests `add_proxy` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    delegate_ss58 = mocker.Mock(spec=str)
    proxy_type = mocker.Mock(spec=str)
    delay = mocker.Mock(spec=int)
    mocked_add_proxy_extrinsic = mocker.patch.object(
        subtensor_module, "add_proxy_extrinsic"
    )

    # call
    response = subtensor.add_proxy(
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        proxy_type=proxy_type,
        delay=delay,
    )

    # asserts
    mocked_add_proxy_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        proxy_type=proxy_type,
        delay=delay,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_add_proxy_extrinsic.return_value


def test_announce_proxy(mocker, subtensor):
    """Tests `announce_proxy` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    real_account_ss58 = mocker.Mock(spec=str)
    call_hash = mocker.Mock(spec=str)
    mocked_announce_extrinsic = mocker.patch.object(
        subtensor_module, "announce_extrinsic"
    )

    # call
    response = subtensor.announce_proxy(
        wallet=wallet,
        real_account_ss58=real_account_ss58,
        call_hash=call_hash,
    )

    # asserts
    mocked_announce_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        real_account_ss58=real_account_ss58,
        call_hash=call_hash,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_announce_extrinsic.return_value


def test_create_pure_proxy(mocker, subtensor):
    """Tests `create_pure_proxy` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    proxy_type = mocker.Mock(spec=str)
    delay = mocker.Mock(spec=int)
    index = mocker.Mock(spec=int)
    mocked_create_pure_proxy_extrinsic = mocker.patch.object(
        subtensor_module, "create_pure_proxy_extrinsic"
    )

    # call
    response = subtensor.create_pure_proxy(
        wallet=wallet,
        proxy_type=proxy_type,
        delay=delay,
        index=index,
    )

    # asserts
    mocked_create_pure_proxy_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        proxy_type=proxy_type,
        delay=delay,
        index=index,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_create_pure_proxy_extrinsic.return_value


def test_kill_pure_proxy(mocker, subtensor):
    """Tests `kill_pure_proxy` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    pure_proxy_ss58 = mocker.Mock(spec=str)
    spawner = mocker.Mock(spec=str)
    proxy_type = mocker.Mock(spec=str)
    index = mocker.Mock(spec=int)
    height = mocker.Mock(spec=int)
    ext_index = mocker.Mock(spec=int)
    mocked_kill_pure_proxy_extrinsic = mocker.patch.object(
        subtensor_module, "kill_pure_proxy_extrinsic"
    )

    # call
    response = subtensor.kill_pure_proxy(
        wallet=wallet,
        pure_proxy_ss58=pure_proxy_ss58,
        spawner=spawner,
        proxy_type=proxy_type,
        index=index,
        height=height,
        ext_index=ext_index,
    )

    # asserts
    mocked_kill_pure_proxy_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        pure_proxy_ss58=pure_proxy_ss58,
        spawner=spawner,
        proxy_type=proxy_type,
        index=index,
        height=height,
        ext_index=ext_index,
        force_proxy_type=subtensor_module.ProxyType.Any,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_kill_pure_proxy_extrinsic.return_value


def test_poke_deposit(mocker, subtensor):
    """Tests `poke_deposit` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    mocked_poke_deposit_extrinsic = mocker.patch.object(
        subtensor_module, "poke_deposit_extrinsic"
    )

    # call
    response = subtensor.poke_deposit(wallet=wallet)

    # asserts
    mocked_poke_deposit_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_poke_deposit_extrinsic.return_value


def test_proxy(mocker, subtensor):
    """Tests `proxy` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    real_account_ss58 = mocker.Mock(spec=str)
    force_proxy_type = mocker.Mock(spec=str)
    call = mocker.Mock(spec=GenericCall)
    mocked_proxy_extrinsic = mocker.patch.object(subtensor_module, "proxy_extrinsic")

    # call
    response = subtensor.proxy(
        wallet=wallet,
        real_account_ss58=real_account_ss58,
        force_proxy_type=force_proxy_type,
        call=call,
    )

    # asserts
    mocked_proxy_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        real_account_ss58=real_account_ss58,
        force_proxy_type=force_proxy_type,
        call=call,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_proxy_extrinsic.return_value


def test_proxy_announced(mocker, subtensor):
    """Tests `proxy_announced` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    delegate_ss58 = mocker.Mock(spec=str)
    real_account_ss58 = mocker.Mock(spec=str)
    force_proxy_type = mocker.Mock(spec=str)
    call = mocker.Mock(spec=GenericCall)
    mocked_proxy_announced_extrinsic = mocker.patch.object(
        subtensor_module, "proxy_announced_extrinsic"
    )

    # call
    response = subtensor.proxy_announced(
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        real_account_ss58=real_account_ss58,
        force_proxy_type=force_proxy_type,
        call=call,
    )

    # asserts
    mocked_proxy_announced_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        real_account_ss58=real_account_ss58,
        force_proxy_type=force_proxy_type,
        call=call,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_proxy_announced_extrinsic.return_value


def test_reject_proxy_announcement(mocker, subtensor):
    """Tests `reject_proxy_announcement` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    delegate_ss58 = mocker.Mock(spec=str)
    call_hash = mocker.Mock(spec=str)
    mocked_reject_announcement_extrinsic = mocker.patch.object(
        subtensor_module, "reject_announcement_extrinsic"
    )

    # call
    response = subtensor.reject_proxy_announcement(
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        call_hash=call_hash,
    )

    # asserts
    mocked_reject_announcement_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        call_hash=call_hash,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_reject_announcement_extrinsic.return_value


def test_remove_proxy_announcement(mocker, subtensor):
    """Tests `remove_proxy_announcement` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    real_account_ss58 = mocker.Mock(spec=str)
    call_hash = mocker.Mock(spec=str)
    mocked_remove_announcement_extrinsic = mocker.patch.object(
        subtensor_module, "remove_announcement_extrinsic"
    )

    # call
    response = subtensor.remove_proxy_announcement(
        wallet=wallet,
        real_account_ss58=real_account_ss58,
        call_hash=call_hash,
    )

    # asserts
    mocked_remove_announcement_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        real_account_ss58=real_account_ss58,
        call_hash=call_hash,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_remove_announcement_extrinsic.return_value


def test_remove_proxies(mocker, subtensor):
    """Tests `remove_proxies` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    mocked_remove_proxies_extrinsic = mocker.patch.object(
        subtensor_module, "remove_proxies_extrinsic"
    )

    # call
    response = subtensor.remove_proxies(wallet=wallet)

    # asserts
    mocked_remove_proxies_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_remove_proxies_extrinsic.return_value


def test_remove_proxy(mocker, subtensor):
    """Tests `remove_proxy` extrinsic call method."""
    # preps
    wallet = mocker.Mock(spec=Wallet)
    delegate_ss58 = mocker.Mock(spec=str)
    proxy_type = mocker.Mock(spec=str)
    delay = mocker.Mock(spec=int)
    mocked_remove_proxy_extrinsic = mocker.patch.object(
        subtensor_module, "remove_proxy_extrinsic"
    )

    # call
    response = subtensor.remove_proxy(
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        proxy_type=proxy_type,
        delay=delay,
    )

    # asserts
    mocked_remove_proxy_extrinsic.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        delegate_ss58=delegate_ss58,
        proxy_type=proxy_type,
        delay=delay,
        mev_protection=DEFAULT_MEV_PROTECTION,
        period=DEFAULT_PERIOD,
        raise_error=False,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response == mocked_remove_proxy_extrinsic.return_value


def test_blocks_until_next_epoch_uses_default_tempo(subtensor, mocker):
    """Test blocks_until_next_epoch uses self.tempo when tempo is None."""
    # Prep
    netuid = 0
    block = 20
    tempo = 100

    spy_get_current_block = mocker.spy(subtensor, "get_current_block")
    spy_tempo = mocker.spy(subtensor, "tempo")

    # Call
    result = subtensor.blocks_until_next_epoch(netuid=netuid, tempo=tempo, block=block)

    # Assert
    spy_get_current_block.assert_not_called()
    spy_tempo.assert_not_called()
    assert result is not None
    assert isinstance(result, int)


def test_get_stake_info_for_coldkeys_none(subtensor, mocker):
    """Tests get_stake_info_for_coldkeys method when query_runtime_api returns None."""
    # Preps
    fake_coldkey_ss58s = ["coldkey1", "coldkey2"]
    fake_block = 123

    mocked_query_runtime_api = mocker.patch.object(
        subtensor, "query_runtime_api", return_value=None
    )

    # Call
    result = subtensor.get_stake_info_for_coldkeys(
        coldkey_ss58s=fake_coldkey_ss58s, block=fake_block
    )

    # Asserts
    assert result == {}
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="StakeInfoRuntimeApi",
        method="get_stake_info_for_coldkeys",
        params=[fake_coldkey_ss58s],
        block=fake_block,
    )


def test_get_stake_info_for_coldkeys_success(subtensor, mocker):
    """Tests get_stake_info_for_coldkeys method when query_runtime_api returns data."""
    # Preps
    fake_coldkey_ss58s = ["coldkey1", "coldkey2"]
    fake_block = 123

    fake_ck1 = b"\x16:\xech\r\xde,g\x03R1\xb9\x88q\xe79\xb8\x88\x93\xae\xd2)?*\rp\xb2\xe62\xads\x1c"
    fake_ck2 = b"\x17:\xech\r\xde,g\x03R1\xb9\x88q\xe79\xb8\x88\x93\xae\xd2)?*\rp\xb2\xe62\xads\x1d"
    fake_decoded_ck1 = "decoded_coldkey1"
    fake_decoded_ck2 = "decoded_coldkey2"

    stake_info_dict_1 = {
        "netuid": 5,
        "hotkey": b"\x16:\xech\r\xde,g\x03R1\xb9\x88q\xe79\xb8\x88\x93\xae\xd2)?*\rp\xb2\xe62\xads\x1c",
        "coldkey": fake_ck1,
        "stake": 1000,
        "locked": 0,
        "emission": 100,
        "drain": 0,
        "is_registered": True,
    }
    stake_info_dict_2 = {
        "netuid": 14,
        "hotkey": b"\x17:\xech\r\xde,g\x03R1\xb9\x88q\xe79\xb8\x88\x93\xae\xd2)?*\rp\xb2\xe62\xads\x1d",
        "coldkey": fake_ck2,
        "stake": 2000,
        "locked": 0,
        "emission": 200,
        "drain": 0,
        "is_registered": False,
    }

    fake_query_result = [
        (fake_ck1, [stake_info_dict_1]),
        (fake_ck2, [stake_info_dict_2]),
    ]

    mocked_query_runtime_api = mocker.patch.object(
        subtensor, "query_runtime_api", return_value=fake_query_result
    )

    mocked_decode_account_id = mocker.patch.object(
        subtensor_module,
        "decode_account_id",
        side_effect=[fake_decoded_ck1, fake_decoded_ck2],
    )

    mock_stake_info_1 = mocker.Mock(spec=StakeInfo)
    mock_stake_info_2 = mocker.Mock(spec=StakeInfo)
    mocked_stake_info_list_from_dicts = mocker.patch.object(
        subtensor_module.StakeInfo,
        "list_from_dicts",
        side_effect=[[mock_stake_info_1], [mock_stake_info_2]],
    )

    # Call
    result = subtensor.get_stake_info_for_coldkeys(
        coldkey_ss58s=fake_coldkey_ss58s, block=fake_block
    )

    # Asserts
    assert result == {
        fake_decoded_ck1: [mock_stake_info_1],
        fake_decoded_ck2: [mock_stake_info_2],
    }
    mocked_query_runtime_api.assert_called_once_with(
        runtime_api="StakeInfoRuntimeApi",
        method="get_stake_info_for_coldkeys",
        params=[fake_coldkey_ss58s],
        block=fake_block,
    )
    mocked_decode_account_id.assert_has_calls(
        [mocker.call(fake_ck1), mocker.call(fake_ck2)]
    )
    mocked_stake_info_list_from_dicts.assert_has_calls(
        [mocker.call([stake_info_dict_1]), mocker.call([stake_info_dict_2])]
    )
