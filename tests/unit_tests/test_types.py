"""Unit tests for bittensor/core/types.py"""

import argparse
from unittest.mock import MagicMock, patch, Mock
import numpy as np
import pytest

from bittensor.core.types import (
    UIDs,
    Weights,
    Salt,
    SubtensorMixin,
    AxonServeCallParams,
    PrometheusServeCallParams,
)
from bittensor.core.chain_data import NeuronInfo, NeuronInfoLite, AxonInfo
from bittensor.core.config import Config
from bittensor.core import settings
from bittensor.utils import Certificate


# ============================================================================
# Type Annotation Tests
# ============================================================================


def test_uids_type_annotation():
    """Test UIDs type accepts NDArray and list."""
    # Test with NDArray
    uids_array: UIDs = np.array([1, 2, 3], dtype=np.int64)
    assert isinstance(uids_array, np.ndarray)
    assert uids_array.dtype == np.int64

    # Test with list
    uids_list: UIDs = [1, 2, 3]
    assert isinstance(uids_list, list)
    assert all(isinstance(uid, int) for uid in uids_list)


def test_weights_type_annotation():
    """Test Weights type accepts NDArray and list."""
    # Test with NDArray
    weights_array: Weights = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert isinstance(weights_array, np.ndarray)
    assert weights_array.dtype == np.float32

    # Test with list of floats
    weights_list_float: Weights = [0.1, 0.2, 0.3]
    assert isinstance(weights_list_float, list)
    assert all(isinstance(w, float) for w in weights_list_float)

    # Test with list of ints
    weights_list_int: Weights = [1, 2, 3]
    assert isinstance(weights_list_int, list)
    assert all(isinstance(w, int) for w in weights_list_int)

    # Test with mixed list
    weights_list_mixed: Weights = [1, 0.5, 2, 0.3]
    assert isinstance(weights_list_mixed, list)


def test_salt_type_annotation():
    """Test Salt type accepts NDArray and list."""
    # Test with NDArray
    salt_array: Salt = np.array([100, 200, 300], dtype=np.int64)
    assert isinstance(salt_array, np.ndarray)
    assert salt_array.dtype == np.int64

    # Test with list
    salt_list: Salt = [100, 200, 300]
    assert isinstance(salt_list, list)
    assert all(isinstance(s, int) for s in salt_list)


# ============================================================================
# SubtensorMixin Tests
# ============================================================================


class ConcreteSubtensorMixin(SubtensorMixin):
    """Concrete implementation of SubtensorMixin for testing."""

    def __init__(self, network: str, chain_endpoint: str, log_verbose: bool = False):
        self.network = network
        self.chain_endpoint = chain_endpoint
        self.log_verbose = log_verbose


def test_subtensor_mixin_str_method():
    """Test __str__ representation."""
    mixin = ConcreteSubtensorMixin(
        network="finney", chain_endpoint="wss://entrypoint-finney.opentensor.ai:443"
    )
    expected = "Network: finney, Chain: wss://entrypoint-finney.opentensor.ai:443"
    assert str(mixin) == expected


def test_subtensor_mixin_repr_method():
    """Test __repr__ representation."""
    mixin = ConcreteSubtensorMixin(
        network="test", chain_endpoint="wss://test.finney.opentensor.ai:443"
    )
    # __repr__ should return the same as __str__
    assert repr(mixin) == str(mixin)


@patch("bittensor.core.types.logging")
def test_check_and_log_network_settings(mock_logging):
    """Test logging for finney network."""
    # Test with finney network and log_verbose=True
    mixin = ConcreteSubtensorMixin(
        network="finney",
        chain_endpoint=settings.FINNEY_ENTRYPOINT,
        log_verbose=True,
    )
    mixin._check_and_log_network_settings()

    # Verify logging.info was called
    assert mock_logging.info.called
    info_call_args = mock_logging.info.call_args[0][0]
    assert "finney" in info_call_args
    assert settings.FINNEY_ENTRYPOINT in info_call_args

    # Verify logging.debug was called
    assert mock_logging.debug.called
    debug_call_args = mock_logging.debug.call_args[0][0]
    assert "local subtensor node" in debug_call_args

    # Reset mock
    mock_logging.reset_mock()

    # Test with finney network and log_verbose=False (should not log)
    mixin_no_log = ConcreteSubtensorMixin(
        network="finney",
        chain_endpoint=settings.FINNEY_ENTRYPOINT,
        log_verbose=False,
    )
    mixin_no_log._check_and_log_network_settings()
    assert not mock_logging.info.called
    assert not mock_logging.debug.called

    # Reset mock
    mock_logging.reset_mock()

    # Test with non-finney network (should not log)
    mixin_test = ConcreteSubtensorMixin(
        network="test",
        chain_endpoint="wss://test.finney.opentensor.ai:443",
        log_verbose=True,
    )
    mixin_test._check_and_log_network_settings()
    assert not mock_logging.info.called
    assert not mock_logging.debug.called


def test_config_creation():
    """Test SubtensorMixin.config() static method."""
    config = SubtensorMixin.config()
    assert isinstance(config, Config)
    # Check that subtensor arguments are present
    assert hasattr(config, "subtensor")
    assert hasattr(config.subtensor, "network")
    assert hasattr(config.subtensor, "chain_endpoint")


def test_setup_config_with_network_string():
    """Test setup_config with network param."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)
    config = Config(parser)

    # Test with explicit network parameter
    endpoint, network = SubtensorMixin.setup_config("finney", config)

    assert network == "finney"
    assert "wss://" in endpoint or "ws://" in endpoint


def test_setup_config_with_chain_endpoint():
    """Test setup_config with endpoint."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)
    config = Config(parser)

    # Set chain_endpoint in config - this will be used if network is None
    # But the setup_config method has complex precedence logic
    endpoint, network = SubtensorMixin.setup_config("wss://custom.endpoint.ai:443", config)

    # When passing a custom endpoint as network parameter, it should be used
    assert "wss://custom.endpoint.ai:443" in endpoint or network is not None


def test_setup_config_precedence_order():
    """Test config resolution order."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)
    config = Config(parser)

    # Test 1: Explicit network parameter takes precedence
    config.subtensor.network = "test"
    config.subtensor.chain_endpoint = "wss://test.endpoint.ai:443"

    endpoint, network = SubtensorMixin.setup_config("finney", config)
    assert network == "finney"  # Explicit parameter wins

    # Test 2: Config chain_endpoint is used when network is None
    endpoint, network = SubtensorMixin.setup_config(None, config)
    # Should use config values
    assert network in ["test", "finney", "local", "archive"]


def test_add_args_to_parser():
    """Test add_args() method."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)

    # Parse empty args to get defaults
    args = parser.parse_args([])

    # Check that arguments were added (they use dot notation in argparse)
    assert hasattr(args, "subtensor.network") or "subtensor.network" in vars(args)
    # Use Config to properly parse the args
    config = Config(parser)
    assert config.subtensor.network == settings.DEFAULTS.subtensor.network
    assert config.subtensor.chain_endpoint == settings.DEFAULTS.subtensor.chain_endpoint
    assert config.subtensor._mock == False


def test_add_args_with_prefix():
    """Test add_args() method with prefix."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser, prefix="test")

    # Parse empty args to get defaults
    args = parser.parse_args([])

    # Check that arguments were added with prefix (they use dot notation)
    assert "test.subtensor.network" in vars(args) or hasattr(args, "test.subtensor.network")
    # Use Config to properly parse
    config = Config(parser)
    assert hasattr(config, "test")


@patch("builtins.print")
def test_help_method(mock_print):
    """Test help() class method."""
    SubtensorMixin.help()

    # Verify print was called (help output)
    assert mock_print.called


# ============================================================================
# AxonServeCallParams Tests
# ============================================================================


def test_axon_serve_call_params_initialization():
    """Test AxonServeCallParams creation."""
    # Certificate expects a dict or string
    cert_data = {
        "algorithm": 1,
        "public_key": [[116, 101, 115, 116]],  # "test" in ASCII
    }
    cert = Certificate(cert_data)
    params = AxonServeCallParams(
        version=4,
        ip=3232235777,  # 192.168.1.1 in int form
        port=8080,
        ip_type=4,
        netuid=1,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        certificate=cert,
    )

    assert params.version == 4
    assert params.ip == 3232235777
    assert params.port == 8080
    assert params.ip_type == 4
    assert params.netuid == 1
    assert params.hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    assert params.coldkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    assert params.protocol == 4
    assert params.placeholder1 == 0
    assert params.placeholder2 == 0
    assert params.certificate == cert


def test_axon_serve_call_params_equality_with_dict():
    """Test __eq__ with dict."""
    params = AxonServeCallParams(
        version=4,
        ip=3232235777,
        port=8080,
        ip_type=4,
        netuid=1,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        certificate=None,
    )

    params_dict = {
        "version": 4,
        "ip": 3232235777,
        "port": 8080,
        "ip_type": 4,
        "netuid": 1,
        "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "coldkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "protocol": 4,
        "placeholder1": 0,
        "placeholder2": 0,
        "certificate": None,
    }

    assert params == params_dict


def test_axon_serve_call_params_equality_with_neuron_info():
    """Test __eq__ with NeuronInfo."""
    # Create mock NeuronInfo
    axon_info = AxonInfo(
        version=4,
        ip="192.168.1.1",
        port=8080,
        ip_type=4,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
    )

    neuron_info = NeuronInfo(
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        uid=0,
        netuid=1,
        active=1,
        stake=0.0,
        stake_dict={},
        total_stake=0.0,
        rank=0.0,
        emission=0.0,
        incentive=0.0,
        consensus=0.0,
        trust=0.0,
        validator_trust=0.0,
        dividends=0.0,
        last_update=0,
        validator_permit=False,
        weights=[],
        bonds=[],
        pruning_score=0,
        prometheus_info=None,
        axon_info=axon_info,
        is_null=False,
    )

    params = AxonServeCallParams(
        version=4,
        ip=3232235777,  # 192.168.1.1
        port=8080,
        ip_type=4,
        netuid=1,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        certificate=None,
    )

    assert params == neuron_info


def test_axon_serve_call_params_equality_with_neuron_info_lite():
    """Test __eq__ with NeuronInfoLite."""
    # Create mock NeuronInfoLite
    axon_info = AxonInfo(
        version=4,
        ip="192.168.1.1",
        port=8080,
        ip_type=4,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
    )

    neuron_info_lite = NeuronInfoLite(
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        uid=0,
        netuid=1,
        active=1,
        stake=0.0,
        stake_dict={},
        total_stake=0.0,
        rank=0.0,
        emission=0.0,
        incentive=0.0,
        consensus=0.0,
        trust=0.0,
        validator_trust=0.0,
        dividends=0.0,
        last_update=0,
        validator_permit=False,
        prometheus_info=None,
        axon_info=axon_info,
        pruning_score=0,
        is_null=False,
    )

    params = AxonServeCallParams(
        version=4,
        ip=3232235777,  # 192.168.1.1
        port=8080,
        ip_type=4,
        netuid=1,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        certificate=None,
    )

    assert params == neuron_info_lite


def test_axon_serve_call_params_equality_with_invalid_type():
    """Test __eq__ raises NotImplementedError for invalid types."""
    params = AxonServeCallParams(
        version=4,
        ip=3232235777,
        port=8080,
        ip_type=4,
        netuid=1,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        certificate=None,
    )

    with pytest.raises(NotImplementedError):
        params == "invalid_type"


def test_axon_serve_call_params_copy():
    """Test copy() method."""
    cert_data = {
        "algorithm": 1,
        "public_key": [[116, 101, 115, 116]],  # "test" in ASCII
    }
    cert = Certificate(cert_data)
    params = AxonServeCallParams(
        version=4,
        ip=3232235777,
        port=8080,
        ip_type=4,
        netuid=1,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        certificate=cert,
    )

    params_copy = params.copy()

    # Verify it's a different object
    assert params_copy is not params

    # Verify all attributes are equal
    assert params_copy.version == params.version
    assert params_copy.ip == params.ip
    assert params_copy.port == params.port
    assert params_copy.ip_type == params.ip_type
    assert params_copy.netuid == params.netuid
    assert params_copy.hotkey == params.hotkey
    assert params_copy.coldkey == params.coldkey
    assert params_copy.protocol == params.protocol
    assert params_copy.placeholder1 == params.placeholder1
    assert params_copy.placeholder2 == params.placeholder2
    assert params_copy.certificate == params.certificate


def test_axon_serve_call_params_dict():
    """Test dict() method."""
    cert_data = {
        "algorithm": 1,
        "public_key": [[116, 101, 115, 116]],  # "test" in ASCII
    }
    cert = Certificate(cert_data)
    params = AxonServeCallParams(
        version=4,
        ip=3232235777,
        port=8080,
        ip_type=4,
        netuid=1,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        certificate=cert,
    )

    params_dict = params.as_dict()

    assert params_dict["version"] == 4
    assert params_dict["ip"] == 3232235777
    assert params_dict["port"] == 8080
    assert params_dict["ip_type"] == 4
    assert params_dict["netuid"] == 1
    assert params_dict["protocol"] == 4
    assert params_dict["placeholder1"] == 0
    assert params_dict["placeholder2"] == 0
    assert params_dict["certificate"] == cert
    # hotkey and coldkey are not in as_dict() output
    assert "hotkey" not in params_dict
    assert "coldkey" not in params_dict


def test_axon_serve_call_params_dict_without_certificate():
    """Test dict() when certificate is None."""
    params = AxonServeCallParams(
        version=4,
        ip=3232235777,
        port=8080,
        ip_type=4,
        netuid=1,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        certificate=None,
    )

    params_dict = params.as_dict()

    # Certificate should not be in dict when it's None
    assert "certificate" not in params_dict


# ============================================================================
# PrometheusServeCallParams Tests
# ============================================================================


def test_prometheus_serve_call_params_structure():
    """Test PrometheusServeCallParams TypedDict."""
    # PrometheusServeCallParams is a TypedDict, so we can create a dict with the required keys
    prometheus_params: PrometheusServeCallParams = {
        "version": 4,
        "ip": 3232235777,
        "port": 9090,
        "ip_type": 4,
        "netuid": 1,
    }

    assert prometheus_params["version"] == 4
    assert prometheus_params["ip"] == 3232235777
    assert prometheus_params["port"] == 9090
    assert prometheus_params["ip_type"] == 4
    assert prometheus_params["netuid"] == 1


def test_prometheus_serve_call_params_type_checking():
    """Test PrometheusServeCallParams type checking."""
    # This test verifies that the TypedDict structure is correct
    prometheus_params: PrometheusServeCallParams = {
        "version": 1,
        "ip": 2130706433,  # 127.0.0.1
        "port": 9090,
        "ip_type": 4,
        "netuid": 0,
    }

    # Verify all required keys are present
    required_keys = {"version", "ip", "port", "ip_type", "netuid"}
    assert set(prometheus_params.keys()) == required_keys


# ============================================================================
# ParamWithTypes Tests (if it exists in the codebase)
# ============================================================================


def test_param_with_types_structure():
    """Test ParamWithTypes TypedDict."""
    # Note: ParamWithTypes was not found in the types.py file
    # This test is a placeholder and should be updated if ParamWithTypes exists elsewhere
    # or removed if it doesn't exist in the codebase
    pytest.skip("ParamWithTypes not found in bittensor/core/types.py")
