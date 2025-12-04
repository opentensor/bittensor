"""Unit tests for bittensor/core/types.py"""

import argparse
from unittest.mock import patch
import pytest

from bittensor.core.types import (
    SubtensorMixin,
    AxonServeCallParams,
    PrometheusServeCallParams,
)
from bittensor.core.chain_data import NeuronInfo, NeuronInfoLite, AxonInfo
from bittensor.core.config import Config
from bittensor.core import settings
from bittensor.utils import Certificate


# ============================================================================
# Test Constants
# ============================================================================

# Network and connection constants
TEST_IP_STRING = "192.168.1.1"
TEST_IP_INT = 3232235777  # 192.168.1.1 in integer form
TEST_PORT = 8080
TEST_IP_TYPE = 4
TEST_NETUID = 1
TEST_VERSION = 4
TEST_PROTOCOL = 4

# Key constants
TEST_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
TEST_COLDKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

# Certificate constants
TEST_CERT_DATA = {
    "algorithm": 1,
    "public_key": [[116, 101, 115, 116]],  # "test" in ASCII
}

# Placeholder constants
TEST_PLACEHOLDER1 = 0
TEST_PLACEHOLDER2 = 0


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
    """Test __str__ returns correctly formatted string with network and endpoint."""
    mixin = ConcreteSubtensorMixin(
        network="finney", 
        chain_endpoint=settings.FINNEY_ENTRYPOINT
    )
    result = str(mixin)
    
    # Verify format and content
    assert result == f"Network: finney, Chain: {settings.FINNEY_ENTRYPOINT}"
    assert "Network: finney" in result
    assert settings.FINNEY_ENTRYPOINT in result


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
    """Test SubtensorMixin.config() static method creates config with default values."""
    config = SubtensorMixin.config()
    
    # Verify config object is created
    assert isinstance(config, Config)
    
    # Verify subtensor config has expected default values
    assert config.subtensor.network == settings.DEFAULTS.subtensor.network
    assert config.subtensor.chain_endpoint == settings.DEFAULTS.subtensor.chain_endpoint
    assert config.subtensor._mock == False


def test_setup_config_with_network_string():
    """Test setup_config with network param."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)
    config = Config(parser)

    # Test with explicit network parameter
    endpoint, network = SubtensorMixin.setup_config("finney", config)

    assert network == "finney"
    assert settings.FINNEY_ENTRYPOINT in endpoint


def test_setup_config_with_chain_endpoint():
    """Test setup_config with custom endpoint."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)
    config = Config(parser)

    # When passing a custom endpoint as the network parameter
    endpoint, network = SubtensorMixin.setup_config("wss://custom.endpoint.ai:443", config)

    # Should use the custom endpoint and return 'unknown' as network
    assert "wss://custom.endpoint.ai:443" in endpoint
    assert network == "unknown"


def test_setup_config_precedence_order():
    """Test config resolution order."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)
    config = Config(parser)

    # Test 1: Explicit network parameter takes precedence over config
    config.subtensor.network = "test"
    config.subtensor.chain_endpoint = "wss://test.endpoint.ai:443"

    endpoint, network = SubtensorMixin.setup_config("finney", config)
    assert network == "finney"  # Explicit parameter wins
    assert settings.FINNEY_ENTRYPOINT in endpoint  # Should use finney endpoint

    # Test 2: Config network is used when network parameter is None
    endpoint, network = SubtensorMixin.setup_config(None, config)
    # Should use config.subtensor.network value which is "test"
    assert network == "test"
    assert settings.FINNEY_TEST_ENTRYPOINT in endpoint


def test_add_args_to_parser():
    """Test add_args() adds subtensor arguments with correct defaults to parser."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)

    # Parse with default values
    config = Config(parser)
    
    # Verify all subtensor arguments have correct default values
    assert config.subtensor.network == settings.DEFAULTS.subtensor.network
    assert config.subtensor.chain_endpoint == settings.DEFAULTS.subtensor.chain_endpoint
    assert config.subtensor._mock == False
    
    # Verify we can override values via command line args
    args = ["--subtensor.network", "test", "--subtensor.chain_endpoint", "wss://custom.ai:443"]
    config_override = Config(parser, args=args)
    assert config_override.subtensor.network == "test"
    assert config_override.subtensor.chain_endpoint == "wss://custom.ai:443"


# ============================================================================
# AxonServeCallParams Tests
# ============================================================================


def test_axon_serve_call_params_initialization():
    """Test AxonServeCallParams creation."""
    cert = Certificate(TEST_CERT_DATA)
    params = AxonServeCallParams(
        version=TEST_VERSION,
        ip=TEST_IP_INT,
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        netuid=TEST_NETUID,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=TEST_PLACEHOLDER1,
        placeholder2=TEST_PLACEHOLDER2,
        certificate=cert,
    )

    assert params.version == TEST_VERSION
    assert params.ip == TEST_IP_INT
    assert params.port == TEST_PORT
    assert params.ip_type == TEST_IP_TYPE
    assert params.netuid == TEST_NETUID
    assert params.hotkey == TEST_HOTKEY
    assert params.coldkey == TEST_COLDKEY
    assert params.protocol == TEST_PROTOCOL
    assert params.placeholder1 == TEST_PLACEHOLDER1
    assert params.placeholder2 == TEST_PLACEHOLDER2
    assert params.certificate == cert


def test_axon_serve_call_params_equality_with_dict():
    """Test __eq__ compares all attributes correctly with dict."""
    params = AxonServeCallParams(
        version=TEST_VERSION,
        ip=TEST_IP_INT,
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        netuid=TEST_NETUID,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=TEST_PLACEHOLDER1,
        placeholder2=TEST_PLACEHOLDER2,
        certificate=None,
    )

    # Test equality with matching dict
    matching_dict = {
        "version": TEST_VERSION,
        "ip": TEST_IP_INT,
        "port": TEST_PORT,
        "ip_type": TEST_IP_TYPE,
        "netuid": TEST_NETUID,
        "hotkey": TEST_HOTKEY,
        "coldkey": TEST_COLDKEY,
        "protocol": TEST_PROTOCOL,
        "placeholder1": TEST_PLACEHOLDER1,
        "placeholder2": TEST_PLACEHOLDER2,
        "certificate": None,
    }
    assert params == matching_dict
    
    # Test inequality with different dict
    different_dict = matching_dict.copy()
    different_dict["port"] = 9999
    assert params != different_dict


def test_axon_serve_call_params_equality_with_neuron_info():
    """Test __eq__ correctly compares AxonServeCallParams with NeuronInfo by extracting axon_info."""
    axon_info = AxonInfo(
        version=TEST_VERSION,
        ip=TEST_IP_STRING,
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=TEST_PLACEHOLDER1,
        placeholder2=TEST_PLACEHOLDER2,
    )

    neuron_info = NeuronInfo(
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        uid=0,
        netuid=TEST_NETUID,
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

    # Create params that match the neuron_info's axon_info
    params = AxonServeCallParams(
        version=TEST_VERSION,
        ip=TEST_IP_INT,  # Note: IP is converted to int for comparison
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        netuid=TEST_NETUID,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=TEST_PLACEHOLDER1,
        placeholder2=TEST_PLACEHOLDER2,
        certificate=None,
    )

    # Test equality - should compare params with neuron_info.axon_info
    assert params == neuron_info
    
    # Test inequality with different neuron_info
    different_axon = AxonInfo(
        version=TEST_VERSION,
        ip=TEST_IP_STRING,
        port=9999,  # Different port
        ip_type=TEST_IP_TYPE,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=TEST_PLACEHOLDER1,
        placeholder2=TEST_PLACEHOLDER2,
    )
    different_neuron = NeuronInfo(
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        uid=0,
        netuid=TEST_NETUID,
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
        axon_info=different_axon,
        is_null=False,
    )
    assert params != different_neuron


def test_axon_serve_call_params_copy():
    """Test copy() creates independent copy with same values."""
    cert = Certificate(TEST_CERT_DATA)
    params = AxonServeCallParams(
        version=TEST_VERSION,
        ip=TEST_IP_INT,
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        netuid=TEST_NETUID,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=TEST_PLACEHOLDER1,
        placeholder2=TEST_PLACEHOLDER2,
        certificate=cert,
    )

    params_copy = params.copy()

    # Verify it's a different object (not same reference)
    assert params_copy is not params
    assert id(params_copy) != id(params)

    # Verify all attributes have exact same values
    assert params_copy.version == TEST_VERSION
    assert params_copy.ip == TEST_IP_INT
    assert params_copy.port == TEST_PORT
    assert params_copy.ip_type == TEST_IP_TYPE
    assert params_copy.netuid == TEST_NETUID
    assert params_copy.hotkey == TEST_HOTKEY
    assert params_copy.coldkey == TEST_COLDKEY
    assert params_copy.protocol == TEST_PROTOCOL
    assert params_copy.placeholder1 == TEST_PLACEHOLDER1
    assert params_copy.placeholder2 == TEST_PLACEHOLDER2
    assert params_copy.certificate == cert
    
    # Verify modifying copy doesn't affect original
    params_copy.port = 9999
    assert params.port == TEST_PORT
    assert params_copy.port == 9999


def test_axon_serve_call_params_dict():
    """Test dict() method."""
    cert = Certificate(TEST_CERT_DATA)
    params = AxonServeCallParams(
        version=TEST_VERSION,
        ip=TEST_IP_INT,
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        netuid=TEST_NETUID,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=TEST_PLACEHOLDER1,
        placeholder2=TEST_PLACEHOLDER2,
        certificate=cert,
    )

    params_dict = params.as_dict()

    assert params_dict["version"] == TEST_VERSION
    assert params_dict["ip"] == TEST_IP_INT
    assert params_dict["port"] == TEST_PORT
    assert params_dict["ip_type"] == TEST_IP_TYPE
    assert params_dict["netuid"] == TEST_NETUID
    assert params_dict["protocol"] == TEST_PROTOCOL
    assert params_dict["placeholder1"] == TEST_PLACEHOLDER1
    assert params_dict["placeholder2"] == TEST_PLACEHOLDER2
    assert params_dict["certificate"] == cert
    # hotkey and coldkey are not in as_dict() output
    assert "hotkey" not in params_dict
    assert "coldkey" not in params_dict


def test_axon_serve_call_params_dict_without_certificate():
    """Test as_dict() excludes certificate when None and includes all other fields."""
    params = AxonServeCallParams(
        version=TEST_VERSION,
        ip=TEST_IP_INT,
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        netuid=TEST_NETUID,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=TEST_PLACEHOLDER1,
        placeholder2=TEST_PLACEHOLDER2,
        certificate=None,
    )

    params_dict = params.as_dict()

    # Verify certificate is excluded when None
    assert "certificate" not in params_dict
    
    # Verify all other expected fields are present with correct values
    assert params_dict["version"] == TEST_VERSION
    assert params_dict["ip"] == TEST_IP_INT
    assert params_dict["port"] == TEST_PORT
    assert params_dict["ip_type"] == TEST_IP_TYPE
    assert params_dict["netuid"] == TEST_NETUID
    assert params_dict["protocol"] == TEST_PROTOCOL
    assert params_dict["placeholder1"] == TEST_PLACEHOLDER1
    assert params_dict["placeholder2"] == TEST_PLACEHOLDER2
    
    # Verify hotkey and coldkey are NOT in as_dict() output (by design)
    assert "hotkey" not in params_dict
    assert "coldkey" not in params_dict


# ============================================================================
# PrometheusServeCallParams Tests
# ============================================================================


def test_prometheus_serve_call_params_structure():
    """Test PrometheusServeCallParams TypedDict has all required fields and correct types."""
    # Create PrometheusServeCallParams with all required fields
    prometheus_params: PrometheusServeCallParams = {
        "version": TEST_VERSION,
        "ip": TEST_IP_INT,
        "port": 9090,
        "ip_type": TEST_IP_TYPE,
        "netuid": TEST_NETUID,
    }

    # Verify all required fields are present with correct values
    assert prometheus_params["version"] == TEST_VERSION
    assert prometheus_params["ip"] == TEST_IP_INT
    assert prometheus_params["port"] == 9090
    assert prometheus_params["ip_type"] == TEST_IP_TYPE
    assert prometheus_params["netuid"] == TEST_NETUID
    
    # Verify exactly 5 keys (no more, no less)
    assert len(prometheus_params) == 5
    assert set(prometheus_params.keys()) == {"version", "ip", "port", "ip_type", "netuid"}
    
    # Verify types are correct
    assert isinstance(prometheus_params["version"], int)
    assert isinstance(prometheus_params["ip"], int)
    assert isinstance(prometheus_params["port"], int)
    assert isinstance(prometheus_params["ip_type"], int)
    assert isinstance(prometheus_params["netuid"], int)
