"""Unit tests for bittensor/core/types.py"""

import argparse
from unittest.mock import patch

from bittensor.core.types import (
    SubtensorMixin,
    AxonServeCallParams,
    PrometheusServeCallParams,
)
from bittensor.core.chain_data import NeuronInfo, AxonInfo
from bittensor.core.config import Config
from bittensor.core import settings
from bittensor.utils import Certificate


# Test constants
TEST_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
TEST_COLDKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
TEST_IP_INT = 3232235777  # 192.168.1.1
TEST_IP_STRING = "192.168.1.1"
TEST_PORT = 8080
TEST_VERSION = 4
TEST_IP_TYPE = 4
TEST_NETUID = 1
TEST_PROTOCOL = 4
TEST_CERT_DATA = {"algorithm": 1, "public_key": [[116, 101, 115, 116]]}


class ConcreteSubtensorMixin(SubtensorMixin):
    """Concrete implementation for testing SubtensorMixin."""

    def __init__(self, network: str, chain_endpoint: str, log_verbose: bool = False):
        self.network = network
        self.chain_endpoint = chain_endpoint
        self.log_verbose = log_verbose


def test_subtensor_mixin_str():
    """Test __str__ formatting."""
    mixin = ConcreteSubtensorMixin("finney", settings.FINNEY_ENTRYPOINT)
    assert str(mixin) == f"Network: finney, Chain: {settings.FINNEY_ENTRYPOINT}"


@patch("bittensor.core.types.logging")
def test_check_and_log_network_settings(mock_logging):
    """Test logging behavior for finney network."""
    # Should log when finney + log_verbose=True
    mixin = ConcreteSubtensorMixin("finney", settings.FINNEY_ENTRYPOINT, log_verbose=True)
    mixin._check_and_log_network_settings()
    assert mock_logging.info.called
    assert mock_logging.debug.called

    mock_logging.reset_mock()

    # Should not log when log_verbose=False
    mixin = ConcreteSubtensorMixin("finney", settings.FINNEY_ENTRYPOINT, log_verbose=False)
    mixin._check_and_log_network_settings()
    assert not mock_logging.info.called

    mock_logging.reset_mock()

    # Should not log for non-finney networks
    mixin = ConcreteSubtensorMixin("test", settings.FINNEY_TEST_ENTRYPOINT, log_verbose=True)
    mixin._check_and_log_network_settings()
    assert not mock_logging.info.called


def test_config_creation():
    """Test config() creates Config with default values."""
    config = SubtensorMixin.config()
    assert isinstance(config, Config)
    assert config.subtensor.network == settings.DEFAULTS.subtensor.network
    assert config.subtensor.chain_endpoint == settings.DEFAULTS.subtensor.chain_endpoint
    assert config.subtensor._mock is False


def test_setup_config_with_network():
    """Test setup_config with network name."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)
    config = Config(parser)

    endpoint, network = SubtensorMixin.setup_config("finney", config)
    assert network == "finney"
    assert endpoint == settings.FINNEY_ENTRYPOINT


def test_setup_config_with_custom_endpoint():
    """Test setup_config with custom endpoint URL."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)
    config = Config(parser)

    endpoint, network = SubtensorMixin.setup_config("wss://custom.ai:443", config)
    assert endpoint == "wss://custom.ai:443"
    assert network == "unknown"


def test_setup_config_precedence():
    """Test parameter takes precedence over config."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)
    config = Config(parser)
    config.subtensor.network = "test"

    # Explicit parameter should override config
    endpoint, network = SubtensorMixin.setup_config("finney", config)
    assert network == "finney"
    assert endpoint == settings.FINNEY_ENTRYPOINT

    # None parameter should use config
    endpoint, network = SubtensorMixin.setup_config(None, config)
    assert network == "test"
    assert endpoint == settings.FINNEY_TEST_ENTRYPOINT


def test_add_args():
    """Test add_args() adds arguments with defaults and allows overrides."""
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)

    # Check defaults
    config = Config(parser)
    assert config.subtensor.network == settings.DEFAULTS.subtensor.network
    assert config.subtensor.chain_endpoint == settings.DEFAULTS.subtensor.chain_endpoint
    assert config.subtensor._mock is False

    # Check overrides work
    config = Config(parser, args=["--subtensor.network", "test"])
    assert config.subtensor.network == "test"


def test_axon_serve_call_params_init():
    """Test AxonServeCallParams initialization with all fields."""
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
        placeholder1=0,
        placeholder2=0,
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
    assert params.placeholder1 == 0
    assert params.placeholder2 == 0
    assert params.certificate == cert


def test_axon_serve_call_params_equality():
    """Test __eq__ correctly compares with dict for equality and inequality."""
    params = AxonServeCallParams(
        version=TEST_VERSION,
        ip=TEST_IP_INT,
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        netuid=TEST_NETUID,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=0,
        placeholder2=0,
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
        "placeholder1": 0,
        "placeholder2": 0,
        "certificate": None,
    }
    assert params == matching_dict

    # Test inequality with different port
    different_dict = matching_dict.copy()
    different_dict["port"] = 9999
    assert params != different_dict


def test_axon_serve_call_params_with_neuron_info():
    """Test __eq__ correctly compares AxonServeCallParams with NeuronInfo by extracting axon_info."""
    axon_info = AxonInfo(
        version=TEST_VERSION,
        ip=TEST_IP_STRING,
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=0,
        placeholder2=0,
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

    params = AxonServeCallParams(
        version=TEST_VERSION,
        ip=TEST_IP_INT,
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        netuid=TEST_NETUID,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=0,
        placeholder2=0,
        certificate=None,
    )
    # Should compare params with neuron_info.axon_info
    assert params == neuron_info


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
        placeholder1=0,
        placeholder2=0,
        certificate=cert,
    )

    params_copy = params.copy()
    
    # Verify different object
    assert params_copy is not params
    assert id(params_copy) != id(params)
    
    # Verify same values
    assert params_copy.port == TEST_PORT
    assert params_copy.version == TEST_VERSION
    assert params_copy.certificate == cert

    # Verify modifying copy doesn't affect original
    params_copy.port = 9999
    assert params.port == TEST_PORT
    assert params_copy.port == 9999


def test_axon_serve_call_params_as_dict():
    """Test as_dict() includes all fields except hotkey/coldkey."""
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
        placeholder1=0,
        placeholder2=0,
        certificate=cert,
    )

    result = params.as_dict()
    
    # Verify all expected fields are present
    assert result["version"] == TEST_VERSION
    assert result["ip"] == TEST_IP_INT
    assert result["port"] == TEST_PORT
    assert result["ip_type"] == TEST_IP_TYPE
    assert result["netuid"] == TEST_NETUID
    assert result["protocol"] == TEST_PROTOCOL
    assert result["placeholder1"] == 0
    assert result["placeholder2"] == 0
    assert result["certificate"] == cert
    
    # Verify hotkey and coldkey are excluded by design
    assert "hotkey" not in result
    assert "coldkey" not in result


def test_axon_serve_call_params_as_dict_no_cert():
    """Test as_dict() excludes None certificate but includes all other fields."""
    params = AxonServeCallParams(
        version=TEST_VERSION,
        ip=TEST_IP_INT,
        port=TEST_PORT,
        ip_type=TEST_IP_TYPE,
        netuid=TEST_NETUID,
        hotkey=TEST_HOTKEY,
        coldkey=TEST_COLDKEY,
        protocol=TEST_PROTOCOL,
        placeholder1=0,
        placeholder2=0,
        certificate=None,
    )

    result = params.as_dict()
    
    # Verify certificate is excluded when None
    assert "certificate" not in result
    
    # Verify all other fields are present
    assert result["version"] == TEST_VERSION
    assert result["ip"] == TEST_IP_INT
    assert result["port"] == TEST_PORT
    assert result["ip_type"] == TEST_IP_TYPE
    assert result["netuid"] == TEST_NETUID
    assert result["protocol"] == TEST_PROTOCOL


def test_prometheus_serve_call_params():
    """Test PrometheusServeCallParams TypedDict has all required fields with correct types."""
    params: PrometheusServeCallParams = {
        "version": TEST_VERSION,
        "ip": TEST_IP_INT,
        "port": 9090,
        "ip_type": TEST_IP_TYPE,
        "netuid": TEST_NETUID,
    }

    # Verify all field values
    assert params["version"] == TEST_VERSION
    assert params["ip"] == TEST_IP_INT
    assert params["port"] == 9090
    assert params["ip_type"] == TEST_IP_TYPE
    assert params["netuid"] == TEST_NETUID
    
    # Verify exactly 5 keys, no more, no less
    assert len(params) == 5
    assert set(params.keys()) == {"version", "ip", "port", "ip_type", "netuid"}
    
    # Verify all values are integers
    assert all(isinstance(v, int) for v in params.values())
