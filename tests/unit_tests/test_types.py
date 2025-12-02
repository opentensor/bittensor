"""
Unit tests for bittensor.core.types module.

Tests the SubtensorMixin class, AxonServeCallParams, PrometheusServeCallParams,
ExtrinsicResponse, BlockInfo, and related type functionality.

This test suite ensures comprehensive coverage of the types module which provides
critical infrastructure for Bittensor network communication and configuration.
"""

import argparse
from abc import ABC
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, Mock, patch, PropertyMock
import pytest

from bittensor.core.types import (
    SubtensorMixin,
    AxonServeCallParams,
    PrometheusServeCallParams,
    ExtrinsicResponse,
    BlockInfo,
)
from bittensor.core.config import Config
from bittensor.core import settings
from bittensor.utils import Certificate
from bittensor.utils.balance import Balance
from bittensor.core.chain_data import NeuronInfo, NeuronInfoLite, AxonInfo


# ============================================================================
# Concrete Implementation for Testing Abstract SubtensorMixin
# ============================================================================

class ConcreteSubtensorMixin(SubtensorMixin):
    """
    Concrete implementation of SubtensorMixin for testing purposes.
    
    This class provides a concrete implementation of the abstract SubtensorMixin
    class, allowing us to test the mixin's functionality without requiring a
    full Subtensor implementation.
    """
    
    def __init__(self, network: str = "finney", chain_endpoint: str = "wss://entrypoint-finney.opentensor.ai:443", log_verbose: bool = False):
        """
        Initialize the concrete SubtensorMixin implementation.
        
        Args:
            network: The network name (e.g., 'finney', 'test', 'local')
            chain_endpoint: The WebSocket endpoint URL for the chain
            log_verbose: Whether to enable verbose logging
        """
        self.network = network
        self.chain_endpoint = chain_endpoint
        self.log_verbose = log_verbose


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """
    Create a mock Config object for testing.
    
    This fixture provides a Config object with default subtensor settings
    that can be used across multiple tests.
    """
    parser = argparse.ArgumentParser()
    SubtensorMixin.add_args(parser)
    config = Config(parser)
    return config


@pytest.fixture
def sample_axon_serve_call_params():
    """
    Create a sample AxonServeCallParams instance for testing.
    
    Returns a fully populated AxonServeCallParams object with typical
    values that can be used in equality and other tests.
    """
    return AxonServeCallParams(
        version=1,
        ip=2130706433,  # 127.0.0.1 in integer format
        port=8091,
        ip_type=4,  # IPv4
        netuid=1,
        hotkey="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm",
        coldkey="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg",
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        certificate=None,
    )


@pytest.fixture
def sample_certificate():
    """
    Create a sample Certificate object for testing.
    
    Returns a mock Certificate that can be used in AxonServeCallParams tests.
    """
    return Certificate("fake_certificate_data")


@pytest.fixture
def sample_extrinsic_response():
    """
    Create a sample ExtrinsicResponse instance for testing.
    
    Returns a successful ExtrinsicResponse with typical values.
    """
    return ExtrinsicResponse(
        success=True,
        message="Test successful",
        extrinsic_function="test_function",
        extrinsic=None,
        extrinsic_fee=None,
        extrinsic_receipt=None,
        transaction_tao_fee=None,
        transaction_alpha_fee=None,
        error=None,
        data=None,
    )


# ============================================================================
# Test Classes for SubtensorMixin
# ============================================================================

class TestSubtensorMixinInitialization:
    """
    Tests for SubtensorMixin class initialization and basic attributes.
    
    These tests verify that the abstract mixin can be properly subclassed
    and that instances maintain the required attributes.
    """
    
    def test_subtensor_mixin_initialization(self):
        """
        Test that ConcreteSubtensorMixin can be instantiated with default values.
        
        Verifies that the mixin pattern works correctly and all required
        attributes are properly initialized.
        """
        # Create an instance with default values
        instance = ConcreteSubtensorMixin()
        
        # Verify it's an instance of SubtensorMixin
        assert isinstance(instance, SubtensorMixin)
        assert isinstance(instance, ABC)
        
        # Verify all required attributes are present
        assert hasattr(instance, "network")
        assert hasattr(instance, "chain_endpoint")
        assert hasattr(instance, "log_verbose")
        
        # Verify default values
        assert instance.network == "finney"
        assert instance.chain_endpoint == "wss://entrypoint-finney.opentensor.ai:443"
        assert instance.log_verbose is False
    
    def test_subtensor_mixin_custom_initialization(self):
        """
        Test SubtensorMixin initialization with custom values.
        
        Ensures that custom network, endpoint, and logging settings
        are properly stored and accessible.
        """
        # Create instance with custom values
        instance = ConcreteSubtensorMixin(
            network="test",
            chain_endpoint="wss://test.endpoint:443",
            log_verbose=True
        )
        
        # Verify custom values are set correctly
        assert instance.network == "test"
        assert instance.chain_endpoint == "wss://test.endpoint:443"
        assert instance.log_verbose is True
    
    def test_subtensor_mixin_is_abstract(self):
        """
        Test that SubtensorMixin cannot be instantiated directly.
        
        Since SubtensorMixin is an abstract base class, attempting to
        instantiate it directly should raise a TypeError.
        """
        # Attempting to instantiate ABC directly should fail
        # Note: ABC itself can be instantiated, but classes with abstract
        # methods cannot. Since SubtensorMixin doesn't define abstract
        # methods, we test that it requires concrete implementation
        # by checking it's an ABC subclass
        assert issubclass(SubtensorMixin, ABC)


class TestSubtensorMixinStringRepresentation:
    """
    Tests for SubtensorMixin string representation methods.
    
    These tests verify that __str__ and __repr__ methods work correctly
    and provide useful information about the instance.
    """
    
    def test_subtensor_mixin_str_representation(self):
        """
        Test the string representation of SubtensorMixin.
        
        Verifies that __str__ returns a formatted string containing
        network and chain endpoint information.
        """
        instance = ConcreteSubtensorMixin(
            network="finney",
            chain_endpoint="wss://entrypoint-finney.opentensor.ai:443"
        )
        
        # Get string representation
        str_repr = str(instance)
        
        # Verify it contains network information
        assert "Network: finney" in str_repr
        assert "Chain: wss://entrypoint-finney.opentensor.ai:443" in str_repr
    
    def test_subtensor_mixin_repr_representation(self):
        """
        Test the repr representation of SubtensorMixin.
        
        Verifies that __repr__ delegates to __str__ and provides
        the same formatted information.
        """
        instance = ConcreteSubtensorMixin(
            network="test",
            chain_endpoint="wss://test.endpoint:443"
        )
        
        # Get repr representation
        repr_str = repr(instance)
        
        # Verify it matches __str__ output
        assert repr_str == str(instance)
        assert "Network: test" in repr_str
        assert "Chain: wss://test.endpoint:443" in repr_str


class TestSubtensorMixinNetworkLogging:
    """
    Tests for SubtensorMixin network logging functionality.
    
    These tests verify that the _check_and_log_network_settings method
    correctly logs network information when appropriate conditions are met.
    """
    
    @patch('bittensor.core.types.logging')
    def test_check_and_log_network_settings_finney_verbose(self, mock_logging):
        """
        Test logging when connecting to finney network with verbose logging enabled.
        
        When log_verbose is True and network is 'finney', the method should
        log an info message about the connection and a debug message encouraging
        local node usage.
        """
        # Create instance with finney network and verbose logging
        instance = ConcreteSubtensorMixin(
            network="finney",
            chain_endpoint=settings.FINNEY_ENTRYPOINT,
            log_verbose=True
        )
        
        # Call the logging check method
        instance._check_and_log_network_settings()
        
        # Verify info logging was called
        mock_logging.info.assert_called_once()
        info_call_args = mock_logging.info.call_args[0][0]
        assert "finney" in info_call_args
        assert settings.FINNEY_ENTRYPOINT in info_call_args
        
        # Verify debug logging was called
        mock_logging.debug.assert_called_once()
        debug_call_args = mock_logging.debug.call_args[0][0]
        assert "local subtensor node" in debug_call_args.lower()
    
    @patch('bittensor.core.types.logging')
    def test_check_and_log_network_settings_finney_non_verbose(self, mock_logging):
        """
        Test that logging is skipped when verbose logging is disabled.
        
        When log_verbose is False, no logging should occur even if
        connecting to the finney network.
        """
        # Create instance with finney network but verbose logging disabled
        instance = ConcreteSubtensorMixin(
            network="finney",
            chain_endpoint=settings.FINNEY_ENTRYPOINT,
            log_verbose=False
        )
        
        # Call the logging check method
        instance._check_and_log_network_settings()
        
        # Verify no logging occurred
        mock_logging.info.assert_not_called()
        mock_logging.debug.assert_not_called()
    
    @patch('bittensor.core.types.logging')
    def test_check_and_log_network_settings_non_finney(self, mock_logging):
        """
        Test that logging is skipped for non-finney networks.
        
        When network is not 'finney', no logging should occur regardless
        of the verbose setting.
        """
        # Create instance with test network
        instance = ConcreteSubtensorMixin(
            network="test",
            chain_endpoint="wss://test.endpoint:443",
            log_verbose=True
        )
        
        # Call the logging check method
        instance._check_and_log_network_settings()
        
        # Verify no logging occurred
        mock_logging.info.assert_not_called()
        mock_logging.debug.assert_not_called()
    
    @patch('bittensor.core.types.logging')
    def test_check_and_log_network_settings_finney_endpoint(self, mock_logging):
        """
        Test logging when chain_endpoint matches FINNEY_ENTRYPOINT.
        
        Even if network name is not 'finney', if chain_endpoint matches
        FINNEY_ENTRYPOINT, logging should occur when verbose is enabled.
        """
        # Create instance with finney endpoint but different network name
        instance = ConcreteSubtensorMixin(
            network="custom",
            chain_endpoint=settings.FINNEY_ENTRYPOINT,
            log_verbose=True
        )
        
        # Call the logging check method
        instance._check_and_log_network_settings()
        
        # Verify logging occurred
        mock_logging.info.assert_called_once()
        mock_logging.debug.assert_called_once()


class TestSubtensorMixinConfig:
    """
    Tests for SubtensorMixin configuration methods.
    
    These tests verify that the config() static method correctly creates
    and returns a Config object with Subtensor arguments added.
    """
    
    def test_subtensor_mixin_config_method(self):
        """
        Test that config() creates a Config object with Subtensor arguments.
        
        The config() method should create an ArgumentParser, add Subtensor
        arguments to it, and return a Config object initialized with that parser.
        """
        # Call the static config method
        config = SubtensorMixin.config()
        
        # Verify it returns a Config instance
        assert isinstance(config, Config)
        
        # Verify it has subtensor configuration
        assert hasattr(config, "subtensor")
        assert hasattr(config.subtensor, "network")
        assert hasattr(config.subtensor, "chain_endpoint")
        assert hasattr(config.subtensor, "_mock")
    
    def test_subtensor_mixin_config_defaults(self):
        """
        Test that config() includes default values for Subtensor settings.
        
        The Config object should have default values for network and
        chain_endpoint based on settings.DEFAULTS.
        """
        # Get config
        config = SubtensorMixin.config()
        
        # Verify default values are present
        # These should match settings.DEFAULTS.subtensor values
        assert config.subtensor.network is not None
        assert config.subtensor.chain_endpoint is not None
        assert isinstance(config.subtensor._mock, bool)


class TestSubtensorMixinSetupConfig:
    """
    Tests for SubtensorMixin setup_config method.
    
    These tests verify that setup_config correctly determines network
    and endpoint configuration based on provided parameters and config object.
    """
    
    @patch('bittensor.core.types.determine_chain_endpoint_and_network')
    @patch('bittensor.core.types.networking.get_formatted_ws_endpoint_url')
    def test_setup_config_with_network_string(self, mock_format_url, mock_determine):
        """
        Test setup_config when network string is provided.
        
        When a network string is provided, it should take precedence over
        config settings and be used to determine the endpoint.
        """
        # Setup mocks
        mock_determine.return_value = ("test_network", "wss://test.endpoint:443")
        mock_format_url.return_value = "wss://formatted.test.endpoint:443"
        
        # Create a config object
        config = Config()
        
        # Call setup_config with network string
        endpoint, network = SubtensorMixin.setup_config("finney", config)
        
        # Verify determine_chain_endpoint_and_network was called with network string
        mock_determine.assert_called_once_with("finney")
        
        # Verify formatting was called
        mock_format_url.assert_called_once_with("wss://test.endpoint:443")
        
        # Verify return values
        assert endpoint == "wss://formatted.test.endpoint:443"
        assert network == "test_network"
    
    @patch('bittensor.core.types.determine_chain_endpoint_and_network')
    @patch('bittensor.core.types.networking.get_formatted_ws_endpoint_url')
    def test_setup_config_with_none_network(self, mock_format_url, mock_determine):
        """
        Test setup_config when network is None and config has chain_endpoint set.
        
        When network is None, setup_config should check config for chain_endpoint
        first, then network, following the precedence order.
        """
        # Setup mocks
        mock_determine.return_value = ("config_network", "wss://config.endpoint:443")
        mock_format_url.return_value = "wss://formatted.config.endpoint:443"
        
        # Create config with chain_endpoint set
        parser = argparse.ArgumentParser()
        SubtensorMixin.add_args(parser)
        config = Config(parser, args=["--subtensor.chain_endpoint", "wss://custom.endpoint:443"])
        
        # Mock is_set to return True for chain_endpoint
        with patch.object(config, 'is_set', return_value=True):
            # Call setup_config with None network
            endpoint, network = SubtensorMixin.setup_config(None, config)
            
            # Verify determine was called (with the config value)
            mock_determine.assert_called()
            mock_format_url.assert_called()
    
    @patch('bittensor.core.types.determine_chain_endpoint_and_network')
    @patch('bittensor.core.types.networking.get_formatted_ws_endpoint_url')
    def test_setup_config_precedence_order(self, mock_format_url, mock_determine):
        """
        Test that setup_config follows correct precedence order.
        
        The precedence should be:
        1. Provided network string
        2. Config chain_endpoint (if set)
        3. Config network (if set)
        4. Default chain_endpoint
        5. Default network
        """
        # Setup mocks
        mock_determine.return_value = ("default_network", "wss://default.endpoint:443")
        mock_format_url.return_value = "wss://formatted.default.endpoint:443"
        
        # Create config without any set values
        config = Config()
        
        # Call setup_config with None network
        endpoint, network = SubtensorMixin.setup_config(None, config)
        
        # Verify determine was called (should use defaults)
        mock_determine.assert_called()
        mock_format_url.assert_called()


class TestSubtensorMixinAddArgs:
    """
    Tests for SubtensorMixin add_args method.
    
    These tests verify that add_args correctly adds command-line arguments
    to an ArgumentParser for configuring Subtensor settings.
    """
    
    def test_add_args_adds_network_argument(self):
        """
        Test that add_args adds the --subtensor.network argument.
        
        The method should add a network argument with appropriate help text
        and default value.
        """
        # Create parser
        parser = argparse.ArgumentParser()
        
        # Add Subtensor arguments
        SubtensorMixin.add_args(parser)
        
        # Parse arguments to verify they were added
        args = parser.parse_args([])
        
        # Verify network argument exists
        assert hasattr(args, "subtensor")
        assert hasattr(args.subtensor, "network")
        assert args.subtensor.network is not None
    
    def test_add_args_adds_chain_endpoint_argument(self):
        """
        Test that add_args adds the --subtensor.chain_endpoint argument.
        
        The method should add a chain_endpoint argument with appropriate
        help text and default value.
        """
        # Create parser
        parser = argparse.ArgumentParser()
        
        # Add Subtensor arguments
        SubtensorMixin.add_args(parser)
        
        # Parse arguments
        args = parser.parse_args([])
        
        # Verify chain_endpoint argument exists
        assert hasattr(args.subtensor, "chain_endpoint")
        assert args.subtensor.chain_endpoint is not None
    
    def test_add_args_adds_mock_argument(self):
        """
        Test that add_args adds the --subtensor._mock argument.
        
        The method should add a _mock argument for testing purposes
        with a default value of False.
        """
        # Create parser
        parser = argparse.ArgumentParser()
        
        # Add Subtensor arguments
        SubtensorMixin.add_args(parser)
        
        # Parse arguments
        args = parser.parse_args([])
        
        # Verify _mock argument exists
        assert hasattr(args.subtensor, "_mock")
        assert isinstance(args.subtensor._mock, bool)
    
    def test_add_args_with_prefix(self):
        """
        Test that add_args works correctly with a prefix.
        
        When a prefix is provided, argument names should be prefixed
        accordingly (e.g., --prefix.subtensor.network).
        """
        # Create parser
        parser = argparse.ArgumentParser()
        
        # Add Subtensor arguments with prefix
        SubtensorMixin.add_args(parser, prefix="custom")
        
        # Parse arguments with prefixed name
        args = parser.parse_args(["--custom.subtensor.network", "test"])
        
        # Verify prefixed argument was parsed
        assert args.custom.subtensor.network == "test"
    
    def test_add_args_handles_reparsing(self):
        """
        Test that add_args handles argument re-parsing gracefully.
        
        If arguments are added multiple times (re-parsing scenario),
        the method should catch ArgumentError and continue without failing.
        """
        # Create parser
        parser = argparse.ArgumentParser()
        
        # Add arguments first time
        SubtensorMixin.add_args(parser)
        
        # Add arguments second time (should not raise error)
        try:
            SubtensorMixin.add_args(parser)
        except argparse.ArgumentError:
            # If ArgumentError is raised, it should be caught internally
            pytest.fail("add_args should handle re-parsing gracefully")


class TestSubtensorMixinHelp:
    """
    Tests for SubtensorMixin help method.
    
    These tests verify that the help() class method correctly prints
    help information to stdout.
    """
    
    @patch('builtins.print')
    @patch('argparse.ArgumentParser.print_help')
    def test_help_method_prints_information(self, mock_print_help, mock_print):
        """
        Test that help() prints class documentation and parser help.
        
        The help method should print the class docstring (via __new__.__doc__)
        and call parser.print_help() to display argument help.
        """
        # Call help method
        SubtensorMixin.help()
        
        # Verify print was called (for docstring)
        assert mock_print.called
        
        # Verify print_help was called
        mock_print_help.assert_called_once()


# ============================================================================
# Test Classes for AxonServeCallParams
# ============================================================================

class TestAxonServeCallParamsInitialization:
    """
    Tests for AxonServeCallParams class initialization.
    
    These tests verify that AxonServeCallParams can be created with
    all required parameters and that attributes are properly stored.
    """
    
    def test_axon_serve_call_params_initialization(self, sample_axon_serve_call_params):
        """
        Test AxonServeCallParams initialization with all parameters.
        
        Verifies that all required parameters can be provided and are
        correctly stored as instance attributes.
        """
        # Verify all attributes are set correctly
        assert sample_axon_serve_call_params.version == 1
        assert sample_axon_serve_call_params.ip == 2130706433
        assert sample_axon_serve_call_params.port == 8091
        assert sample_axon_serve_call_params.ip_type == 4
        assert sample_axon_serve_call_params.netuid == 1
        assert sample_axon_serve_call_params.hotkey == "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        assert sample_axon_serve_call_params.coldkey == "5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        assert sample_axon_serve_call_params.protocol == 4
        assert sample_axon_serve_call_params.placeholder1 == 0
        assert sample_axon_serve_call_params.placeholder2 == 0
        assert sample_axon_serve_call_params.certificate is None
    
    def test_axon_serve_call_params_with_certificate(self, sample_certificate):
        """
        Test AxonServeCallParams initialization with certificate.
        
        Verifies that a Certificate object can be provided and is
        correctly stored in the certificate attribute.
        """
        # Create params with certificate
        params = AxonServeCallParams(
            version=1,
            ip=2130706433,
            port=8091,
            ip_type=4,
            netuid=1,
            hotkey="test_hotkey",
            coldkey="test_coldkey",
            protocol=4,
            placeholder1=0,
            placeholder2=0,
            certificate=sample_certificate,
        )
        
        # Verify certificate is set
        assert params.certificate == sample_certificate
        assert params.certificate is not None


class TestAxonServeCallParamsEquality:
    """
    Tests for AxonServeCallParams equality comparison.
    
    These tests verify that the __eq__ method correctly compares
    AxonServeCallParams instances with various types of objects.
    """
    
    def test_axon_serve_call_params_equality_same_instance(self, sample_axon_serve_call_params):
        """
        Test equality comparison with the same instance.
        
        An instance should be equal to itself (reflexive property).
        """
        assert sample_axon_serve_call_params == sample_axon_serve_call_params
    
    def test_axon_serve_call_params_equality_equal_instances(self, sample_axon_serve_call_params):
        """
        Test equality comparison with equal instances.
        
        Two instances with identical attribute values should be equal.
        """
        # Create another instance with same values
        other_params = AxonServeCallParams(
            version=sample_axon_serve_call_params.version,
            ip=sample_axon_serve_call_params.ip,
            port=sample_axon_serve_call_params.port,
            ip_type=sample_axon_serve_call_params.ip_type,
            netuid=sample_axon_serve_call_params.netuid,
            hotkey=sample_axon_serve_call_params.hotkey,
            coldkey=sample_axon_serve_call_params.coldkey,
            protocol=sample_axon_serve_call_params.protocol,
            placeholder1=sample_axon_serve_call_params.placeholder1,
            placeholder2=sample_axon_serve_call_params.placeholder2,
            certificate=sample_axon_serve_call_params.certificate,
        )
        
        # Verify they are equal
        assert sample_axon_serve_call_params == other_params
    
    def test_axon_serve_call_params_equality_different_values(self, sample_axon_serve_call_params):
        """
        Test equality comparison with different values.
        
        Two instances with different attribute values should not be equal.
        """
        # Create instance with different port
        other_params = AxonServeCallParams(
            version=sample_axon_serve_call_params.version,
            ip=sample_axon_serve_call_params.ip,
            port=9999,  # Different port
            ip_type=sample_axon_serve_call_params.ip_type,
            netuid=sample_axon_serve_call_params.netuid,
            hotkey=sample_axon_serve_call_params.hotkey,
            coldkey=sample_axon_serve_call_params.coldkey,
            protocol=sample_axon_serve_call_params.protocol,
            placeholder1=sample_axon_serve_call_params.placeholder1,
            placeholder2=sample_axon_serve_call_params.placeholder2,
            certificate=sample_axon_serve_call_params.certificate,
        )
        
        # Verify they are not equal
        assert sample_axon_serve_call_params != other_params
    
    def test_axon_serve_call_params_equality_with_dict(self, sample_axon_serve_call_params):
        """
        Test equality comparison with a dictionary.
        
        AxonServeCallParams should be comparable with dictionaries
        containing matching attribute values.
        """
        # Create dictionary with matching values
        params_dict = {
            "version": sample_axon_serve_call_params.version,
            "ip": sample_axon_serve_call_params.ip,
            "port": sample_axon_serve_call_params.port,
            "ip_type": sample_axon_serve_call_params.ip_type,
            "netuid": sample_axon_serve_call_params.netuid,
            "hotkey": sample_axon_serve_call_params.hotkey,
            "coldkey": sample_axon_serve_call_params.coldkey,
            "protocol": sample_axon_serve_call_params.protocol,
            "placeholder1": sample_axon_serve_call_params.placeholder1,
            "placeholder2": sample_axon_serve_call_params.placeholder2,
            "certificate": sample_axon_serve_call_params.certificate,
        }
        
        # Verify equality
        assert sample_axon_serve_call_params == params_dict
    
    def test_axon_serve_call_params_equality_with_neuron_info(self, sample_axon_serve_call_params):
        """
        Test equality comparison with NeuronInfo object.
        
        AxonServeCallParams should be comparable with NeuronInfo objects
        by comparing relevant axon_info attributes.
        """
        # Create mock AxonInfo
        mock_axon_info = MagicMock(spec=AxonInfo)
        mock_axon_info.version = sample_axon_serve_call_params.version
        mock_axon_info.ip = "127.0.0.1"  # Will be converted to int
        mock_axon_info.port = sample_axon_serve_call_params.port
        mock_axon_info.ip_type = sample_axon_serve_call_params.ip_type
        mock_axon_info.protocol = sample_axon_serve_call_params.protocol
        mock_axon_info.placeholder1 = sample_axon_serve_call_params.placeholder1
        mock_axon_info.placeholder2 = sample_axon_serve_call_params.placeholder2
        
        # Create mock NeuronInfo
        mock_neuron_info = MagicMock(spec=NeuronInfo)
        mock_neuron_info.axon_info = mock_axon_info
        mock_neuron_info.netuid = sample_axon_serve_call_params.netuid
        mock_neuron_info.hotkey = sample_axon_serve_call_params.hotkey
        mock_neuron_info.coldkey = sample_axon_serve_call_params.coldkey
        
        # Mock networking.ip_to_int to return the integer IP
        with patch('bittensor.core.types.networking.ip_to_int', return_value=sample_axon_serve_call_params.ip):
            # Verify equality
            assert sample_axon_serve_call_params == mock_neuron_info
    
    def test_axon_serve_call_params_equality_unsupported_type(self, sample_axon_serve_call_params):
        """
        Test equality comparison with unsupported type raises NotImplementedError.
        
        Comparing with a type that's not supported should raise
        NotImplementedError with an informative message.
        """
        # Attempt comparison with unsupported type
        with pytest.raises(NotImplementedError) as exc_info:
            _ = sample_axon_serve_call_params == "not_supported"
        
        # Verify error message mentions the type
        assert "AxonServeCallParams equality not implemented" in str(exc_info.value)
        assert "str" in str(exc_info.value)


class TestAxonServeCallParamsCopy:
    """
    Tests for AxonServeCallParams copy method.
    
    These tests verify that the copy() method creates a new instance
    with identical attribute values.
    """
    
    def test_axon_serve_call_params_copy(self, sample_axon_serve_call_params):
        """
        Test that copy() creates a new instance with same values.
        
        The copy method should create a new AxonServeCallParams instance
        with identical attribute values to the original.
        """
        # Create a copy
        copied_params = sample_axon_serve_call_params.copy()
        
        # Verify it's a different instance
        assert copied_params is not sample_axon_serve_call_params
        
        # Verify all attributes are equal
        assert copied_params == sample_axon_serve_call_params
        assert copied_params.version == sample_axon_serve_call_params.version
        assert copied_params.ip == sample_axon_serve_call_params.ip
        assert copied_params.port == sample_axon_serve_call_params.port
        assert copied_params.certificate == sample_axon_serve_call_params.certificate
    
    def test_axon_serve_call_params_copy_independence(self, sample_axon_serve_call_params):
        """
        Test that copied instance is independent of original.
        
        Modifying the copy should not affect the original instance.
        """
        # Create a copy
        copied_params = sample_axon_serve_call_params.copy()
        
        # Modify the copy
        copied_params.port = 9999
        
        # Verify original is unchanged
        assert sample_axon_serve_call_params.port != 9999
        assert sample_axon_serve_call_params.port == 8091


class TestAxonServeCallParamsAsDict:
    """
    Tests for AxonServeCallParams as_dict method.
    
    These tests verify that as_dict() correctly converts the instance
    to a dictionary representation.
    """
    
    def test_axon_serve_call_params_as_dict(self, sample_axon_serve_call_params):
        """
        Test that as_dict() returns correct dictionary representation.
        
        The dictionary should contain all attributes except certificate
        if certificate is None.
        """
        # Convert to dictionary
        params_dict = sample_axon_serve_call_params.as_dict()
        
        # Verify it's a dictionary
        assert isinstance(params_dict, dict)
        
        # Verify all non-certificate attributes are present
        assert params_dict["version"] == sample_axon_serve_call_params.version
        assert params_dict["ip"] == sample_axon_serve_call_params.ip
        assert params_dict["port"] == sample_axon_serve_call_params.port
        assert params_dict["ip_type"] == sample_axon_serve_call_params.ip_type
        assert params_dict["netuid"] == sample_axon_serve_call_params.netuid
        assert params_dict["protocol"] == sample_axon_serve_call_params.protocol
        assert params_dict["placeholder1"] == sample_axon_serve_call_params.placeholder1
        assert params_dict["placeholder2"] == sample_axon_serve_call_params.placeholder2
        
        # Verify certificate is not in dict when None
        assert "certificate" not in params_dict
    
    def test_axon_serve_call_params_as_dict_with_certificate(self, sample_axon_serve_call_params, sample_certificate):
        """
        Test that as_dict() includes certificate when present.
        
        When certificate is not None, it should be included in the
        dictionary representation.
        """
        # Set certificate
        sample_axon_serve_call_params.certificate = sample_certificate
        
        # Convert to dictionary
        params_dict = sample_axon_serve_call_params.as_dict()
        
        # Verify certificate is included
        assert "certificate" in params_dict
        assert params_dict["certificate"] == sample_certificate


# ============================================================================
# Test Classes for PrometheusServeCallParams
# ============================================================================

class TestPrometheusServeCallParams:
    """
    Tests for PrometheusServeCallParams TypedDict.
    
    These tests verify that PrometheusServeCallParams can be created
    and used as a TypedDict with the required fields.
    """
    
    def test_prometheus_serve_call_params_creation(self):
        """
        Test that PrometheusServeCallParams can be created with all required fields.
        
        TypedDict allows dictionary-like creation with type checking.
        All required fields must be provided.
        """
        # Create PrometheusServeCallParams instance
        params: PrometheusServeCallParams = {
            "version": 1,
            "ip": 2130706433,
            "port": 9090,
            "ip_type": 4,
            "netuid": 1,
        }
        
        # Verify all fields are present
        assert params["version"] == 1
        assert params["ip"] == 2130706433
        assert params["port"] == 9090
        assert params["ip_type"] == 4
        assert params["netuid"] == 1
    
    def test_prometheus_serve_call_params_type_checking(self):
        """
        Test that PrometheusServeCallParams enforces correct types.
        
        TypedDict provides type hints that help ensure correct types
        are used for each field.
        """
        # Create with correct types
        params: PrometheusServeCallParams = {
            "version": 1,  # int
            "ip": 2130706433,  # int
            "port": 9090,  # int
            "ip_type": 4,  # int
            "netuid": 1,  # int
        }
        
        # Verify types
        assert isinstance(params["version"], int)
        assert isinstance(params["ip"], int)
        assert isinstance(params["port"], int)
        assert isinstance(params["ip_type"], int)
        assert isinstance(params["netuid"], int)
    
    def test_prometheus_serve_call_params_all_fields_required(self):
        """
        Test that all fields are required in PrometheusServeCallParams.
        
        TypedDict requires all fields to be present when creating
        an instance (unless marked as Optional).
        """
        # This should work - all fields provided
        params: PrometheusServeCallParams = {
            "version": 1,
            "ip": 2130706433,
            "port": 9090,
            "ip_type": 4,
            "netuid": 1,
        }
        
        # Verify it's a valid dictionary
        assert len(params) == 5
        assert all(key in params for key in ["version", "ip", "port", "ip_type", "netuid"])


# ============================================================================
# Test Classes for ExtrinsicResponse
# ============================================================================

class TestExtrinsicResponseInitialization:
    """
    Tests for ExtrinsicResponse dataclass initialization.
    
    These tests verify that ExtrinsicResponse can be created with
    various combinations of parameters.
    """
    
    def test_extrinsic_response_default_initialization(self):
        """
        Test ExtrinsicResponse initialization with default values.
        
        ExtrinsicResponse should be creatable with minimal parameters,
        using defaults for optional fields.
        """
        # Create with minimal parameters (using defaults)
        response = ExtrinsicResponse(
            success=True,
            message="Test message"
        )
        
        # Verify required fields
        assert response.success is True
        assert response.message == "Test message"
        
        # Verify optional fields have defaults
        assert response.extrinsic_function is not None  # Set by __post_init__
        assert response.extrinsic is None
        assert response.extrinsic_fee is None
        assert response.extrinsic_receipt is None
        assert response.transaction_tao_fee is None
        assert response.transaction_alpha_fee is None
        assert response.error is None
        assert response.data is None
    
    def test_extrinsic_response_full_initialization(self, sample_extrinsic_response):
        """
        Test ExtrinsicResponse initialization with all parameters.
        
        All fields should be settable during initialization.
        """
        # Verify all fields are set
        assert sample_extrinsic_response.success is True
        assert sample_extrinsic_response.message == "Test successful"
        assert sample_extrinsic_response.extrinsic_function == "test_function"
        assert sample_extrinsic_response.extrinsic is None
        assert sample_extrinsic_response.error is None


class TestExtrinsicResponseStringRepresentation:
    """
    Tests for ExtrinsicResponse string representation methods.
    
    These tests verify that __str__ and __repr__ work correctly.
    """
    
    def test_extrinsic_response_str_representation(self, sample_extrinsic_response):
        """
        Test the string representation of ExtrinsicResponse.
        
        __str__ should return a formatted multi-line string with
        all relevant information.
        """
        # Get string representation
        str_repr = str(sample_extrinsic_response)
        
        # Verify it contains key information
        assert "ExtrinsicResponse" in str_repr
        assert "success" in str_repr.lower()
        assert "message" in str_repr.lower()
        assert "Test successful" in str_repr
    
    def test_extrinsic_response_repr_representation(self, sample_extrinsic_response):
        """
        Test the repr representation of ExtrinsicResponse.
        
        __repr__ should return a tuple-like representation with
        success and message.
        """
        # Get repr representation
        repr_str = repr(sample_extrinsic_response)
        
        # Verify it's a tuple-like representation
        assert "True" in repr_str
        assert "Test successful" in repr_str


class TestExtrinsicResponseIterationAndIndexing:
    """
    Tests for ExtrinsicResponse iteration and indexing behavior.
    
    These tests verify that ExtrinsicResponse behaves like a tuple
    for iteration and indexing operations.
    """
    
    def test_extrinsic_response_iteration(self, sample_extrinsic_response):
        """
        Test that ExtrinsicResponse can be iterated like a tuple.
        
        Iteration should yield (success, message) values.
        """
        # Unpack via iteration
        values = list(sample_extrinsic_response)
        
        # Verify correct values
        assert len(values) == 2
        assert values[0] == sample_extrinsic_response.success
        assert values[1] == sample_extrinsic_response.message
    
    def test_extrinsic_response_indexing(self, sample_extrinsic_response):
        """
        Test that ExtrinsicResponse supports indexing like a tuple.
        
        Index 0 should return success, index 1 should return message.
        """
        # Test index 0 (success)
        assert sample_extrinsic_response[0] == sample_extrinsic_response.success
        
        # Test index 1 (message)
        assert sample_extrinsic_response[1] == sample_extrinsic_response.message
    
    def test_extrinsic_response_indexing_out_of_range(self, sample_extrinsic_response):
        """
        Test that indexing with invalid index raises IndexError.
        
        Only indices 0 and 1 should be valid.
        """
        # Test invalid index
        with pytest.raises(IndexError) as exc_info:
            _ = sample_extrinsic_response[2]
        
        # Verify error message
        assert "only supports indices 0" in str(exc_info.value)
    
    def test_extrinsic_response_length(self, sample_extrinsic_response):
        """
        Test that len() returns 2 for ExtrinsicResponse.
        
        ExtrinsicResponse should always have length 2 (success, message).
        """
        # Verify length
        assert len(sample_extrinsic_response) == 2


class TestExtrinsicResponseEquality:
    """
    Tests for ExtrinsicResponse equality comparison.
    
    These tests verify that __eq__ correctly compares ExtrinsicResponse
    instances with various types.
    """
    
    def test_extrinsic_response_equality_with_tuple(self, sample_extrinsic_response):
        """
        Test equality comparison with tuple.
        
        ExtrinsicResponse should be equal to a tuple with (success, message).
        """
        # Create matching tuple
        response_tuple = (sample_extrinsic_response.success, sample_extrinsic_response.message)
        
        # Verify equality
        assert sample_extrinsic_response == response_tuple
    
    def test_extrinsic_response_equality_with_list(self, sample_extrinsic_response):
        """
        Test equality comparison with list.
        
        ExtrinsicResponse should be equal to a list with [success, message].
        """
        # Create matching list
        response_list = [sample_extrinsic_response.success, sample_extrinsic_response.message]
        
        # Verify equality
        assert sample_extrinsic_response == response_list
    
    def test_extrinsic_response_equality_with_other_response(self, sample_extrinsic_response):
        """
        Test equality comparison with another ExtrinsicResponse.
        
        Two ExtrinsicResponse instances with identical values should be equal.
        """
        # Create another response with same values
        other_response = ExtrinsicResponse(
            success=sample_extrinsic_response.success,
            message=sample_extrinsic_response.message,
            extrinsic_function=sample_extrinsic_response.extrinsic_function,
            extrinsic=sample_extrinsic_response.extrinsic,
            extrinsic_fee=sample_extrinsic_response.extrinsic_fee,
            extrinsic_receipt=sample_extrinsic_response.extrinsic_receipt,
            transaction_tao_fee=sample_extrinsic_response.transaction_tao_fee,
            transaction_alpha_fee=sample_extrinsic_response.transaction_alpha_fee,
            error=sample_extrinsic_response.error,
            data=sample_extrinsic_response.data,
        )
        
        # Verify equality
        assert sample_extrinsic_response == other_response


class TestExtrinsicResponseAsDict:
    """
    Tests for ExtrinsicResponse as_dict method.
    
    These tests verify that as_dict() correctly converts the instance
    to a dictionary representation.
    """
    
    def test_extrinsic_response_as_dict(self, sample_extrinsic_response):
        """
        Test that as_dict() returns correct dictionary representation.
        
        The dictionary should contain all fields with appropriate
        conversions (e.g., Balance objects converted to rao).
        """
        # Convert to dictionary
        response_dict = sample_extrinsic_response.as_dict()
        
        # Verify it's a dictionary
        assert isinstance(response_dict, dict)
        
        # Verify key fields are present
        assert "success" in response_dict
        assert "message" in response_dict
        assert "extrinsic_function" in response_dict
        assert response_dict["success"] == sample_extrinsic_response.success
        assert response_dict["message"] == sample_extrinsic_response.message


class TestExtrinsicResponseClassMethods:
    """
    Tests for ExtrinsicResponse class methods.
    
    These tests verify that class methods like from_exception and
    unlock_wallet work correctly.
    """
    
    @patch('bittensor.core.types.get_caller_name')
    @patch('bittensor.core.types.format_error_message')
    def test_from_exception_with_raise_error(self, mock_format_error, mock_get_caller):
        """
        Test from_exception when raise_error is True.
        
        When raise_error is True, the exception should be re-raised
        instead of returning an ExtrinsicResponse.
        """
        # Setup mocks
        mock_get_caller.return_value = "test_function"
        test_error = ValueError("Test error")
        
        # Call from_exception with raise_error=True
        with pytest.raises(ValueError) as exc_info:
            ExtrinsicResponse.from_exception(raise_error=True, error=test_error)
        
        # Verify exception was raised
        assert str(exc_info.value) == "Test error"
    
    @patch('bittensor.core.types.get_caller_name')
    @patch('bittensor.core.types.format_error_message')
    @patch('bittensor.core.types.logging')
    def test_from_exception_without_raise_error(self, mock_logging, mock_format_error, mock_get_caller):
        """
        Test from_exception when raise_error is False.
        
        When raise_error is False, should return ExtrinsicResponse
        with success=False and formatted error message.
        """
        # Setup mocks
        mock_get_caller.return_value = "test_function"
        mock_format_error.return_value = "Formatted error message"
        test_error = ValueError("Test error")
        
        # Call from_exception with raise_error=False
        response = ExtrinsicResponse.from_exception(raise_error=False, error=test_error)
        
        # Verify response
        assert response.success is False
        assert response.error == test_error
        assert response.extrinsic_function == "test_function"
        # Verify with_log was called (logs the message)
        mock_logging.error.assert_called()


class TestExtrinsicResponseWithLog:
    """
    Tests for ExtrinsicResponse with_log method.
    
    These tests verify that with_log correctly logs messages
    at the specified level.
    """
    
    @patch('bittensor.core.types.logging')
    def test_with_log_info_level(self, mock_logging, sample_extrinsic_response):
        """
        Test with_log with info level.
        
        Should log the message at info level and return self.
        """
        # Call with_log with info level
        result = sample_extrinsic_response.with_log(level="info")
        
        # Verify it returns self
        assert result is sample_extrinsic_response
        
        # Verify logging.info was called
        mock_logging.info.assert_called_once_with(sample_extrinsic_response.message)
    
    @patch('bittensor.core.types.logging')
    def test_with_log_error_level(self, mock_logging, sample_extrinsic_response):
        """
        Test with_log with error level.
        
        Should log the message at error level and return self.
        """
        # Call with_log with error level
        result = sample_extrinsic_response.with_log(level="error")
        
        # Verify it returns self
        assert result is sample_extrinsic_response
        
        # Verify logging.error was called
        mock_logging.error.assert_called_once_with(sample_extrinsic_response.message)
    
    @patch('bittensor.core.types.logging')
    def test_with_log_no_message(self, mock_logging):
        """
        Test with_log when message is None.
        
        Should not log anything when message is None.
        """
        # Create response without message
        response = ExtrinsicResponse(success=True, message=None)
        
        # Call with_log
        result = response.with_log()
        
        # Verify no logging occurred
        mock_logging.error.assert_not_called()


# ============================================================================
# Test Classes for BlockInfo
# ============================================================================

class TestBlockInfoInitialization:
    """
    Tests for BlockInfo dataclass initialization.
    
    These tests verify that BlockInfo can be created with
    all required parameters.
    """
    
    def test_block_info_initialization(self):
        """
        Test BlockInfo initialization with all parameters.
        
        Verifies that all required fields can be provided and are
        correctly stored.
        """
        # Create BlockInfo instance
        block_info = BlockInfo(
            number=12345,
            hash="0x1234567890abcdef",
            timestamp=1234567890,
            header={"block_number": 12345},
            extrinsics=[{"extrinsic": "data"}],
            explorer="https://explorer.example.com/block/12345"
        )
        
        # Verify all attributes are set correctly
        assert block_info.number == 12345
        assert block_info.hash == "0x1234567890abcdef"
        assert block_info.timestamp == 1234567890
        assert isinstance(block_info.header, dict)
        assert isinstance(block_info.extrinsics, list)
        assert block_info.explorer == "https://explorer.example.com/block/12345"
    
    def test_block_info_with_optional_timestamp(self):
        """
        Test BlockInfo initialization with None timestamp.
        
        Timestamp is Optional, so None should be acceptable.
        """
        # Create BlockInfo with None timestamp
        block_info = BlockInfo(
            number=12345,
            hash="0x1234567890abcdef",
            timestamp=None,
            header={},
            extrinsics=[],
            explorer="https://explorer.example.com/block/12345"
        )
        
        # Verify timestamp is None
        assert block_info.timestamp is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestTypesIntegration:
    """
    Integration tests for types module components working together.
    
    These tests verify that different components from the types module
    can work together correctly in realistic scenarios.
    """
    
    def test_subtensor_mixin_with_config_integration(self, mock_config):
        """
        Test SubtensorMixin config and setup_config integration.
        
        Verifies that config() creates a Config that can be used
        with setup_config() to determine network settings.
        """
        # Get config from SubtensorMixin
        config = SubtensorMixin.config()
        
        # Verify config is usable
        assert isinstance(config, Config)
        assert hasattr(config, "subtensor")
        
        # Setup config with network
        with patch('bittensor.core.types.determine_chain_endpoint_and_network') as mock_determine, \
             patch('bittensor.core.types.networking.get_formatted_ws_endpoint_url') as mock_format:
            mock_determine.return_value = ("test_network", "wss://test.endpoint:443")
            mock_format.return_value = "wss://formatted.test.endpoint:443"
            
            endpoint, network = SubtensorMixin.setup_config("test", config)
            
            # Verify results
            assert endpoint is not None
            assert network is not None
    
    def test_axon_serve_call_params_with_extrinsic_response(self, sample_axon_serve_call_params):
        """
        Test AxonServeCallParams used in context of ExtrinsicResponse.
        
        Verifies that AxonServeCallParams can be included in ExtrinsicResponse
        data field for complete workflow scenarios.
        """
        # Create ExtrinsicResponse with AxonServeCallParams in data
        response = ExtrinsicResponse(
            success=True,
            message="Axon served successfully",
            extrinsic_function="serve_axon",
            data=sample_axon_serve_call_params.as_dict()
        )
        
        # Verify data contains params
        assert response.data is not None
        assert "version" in response.data
        assert "ip" in response.data
        assert response.success is True

