"""
Unit tests for bittensor/core/settings.py

Tests all configuration constants, network mappings, environment variable handling,
and default configurations used throughout the Bittensor codebase.
"""

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from munch import Munch

from bittensor.core import settings


class TestNetworkConstants:
    """Test network-related constants and mappings."""

    def test_network_constants(self):
        """Verify NETWORKS list contains all expected networks."""
        assert settings.NETWORKS == ["finney", "test", "archive", "local", "latent-lite"]
        assert len(settings.NETWORKS) == 5
        assert "finney" in settings.NETWORKS
        assert "test" in settings.NETWORKS
        assert "archive" in settings.NETWORKS
        assert "local" in settings.NETWORKS
        assert "latent-lite" in settings.NETWORKS

    def test_network_map_completeness(self):
        """Ensure all networks have corresponding endpoints in NETWORK_MAP."""
        for network in settings.NETWORKS:
            assert network in settings.NETWORK_MAP, f"Network '{network}' missing from NETWORK_MAP"
            endpoint = settings.NETWORK_MAP[network]
            assert endpoint is not None
            assert isinstance(endpoint, str)
            # Verify endpoint format (should be ws:// or wss://)
            assert endpoint.startswith("ws://") or endpoint.startswith("wss://")

    def test_reverse_network_map_consistency(self):
        """Verify bidirectional mapping between NETWORK_MAP and REVERSE_NETWORK_MAP."""
        # Check that every entry in NETWORK_MAP has a reverse entry
        for network, endpoint in settings.NETWORK_MAP.items():
            assert endpoint in settings.REVERSE_NETWORK_MAP, \
                f"Endpoint '{endpoint}' for network '{network}' missing from REVERSE_NETWORK_MAP"
            assert settings.REVERSE_NETWORK_MAP[endpoint] == network, \
                f"Reverse mapping mismatch for {network} -> {endpoint}"
        
        # Check that every entry in REVERSE_NETWORK_MAP has a forward entry
        for endpoint, network in settings.REVERSE_NETWORK_MAP.items():
            assert network in settings.NETWORK_MAP, \
                f"Network '{network}' from REVERSE_NETWORK_MAP missing from NETWORK_MAP"
            assert settings.NETWORK_MAP[network] == endpoint, \
                f"Forward mapping mismatch for {endpoint} -> {network}"
        
        # Verify counts match
        assert len(settings.NETWORK_MAP) == len(settings.REVERSE_NETWORK_MAP)

    def test_default_network_and_endpoint(self):
        """Test DEFAULT_NETWORK and DEFAULT_ENDPOINT values."""
        assert settings.DEFAULT_NETWORK == "finney"
        assert settings.DEFAULT_NETWORK == settings.NETWORKS[0]
        assert settings.DEFAULT_ENDPOINT == settings.NETWORK_MAP[settings.DEFAULT_NETWORK]
        assert settings.DEFAULT_ENDPOINT == settings.FINNEY_ENTRYPOINT

    def test_network_explorer_map(self):
        """Test NETWORK_EXPLORER_MAP entries."""
        assert "opentensor" in settings.NETWORK_EXPLORER_MAP
        assert "taostats" in settings.NETWORK_EXPLORER_MAP
        
        # Check opentensor explorer entries
        opentensor = settings.NETWORK_EXPLORER_MAP["opentensor"]
        assert "local" in opentensor
        assert "endpoint" in opentensor
        assert "finney" in opentensor
        assert all(isinstance(url, str) for url in opentensor.values())
        assert all(url.startswith("https://") for url in opentensor.values())
        
        # Check taostats explorer entries
        taostats = settings.NETWORK_EXPLORER_MAP["taostats"]
        assert "local" in taostats
        assert "endpoint" in taostats
        assert "finney" in taostats
        assert all(isinstance(url, str) for url in taostats.values())


class TestEnvironmentVariables:
    """Test environment variable handling and overrides."""

    def test_environment_variable_overrides(self):
        """Test LOCAL_ENTRYPOINT can be overridden by BT_SUBTENSOR_CHAIN_ENDPOINT env var."""
        # Test default value when env var is not set
        with patch.dict(os.environ, {}, clear=False):
            # Remove the env var if it exists
            os.environ.pop("BT_SUBTENSOR_CHAIN_ENDPOINT", None)
            # Re-import to get fresh value
            import importlib
            importlib.reload(settings)
            # Should default to local endpoint
            assert "127.0.0.1:9944" in settings.LOCAL_ENTRYPOINT or \
                   settings.LOCAL_ENTRYPOINT == "ws://127.0.0.1:9944"

    def test_read_only_mode(self):
        """Test READ_ONLY environment variable behavior."""
        # Test when READ_ONLY is "1"
        with patch.dict(os.environ, {"READ_ONLY": "1"}):
            import importlib
            importlib.reload(settings)
            assert settings.READ_ONLY is True
        
        # Test when READ_ONLY is not "1"
        with patch.dict(os.environ, {"READ_ONLY": "0"}):
            import importlib
            importlib.reload(settings)
            assert settings.READ_ONLY is False
        
        # Test when READ_ONLY is not set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("READ_ONLY", None)
            import importlib
            importlib.reload(settings)
            assert settings.READ_ONLY is False


class TestDirectoryConfiguration:
    """Test directory paths and creation logic."""

    def test_directory_creation(self):
        """Test WALLETS_DIR and MINERS_DIR are properly configured."""
        assert isinstance(settings.WALLETS_DIR, Path)
        assert isinstance(settings.MINERS_DIR, Path)
        assert settings.WALLETS_DIR == settings.USER_BITTENSOR_DIR / "wallets"
        assert settings.MINERS_DIR == settings.USER_BITTENSOR_DIR / "miners"
        assert settings.USER_BITTENSOR_DIR == settings.HOME_DIR / ".bittensor"
        
        # Verify directories exist (unless READ_ONLY mode)
        if not settings.READ_ONLY:
            assert settings.WALLETS_DIR.exists()
            assert settings.MINERS_DIR.exists()


class TestVersionHandling:
    """Test version parsing and conversion."""

    def test_version_parsing(self):
        """Test __version__ extraction and parsing."""
        assert isinstance(settings.__version__, str)
        # Version should match pattern X.Y.Z
        version_pattern = re.compile(r"^\d+\.\d+\.\d+$")
        assert version_pattern.match(settings.__version__), \
            f"Version '{settings.__version__}' doesn't match X.Y.Z pattern"
        
        # Verify version components
        parts = settings.__version__.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_version_as_int_conversion(self):
        """Verify version_as_int calculation."""
        assert isinstance(settings.version_as_int, int)
        assert settings.version_as_int > 0
        assert settings.version_as_int < 2**31, "version_as_int should fit in int32"
        
        # Verify calculation matches expected formula
        version_parts = settings.__version__.split(".")
        version_info = tuple(int(part) for part in version_parts)
        expected = sum(
            e * (1000**i) for i, e in enumerate(reversed(version_info))
        )
        assert settings.version_as_int == expected


class TestConstants:
    """Test various constant values."""

    def test_ss58_format_constant(self):
        """Verify SS58_FORMAT = 42 (from bittensor_wallet)."""
        # SS58_FORMAT is imported from bittensor_wallet, verify it's 42
        from bittensor_wallet.utils import SS58_FORMAT
        assert SS58_FORMAT == 42

    def test_blocktime_constant(self):
        """Verify BLOCKTIME = 12."""
        assert settings.BLOCKTIME == 12
        assert isinstance(settings.BLOCKTIME, int)

    def test_currency_symbols(self):
        """Test TAO_SYMBOL and RAO_SYMBOL."""
        assert settings.TAO_SYMBOL == chr(0x03C4)  # Greek letter tau (τ)
        assert settings.RAO_SYMBOL == chr(0x03C1)  # Greek letter rho (ρ)
        assert isinstance(settings.TAO_SYMBOL, str)
        assert isinstance(settings.RAO_SYMBOL, str)
        assert len(settings.TAO_SYMBOL) == 1
        assert len(settings.RAO_SYMBOL) == 1

    def test_ss58_address_length(self):
        """Test SS58_ADDRESS_LENGTH constant."""
        assert settings.SS58_ADDRESS_LENGTH == 48
        assert isinstance(settings.SS58_ADDRESS_LENGTH, int)

    def test_default_period(self):
        """Test DEFAULT_PERIOD constant for extrinsics Era."""
        assert settings.DEFAULT_PERIOD == 128
        assert isinstance(settings.DEFAULT_PERIOD, int)

    def test_root_tao_stake_weight(self):
        """Test ROOT_TAO_STAKE_WEIGHT constant."""
        assert settings.ROOT_TAO_STAKE_WEIGHT == 0.18
        assert isinstance(settings.ROOT_TAO_STAKE_WEIGHT, float)


class TestTypeRegistry:
    """Test type registry configuration."""

    def test_type_registry_balance_override(self):
        """Verify Balance type is u64 not u128."""
        assert "types" in settings.TYPE_REGISTRY
        assert "Balance" in settings.TYPE_REGISTRY["types"]
        assert settings.TYPE_REGISTRY["types"]["Balance"] == "u64"
        # Explicitly verify it's NOT u128 (the default)
        assert settings.TYPE_REGISTRY["types"]["Balance"] != "u128"


class TestDefaultsStructure:
    """Test DEFAULTS munch object structure and values."""

    def test_defaults_structure(self):
        """Test DEFAULTS munch object structure."""
        assert isinstance(settings.DEFAULTS, Munch)
        assert "axon" in settings.DEFAULTS
        assert "logging" in settings.DEFAULTS
        assert "priority" in settings.DEFAULTS
        assert "subtensor" in settings.DEFAULTS
        assert "wallet" in settings.DEFAULTS
        assert "config" in settings.DEFAULTS
        assert "strict" in settings.DEFAULTS
        assert "no_version_checking" in settings.DEFAULTS

    def test_defaults_axon_config(self):
        """Verify axon default values."""
        axon = settings.DEFAULTS.axon
        assert isinstance(axon, Munch)
        
        # Check default port (unless overridden by env var)
        assert "port" in axon
        assert isinstance(axon.port, int)
        if not os.getenv("BT_AXON_PORT"):
            assert axon.port == 8091
        
        # Check default IP
        assert "ip" in axon
        if not os.getenv("BT_AXON_IP"):
            assert axon.ip == "[::]"
        
        # Check external port and IP
        assert "external_port" in axon
        assert "external_ip" in axon
        
        # Check max_workers
        assert "max_workers" in axon
        assert isinstance(axon.max_workers, int)
        if not os.getenv("BT_AXON_MAX_WORKERS"):
            assert axon.max_workers == 10

    def test_defaults_logging_config(self):
        """Verify logging default values."""
        logging = settings.DEFAULTS.logging
        assert isinstance(logging, Munch)
        
        # Check boolean flags
        assert "debug" in logging
        assert "trace" in logging
        assert "info" in logging
        assert "record_log" in logging
        assert isinstance(logging.debug, bool)
        assert isinstance(logging.trace, bool)
        assert isinstance(logging.info, bool)
        assert isinstance(logging.record_log, bool)
        
        # Check logging_dir
        assert "logging_dir" in logging
        if settings.READ_ONLY:
            assert logging.logging_dir is None
        elif not os.getenv("BT_LOGGING_LOGGING_DIR"):
            assert logging.logging_dir == str(settings.MINERS_DIR)
        
        # Check enable_third_party_loggers
        assert "enable_third_party_loggers" in logging

    def test_defaults_priority_config(self):
        """Verify priority threadpool defaults."""
        priority = settings.DEFAULTS.priority
        assert isinstance(priority, Munch)
        
        # Check max_workers
        assert "max_workers" in priority
        assert isinstance(priority.max_workers, int)
        if not os.getenv("BT_PRIORITY_MAX_WORKERS"):
            assert priority.max_workers == 5
        
        # Check maxsize
        assert "maxsize" in priority
        assert isinstance(priority.maxsize, int)
        if not os.getenv("BT_PRIORITY_MAXSIZE"):
            assert priority.maxsize == 10

    def test_defaults_subtensor_config(self):
        """Verify subtensor defaults."""
        subtensor = settings.DEFAULTS.subtensor
        assert isinstance(subtensor, Munch)
        
        # Check chain_endpoint
        assert "chain_endpoint" in subtensor
        if not os.getenv("BT_SUBTENSOR_CHAIN_ENDPOINT"):
            assert subtensor.chain_endpoint == settings.DEFAULT_ENDPOINT
        
        # Check network
        assert "network" in subtensor
        if not os.getenv("BT_SUBTENSOR_NETWORK"):
            assert subtensor.network == settings.DEFAULT_NETWORK
        
        # Check _mock flag
        assert "_mock" in subtensor
        assert subtensor._mock is False

    def test_defaults_wallet_config(self):
        """Verify wallet defaults."""
        wallet = settings.DEFAULTS.wallet
        assert isinstance(wallet, Munch)
        
        # Check name
        assert "name" in wallet
        if not os.getenv("BT_WALLET_NAME"):
            assert wallet.name == "default"
        
        # Check hotkey
        assert "hotkey" in wallet
        if not os.getenv("BT_WALLET_HOTKEY"):
            assert wallet.hotkey == "default"
        
        # Check path
        assert "path" in wallet
        if not os.getenv("BT_WALLET_PATH"):
            assert wallet.path == str(settings.WALLETS_DIR)

    def test_defaults_global_flags(self):
        """Test global default flags."""
        assert settings.DEFAULTS.config is False
        assert settings.DEFAULTS.strict is False
        assert settings.DEFAULTS.no_version_checking is False


class TestEndpointConstants:
    """Test individual endpoint constants."""

    def test_finney_entrypoint(self):
        """Test FINNEY_ENTRYPOINT constant."""
        assert settings.FINNEY_ENTRYPOINT == "wss://entrypoint-finney.opentensor.ai:443"
        assert settings.FINNEY_ENTRYPOINT.startswith("wss://")

    def test_finney_test_entrypoint(self):
        """Test FINNEY_TEST_ENTRYPOINT constant."""
        assert settings.FINNEY_TEST_ENTRYPOINT == "wss://test.finney.opentensor.ai:443"
        assert settings.FINNEY_TEST_ENTRYPOINT.startswith("wss://")

    def test_archive_entrypoint(self):
        """Test ARCHIVE_ENTRYPOINT constant."""
        assert settings.ARCHIVE_ENTRYPOINT == "wss://archive.chain.opentensor.ai:443"
        assert settings.ARCHIVE_ENTRYPOINT.startswith("wss://")

    def test_latent_lite_entrypoint(self):
        """Test LATENT_LITE_ENTRYPOINT constant."""
        assert settings.LATENT_LITE_ENTRYPOINT == "wss://lite.sub.latent.to:443"
        assert settings.LATENT_LITE_ENTRYPOINT.startswith("wss://")


class TestMiscellaneousConstants:
    """Test miscellaneous constants."""

    def test_pipaddress(self):
        """Test PIPADDRESS constant."""
        assert settings.PIPADDRESS == "https://pypi.org/pypi/bittensor/json"
        assert settings.PIPADDRESS.startswith("https://")

    def test_tao_app_block_explorer(self):
        """Test TAO_APP_BLOCK_EXPLORER constant."""
        assert settings.TAO_APP_BLOCK_EXPLORER == "https://www.tao.app/block/"
        assert settings.TAO_APP_BLOCK_EXPLORER.startswith("https://")
