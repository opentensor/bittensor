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

    def test_network_map_completeness(self):
        """Ensure all networks have corresponding endpoints in NETWORK_MAP."""
        for network in settings.NETWORKS:
            assert network in settings.NETWORK_MAP
            endpoint = settings.NETWORK_MAP[network]
            assert endpoint.startswith("ws://") or endpoint.startswith("wss://")

    def test_reverse_network_map_consistency(self):
        """Verify bidirectional mapping between NETWORK_MAP and REVERSE_NETWORK_MAP."""
        for network, endpoint in settings.NETWORK_MAP.items():
            assert settings.REVERSE_NETWORK_MAP[endpoint] == network
        assert len(settings.NETWORK_MAP) == len(settings.REVERSE_NETWORK_MAP)

    def test_default_network_and_endpoint(self):
        """Test DEFAULT_NETWORK and DEFAULT_ENDPOINT values."""
        assert settings.DEFAULT_NETWORK == "finney"
        assert settings.DEFAULT_ENDPOINT == settings.NETWORK_MAP[settings.DEFAULT_NETWORK]


class TestEnvironmentVariables:
    """Test environment variable handling and overrides."""

    def test_read_only_mode(self):
        """Test READ_ONLY environment variable behavior."""
        import importlib
        
        with patch.dict(os.environ, {"READ_ONLY": "1"}):
            importlib.reload(settings)
            assert settings.READ_ONLY is True
        
        with patch.dict(os.environ, {"READ_ONLY": "0"}):
            importlib.reload(settings)
            assert settings.READ_ONLY is False


class TestDirectoryConfiguration:
    """Test directory paths and creation logic."""

    def test_directory_creation(self):
        """Test WALLETS_DIR and MINERS_DIR are properly configured."""
        assert settings.WALLETS_DIR == settings.USER_BITTENSOR_DIR / "wallets"
        assert settings.MINERS_DIR == settings.USER_BITTENSOR_DIR / "miners"
        if not settings.READ_ONLY:
            assert settings.WALLETS_DIR.exists()
            assert settings.MINERS_DIR.exists()


class TestVersionHandling:
    """Test version parsing and conversion."""

    def test_version_parsing(self):
        """Test __version__ extraction and parsing."""
        version_pattern = re.compile(r"^\d+\.\d+\.\d+$")
        assert version_pattern.match(settings.__version__)

    def test_version_as_int_conversion(self):
        """Verify version_as_int calculation."""
        version_parts = settings.__version__.split(".")
        version_info = tuple(int(part) for part in version_parts)
        expected = sum(e * (1000**i) for i, e in enumerate(reversed(version_info)))
        assert settings.version_as_int == expected


class TestConstants:
    """Test various constant values."""

    def test_blocktime_constant(self):
        """Verify BLOCKTIME = 12."""
        assert settings.BLOCKTIME == 12

    def test_currency_symbols(self):
        """Test TAO_SYMBOL and RAO_SYMBOL."""
        assert settings.TAO_SYMBOL == chr(0x03C4)
        assert settings.RAO_SYMBOL == chr(0x03C1)

    def test_ss58_address_length(self):
        """Test SS58_ADDRESS_LENGTH constant."""
        assert settings.SS58_ADDRESS_LENGTH == 48

    def test_default_period(self):
        """Test DEFAULT_PERIOD constant for extrinsics Era."""
        assert settings.DEFAULT_PERIOD == 128


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
        required_keys = ["axon", "logging", "priority", "subtensor", "wallet"]
        for key in required_keys:
            assert key in settings.DEFAULTS

    def test_defaults_axon_config(self):
        """Verify axon default values."""
        axon = settings.DEFAULTS.axon
        if not os.getenv("BT_AXON_PORT"):
            assert axon.port == 8091
        if not os.getenv("BT_AXON_IP"):
            assert axon.ip == "[::]" 
        if not os.getenv("BT_AXON_MAX_WORKERS"):
            assert axon.max_workers == 10

    def test_defaults_subtensor_config(self):
        """Verify subtensor defaults."""
        subtensor = settings.DEFAULTS.subtensor
        if not os.getenv("BT_SUBTENSOR_CHAIN_ENDPOINT"):
            assert subtensor.chain_endpoint == settings.DEFAULT_ENDPOINT
        if not os.getenv("BT_SUBTENSOR_NETWORK"):
            assert subtensor.network == settings.DEFAULT_NETWORK
        assert subtensor._mock is False

    def test_defaults_wallet_config(self):
        """Verify wallet defaults."""
        wallet = settings.DEFAULTS.wallet
        if not os.getenv("BT_WALLET_NAME"):
            assert wallet.name == "default"
        if not os.getenv("BT_WALLET_HOTKEY"):
            assert wallet.hotkey == "default"


class TestMiscellaneousConstants:
    """Test miscellaneous constants."""

    def test_pipaddress(self):
        """Test PIPADDRESS constant."""
        assert settings.PIPADDRESS == "https://pypi.org/pypi/bittensor/json"

    def test_tao_app_block_explorer(self):
        """Test TAO_APP_BLOCK_EXPLORER constant."""
        assert settings.TAO_APP_BLOCK_EXPLORER == "https://www.tao.app/block/"
