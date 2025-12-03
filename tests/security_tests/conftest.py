"""
Pytest configuration and shared fixtures for security tests.
"""

import pytest
from unittest.mock import Mock
from bittensor_wallet import Wallet, Keypair


@pytest.fixture
def test_keypair():
    """Create a test keypair"""
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


@pytest.fixture
def mock_wallet():
    """Create a mock wallet for testing"""
    wallet = Mock(spec=Wallet)
    wallet.hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet.coldkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet.coldkeypub = wallet.coldkey
    wallet.name = "test_wallet"
    return wallet


@pytest.fixture
def valid_ss58_address():
    """Return a valid SS58 address for testing"""
    return "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


@pytest.fixture
def another_valid_ss58_address():
    """Return another valid SS58 address for testing"""
    return "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "race_condition: mark test as testing race conditions"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as stress/load test"
    )
