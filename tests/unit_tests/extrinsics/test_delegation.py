import pytest
from unittest.mock import MagicMock, patch
from bittensor.subtensor import Subtensor
from bittensor.wallet import wallet as Wallet
from bittensor.utils.balance import Balance
from bittensor.errors import (
    NominationError,
    NotDelegateError,
    NotRegisteredError,
    StakeError,
)


@pytest.fixture
def mock_subtensor():
    mock = MagicMock(spec=Subtensor)
    mock.network = "magic_mock"
    return mock


@pytest.fixture
def mock_wallet():
    mock = MagicMock(spec=Wallet)
    mock.hotkey.ss58_address = "fake_hotkey_address"
    mock.coldkey.ss58_address = "fake_coldkey_address"
    mock.coldkey = MagicMock()
    mock.hotkey = MagicMock()
    mock.name = "fake_wallet_name"
    mock.hotkey_str = "fake_hotkey_str"
    return mock
