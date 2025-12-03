"""Unit tests for bittensor.core.chain_data.delegate_info_lite module."""
import pytest
from unittest.mock import patch
from bittensor.core.chain_data.delegate_info_lite import DelegateInfoLite
from bittensor.utils.balance import Balance
MOCK_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
MOCK_OWNER = "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"
class TestDelegateInfoLiteCreation:
    """Tests for DelegateInfoLite dataclass creation."""
    def test_delegate_info_lite_basic_creation(self):
        """Test creating a DelegateInfoLite instance with valid data."""
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=0.18,
            nominators=100,
            owner_ss58=MOCK_OWNER,
            registrations=[1, 2, 3],
            validator_permits=[1, 2],
            return_per_1000=Balance.from_tao(0.5),
            total_daily_return=Balance.from_tao(100.0),
        )
        assert delegate.delegate_ss58 == MOCK_HOTKEY
        assert delegate.take == 0.18
        assert delegate.nominators == 100
        assert delegate.owner_ss58 == MOCK_OWNER
        assert delegate.registrations == [1, 2, 3]
        assert delegate.validator_permits == [1, 2]
    def test_delegate_info_lite_zero_nominators(self):
        """Test DelegateInfoLite with zero nominators."""
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=0.0,
            nominators=0,
            owner_ss58=MOCK_OWNER,
            registrations=[],
            validator_permits=[],
            return_per_1000=Balance.from_rao(0),
            total_daily_return=Balance.from_rao(0),
        )
        assert delegate.nominators == 0
        assert delegate.take == 0.0
    def test_delegate_info_lite_large_nominators(self):
        """Test DelegateInfoLite with large number of nominators."""
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=0.18,
            nominators=10000,
            owner_ss58=MOCK_OWNER,
            registrations=[1],
            validator_permits=[1],
            return_per_1000=Balance.from_tao(1.0),
            total_daily_return=Balance.from_tao(10000.0),
        )
        assert delegate.nominators == 10000
    def test_delegate_info_lite_max_take(self):
        """Test DelegateInfoLite with maximum take value."""
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=1.0,
            nominators=50,
            owner_ss58=MOCK_OWNER,
            registrations=[1],
            validator_permits=[1],
            return_per_1000=Balance.from_tao(0.1),
            total_daily_return=Balance.from_tao(50.0),
        )
        assert delegate.take == 1.0
    def test_delegate_info_lite_empty_lists(self):
        """Test DelegateInfoLite with empty registrations and permits."""
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=0.1,
            nominators=10,
            owner_ss58=MOCK_OWNER,
            registrations=[],
            validator_permits=[],
            return_per_1000=Balance.from_rao(100000000),
            total_daily_return=Balance.from_rao(1000000000),
        )
        assert delegate.registrations == []
        assert delegate.validator_permits == []
class TestDelegateInfoLiteFromDict:
    """Tests for DelegateInfoLite._from_dict method."""
    @patch("bittensor.core.chain_data.delegate_info_lite.decode_account_id")
    def test_from_dict_basic(self, mock_decode):
        """Test _from_dict with basic valid data."""
        mock_decode.side_effect = lambda x: x
        decoded = {
            "delegate_ss58": MOCK_HOTKEY,
            "take": 11796,
            "nominators": 100,
            "owner_ss58": MOCK_OWNER,
            "registrations": [1, 2],
            "validator_permits": [1],
            "return_per_1000": 500000000,
            "total_daily_return": 100000000000,
        }
        result = DelegateInfoLite._from_dict(decoded)
        assert result.delegate_ss58 == MOCK_HOTKEY
        assert result.nominators == 100
        assert result.owner_ss58 == MOCK_OWNER
        assert result.registrations == [1, 2]
        assert result.validator_permits == [1]
    @patch("bittensor.core.chain_data.delegate_info_lite.decode_account_id")
    def test_from_dict_zero_values(self, mock_decode):
        """Test _from_dict with zero values."""
        mock_decode.side_effect = lambda x: x
        decoded = {
            "delegate_ss58": MOCK_HOTKEY,
            "take": 0,
            "nominators": 0,
            "owner_ss58": MOCK_OWNER,
            "registrations": [],
            "validator_permits": [],
            "return_per_1000": 0,
            "total_daily_return": 0,
        }
        result = DelegateInfoLite._from_dict(decoded)
        assert result.nominators == 0
        assert result.take == 0.0
        assert result.return_per_1000.rao == 0
        assert result.total_daily_return.rao == 0
    @patch("bittensor.core.chain_data.delegate_info_lite.decode_account_id")
    def test_from_dict_large_values(self, mock_decode):
        """Test _from_dict with large values."""
        mock_decode.side_effect = lambda x: x
        decoded = {
            "delegate_ss58": MOCK_HOTKEY,
            "take": 65535,
            "nominators": 1000000,
            "owner_ss58": MOCK_OWNER,
            "registrations": list(range(100)),
            "validator_permits": list(range(50)),
            "return_per_1000": 10000000000000,
            "total_daily_return": 100000000000000,
        }
        result = DelegateInfoLite._from_dict(decoded)
        assert result.nominators == 1000000
        assert len(result.registrations) == 100
        assert len(result.validator_permits) == 50
    @patch("bittensor.core.chain_data.delegate_info_lite.decode_account_id")
    def test_from_dict_take_normalization(self, mock_decode):
        """Test that take value is properly normalized."""
        mock_decode.side_effect = lambda x: x
        decoded = {
            "delegate_ss58": MOCK_HOTKEY,
            "take": 32768,
            "nominators": 10,
            "owner_ss58": MOCK_OWNER,
            "registrations": [1],
            "validator_permits": [1],
            "return_per_1000": 100000000,
            "total_daily_return": 1000000000,
        }
        result = DelegateInfoLite._from_dict(decoded)
        assert 0.0 <= result.take <= 1.0
class TestDelegateInfoLiteBalance:
    """Tests for balance-related functionality in DelegateInfoLite."""
    def test_return_per_1000_balance_type(self):
        """Test that return_per_1000 is a Balance object."""
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=0.18,
            nominators=100,
            owner_ss58=MOCK_OWNER,
            registrations=[1],
            validator_permits=[1],
            return_per_1000=Balance.from_tao(0.5),
            total_daily_return=Balance.from_tao(100.0),
        )
        assert isinstance(delegate.return_per_1000, Balance)
        assert isinstance(delegate.total_daily_return, Balance)
    def test_balance_from_rao(self):
        """Test creating DelegateInfoLite with Balance from rao."""
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=0.1,
            nominators=50,
            owner_ss58=MOCK_OWNER,
            registrations=[],
            validator_permits=[],
            return_per_1000=Balance.from_rao(1000000000),
            total_daily_return=Balance.from_rao(10000000000),
        )
        assert delegate.return_per_1000.rao == 1000000000
        assert delegate.total_daily_return.rao == 10000000000
    def test_balance_tao_conversion(self):
        """Test tao conversion in balance fields."""
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=0.18,
            nominators=100,
            owner_ss58=MOCK_OWNER,
            registrations=[1],
            validator_permits=[1],
            return_per_1000=Balance.from_tao(1.5),
            total_daily_return=Balance.from_tao(150.0),
        )
        assert delegate.return_per_1000.tao == 1.5
        assert delegate.total_daily_return.tao == 150.0
class TestDelegateInfoLiteValidation:
    """Tests for validation scenarios in DelegateInfoLite."""
    def test_multiple_registrations(self):
        """Test DelegateInfoLite with multiple subnet registrations."""
        registrations = [1, 2, 3, 5, 8, 13, 21]
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=0.18,
            nominators=100,
            owner_ss58=MOCK_OWNER,
            registrations=registrations,
            validator_permits=[1, 2, 3],
            return_per_1000=Balance.from_tao(0.5),
            total_daily_return=Balance.from_tao(100.0),
        )
        assert delegate.registrations == registrations
        assert len(delegate.registrations) == 7
    def test_validator_permits_subset_of_registrations(self):
        """Test that validator_permits can be a subset of registrations."""
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=0.18,
            nominators=100,
            owner_ss58=MOCK_OWNER,
            registrations=[1, 2, 3, 4, 5],
            validator_permits=[1, 3],
            return_per_1000=Balance.from_tao(0.5),
            total_daily_return=Balance.from_tao(100.0),
        )
        assert all(p in delegate.registrations for p in delegate.validator_permits)
    def test_delegate_info_lite_attributes_access(self):
        """Test accessing all attributes of DelegateInfoLite."""
        delegate = DelegateInfoLite(
            delegate_ss58=MOCK_HOTKEY,
            take=0.18,
            nominators=100,
            owner_ss58=MOCK_OWNER,
            registrations=[1, 2],
            validator_permits=[1],
            return_per_1000=Balance.from_tao(0.5),
            total_daily_return=Balance.from_tao(100.0),
        )
        attrs = [
            "delegate_ss58",
            "take",
            "nominators",
            "owner_ss58",
            "registrations",
            "validator_permits",
            "return_per_1000",
            "total_daily_return",
        ]
        for attr in attrs:
            assert hasattr(delegate, attr)
            assert getattr(delegate, attr) is not None
