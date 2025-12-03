"""Unit tests for bittensor.core.chain_data.delegate_info module."""
import pytest
from unittest.mock import patch, MagicMock
from bittensor.core.chain_data.delegate_info import (
    DelegateInfoBase,
    DelegateInfo,
    DelegatedInfo,
)
from bittensor.utils.balance import Balance
MOCK_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
MOCK_COLDKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
MOCK_OWNER = "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"
class TestDelegateInfoBase:
    """Tests for DelegateInfoBase dataclass."""
    def test_delegate_info_base_creation(self):
        """Test creating a DelegateInfoBase instance with valid data."""
        delegate = DelegateInfoBase(
            hotkey_ss58=MOCK_HOTKEY,
            owner_ss58=MOCK_OWNER,
            take=0.18,
            validator_permits=[1, 2, 3],
            registrations=[1, 2],
            return_per_1000=Balance.from_tao(0.5),
            total_daily_return=Balance.from_tao(100.0),
        )
        assert delegate.hotkey_ss58 == MOCK_HOTKEY
        assert delegate.owner_ss58 == MOCK_OWNER
        assert delegate.take == 0.18
        assert delegate.validator_permits == [1, 2, 3]
        assert delegate.registrations == [1, 2]
        assert delegate.return_per_1000.tao == 0.5
        assert delegate.total_daily_return.tao == 100.0
    def test_delegate_info_base_empty_permits(self):
        """Test DelegateInfoBase with empty validator permits."""
        delegate = DelegateInfoBase(
            hotkey_ss58=MOCK_HOTKEY,
            owner_ss58=MOCK_OWNER,
            take=0.0,
            validator_permits=[],
            registrations=[],
            return_per_1000=Balance.from_rao(0),
            total_daily_return=Balance.from_rao(0),
        )
        assert delegate.validator_permits == []
        assert delegate.registrations == []
        assert delegate.take == 0.0
    def test_delegate_info_base_max_take(self):
        """Test DelegateInfoBase with maximum take value."""
        delegate = DelegateInfoBase(
            hotkey_ss58=MOCK_HOTKEY,
            owner_ss58=MOCK_OWNER,
            take=1.0,
            validator_permits=[1],
            registrations=[1],
            return_per_1000=Balance.from_tao(1.0),
            total_daily_return=Balance.from_tao(1000.0),
        )
        assert delegate.take == 1.0
class TestDelegateInfo:
    """Tests for DelegateInfo dataclass."""
    def test_delegate_info_creation(self):
        """Test creating a DelegateInfo instance with valid data."""
        total_stake = {1: Balance.from_tao(100.0), 2: Balance.from_tao(200.0)}
        nominators = {
            MOCK_COLDKEY: {1: Balance.from_tao(50.0), 2: Balance.from_tao(100.0)},
        }
        delegate = DelegateInfo(
            hotkey_ss58=MOCK_HOTKEY,
            owner_ss58=MOCK_OWNER,
            take=0.18,
            validator_permits=[1, 2],
            registrations=[1, 2],
            return_per_1000=Balance.from_tao(0.5),
            total_daily_return=Balance.from_tao(100.0),
            total_stake=total_stake,
            nominators=nominators,
        )
        assert delegate.hotkey_ss58 == MOCK_HOTKEY
        assert delegate.total_stake == total_stake
        assert delegate.nominators == nominators
        assert len(delegate.nominators) == 1
    def test_delegate_info_empty_nominators(self):
        """Test DelegateInfo with no nominators."""
        delegate = DelegateInfo(
            hotkey_ss58=MOCK_HOTKEY,
            owner_ss58=MOCK_OWNER,
            take=0.18,
            validator_permits=[],
            registrations=[],
            return_per_1000=Balance.from_rao(0),
            total_daily_return=Balance.from_rao(0),
            total_stake={},
            nominators={},
        )
        assert delegate.nominators == {}
        assert delegate.total_stake == {}
    def test_delegate_info_multiple_nominators(self):
        """Test DelegateInfo with multiple nominators."""
        nominators = {
            MOCK_COLDKEY: {1: Balance.from_tao(100.0)},
            MOCK_OWNER: {1: Balance.from_tao(200.0)},
            MOCK_HOTKEY: {1: Balance.from_tao(50.0)},
        }
        delegate = DelegateInfo(
            hotkey_ss58=MOCK_HOTKEY,
            owner_ss58=MOCK_OWNER,
            take=0.1,
            validator_permits=[1],
            registrations=[1],
            return_per_1000=Balance.from_tao(1.0),
            total_daily_return=Balance.from_tao(500.0),
            total_stake={1: Balance.from_tao(350.0)},
            nominators=nominators,
        )
        assert len(delegate.nominators) == 3
    @patch("bittensor.core.chain_data.delegate_info.decode_account_id")
    def test_delegate_info_from_dict(self, mock_decode):
        """Test DelegateInfo._from_dict method."""
        mock_decode.side_effect = lambda x: x if x else ""
        decoded = {
            "delegate_ss58": MOCK_HOTKEY,
            "owner_ss58": MOCK_OWNER,
            "take": 11796,
            "validator_permits": [1, 2],
            "registrations": [1],
            "return_per_1000": 500000000,
            "total_daily_return": 100000000000,
            "nominators": [
                [MOCK_COLDKEY, [(1, 100000000000), (2, 200000000000)]],
            ],
        }
        result = DelegateInfo._from_dict(decoded)
        assert result is not None
        assert result.hotkey_ss58 == MOCK_HOTKEY
        assert result.owner_ss58 == MOCK_OWNER
        assert len(result.nominators) == 1
        assert 1 in result.total_stake
        assert 2 in result.total_stake
    @patch("bittensor.core.chain_data.delegate_info.decode_account_id")
    def test_delegate_info_from_dict_empty_nominators(self, mock_decode):
        """Test DelegateInfo._from_dict with empty nominators."""
        mock_decode.side_effect = lambda x: x if x else ""
        decoded = {
            "delegate_ss58": MOCK_HOTKEY,
            "owner_ss58": MOCK_OWNER,
            "take": 0,
            "validator_permits": [],
            "registrations": [],
            "return_per_1000": 0,
            "total_daily_return": 0,
            "nominators": [],
        }
        result = DelegateInfo._from_dict(decoded)
        assert result is not None
        assert result.nominators == {}
        assert result.total_stake == {}
    @patch("bittensor.core.chain_data.delegate_info.decode_account_id")
    def test_delegate_info_from_dict_missing_optional_fields(self, mock_decode):
        """Test DelegateInfo._from_dict with missing optional fields."""
        mock_decode.side_effect = lambda x: x if x else ""
        decoded = {
            "delegate_ss58": MOCK_HOTKEY,
            "owner_ss58": MOCK_OWNER,
            "take": 5000,
            "return_per_1000": 0,
            "total_daily_return": 0,
        }
        result = DelegateInfo._from_dict(decoded)
        assert result is not None
        assert result.validator_permits == []
        assert result.registrations == []
class TestDelegatedInfo:
    """Tests for DelegatedInfo dataclass."""
    def test_delegated_info_creation(self):
        """Test creating a DelegatedInfo instance."""
        delegated = DelegatedInfo(
            hotkey_ss58=MOCK_HOTKEY,
            owner_ss58=MOCK_OWNER,
            take=0.18,
            validator_permits=[1, 2],
            registrations=[1],
            return_per_1000=Balance.from_tao(0.5),
            total_daily_return=Balance.from_tao(100.0),
            netuid=1,
            stake=Balance.from_tao(500.0),
        )
        assert delegated.netuid == 1
        assert delegated.stake.tao == 500.0
        assert delegated.hotkey_ss58 == MOCK_HOTKEY
    def test_delegated_info_zero_stake(self):
        """Test DelegatedInfo with zero stake."""
        delegated = DelegatedInfo(
            hotkey_ss58=MOCK_HOTKEY,
            owner_ss58=MOCK_OWNER,
            take=0.0,
            validator_permits=[],
            registrations=[],
            return_per_1000=Balance.from_rao(0),
            total_daily_return=Balance.from_rao(0),
            netuid=0,
            stake=Balance.from_rao(0),
        )
        assert delegated.stake.rao == 0
        assert delegated.netuid == 0
    @patch("bittensor.core.chain_data.delegate_info.decode_account_id")
    def test_delegated_info_from_dict(self, mock_decode):
        """Test DelegatedInfo._from_dict method."""
        mock_decode.side_effect = lambda x: x if x else ""
        delegate_info = {
            "delegate_ss58": MOCK_HOTKEY,
            "owner_ss58": MOCK_OWNER,
            "take": 11796,
            "validator_permits": [1],
            "registrations": [1],
            "return_per_1000": 500000000,
            "total_daily_return": 100000000000,
        }
        decoded = (delegate_info, (1, 500000000000))
        result = DelegatedInfo._from_dict(decoded)
        assert result is not None
        assert result.netuid == 1
        assert result.hotkey_ss58 == MOCK_HOTKEY
    @patch("bittensor.core.chain_data.delegate_info.decode_account_id")
    def test_delegated_info_from_dict_different_netuids(self, mock_decode):
        """Test DelegatedInfo._from_dict with different netuid values."""
        mock_decode.side_effect = lambda x: x if x else ""
        for netuid in [0, 1, 10, 100, 255]:
            delegate_info = {
                "delegate_ss58": MOCK_HOTKEY,
                "owner_ss58": MOCK_OWNER,
                "take": 5000,
                "validator_permits": [],
                "registrations": [],
                "return_per_1000": 0,
                "total_daily_return": 0,
            }
            decoded = (delegate_info, (netuid, 1000000000))
            result = DelegatedInfo._from_dict(decoded)
            assert result.netuid == netuid
class TestDelegateInfoIntegration:
    """Integration tests for delegate info classes."""
    def test_delegate_info_inheritance(self):
        """Test that DelegateInfo inherits from DelegateInfoBase."""
        assert issubclass(DelegateInfo, DelegateInfoBase)
    def test_delegated_info_inheritance(self):
        """Test that DelegatedInfo inherits from DelegateInfoBase."""
        assert issubclass(DelegatedInfo, DelegateInfoBase)
    def test_delegate_info_balance_operations(self):
        """Test balance operations within DelegateInfo."""
        stake1 = Balance.from_tao(100.0)
        stake2 = Balance.from_tao(200.0)
        total_stake = {1: stake1, 2: stake2}
        delegate = DelegateInfo(
            hotkey_ss58=MOCK_HOTKEY,
            owner_ss58=MOCK_OWNER,
            take=0.18,
            validator_permits=[1, 2],
            registrations=[1, 2],
            return_per_1000=Balance.from_tao(0.5),
            total_daily_return=Balance.from_tao(100.0),
            total_stake=total_stake,
            nominators={},
        )
        total = sum(delegate.total_stake.values())
        assert total.tao == 300.0
    def test_delegate_info_take_percentage_range(self):
        """Test take percentage is within valid range."""
        for take in [0.0, 0.1, 0.18, 0.5, 1.0]:
            delegate = DelegateInfoBase(
                hotkey_ss58=MOCK_HOTKEY,
                owner_ss58=MOCK_OWNER,
                take=take,
                validator_permits=[],
                registrations=[],
                return_per_1000=Balance.from_rao(0),
                total_daily_return=Balance.from_rao(0),
            )
            assert 0.0 <= delegate.take <= 1.0
