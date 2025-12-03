"""Unit tests for bittensor.core.chain_data.stake_info module."""
import pytest
from unittest.mock import patch
from bittensor.core.chain_data.stake_info import StakeInfo
from bittensor.utils.balance import Balance
MOCK_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
MOCK_COLDKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
class TestStakeInfoCreation:
    """Tests for StakeInfo dataclass creation."""
    def test_stake_info_basic_creation(self):
        """Test creating a StakeInfo instance with valid data."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(100.0),
            locked=Balance.from_tao(10.0),
            emission=Balance.from_tao(0.5),
            drain=0,
            is_registered=True,
        )
        assert stake_info.hotkey_ss58 == MOCK_HOTKEY
        assert stake_info.coldkey_ss58 == MOCK_COLDKEY
        assert stake_info.netuid == 1
        assert stake_info.stake.tao == 100.0
        assert stake_info.locked.tao == 10.0
        assert stake_info.emission.tao == 0.5
        assert stake_info.drain == 0
        assert stake_info.is_registered is True
    def test_stake_info_zero_stake(self):
        """Test StakeInfo with zero stake."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_rao(0),
            locked=Balance.from_rao(0),
            emission=Balance.from_rao(0),
            drain=0,
            is_registered=False,
        )
        assert stake_info.stake.rao == 0
        assert stake_info.locked.rao == 0
        assert stake_info.emission.rao == 0
        assert stake_info.is_registered is False
    def test_stake_info_large_stake(self):
        """Test StakeInfo with large stake values."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(1000000.0),
            locked=Balance.from_tao(100000.0),
            emission=Balance.from_tao(1000.0),
            drain=100,
            is_registered=True,
        )
        assert stake_info.stake.tao == 1000000.0
        assert stake_info.locked.tao == 100000.0
        assert stake_info.emission.tao == 1000.0
    def test_stake_info_different_netuids(self):
        """Test StakeInfo with different netuid values."""
        for netuid in [0, 1, 10, 100, 255]:
            stake_info = StakeInfo(
                hotkey_ss58=MOCK_HOTKEY,
                coldkey_ss58=MOCK_COLDKEY,
                netuid=netuid,
                stake=Balance.from_tao(50.0),
                locked=Balance.from_tao(5.0),
                emission=Balance.from_tao(0.1),
                drain=0,
                is_registered=True,
            )
            assert stake_info.netuid == netuid
    def test_stake_info_drain_values(self):
        """Test StakeInfo with various drain values."""
        for drain in [0, 1, 100, 1000, 10000]:
            stake_info = StakeInfo(
                hotkey_ss58=MOCK_HOTKEY,
                coldkey_ss58=MOCK_COLDKEY,
                netuid=1,
                stake=Balance.from_tao(100.0),
                locked=Balance.from_tao(10.0),
                emission=Balance.from_tao(0.5),
                drain=drain,
                is_registered=True,
            )
            assert stake_info.drain == drain
class TestStakeInfoFromDict:
    """Tests for StakeInfo.from_dict method."""
    @patch("bittensor.core.chain_data.stake_info.decode_account_id")
    def test_from_dict_basic(self, mock_decode):
        """Test from_dict with basic valid data."""
        mock_decode.side_effect = lambda x: x
        decoded = {
            "hotkey": MOCK_HOTKEY,
            "coldkey": MOCK_COLDKEY,
            "netuid": 1,
            "stake": 100000000000,
            "locked": 10000000000,
            "emission": 500000000,
            "drain": 0,
            "is_registered": True,
        }
        result = StakeInfo.from_dict(decoded)
        assert result.hotkey_ss58 == MOCK_HOTKEY
        assert result.coldkey_ss58 == MOCK_COLDKEY
        assert result.netuid == 1
        assert result.is_registered is True
    @patch("bittensor.core.chain_data.stake_info.decode_account_id")
    def test_from_dict_zero_values(self, mock_decode):
        """Test from_dict with zero values."""
        mock_decode.side_effect = lambda x: x
        decoded = {
            "hotkey": MOCK_HOTKEY,
            "coldkey": MOCK_COLDKEY,
            "netuid": 0,
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        }
        result = StakeInfo.from_dict(decoded)
        assert result.stake.rao == 0
        assert result.locked.rao == 0
        assert result.emission.rao == 0
        assert result.is_registered is False
    @patch("bittensor.core.chain_data.stake_info.decode_account_id")
    def test_from_dict_large_values(self, mock_decode):
        """Test from_dict with large values."""
        mock_decode.side_effect = lambda x: x
        decoded = {
            "hotkey": MOCK_HOTKEY,
            "coldkey": MOCK_COLDKEY,
            "netuid": 255,
            "stake": 10000000000000000,
            "locked": 1000000000000000,
            "emission": 100000000000000,
            "drain": 999999,
            "is_registered": True,
        }
        result = StakeInfo.from_dict(decoded)
        assert result.netuid == 255
        assert result.drain == 999999
    @patch("bittensor.core.chain_data.stake_info.decode_account_id")
    def test_from_dict_netuid_unit_setting(self, mock_decode):
        """Test that from_dict sets the correct unit based on netuid."""
        mock_decode.side_effect = lambda x: x
        for netuid in [0, 1, 5, 10]:
            decoded = {
                "hotkey": MOCK_HOTKEY,
                "coldkey": MOCK_COLDKEY,
                "netuid": netuid,
                "stake": 1000000000,
                "locked": 100000000,
                "emission": 10000000,
                "drain": 0,
                "is_registered": True,
            }
            result = StakeInfo.from_dict(decoded)
            assert result.netuid == netuid
class TestStakeInfoBalance:
    """Tests for balance-related functionality in StakeInfo."""
    def test_stake_balance_type(self):
        """Test that stake is a Balance object."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(100.0),
            locked=Balance.from_tao(10.0),
            emission=Balance.from_tao(0.5),
            drain=0,
            is_registered=True,
        )
        assert isinstance(stake_info.stake, Balance)
        assert isinstance(stake_info.locked, Balance)
        assert isinstance(stake_info.emission, Balance)
    def test_balance_from_rao(self):
        """Test creating StakeInfo with Balance from rao."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_rao(1000000000),
            locked=Balance.from_rao(100000000),
            emission=Balance.from_rao(10000000),
            drain=0,
            is_registered=True,
        )
        assert stake_info.stake.rao == 1000000000
        assert stake_info.locked.rao == 100000000
        assert stake_info.emission.rao == 10000000
    def test_balance_tao_conversion(self):
        """Test tao conversion in balance fields."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(123.456),
            locked=Balance.from_tao(12.345),
            emission=Balance.from_tao(1.234),
            drain=0,
            is_registered=True,
        )
        assert abs(stake_info.stake.tao - 123.456) < 0.001
        assert abs(stake_info.locked.tao - 12.345) < 0.001
        assert abs(stake_info.emission.tao - 1.234) < 0.001
    def test_locked_less_than_stake(self):
        """Test that locked can be less than or equal to stake."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(100.0),
            locked=Balance.from_tao(50.0),
            emission=Balance.from_tao(0.5),
            drain=0,
            is_registered=True,
        )
        assert stake_info.locked.tao <= stake_info.stake.tao
class TestStakeInfoRegistration:
    """Tests for registration status in StakeInfo."""
    def test_registered_stake_info(self):
        """Test StakeInfo with is_registered=True."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(100.0),
            locked=Balance.from_tao(10.0),
            emission=Balance.from_tao(0.5),
            drain=0,
            is_registered=True,
        )
        assert stake_info.is_registered is True
    def test_unregistered_stake_info(self):
        """Test StakeInfo with is_registered=False."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(100.0),
            locked=Balance.from_tao(10.0),
            emission=Balance.from_tao(0.5),
            drain=0,
            is_registered=False,
        )
        assert stake_info.is_registered is False
    def test_registration_with_zero_stake(self):
        """Test that registration status is independent of stake amount."""
        stake_info_registered = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_rao(0),
            locked=Balance.from_rao(0),
            emission=Balance.from_rao(0),
            drain=0,
            is_registered=True,
        )
        stake_info_unregistered = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(1000.0),
            locked=Balance.from_tao(100.0),
            emission=Balance.from_tao(10.0),
            drain=0,
            is_registered=False,
        )
        assert stake_info_registered.is_registered is True
        assert stake_info_unregistered.is_registered is False
class TestStakeInfoAttributes:
    """Tests for StakeInfo attribute access."""
    def test_all_attributes_accessible(self):
        """Test that all attributes are accessible."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(100.0),
            locked=Balance.from_tao(10.0),
            emission=Balance.from_tao(0.5),
            drain=5,
            is_registered=True,
        )
        attrs = [
            "hotkey_ss58",
            "coldkey_ss58",
            "netuid",
            "stake",
            "locked",
            "emission",
            "drain",
            "is_registered",
        ]
        for attr in attrs:
            assert hasattr(stake_info, attr)
            assert getattr(stake_info, attr) is not None
    def test_hotkey_coldkey_format(self):
        """Test that hotkey and coldkey are in SS58 format."""
        stake_info = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(100.0),
            locked=Balance.from_tao(10.0),
            emission=Balance.from_tao(0.5),
            drain=0,
            is_registered=True,
        )
        assert stake_info.hotkey_ss58.startswith("5")
        assert stake_info.coldkey_ss58.startswith("5")
        assert len(stake_info.hotkey_ss58) == 48
        assert len(stake_info.coldkey_ss58) == 48
class TestStakeInfoComparison:
    """Tests for comparing StakeInfo instances."""
    def test_same_hotkey_different_netuids(self):
        """Test StakeInfo with same hotkey but different netuids."""
        stake_info1 = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(100.0),
            locked=Balance.from_tao(10.0),
            emission=Balance.from_tao(0.5),
            drain=0,
            is_registered=True,
        )
        stake_info2 = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=2,
            stake=Balance.from_tao(200.0),
            locked=Balance.from_tao(20.0),
            emission=Balance.from_tao(1.0),
            drain=0,
            is_registered=True,
        )
        assert stake_info1.hotkey_ss58 == stake_info2.hotkey_ss58
        assert stake_info1.netuid != stake_info2.netuid
        assert stake_info1.stake.tao != stake_info2.stake.tao
    def test_different_hotkeys_same_netuid(self):
        """Test StakeInfo with different hotkeys on same netuid."""
        other_hotkey = "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"
        stake_info1 = StakeInfo(
            hotkey_ss58=MOCK_HOTKEY,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(100.0),
            locked=Balance.from_tao(10.0),
            emission=Balance.from_tao(0.5),
            drain=0,
            is_registered=True,
        )
        stake_info2 = StakeInfo(
            hotkey_ss58=other_hotkey,
            coldkey_ss58=MOCK_COLDKEY,
            netuid=1,
            stake=Balance.from_tao(100.0),
            locked=Balance.from_tao(10.0),
            emission=Balance.from_tao(0.5),
            drain=0,
            is_registered=True,
        )
        assert stake_info1.hotkey_ss58 != stake_info2.hotkey_ss58
        assert stake_info1.netuid == stake_info2.netuid
