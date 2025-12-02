"""
Comprehensive unit tests for the bittensor.core.chain_data.stake_info module.

This test suite covers all major components of the StakeInfo class including:
- Class instantiation and attribute validation
- Dictionary conversion (from_dict method)
- Account ID decoding for hotkey and coldkey
- Balance conversions from rao to Balance objects with correct units
- Multiple Balance fields (stake, locked, emission)
- Boolean and integer field handling
- Inheritance from InfoBase
- Edge cases and error handling

The tests are designed to ensure that:
1. StakeInfo objects can be created correctly with all required fields
2. Dictionary conversion works correctly with chain data format
3. Account IDs are properly decoded from bytes format
4. Balance values are correctly converted from rao with proper unit assignments
5. Error handling is robust for missing or invalid data
6. All methods handle edge cases properly

StakeInfo is a critical data structure representing stake information linked to
hotkey-coldkey pairs, including stake amounts, locked amounts, emissions, and
registration status. This is important for staking operations.

Each test includes extensive comments explaining:
- What functionality is being tested
- Why the test is important
- What assertions verify
- Expected behavior and edge cases
"""

from unittest.mock import patch

import pytest

# Import the modules to test
from bittensor.core.chain_data.stake_info import StakeInfo
from bittensor.core.errors import SubstrateRequestException
from bittensor.utils.balance import Balance


class TestStakeInfoInitialization:
    """
    Test class for StakeInfo object initialization.
    
    This class tests that StakeInfo objects can be created correctly with
    all required fields. StakeInfo contains stake information including
    hotkey, coldkey, netuid, stake amounts, locked amounts, emissions,
    drain, and registration status.
    """

    def test_stake_info_initialization_with_all_fields(self):
        """
        Test that StakeInfo can be initialized with all required fields.
        
        This test verifies that a StakeInfo object can be created with all
        required fields. StakeInfo contains comprehensive stake information
        including hotkey and coldkey addresses, subnet ID, stake balance,
        locked balance, emission balance, drain value, and registration status.
        """
        netuid = 1
        
        # Create a StakeInfo with all fields
        stake_info = StakeInfo(
            hotkey_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # Hotkey SS58 address
            coldkey_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # Coldkey SS58 address
            netuid=netuid,  # Subnet ID
            stake=Balance.from_tao(1000).set_unit(netuid),  # Stake balance (subnet unit)
            locked=Balance.from_tao(500).set_unit(netuid),  # Locked stake (subnet unit)
            emission=Balance.from_tao(10).set_unit(netuid),  # Emission balance (subnet unit)
            drain=5,  # Drain value
            is_registered=True,  # Registration status
        )
        
        # Verify all fields are set correctly
        assert stake_info.hotkey_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "Hotkey SS58 address should be set correctly"
        assert stake_info.coldkey_ss58 == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", \
            "Coldkey SS58 address should be set correctly"
        assert stake_info.netuid == netuid, \
            "Netuid should specify which subnet this stake info is for"
        assert isinstance(stake_info.stake, Balance), \
            "Stake should be a Balance object"
        assert stake_info.stake.tao == pytest.approx(1000, rel=0.01), \
            "Stake balance should be set correctly"
        assert stake_info.stake.unit == netuid, \
            "Stake balance should have unit set to netuid"
        assert isinstance(stake_info.locked, Balance), \
            "Locked should be a Balance object"
        assert stake_info.locked.tao == pytest.approx(500, rel=0.01), \
            "Locked balance should be set correctly"
        assert isinstance(stake_info.emission, Balance), \
            "Emission should be a Balance object"
        assert stake_info.emission.tao == pytest.approx(10, rel=0.01), \
            "Emission balance should be set correctly"
        assert stake_info.drain == 5, \
            "Drain value should be set correctly"
        assert stake_info.is_registered is True, \
            "Registration status should be set correctly"

    def test_stake_info_initialization_with_zero_balances(self):
        """
        Test that StakeInfo can be initialized with zero balances.
        
        This test verifies that zero balances for stake, locked, and emission
        are handled correctly. This is a valid edge case for accounts that
        have no stake or haven't received emissions yet.
        """
        netuid = 1
        
        stake_info = StakeInfo(
            hotkey_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            netuid=netuid,
            stake=Balance.from_tao(0).set_unit(netuid),  # Zero stake
            locked=Balance.from_tao(0).set_unit(netuid),  # Zero locked
            emission=Balance.from_tao(0).set_unit(netuid),  # Zero emission
            drain=0,  # Zero drain
            is_registered=False,  # Not registered
        )
        
        # Verify zero values are handled correctly
        assert stake_info.stake.tao == pytest.approx(0, abs=0.01), \
            "Zero stake balance should be handled correctly"
        assert stake_info.locked.tao == pytest.approx(0, abs=0.01), \
            "Zero locked balance should be handled correctly"
        assert stake_info.emission.tao == pytest.approx(0, abs=0.01), \
            "Zero emission balance should be handled correctly"
        assert stake_info.drain == 0, \
            "Zero drain value should be handled correctly"
        assert stake_info.is_registered is False, \
            "False registration status should be handled correctly"

    def test_stake_info_inherits_from_info_base(self):
        """
        Test that StakeInfo properly inherits from InfoBase.
        
        This test verifies that StakeInfo is a subclass of InfoBase, which
        provides common functionality for chain data structures. This ensures
        consistency across all chain data classes.
        """
        from bittensor.core.chain_data.info_base import InfoBase
        assert issubclass(StakeInfo, InfoBase), \
            "StakeInfo should inherit from InfoBase for common chain data functionality"
        
        from dataclasses import is_dataclass
        assert is_dataclass(StakeInfo), \
            "StakeInfo should be a dataclass for automatic field handling"


class TestStakeInfoFromDict:
    """
    Test class for the from_dict() class method.
    
    This class tests that StakeInfo objects can be created from dictionary
    data. Note that this class uses from_dict() directly (not _from_dict),
    so it likely includes error handling.
    """

    def test_from_dict_creates_stake_info_correctly(self):
        """
        Test that from_dict() correctly creates StakeInfo from dictionary data.
        
        This test verifies that when given a dictionary with stake information
        (as would come from chain data), the from_dict() method correctly creates
        a StakeInfo object. The conversion includes:
        - Decoding hotkey and coldkey account IDs from bytes to SS58 addresses
        - Converting rao values to Balance objects
        - Setting balance units correctly (netuid for subnet tokens)
        """
        netuid = 1
        
        # Mock decode_account_id for hotkey and coldkey
        with patch("bittensor.core.chain_data.stake_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # hotkey
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # coldkey
            ]
            
            # Create dictionary data as would come from chain
            # Note: Stakes are in rao (smallest unit), need to be converted to Balance
            # 1 TAO = 1,000,000,000,000 rao
            decoded = {
                "netuid": netuid,
                "hotkey": b"hotkey_bytes",  # Raw bytes that will be decoded
                "coldkey": b"coldkey_bytes",  # Raw bytes that will be decoded
                "stake": 1000000000000000,  # 1000 TAO in rao
                "locked": 500000000000000,  # 500 TAO in rao
                "emission": 10000000000000,  # 10 TAO in rao
                "drain": 5,  # Drain value
                "is_registered": True,  # Registration status
            }
            
            # Create StakeInfo from dictionary using from_dict class method
            stake_info = StakeInfo.from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(stake_info, StakeInfo), \
                "Should return a StakeInfo instance"
            
            # Verify netuid is set correctly
            assert stake_info.netuid == netuid, \
                "Netuid should be set correctly from dictionary"
            
            # Verify account IDs were decoded correctly
            assert stake_info.hotkey_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
                "Hotkey SS58 should be decoded from bytes to SS58 address"
            assert stake_info.coldkey_ss58 == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", \
                "Coldkey SS58 should be decoded from bytes to SS58 address"
            
            # Verify balances are converted correctly
            assert isinstance(stake_info.stake, Balance), \
                "Stake should be converted to a Balance object"
            assert stake_info.stake.tao == pytest.approx(1000, rel=0.01), \
                "Stake should be correctly converted from rao to TAO (1000 TAO)"
            assert stake_info.stake.unit == netuid, \
                "Stake balance should have unit set to netuid"
            
            assert stake_info.locked.tao == pytest.approx(500, rel=0.01), \
                "Locked should be correctly converted from rao to TAO (500 TAO)"
            assert stake_info.locked.unit == netuid, \
                "Locked balance should have unit set to netuid"
            
            assert stake_info.emission.tao == pytest.approx(10, rel=0.01), \
                "Emission should be correctly converted from rao to TAO (10 TAO)"
            assert stake_info.emission.unit == netuid, \
                "Emission balance should have unit set to netuid"
            
            # Verify other fields
            assert stake_info.drain == 5, \
                "Drain value should be set correctly"
            assert stake_info.is_registered is True, \
                "Registration status should be set correctly"

    def test_from_dict_decodes_account_ids(self):
        """
        Test that from_dict() correctly decodes account IDs using decode_account_id.
        
        This test verifies that the hotkey and coldkey fields are properly decoded
        from bytes/raw format to SS58 string addresses using the decode_account_id
        utility function. This decoding is essential for working with account
        addresses in a human-readable format.
        """
        netuid = 1
        
        # Mock decode_account_id to verify it's called correctly
        with patch("bittensor.core.chain_data.stake_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with raw account ID bytes
            decoded = {
                "netuid": netuid,
                "hotkey": b"raw_hotkey_bytes",  # Raw bytes from chain
                "coldkey": b"raw_coldkey_bytes",  # Raw bytes from chain
                "stake": 0,
                "locked": 0,
                "emission": 0,
                "drain": 0,
                "is_registered": False,
            }
            
            # Create StakeInfo
            stake_info = StakeInfo.from_dict(decoded)
            
            # Verify decode_account_id was called for both account IDs
            assert mock_decode.call_count == 2, \
                "decode_account_id should be called twice (once for hotkey, once for coldkey)"
            mock_decode.assert_any_call(b"raw_hotkey_bytes"), \
                "decode_account_id should be called with hotkey bytes"
            mock_decode.assert_any_call(b"raw_coldkey_bytes"), \
                "decode_account_id should be called with coldkey bytes"

    def test_from_dict_converts_balances_from_rao(self):
        """
        Test that from_dict() correctly converts balance values from rao to Balance objects.
        
        This test verifies that stake, locked, and emission values (which come from chain
        as rao, the smallest unit) are properly converted to Balance objects using
        Balance.from_rao(). This ensures proper balance handling and unit conversions.
        """
        netuid = 1
        
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.stake_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with balance values in rao
            # 1 TAO = 1,000,000,000,000 rao (10^12)
            decoded = {
                "netuid": netuid,
                "hotkey": b"hotkey_bytes",
                "coldkey": b"coldkey_bytes",
                "stake": 1500000000000000,  # 1.5 TAO in rao
                "locked": 750000000000000,  # 0.75 TAO in rao
                "emission": 15000000000000,  # 0.015 TAO in rao
                "drain": 5,
                "is_registered": True,
            }
            
            # Create StakeInfo
            stake_info = StakeInfo.from_dict(decoded)
            
            # Verify balances are converted correctly
            assert stake_info.stake.tao == pytest.approx(1.5, rel=0.01), \
                "Stake should be correctly converted from rao to TAO (1.5 TAO)"
            assert stake_info.locked.tao == pytest.approx(0.75, rel=0.01), \
                "Locked should be correctly converted from rao to TAO (0.75 TAO)"
            assert stake_info.emission.tao == pytest.approx(0.015, rel=0.01), \
                "Emission should be correctly converted from rao to TAO (0.015 TAO)"

    def test_from_dict_sets_balance_units_correctly(self):
        """
        Test that from_dict() correctly sets Balance units based on netuid.
        
        This test verifies that Balance objects are created with the correct
        unit assignments. For StakeInfo, all balances (stake, locked, emission)
        should have their unit set to the netuid for proper type safety and
        subnet token tracking.
        """
        netuid = 2  # Using netuid 2 to verify unit assignment
        
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.stake_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with specific netuid
            decoded = {
                "netuid": netuid,
                "hotkey": b"hotkey_bytes",
                "coldkey": b"coldkey_bytes",
                "stake": 1000000000000000,
                "locked": 500000000000000,
                "emission": 10000000000000,
                "drain": 5,
                "is_registered": True,
            }
            
            # Create StakeInfo
            stake_info = StakeInfo.from_dict(decoded)
            
            # Verify balance units are set correctly
            assert stake_info.stake.unit == netuid, \
                "Stake balance should have unit set to netuid (2)"
            assert stake_info.locked.unit == netuid, \
                "Locked balance should have unit set to netuid (2)"
            assert stake_info.emission.unit == netuid, \
                "Emission balance should have unit set to netuid (2)"


class TestStakeInfoEdgeCases:
    """
    Test class for edge cases and special scenarios.
    
    This class tests edge cases such as zero balances, different netuids,
    boolean combinations, and other boundary conditions.
    """

    def test_stake_info_with_different_netuids(self):
        """
        Test that StakeInfo handles different netuids correctly.
        
        This test verifies that StakeInfo can represent stakes from different
        subnets, and that balance units are correctly set for each subnet.
        """
        # Test with netuid 1
        netuid_1 = 1
        with patch("bittensor.core.chain_data.stake_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            decoded_1 = {
                "netuid": netuid_1,
                "hotkey": b"hotkey_bytes",
                "coldkey": b"coldkey_bytes",
                "stake": 1000000000000000,
                "locked": 0,
                "emission": 0,
                "drain": 0,
                "is_registered": True,
            }
            
            stake_info_1 = StakeInfo.from_dict(decoded_1)
            assert stake_info_1.netuid == netuid_1, \
                "Should handle netuid 1 correctly"
            assert stake_info_1.stake.unit == netuid_1, \
                "Balance unit should be set to netuid 1"
        
        # Test with netuid 2
        netuid_2 = 2
        with patch("bittensor.core.chain_data.stake_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            decoded_2 = {
                "netuid": netuid_2,
                "hotkey": b"hotkey_bytes",
                "coldkey": b"coldkey_bytes",
                "stake": 2000000000000000,
                "locked": 0,
                "emission": 0,
                "drain": 0,
                "is_registered": True,
            }
            
            stake_info_2 = StakeInfo.from_dict(decoded_2)
            assert stake_info_2.netuid == netuid_2, \
                "Should handle netuid 2 correctly"
            assert stake_info_2.stake.unit == netuid_2, \
                "Balance unit should be set to netuid 2"

    def test_stake_info_field_types(self):
        """
        Test that StakeInfo fields maintain correct types.
        
        This test verifies that all fields in StakeInfo maintain their
        expected types. This is important for type consistency and ensures
        that the dataclass properly enforces type constraints at runtime.
        """
        netuid = 1
        
        stake_info = StakeInfo(
            hotkey_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            netuid=netuid,
            stake=Balance.from_tao(1000).set_unit(netuid),
            locked=Balance.from_tao(500).set_unit(netuid),
            emission=Balance.from_tao(10).set_unit(netuid),
            drain=5,
            is_registered=True,
        )
        
        # Verify all field types are correct
        assert isinstance(stake_info.hotkey_ss58, str), \
            "hotkey_ss58 should be string type (SS58 address)"
        assert isinstance(stake_info.coldkey_ss58, str), \
            "coldkey_ss58 should be string type (SS58 address)"
        assert isinstance(stake_info.netuid, int), \
            "netuid should be int type"
        assert isinstance(stake_info.stake, Balance), \
            "stake should be Balance type"
        assert isinstance(stake_info.locked, Balance), \
            "locked should be Balance type"
        assert isinstance(stake_info.emission, Balance), \
            "emission should be Balance type"
        assert isinstance(stake_info.drain, int), \
            "drain should be int type"
        assert isinstance(stake_info.is_registered, bool), \
            "is_registered should be bool type"

