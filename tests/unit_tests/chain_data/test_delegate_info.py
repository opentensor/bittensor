"""
Comprehensive unit tests for the bittensor.core.chain_data.delegate_info module.

This test suite covers all major components of the delegate_info module including:
- DelegateInfoBase base class structure and inheritance
- DelegateInfo class with total_stake and nominators dictionaries
- DelegatedInfo class for subnet-specific delegate information
- Dictionary conversion methods (_from_dict, from_dict from InfoBase)
- Balance calculations and stake aggregations
- Error handling and edge cases

The tests are designed to ensure that:
1. DelegateInfo objects can be created correctly with all required fields
2. Dictionary conversion works correctly with chain data format
3. Total stake calculations are accurate (summing nominator stakes)
4. Account ID decoding works properly
5. Error handling is robust for missing or invalid data
6. All methods handle edge cases properly

Each test includes extensive comments explaining:
- What functionality is being tested
- Why the test is important
- What assertions verify
- Expected behavior and edge cases
"""

from unittest.mock import patch

import pytest

# Import the modules to test
from bittensor.core.chain_data.delegate_info import (
    DelegatedInfo,
    DelegateInfo,
    DelegateInfoBase,
)
from bittensor.core.errors import SubstrateRequestException
from bittensor.utils.balance import Balance


class TestDelegateInfoBase:
    """
    Test class for DelegateInfoBase base class.
    
    This class tests the base class that contains common delegate information
    fields shared by DelegateInfo and DelegatedInfo. DelegateInfoBase provides
    the foundation for delegate-related data structures in the Bittensor network.
    """

    def test_delegate_info_base_initialization(self):
        """
        Test that DelegateInfoBase can be initialized with all required fields.
        
        This test verifies that a DelegateInfoBase object can be created with all
        required fields. Note that DelegateInfoBase is typically not instantiated
        directly (it's a base class), but we can test its structure to ensure it
        properly defines the common fields used by its subclasses.
        """
        # Create a DelegateInfoBase instance (it's a dataclass, so we can instantiate it)
        # This tests the base class structure even though it's typically used via inheritance
        delegate_base = DelegateInfoBase(
            hotkey_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            owner_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            take=0.18,  # 18% take percentage
            validator_permits=[1, 2, 3],  # Subnets the delegate can validate on
            registrations=[1, 2],  # Subnets the delegate is registered on
            return_per_1000=Balance.from_tao(1.5)  # Return per 1000 TAO staked
        )
        
        # Verify all fields are set correctly
        assert delegate_base.hotkey_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "Hotkey SS58 address should be set correctly"
        assert delegate_base.owner_ss58 == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", \
            "Owner (coldkey) SS58 address should be set correctly"
        assert delegate_base.take == 0.18, \
            "Take percentage should be set correctly (0.18 = 18%)"
        assert delegate_base.validator_permits == [1, 2, 3], \
            "Validator permits list should contain subnets the delegate can validate on"
        assert delegate_base.registrations == [1, 2], \
            "Registrations list should contain subnets the delegate is registered on"
        assert isinstance(delegate_base.return_per_1000, Balance), \
            "Return per 1000 should be a Balance object representing TAO returns"

    def test_delegate_info_base_inherits_from_info_base(self):
        """
        Test that DelegateInfoBase properly inherits from InfoBase.
        
        This test verifies that DelegateInfoBase is a subclass of InfoBase, which
        provides common functionality for chain data structures including the
        from_dict() and list_from_dicts() methods. This inheritance ensures
        consistency across all chain data classes.
        """
        from bittensor.core.chain_data.info_base import InfoBase
        assert issubclass(DelegateInfoBase, InfoBase), \
            "DelegateInfoBase should inherit from InfoBase to get common chain data functionality"
        
        # Verify it's a dataclass (which InfoBase also is)
        from dataclasses import is_dataclass
        assert is_dataclass(DelegateInfoBase), \
            "DelegateInfoBase should be a dataclass for automatic field handling"


class TestDelegateInfo:
    """
    Test class for DelegateInfo class.
    
    This class tests the full DelegateInfo class which extends DelegateInfoBase
    with additional fields for tracking total_stake per subnet and detailed
    nominator stake information. This is the complete delegate information structure.
    """

    def test_delegate_info_initialization_with_all_fields(self):
        """
        Test that DelegateInfo can be initialized with all required fields.
        
        This test verifies that a DelegateInfo object can be created with all
        required fields including the base class fields and the additional
        total_stake and nominators dictionaries that provide detailed stake
        tracking across multiple subnets.
        """
        # Create total_stake dictionary mapping netuid to total stake Balance
        # This represents the total stake delegated to this delegate per subnet
        total_stake = {
            1: Balance.from_tao(1000).set_unit(1),  # 1000 TAO in subnet 1
            2: Balance.from_tao(2000).set_unit(2),  # 2000 TAO in subnet 2
        }
        
        # Create nominators dictionary mapping nominator address to their stake per subnet
        # This provides detailed breakdown of who staked what amount in which subnet
        nominators = {
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty": {
                1: Balance.from_tao(500).set_unit(1),   # 500 TAO in subnet 1
                2: Balance.from_tao(1000).set_unit(2),  # 1000 TAO in subnet 2
            },
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY": {
                1: Balance.from_tao(500).set_unit(1),   # 500 TAO in subnet 1
            }
        }
        
        # Create a DelegateInfo instance with all fields
        delegate_info = DelegateInfo(
            hotkey_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            owner_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            take=0.18,
            validator_permits=[1, 2, 3],
            registrations=[1, 2],
            return_per_1000=Balance.from_tao(1.5),
            total_stake=total_stake,
            nominators=nominators
        )
        
        # Verify base class fields are set correctly
        assert delegate_info.hotkey_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "Hotkey SS58 should be set from base class"
        assert delegate_info.take == 0.18, \
            "Take percentage should be set from base class"
        
        # Verify additional fields specific to DelegateInfo
        assert len(delegate_info.total_stake) == 2, \
            "Total stake dictionary should contain stakes for 2 subnets"
        assert delegate_info.total_stake[1].tao == pytest.approx(1000, rel=0.01), \
            "Total stake for netuid 1 should be 1000 TAO"
        assert delegate_info.total_stake[2].tao == pytest.approx(2000, rel=0.01), \
            "Total stake for netuid 2 should be 2000 TAO"
        
        # Verify nominators dictionary structure
        assert len(delegate_info.nominators) == 2, \
            "Should have 2 nominators with stakes"
        assert "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty" in delegate_info.nominators, \
            "First nominator should be in the dictionary"
        assert 1 in delegate_info.nominators["5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"], \
            "First nominator should have stake in subnet 1"

    def test_delegate_info_from_dict_with_complete_data(self):
        """
        Test that _from_dict() correctly creates DelegateInfo from dictionary data.
        
        This test verifies that when given a dictionary with complete delegate
        information (as would come from chain data), the _from_dict() method
        correctly creates a DelegateInfo object. It also verifies that:
        - Account IDs are properly decoded from bytes to SS58 addresses
        - Stake amounts are converted from rao to Balance objects
        - Total stake is correctly calculated by summing nominator stakes per subnet
        - Multiple nominators with stakes in the same subnet are aggregated correctly
        """
        # Mock decode_account_id to return SS58 addresses
        # This simulates the decoding that happens when converting raw chain data
        with patch("bittensor.core.chain_data.delegate_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # delegate_ss58
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # owner_ss58
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # nominator1
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # nominator2
            ]
            
            # Create dictionary data as would come from chain
            # Note: Stakes are in rao (smallest unit), need to be converted to Balance
            # 1 TAO = 1,000,000,000,000 rao
            decoded = {
                "delegate_ss58": b"delegate_bytes",  # Raw bytes that will be decoded
                "owner_ss58": b"owner_bytes",  # Raw bytes that will be decoded
                "take": 29491,  # u16 value that will be normalized to float (0-1 range)
                "validator_permits": [1, 2, 3],  # List of subnet IDs
                "registrations": [1, 2],  # List of subnet IDs
                "return_per_1000": 1500000000000,  # 1.5 TAO in rao
                "nominators": [
                    (
                        b"nominator1_bytes",  # Raw nominator address bytes
                        [(1, 500000000000), (2, 1000000000000)]  # (netuid, stake_rao) tuples
                        # nominator1 has 0.5 TAO in subnet 1, 1.0 TAO in subnet 2
                    ),
                    (
                        b"nominator2_bytes",
                        [(1, 500000000000)]  # nominator2 has 0.5 TAO in subnet 1
                    )
                ]
            }
            
            # Create DelegateInfo from dictionary using _from_dict class method
            delegate_info = DelegateInfo._from_dict(decoded)
            
            # Verify the object was created successfully
            assert isinstance(delegate_info, DelegateInfo), \
                "Should return a DelegateInfo instance"
            
            # Verify account IDs were decoded correctly
            assert delegate_info.hotkey_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
                "Hotkey should be decoded from bytes to SS58 address"
            assert delegate_info.owner_ss58 == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", \
                "Owner should be decoded from bytes to SS58 address"
            
            # Verify total_stake is calculated correctly
            # Subnet 1: 0.5 TAO (nominator1) + 0.5 TAO (nominator2) = 1.0 TAO
            assert 1 in delegate_info.total_stake, \
                "Total stake should include subnet 1"
            assert delegate_info.total_stake[1].tao == pytest.approx(1.0, rel=0.01), \
                "Total stake for subnet 1 should be sum of both nominators (1.0 TAO)"
            
            # Subnet 2: 1.0 TAO (nominator1 only)
            assert 2 in delegate_info.total_stake, \
                "Total stake should include subnet 2"
            assert delegate_info.total_stake[2].tao == pytest.approx(1.0, rel=0.01), \
                "Total stake for subnet 2 should be from nominator1 only (1.0 TAO)"
            
            # Verify nominators dictionary is populated correctly
            assert len(delegate_info.nominators) == 2, \
                "Should have 2 nominators in the dictionary"

    def test_delegate_info_from_dict_with_empty_nominators(self):
        """
        Test that _from_dict() handles empty nominators list correctly.
        
        This test verifies that when a delegate has no nominators (which is
        a valid state for a new delegate), the _from_dict() method correctly
        creates a DelegateInfo object with empty nominators and total_stake
        dictionaries. This ensures the code handles edge cases gracefully.
        """
        # Mock decode_account_id for delegate and owner only
        with patch("bittensor.core.chain_data.delegate_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with no nominators
            decoded = {
                "delegate_ss58": b"delegate_bytes",
                "owner_ss58": b"owner_bytes",
                "take": 29491,
                "validator_permits": [],
                "registrations": [],
                "return_per_1000": 0,
                "nominators": []  # Empty nominators list - no one has staked yet
            }
            
            # Create DelegateInfo from dictionary
            delegate_info = DelegateInfo._from_dict(decoded)
            
            # Verify nominators dictionary is empty but exists
            assert len(delegate_info.nominators) == 0, \
                "Nominators dictionary should be empty when no nominators present"
            assert isinstance(delegate_info.nominators, dict), \
                "Nominators should still be a dictionary (empty dict)"
            
            # Verify total_stake is empty (no stakes to sum)
            assert len(delegate_info.total_stake) == 0, \
                "Total stake dictionary should be empty when no nominators"

    def test_delegate_info_from_dict_calculates_total_stake_correctly(self):
        """
        Test that _from_dict() correctly calculates total_stake by summing nominator stakes.
        
        This test verifies that when multiple nominators have stakes in the same
        subnet, the total_stake for that subnet is correctly calculated as the
        sum of all nominator stakes. This is critical for accurate stake tracking
        and delegation calculations.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.delegate_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # nominator1
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # nominator2
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # nominator3
            ]
            
            # Create dictionary with multiple nominators having stakes in same subnet
            decoded = {
                "delegate_ss58": b"delegate_bytes",
                "owner_ss58": b"owner_bytes",
                "take": 29491,
                "validator_permits": [1],
                "registrations": [1],
                "return_per_1000": 0,
                "nominators": [
                    (
                        b"nominator1_bytes",
                        [(1, 500000000000)]  # 0.5 TAO in subnet 1
                    ),
                    (
                        b"nominator2_bytes",
                        [(1, 1000000000000)]  # 1.0 TAO in subnet 1
                    ),
                    (
                        b"nominator3_bytes",
                        [(1, 250000000000)]  # 0.25 TAO in subnet 1
                    )
                ]
                # Total should be: 0.5 + 1.0 + 0.25 = 1.75 TAO
            }
            
            # Create DelegateInfo from dictionary
            delegate_info = DelegateInfo._from_dict(decoded)
            
            # Verify total_stake for subnet 1 is sum of all nominators
            assert 1 in delegate_info.total_stake, \
                "Total stake should include subnet 1"
            assert delegate_info.total_stake[1].tao == pytest.approx(1.75, rel=0.01), \
                "Total stake should be sum of all nominator stakes (1.75 TAO)"

    def test_delegate_info_from_dict_sets_balance_units(self):
        """
        Test that _from_dict() correctly sets Balance units to netuid.
        
        This test verifies that when creating Balance objects from stake amounts,
        the Balance.set_unit() method is called with the netuid. This ensures
        that balances are properly tagged with their subnet unit, which is
        important for type safety and calculations.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.delegate_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with stakes in multiple subnets
            decoded = {
                "delegate_ss58": b"delegate_bytes",
                "owner_ss58": b"owner_bytes",
                "take": 29491,
                "validator_permits": [1, 2],
                "registrations": [1, 2],
                "return_per_1000": 0,
                "nominators": [
                    (
                        b"nominator_bytes",
                        [
                            (1, 500000000000),   # Subnet 1
                            (2, 1000000000000)   # Subnet 2
                        ]
                    )
                ]
            }
            
            # Create DelegateInfo
            delegate_info = DelegateInfo._from_dict(decoded)
            
            # Verify balance units are set correctly
            assert delegate_info.total_stake[1].unit == 1, \
                "Total stake for subnet 1 should have unit=1"
            assert delegate_info.total_stake[2].unit == 2, \
                "Total stake for subnet 2 should have unit=2"
            
            # Verify nominator stakes also have correct units
            nominator_stakes = delegate_info.nominators["5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"]
            assert nominator_stakes[1].unit == 1, \
                "Nominator stake in subnet 1 should have unit=1"
            assert nominator_stakes[2].unit == 2, \
                "Nominator stake in subnet 2 should have unit=2"


class TestDelegatedInfo:
    """
    Test class for DelegatedInfo class.
    
    This class tests the DelegatedInfo class which represents delegate
    information specific to a particular subnet. Unlike DelegateInfo which
    contains information across all subnets, DelegatedInfo focuses on a
    single subnet with its specific stake amount.
    """

    def test_delegated_info_initialization(self):
        """
        Test that DelegatedInfo can be initialized with all required fields.
        
        This test verifies that a DelegatedInfo object can be created with all
        required fields including the base class fields (from DelegateInfoBase)
        and the subnet-specific netuid and stake fields. This is useful when
        working with delegate information for a single subnet.
        """
        # Create a DelegatedInfo instance
        delegated_info = DelegatedInfo(
            hotkey_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            owner_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            take=0.18,
            validator_permits=[1, 2],
            registrations=[1, 2],
            return_per_1000=Balance.from_tao(1.5),
            netuid=1,  # Specific subnet this delegated info is for
            stake=Balance.from_tao(1000).set_unit(1)  # Stake amount in this subnet
        )
        
        # Verify base class fields
        assert delegated_info.hotkey_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "Hotkey should be set from base class"
        
        # Verify subnet-specific fields
        assert delegated_info.netuid == 1, \
            "Netuid should specify which subnet this delegated info is for"
        assert delegated_info.stake.tao == pytest.approx(1000, rel=0.01), \
            "Stake should represent the total stake in this specific subnet"
        assert delegated_info.stake.unit == 1, \
            "Stake should have unit set to the netuid for type safety"

    def test_delegated_info_from_dict_with_tuple_input(self):
        """
        Test that _from_dict() correctly handles tuple input format.
        
        This test verifies that DelegatedInfo._from_dict() correctly handles
        the tuple format (delegate_info_dict, (netuid, stake)) that comes
        from chain data for subnet-specific delegate information. This tuple
        format allows combining general delegate info with subnet-specific data.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.delegate_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create tuple format as would come from chain
            # Format: (delegate_info_dict, (netuid, stake_rao))
            # This format allows combining general delegate info with subnet-specific stake
            decoded = (
                {
                    "delegate_ss58": b"delegate_bytes",
                    "owner_ss58": b"owner_bytes",
                    "take": 29491,
                    "validator_permits": [1, 2],
                    "registrations": [1, 2],
                    "return_per_1000": 1500000000000,  # 1.5 TAO in rao
                },
                (1, 1000000000000)  # (netuid, stake_rao) - 1.0 TAO in subnet 1
            )
            
            # Create DelegatedInfo from tuple
            delegated_info = DelegatedInfo._from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(delegated_info, DelegatedInfo), \
                "Should return a DelegatedInfo instance"
            
            # Verify netuid was extracted from tuple
            assert delegated_info.netuid == 1, \
                "Netuid should be extracted from the tuple's second element"
            
            # Verify stake was converted from rao correctly
            assert delegated_info.stake.tao == pytest.approx(1.0, rel=0.01), \
                "Stake should be converted from rao to TAO correctly"
            
            # Verify stake unit is set to netuid
            assert delegated_info.stake.unit == 1, \
                "Stake should have unit set to netuid for proper type tracking"

    def test_delegated_info_from_dict_handles_different_netuids(self):
        """
        Test that _from_dict() correctly handles different subnet IDs.
        
        This test verifies that DelegatedInfo can be created for different
        subnets and that the netuid and stake unit are correctly set for
        each subnet. This ensures proper subnet-specific tracking.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.delegate_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Test with subnet 2
            decoded = (
                {
                    "delegate_ss58": b"delegate_bytes",
                    "owner_ss58": b"owner_bytes",
                    "take": 29491,
                    "validator_permits": [2],
                    "registrations": [2],
                    "return_per_1000": 0,
                },
                (2, 500000000000)  # netuid 2, 0.5 TAO
            )
            
            # Create DelegatedInfo
            delegated_info = DelegatedInfo._from_dict(decoded)
            
            # Verify netuid and unit are correct
            assert delegated_info.netuid == 2, \
                "Netuid should be 2 as specified in tuple"
            assert delegated_info.stake.unit == 2, \
                "Stake unit should match netuid (2)"


class TestDelegateInfoFromDictBaseClass:
    """
    Test class for the from_dict() method inherited from InfoBase.
    
    This class tests that DelegateInfo can use the from_dict() method from
    InfoBase, which includes error handling for missing fields. The from_dict()
    method calls _from_dict() internally but adds exception handling.
    """

    def test_delegate_info_from_dict_with_complete_data(self):
        """
        Test that from_dict() works with complete data.
        
        This test verifies that the from_dict() method (inherited from InfoBase)
        correctly calls _from_dict() when all required fields are present in
        the dictionary. This is the happy path for creating DelegateInfo from chain data.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.delegate_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create complete dictionary data
            decoded = {
                "delegate_ss58": b"delegate_bytes",
                "owner_ss58": b"owner_bytes",
                "take": 29491,
                "validator_permits": [1],
                "registrations": [1],
                "return_per_1000": 1000000000000,
                "nominators": []
            }
            
            # Create DelegateInfo using from_dict (from InfoBase)
            # This method includes error handling
            delegate_info = DelegateInfo.from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(delegate_info, DelegateInfo), \
                "from_dict() should return a DelegateInfo instance"
            assert delegate_info.hotkey_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
                "Hotkey should be decoded correctly"

    def test_delegate_info_from_dict_raises_exception_on_missing_field(self):
        """
        Test that from_dict() raises SubstrateRequestException on missing fields.
        
        This test verifies that when required fields are missing from the
        dictionary, the from_dict() method (inherited from InfoBase) raises
        a SubstrateRequestException with a descriptive message. This helps
        identify data structure issues from the chain early.
        """
        # Create incomplete dictionary (missing required field)
        incomplete_data = {
            "delegate_ss58": b"delegate_bytes",
            "take": 29491,
            # Missing owner_ss58, validator_permits, registrations, etc.
        }
        
        # Verify from_dict raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException) as exc_info:
            DelegateInfo.from_dict(incomplete_data)
        
        # Verify error message mentions missing field
        assert "missing" in str(exc_info.value).lower(), \
            "Error message should mention missing field"
        assert "DelegateInfo" in str(exc_info.value), \
            "Error message should mention DelegateInfo class name"


class TestDelegatedInfoFromDictBaseClass:
    """
    Test class for DelegatedInfo using from_dict() from InfoBase.
    
    This class tests that DelegatedInfo can use the from_dict() method for
    error handling, similar to DelegateInfo.
    """

    def test_delegated_info_from_dict_with_complete_tuple(self):
        """
        Test that from_dict() works with complete tuple data.
        
        This test verifies that DelegatedInfo.from_dict() correctly processes
        the tuple format and creates a DelegatedInfo object when all required
        fields are present.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.delegate_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create complete tuple data
            decoded = (
                {
                    "delegate_ss58": b"delegate_bytes",
                    "owner_ss58": b"owner_bytes",
                    "take": 29491,
                    "validator_permits": [1],
                    "registrations": [1],
                    "return_per_1000": 0,
                },
                (1, 1000000000000)
            )
            
            # Create DelegatedInfo using from_dict
            delegated_info = DelegatedInfo.from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(delegated_info, DelegatedInfo), \
                "from_dict() should return a DelegatedInfo instance"
            assert delegated_info.netuid == 1, \
                "Netuid should be extracted correctly"

