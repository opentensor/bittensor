"""
Comprehensive unit tests for the bittensor.core.chain_data.delegate_info_lite module.

This test suite covers all major components of the DelegateInfoLite class including:
- Class instantiation and attribute validation
- Dictionary conversion methods (_from_dict, from_dict from InfoBase)
- Account ID decoding from bytes to SS58 addresses
- Balance conversion from rao to Balance objects
- Take value normalization using u16_normalized_float
- Inheritance from InfoBase
- Edge cases and error handling

The tests are designed to ensure that:
1. DelegateInfoLite objects can be created correctly with all required fields
2. Dictionary conversion works correctly with chain data format
3. Account IDs are properly decoded from bytes format
4. Balance values are correctly converted from rao
5. Error handling is robust for missing or invalid data
6. All methods handle edge cases properly

Note: DelegateInfoLite is a lighter version of DelegateInfo that doesn't include
detailed nominator stake information. Instead, it only includes the count of
nominators, making it more efficient for listing delegates without full details.

Each test includes extensive comments explaining:
- What functionality is being tested
- Why the test is important
- What assertions verify
- Expected behavior and edge cases
"""

from unittest.mock import patch

import pytest

# Import the modules to test
from bittensor.core.chain_data.delegate_info_lite import DelegateInfoLite
from bittensor.core.errors import SubstrateRequestException
from bittensor.utils.balance import Balance


class TestDelegateInfoLiteInitialization:
    """
    Test class for DelegateInfoLite object initialization.
    
    This class tests that DelegateInfoLite objects can be created correctly with
    all required fields. The lite version is optimized for listing operations
    where full nominator details aren't needed.
    """

    def test_delegate_info_lite_initialization_with_all_fields(self):
        """
        Test that DelegateInfoLite can be initialized with all required fields.
        
        This test verifies that a DelegateInfoLite object can be created with all
        required fields. Note that this is a "lite" version that includes only
        the nominator count (as an integer) rather than detailed nominator stake
        information. This makes it more efficient for listing operations.
        """
        # Create a DelegateInfoLite with all fields
        delegate_lite = DelegateInfoLite(
            delegate_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            take=0.18,  # 18% take percentage (normalized float)
            nominators=10,  # Count of nominators (integer, not detailed stakes)
            owner_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            registrations=[1, 2, 3],  # Subnets the delegate is registered on
            validator_permits=[1, 2],  # Subnets the delegate can validate on
            return_per_1000=Balance.from_tao(1.5)  # Return per 1000 TAO staked
        )
        
        # Verify all fields are set correctly
        assert delegate_lite.delegate_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "Delegate SS58 address should be set correctly"
        assert delegate_lite.take == 0.18, \
            "Take percentage should be set correctly (0.18 = 18%)"
        assert delegate_lite.nominators == 10, \
            "Nominators count should be set correctly"
        assert isinstance(delegate_lite.nominators, int), \
            "Nominators should be an integer representing count, not a dictionary"
        assert delegate_lite.owner_ss58 == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", \
            "Owner SS58 address should be set correctly"
        assert delegate_lite.registrations == [1, 2, 3], \
            "Registrations list should contain subnet IDs"
        assert delegate_lite.validator_permits == [1, 2], \
            "Validator permits list should contain subnet IDs"
        assert isinstance(delegate_lite.return_per_1000, Balance), \
            "Return per 1000 should be a Balance object representing TAO returns"

    def test_delegate_info_lite_with_zero_nominators(self):
        """
        Test that DelegateInfoLite handles zero nominators correctly.
        
        This test verifies that a delegate with no nominators can be represented
        correctly with a nominators count of 0. This is a valid edge case for
        new delegates that haven't received any stakes yet.
        """
        # Create a DelegateInfoLite with zero nominators
        delegate_lite = DelegateInfoLite(
            delegate_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            take=0.0,  # 0% take (could be a special case)
            nominators=0,  # No nominators - delegate hasn't received stakes yet
            owner_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            registrations=[],  # Not registered on any subnet yet
            validator_permits=[],  # No validator permits
            return_per_1000=Balance.from_tao(0)  # No returns yet
        )
        
        # Verify zero nominators is handled correctly
        assert delegate_lite.nominators == 0, \
            "Nominators count should be 0 when no nominators present"
        assert isinstance(delegate_lite.nominators, int), \
            "Nominators should still be an integer type (count), not None or other type"

    def test_delegate_info_lite_inherits_from_info_base(self):
        """
        Test that DelegateInfoLite properly inherits from InfoBase.
        
        This test verifies that DelegateInfoLite is a subclass of InfoBase, which
        provides common functionality for chain data structures including the
        from_dict() and list_from_dicts() methods. This ensures consistency
        across all chain data classes.
        """
        from bittensor.core.chain_data.info_base import InfoBase
        assert issubclass(DelegateInfoLite, InfoBase), \
            "DelegateInfoLite should inherit from InfoBase to get common chain data functionality"
        
        # Verify it's a dataclass (which InfoBase also is)
        from dataclasses import is_dataclass
        assert is_dataclass(DelegateInfoLite), \
            "DelegateInfoLite should be a dataclass for automatic field handling"


class TestDelegateInfoLiteFromDict:
    """
    Test class for the _from_dict() class method.
    
    This class tests that DelegateInfoLite objects can be created from dictionary
    data, which is how chain data is typically received from the substrate
    interface. The _from_dict method handles field decoding and conversion.
    """

    def test_from_dict_creates_delegate_info_lite_correctly(self):
        """
        Test that _from_dict() correctly creates a DelegateInfoLite from dictionary data.
        
        This test verifies that when given a dictionary with delegate lite information
        fields (as would come from chain data), the _from_dict() method correctly
        creates a DelegateInfoLite object with all fields properly decoded and converted.
        This includes account ID decoding, balance conversion, and take normalization.
        """
        # Mock decode_account_id to return SS58 addresses
        # This simulates the decoding that happens when converting raw chain data
        with patch("bittensor.core.chain_data.delegate_info_lite.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # delegate_ss58
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # owner_ss58
            ]
            
            # Create dictionary data as would come from chain
            decoded = {
                "delegate_ss58": b"delegate_bytes",  # Raw bytes that will be decoded to SS58
                "take": 29491,  # u16 value that will be normalized to float (0-1 range)
                "nominators": 10,  # Count of nominators (integer, not detailed stakes)
                "owner_ss58": b"owner_bytes",  # Raw bytes that will be decoded to SS58
                "registrations": [1, 2, 3],  # List of subnet IDs
                "validator_permits": [1, 2],  # List of subnet IDs
                "return_per_1000": 1500000000000  # 1.5 TAO in rao (will be converted to Balance)
            }
            
            # Create DelegateInfoLite from dictionary using _from_dict class method
            delegate_lite = DelegateInfoLite._from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(delegate_lite, DelegateInfoLite), \
                "Should return a DelegateInfoLite instance"
            
            # Verify account IDs were decoded correctly
            assert delegate_lite.delegate_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
                "Delegate SS58 should be decoded from bytes to SS58 address string"
            assert delegate_lite.owner_ss58 == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", \
                "Owner SS58 should be decoded from bytes to SS58 address string"
            
            # Verify nominators count is set correctly
            assert delegate_lite.nominators == 10, \
                "Nominators count should be set correctly from dictionary"
            assert isinstance(delegate_lite.nominators, int), \
                "Nominators should be an integer (count), not a dictionary"
            
            # Verify lists are preserved
            assert delegate_lite.registrations == [1, 2, 3], \
                "Registrations list should be set correctly"
            assert delegate_lite.validator_permits == [1, 2], \
                "Validator permits list should be set correctly"

    def test_from_dict_decodes_account_ids(self):
        """
        Test that _from_dict() correctly decodes account IDs using decode_account_id.
        
        This test verifies that the delegate_ss58 and owner_ss58 fields are
        properly decoded from bytes/raw format to SS58 string addresses using
        the decode_account_id utility function. This decoding is essential for
        working with account addresses in a human-readable format.
        """
        # Mock decode_account_id to verify it's called correctly
        with patch("bittensor.core.chain_data.delegate_info_lite.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with raw account ID bytes
            decoded = {
                "delegate_ss58": b"raw_delegate_bytes",  # Raw bytes from chain
                "take": 29491,
                "nominators": 5,
                "owner_ss58": b"raw_owner_bytes",  # Raw bytes from chain
                "registrations": [1],
                "validator_permits": [1],
                "return_per_1000": 0
            }
            
            # Create DelegateInfoLite
            delegate_lite = DelegateInfoLite._from_dict(decoded)
            
            # Verify decode_account_id was called for both account IDs
            assert mock_decode.call_count == 2, \
                "decode_account_id should be called twice (once for delegate, once for owner)"
            mock_decode.assert_any_call(b"raw_delegate_bytes"), \
                "decode_account_id should be called with delegate bytes"
            mock_decode.assert_any_call(b"raw_owner_bytes"), \
                "decode_account_id should be called with owner bytes"

    def test_from_dict_converts_return_per_1000_from_rao(self):
        """
        Test that _from_dict() correctly converts return_per_1000 from rao to Balance.
        
        This test verifies that the return_per_1000 value (which comes from chain
        as rao, the smallest unit) is properly converted to a Balance object using
        Balance.from_rao(). This ensures proper balance handling and unit conversions.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.delegate_info_lite.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with return_per_1000 in rao
            # 1.5 TAO = 1,500,000,000,000 rao
            # 1 TAO = 1,000,000,000,000 rao (10^12)
            decoded = {
                "delegate_ss58": b"delegate_bytes",
                "take": 29491,
                "nominators": 1,
                "owner_ss58": b"owner_bytes",
                "registrations": [1],
                "validator_permits": [1],
                "return_per_1000": 1500000000000  # 1.5 TAO in rao
            }
            
            # Create DelegateInfoLite
            delegate_lite = DelegateInfoLite._from_dict(decoded)
            
            # Verify return_per_1000 is a Balance object with correct value
            assert isinstance(delegate_lite.return_per_1000, Balance), \
                "return_per_1000 should be converted to a Balance object"
            assert delegate_lite.return_per_1000.tao == pytest.approx(1.5, rel=0.01), \
                "return_per_1000 should be correctly converted from rao to TAO (1.5 TAO)"

    def test_from_dict_normalizes_take_value(self):
        """
        Test that _from_dict() correctly normalizes the take value using u16_normalized_float.
        
        This test verifies that the take value (which comes from chain as a u16 integer)
        is properly normalized to a float percentage (0-1 range) using the
        u16_normalized_float utility function. This normalization converts the raw
        u16 value to a human-readable percentage.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.delegate_info_lite.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with take as u16 value
            # u16_normalized_float converts u16 (0-65535) to float (0.0-1.0)
            decoded = {
                "delegate_ss58": b"delegate_bytes",
                "take": 29491,  # This will be normalized by u16_normalized_float
                "nominators": 1,
                "owner_ss58": b"owner_bytes",
                "registrations": [1],
                "validator_permits": [1],
                "return_per_1000": 0
            }
            
            # Create DelegateInfoLite
            delegate_lite = DelegateInfoLite._from_dict(decoded)
            
            # Verify take is a float (normalized)
            assert isinstance(delegate_lite.take, float), \
                "Take should be normalized to float type"
            assert 0 <= delegate_lite.take <= 1, \
                "Take should be in range 0.0-1.0 after normalization (0-100%)"


class TestDelegateInfoLiteFromDictBaseClass:
    """
    Test class for the from_dict() method inherited from InfoBase.
    
    This class tests that DelegateInfoLite can use the from_dict() method from
    InfoBase, which includes error handling for missing fields. The from_dict()
    method calls _from_dict() internally but adds exception handling.
    """

    def test_from_dict_with_complete_data(self):
        """
        Test that from_dict() works with complete data.
        
        This test verifies that the from_dict() method (inherited from InfoBase)
        correctly calls _from_dict() when all required fields are present in
        the dictionary. This is the happy path for creating DelegateInfoLite
        from chain data with proper error handling wrapper.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.delegate_info_lite.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create complete dictionary data
            decoded = {
                "delegate_ss58": b"delegate_bytes",
                "take": 29491,
                "nominators": 5,
                "owner_ss58": b"owner_bytes",
                "registrations": [1, 2],
                "validator_permits": [1, 2],
                "return_per_1000": 1000000000000  # 1.0 TAO in rao
            }
            
            # Create DelegateInfoLite using from_dict (from InfoBase)
            # This method includes error handling for missing fields
            delegate_lite = DelegateInfoLite.from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(delegate_lite, DelegateInfoLite), \
                "from_dict() should return a DelegateInfoLite instance"
            assert delegate_lite.nominators == 5, \
                "Nominators count should be set correctly from dictionary"

    def test_from_dict_raises_exception_on_missing_field(self):
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
            # Missing nominators, owner_ss58, registrations, validator_permits, return_per_1000
        }
        
        # Verify from_dict raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException) as exc_info:
            DelegateInfoLite.from_dict(incomplete_data)
        
        # Verify error message mentions missing field
        assert "missing" in str(exc_info.value).lower(), \
            "Error message should mention that a field is missing"
        assert "DelegateInfoLite" in str(exc_info.value), \
            "Error message should mention DelegateInfoLite class name"


class TestDelegateInfoLiteEdgeCases:
    """
    Test class for edge cases and special scenarios.
    
    This class tests edge cases such as empty lists, zero values, large counts,
    and other boundary conditions that might occur in real-world usage.
    """

    def test_delegate_info_lite_with_empty_registrations(self):
        """
        Test that DelegateInfoLite handles empty registrations list correctly.
        
        This test verifies that a delegate that is not registered on any subnet
        can be represented correctly with an empty registrations list. This is
        a valid state for delegates that haven't registered yet.
        """
        # Create DelegateInfoLite with empty registrations
        delegate_lite = DelegateInfoLite(
            delegate_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            take=0.0,
            nominators=0,
            owner_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            registrations=[],  # Not registered on any subnet yet
            validator_permits=[],
            return_per_1000=Balance.from_tao(0)
        )
        
        # Verify empty registrations is handled correctly
        assert delegate_lite.registrations == [], \
            "Empty registrations list should be valid (not registered on any subnet)"
        assert isinstance(delegate_lite.registrations, list), \
            "Registrations should still be a list type (empty list)"

    def test_delegate_info_lite_with_empty_validator_permits(self):
        """
        Test that DelegateInfoLite handles empty validator_permits list correctly.
        
        This test verifies that a delegate without validator permits can be
        represented correctly with an empty validator_permits list. This is
        valid for delegates that haven't been granted validator permissions yet.
        """
        # Create DelegateInfoLite with empty validator permits
        delegate_lite = DelegateInfoLite(
            delegate_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            take=0.0,
            nominators=0,
            owner_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            registrations=[1],  # Registered on subnet 1
            validator_permits=[],  # No validator permits granted yet
            return_per_1000=Balance.from_tao(0)
        )
        
        # Verify empty validator permits is handled correctly
        assert delegate_lite.validator_permits == [], \
            "Empty validator_permits list should be valid (no permits granted)"
        assert isinstance(delegate_lite.validator_permits, list), \
            "Validator permits should still be a list type (empty list)"

    def test_delegate_info_lite_field_types(self):
        """
        Test that DelegateInfoLite fields maintain correct types.
        
        This test verifies that all fields in DelegateInfoLite maintain their
        expected types. This is important for type consistency and ensures that
        the dataclass properly enforces type constraints at runtime.
        """
        # Create DelegateInfoLite
        delegate_lite = DelegateInfoLite(
            delegate_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            take=0.18,
            nominators=5,
            owner_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            registrations=[1, 2],
            validator_permits=[1],
            return_per_1000=Balance.from_tao(1.0)
        )
        
        # Verify all field types are correct
        assert isinstance(delegate_lite.delegate_ss58, str), \
            "delegate_ss58 should be string type (SS58 address)"
        assert isinstance(delegate_lite.take, float), \
            "take should be float type (normalized percentage)"
        assert isinstance(delegate_lite.nominators, int), \
            "nominators should be int type (count, not detailed stakes)"
        assert isinstance(delegate_lite.owner_ss58, str), \
            "owner_ss58 should be string type (SS58 address)"
        assert isinstance(delegate_lite.registrations, list), \
            "registrations should be list type (list of subnet IDs)"
        assert isinstance(delegate_lite.validator_permits, list), \
            "validator_permits should be list type (list of subnet IDs)"
        assert isinstance(delegate_lite.return_per_1000, Balance), \
            "return_per_1000 should be Balance type (TAO returns)"

    def test_delegate_info_lite_with_large_nominator_count(self):
        """
        Test that DelegateInfoLite handles large nominator counts correctly.
        
        This test verifies that DelegateInfoLite can handle delegates with
        many nominators (represented as a large integer count). This is
        important for popular delegates that might have hundreds or thousands
        of nominators.
        """
        # Create DelegateInfoLite with large nominator count
        delegate_lite = DelegateInfoLite(
            delegate_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            take=0.18,
            nominators=1000,  # Large number of nominators
            owner_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            registrations=[1, 2, 3],
            validator_permits=[1, 2],
            return_per_1000=Balance.from_tao(10.0)
        )
        
        # Verify large nominator count is handled
        assert delegate_lite.nominators == 1000, \
            "Large nominator count should be stored correctly"
        assert isinstance(delegate_lite.nominators, int), \
            "Nominators should still be integer type regardless of size"

