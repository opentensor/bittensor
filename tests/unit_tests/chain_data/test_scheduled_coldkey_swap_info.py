"""
Comprehensive unit tests for the bittensor.core.chain_data.scheduled_coldkey_swap_info module.

This test suite covers all major components of the ScheduledColdkeySwapInfo class including:
- Class instantiation and attribute validation
- Dictionary conversion (_from_dict, from_dict from InfoBase)
- SS58 encoding for old_coldkey and new_coldkey
- decode_account_id_list class method
- Inheritance from InfoBase
- Edge cases and error handling

The tests are designed to ensure that:
1. ScheduledColdkeySwapInfo objects can be created correctly with all required fields
2. Dictionary conversion works correctly with chain data format
3. Account IDs are properly encoded to SS58 addresses
4. decode_account_id_list method works correctly
5. Error handling is robust for missing or invalid data
6. All methods handle edge cases properly

ScheduledColdkeySwapInfo represents information about scheduled coldkey swaps,
including the old and new coldkey addresses and the arbitration block number.

Each test includes extensive comments explaining:
- What functionality is being tested
- Why the test is important
- What assertions verify
- Expected behavior and edge cases
"""

from unittest.mock import MagicMock, patch

import pytest

# Import the modules to test
from bittensor.core.chain_data.scheduled_coldkey_swap_info import ScheduledColdkeySwapInfo
from bittensor.core.errors import SubstrateRequestException


class TestScheduledColdkeySwapInfoInitialization:
    """
    Test class for ScheduledColdkeySwapInfo object initialization.
    
    This class tests that ScheduledColdkeySwapInfo objects can be created correctly
    with all required fields. ScheduledColdkeySwapInfo contains old coldkey, new
    coldkey, and arbitration block information.
    """

    def test_scheduled_coldkey_swap_info_initialization_with_all_fields(self):
        """
        Test that ScheduledColdkeySwapInfo can be initialized with all required fields.
        
        This test verifies that a ScheduledColdkeySwapInfo object can be created with
        all required fields. ScheduledColdkeySwapInfo contains information about a
        scheduled coldkey swap including the old coldkey address, new coldkey address,
        and the block number at which arbitration will take place.
        """
        # Create a ScheduledColdkeySwapInfo with all fields
        swap_info = ScheduledColdkeySwapInfo(
            old_coldkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # Old coldkey SS58 address
            new_coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # New coldkey SS58 address
            arbitration_block=10000,  # Block number for arbitration
        )
        
        # Verify all fields are set correctly
        assert swap_info.old_coldkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "Old coldkey SS58 address should be set correctly"
        assert swap_info.new_coldkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", \
            "New coldkey SS58 address should be set correctly"
        assert swap_info.arbitration_block == 10000, \
            "Arbitration block number should be set correctly"

    def test_scheduled_coldkey_swap_info_inherits_from_info_base(self):
        """
        Test that ScheduledColdkeySwapInfo properly inherits from InfoBase.
        
        This test verifies that ScheduledColdkeySwapInfo is a subclass of InfoBase,
        which provides common functionality for chain data structures. This ensures
        that ScheduledColdkeySwapInfo can use methods like from_dict() from the base class.
        """
        from bittensor.core.chain_data.info_base import InfoBase
        assert issubclass(ScheduledColdkeySwapInfo, InfoBase), \
            "ScheduledColdkeySwapInfo should inherit from InfoBase for common chain data functionality"
        
        from dataclasses import is_dataclass
        assert is_dataclass(ScheduledColdkeySwapInfo), \
            "ScheduledColdkeySwapInfo should be a dataclass for automatic field handling"


class TestScheduledColdkeySwapInfoFromDict:
    """
    Test class for the _from_dict() class method.
    
    This class tests that ScheduledColdkeySwapInfo objects can be created from dictionary
    data. The conversion includes SS58 encoding of account IDs using ss58_encode.
    """

    def test_from_dict_creates_scheduled_coldkey_swap_info_correctly(self):
        """
        Test that _from_dict() correctly creates ScheduledColdkeySwapInfo from dictionary data.
        
        This test verifies that when given a dictionary with scheduled coldkey swap information
        (as would come from chain data), the _from_dict() method correctly creates a
        ScheduledColdkeySwapInfo object. The conversion includes encoding account IDs
        to SS58 addresses using ss58_encode.
        """
        # Mock ss58_encode for coldkey encoding
        with patch("bittensor.core.chain_data.scheduled_coldkey_swap_info.ss58_encode") as mock_encode:
            mock_encode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # new_coldkey
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # old_coldkey
            ]
            
            # Create dictionary data as would come from chain
            # Note: Account IDs come as raw format that need to be encoded to SS58
            decoded = {
                "old_coldkey": b"old_coldkey_bytes",  # Raw bytes that will be encoded
                "new_coldkey": b"new_coldkey_bytes",  # Raw bytes that will be encoded
                "arbitration_block": 10000,  # Block number
            }
            
            # Create ScheduledColdkeySwapInfo from dictionary using _from_dict class method
            swap_info = ScheduledColdkeySwapInfo._from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(swap_info, ScheduledColdkeySwapInfo), \
                "Should return a ScheduledColdkeySwapInfo instance"
            
            # Verify arbitration_block is set correctly
            assert swap_info.arbitration_block == 10000, \
                "Arbitration block should be set correctly from dictionary"
            
            # Verify ss58_encode was called for both coldkeys
            assert mock_encode.call_count == 2, \
                "ss58_encode should be called twice (once for old_coldkey, once for new_coldkey)"
            mock_encode.assert_any_call(b"old_coldkey_bytes", 42), \
                "ss58_encode should be called with old_coldkey bytes and SS58_FORMAT"
            mock_encode.assert_any_call(b"new_coldkey_bytes", 42), \
                "ss58_encode should be called with new_coldkey bytes and SS58_FORMAT"

    def test_from_dict_encodes_coldkeys_to_ss58(self):
        """
        Test that _from_dict() correctly encodes coldkeys to SS58 addresses.
        
        This test verifies that the old_coldkey and new_coldkey fields are properly
        encoded from raw format to SS58 string addresses using the ss58_encode utility
        function. This encoding is essential for working with account addresses in a
        human-readable format.
        """
        # Mock ss58_encode to verify it's called correctly
        with patch("bittensor.core.chain_data.scheduled_coldkey_swap_info.ss58_encode") as mock_encode:
            mock_encode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # new_coldkey
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # old_coldkey
            ]
            
            # Create dictionary with raw coldkey bytes
            decoded = {
                "old_coldkey": b"raw_old_coldkey_bytes",  # Raw bytes from chain
                "new_coldkey": b"raw_new_coldkey_bytes",  # Raw bytes from chain
                "arbitration_block": 10000,
            }
            
            # Create ScheduledColdkeySwapInfo
            swap_info = ScheduledColdkeySwapInfo._from_dict(decoded)
            
            # Verify ss58_encode was called with correct parameters
            # Note: SS58_FORMAT is 42 (Bittensor format)
            from bittensor_wallet.utils import SS58_FORMAT
            assert mock_encode.call_count == 2, \
                "ss58_encode should be called twice"
            mock_encode.assert_any_call(b"raw_old_coldkey_bytes", SS58_FORMAT), \
                "ss58_encode should be called with old_coldkey bytes and SS58_FORMAT"
            mock_encode.assert_any_call(b"raw_new_coldkey_bytes", SS58_FORMAT), \
                "ss58_encode should be called with new_coldkey bytes and SS58_FORMAT"


class TestScheduledColdkeySwapInfoFromDictBaseClass:
    """
    Test class for the from_dict() method inherited from InfoBase.
    
    This class tests that ScheduledColdkeySwapInfo can use the from_dict() method from
    InfoBase, which includes error handling for missing fields.
    """

    def test_from_dict_with_complete_data(self):
        """
        Test that from_dict() works with complete data.
        
        This test verifies that the from_dict() method (inherited from InfoBase)
        correctly calls _from_dict() when all required fields are present in
        the dictionary. This is the happy path for creating ScheduledColdkeySwapInfo
        from chain data with proper error handling wrapper.
        """
        # Mock ss58_encode
        with patch("bittensor.core.chain_data.scheduled_coldkey_swap_info.ss58_encode") as mock_encode:
            mock_encode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create complete dictionary data
            decoded = {
                "old_coldkey": b"old_coldkey_bytes",
                "new_coldkey": b"new_coldkey_bytes",
                "arbitration_block": 10000,
            }
            
            # Create ScheduledColdkeySwapInfo using from_dict (from InfoBase)
            # This method includes error handling for missing fields
            swap_info = ScheduledColdkeySwapInfo.from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(swap_info, ScheduledColdkeySwapInfo), \
                "from_dict() should return a ScheduledColdkeySwapInfo instance"
            assert swap_info.arbitration_block == 10000, \
                "Arbitration block should be set correctly from dictionary"

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
            "old_coldkey": b"old_coldkey_bytes",
            # Missing new_coldkey and arbitration_block
        }
        
        # Verify from_dict raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException) as exc_info:
            ScheduledColdkeySwapInfo.from_dict(incomplete_data)
        
        # Verify error message mentions missing field
        assert "missing" in str(exc_info.value).lower(), \
            "Error message should mention that a field is missing"
        assert "ScheduledColdkeySwapInfo" in str(exc_info.value), \
            "Error message should mention ScheduledColdkeySwapInfo class name"


class TestScheduledColdkeySwapInfoDecodeAccountIdList:
    """
    Test class for the decode_account_id_list() class method.
    
    This class tests the decode_account_id_list method which decodes a list of
    AccountIds from vec_u8 format. This is a utility method for handling lists
    of account IDs.
    """

    def test_decode_account_id_list_decodes_list_correctly(self):
        """
        Test that decode_account_id_list() correctly decodes a list of AccountIds.
        
        This test verifies that the decode_account_id_list() method correctly
        decodes a list of account IDs from vec_u8 format (list of integers)
        and returns a list of SS58-encoded address strings.
        """
        # Mock from_scale_encoding to return decoded account IDs
        mock_decoded_account_ids = [
            b"account_id_1_bytes",
            b"account_id_2_bytes",
            b"account_id_3_bytes",
        ]
        
        with patch(
            "bittensor.core.chain_data.scheduled_coldkey_swap_info.from_scale_encoding"
        ) as mock_from_scale, patch(
            "bittensor.core.chain_data.scheduled_coldkey_swap_info.ss58_encode"
        ) as mock_encode:
            # Mock from_scale_encoding to return decoded account IDs
            mock_from_scale.return_value = mock_decoded_account_ids
            
            # Mock ss58_encode to return SS58 addresses
            mock_encode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "5GNJqTPyNqANBkUVMN1LPprxXnFouWXoe2wNSmmEoLctxiZY",
            ]
            
            # Create vec_u8 list (list of integers representing bytes)
            vec_u8 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            
            # Call decode_account_id_list
            result = ScheduledColdkeySwapInfo.decode_account_id_list(vec_u8)
            
            # Verify result is a list
            assert isinstance(result, list), \
                "decode_account_id_list should return a list"
            assert len(result) == 3, \
                "Should return 3 SS58 addresses (one per decoded account ID)"
            
            # Verify ss58_encode was called for each account ID
            assert mock_encode.call_count == 3, \
                "ss58_encode should be called once per account ID"
            assert result[0] == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
                "First account ID should be encoded to SS58 address"

    def test_decode_account_id_list_returns_none_on_failed_decode(self):
        """
        Test that decode_account_id_list() returns None when decoding fails.
        
        This test verifies that when from_scale_encoding returns None (indicating
        a failed decode), the decode_account_id_list() method returns None instead
        of raising an exception. This allows graceful handling of invalid data.
        """
        # Mock from_scale_encoding to return None (failed decode)
        with patch(
            "bittensor.core.chain_data.scheduled_coldkey_swap_info.from_scale_encoding"
        ) as mock_from_scale:
            mock_from_scale.return_value = None  # Simulate failed decode
            
            # Create vec_u8 list
            vec_u8 = [1, 2, 3, 4, 5]
            
            # Call decode_account_id_list
            result = ScheduledColdkeySwapInfo.decode_account_id_list(vec_u8)
            
            # Verify result is None
            assert result is None, \
                "decode_account_id_list should return None when decoding fails"

    def test_decode_account_id_list_handles_empty_list(self):
        """
        Test that decode_account_id_list() handles empty list correctly.
        
        This test verifies that when an empty list is passed (or decoding results
        in an empty list), the method returns an empty list of SS58 addresses.
        """
        # Mock from_scale_encoding to return empty list
        with patch(
            "bittensor.core.chain_data.scheduled_coldkey_swap_info.from_scale_encoding"
        ) as mock_from_scale:
            mock_from_scale.return_value = []  # Empty decoded list
            
            # Create vec_u8 list
            vec_u8 = []
            
            # Call decode_account_id_list
            result = ScheduledColdkeySwapInfo.decode_account_id_list(vec_u8)
            
            # Verify result is an empty list
            assert isinstance(result, list), \
                "decode_account_id_list should return a list even when empty"
            assert len(result) == 0, \
                "Should return empty list when decoding results in no account IDs"


class TestScheduledColdkeySwapInfoEdgeCases:
    """
    Test class for edge cases and special scenarios.
    
    This class tests edge cases such as zero arbitration block, same old and new
    coldkeys, and other boundary conditions.
    """

    def test_scheduled_coldkey_swap_info_with_zero_arbitration_block(self):
        """
        Test that ScheduledColdkeySwapInfo handles zero arbitration block correctly.
        
        This test verifies that an arbitration block of zero is handled correctly.
        This might represent an immediate arbitration or special case.
        """
        # Mock ss58_encode
        with patch("bittensor.core.chain_data.scheduled_coldkey_swap_info.ss58_encode") as mock_encode:
            mock_encode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            decoded = {
                "old_coldkey": b"old_coldkey_bytes",
                "new_coldkey": b"new_coldkey_bytes",
                "arbitration_block": 0,  # Zero arbitration block
            }
            
            swap_info = ScheduledColdkeySwapInfo._from_dict(decoded)
            
            # Verify zero arbitration block is handled
            assert swap_info.arbitration_block == 0, \
                "Zero arbitration block should be handled correctly"

    def test_scheduled_coldkey_swap_info_field_types(self):
        """
        Test that ScheduledColdkeySwapInfo fields maintain correct types.
        
        This test verifies that all fields in ScheduledColdkeySwapInfo maintain their
        expected types. This is important for type consistency and ensures that
        the dataclass properly enforces type constraints at runtime.
        """
        swap_info = ScheduledColdkeySwapInfo(
            old_coldkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            new_coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            arbitration_block=10000,
        )
        
        # Verify all field types are correct
        assert isinstance(swap_info.old_coldkey, str), \
            "old_coldkey should be string type (SS58 address)"
        assert isinstance(swap_info.new_coldkey, str), \
            "new_coldkey should be string type (SS58 address)"
        assert isinstance(swap_info.arbitration_block, int), \
            "arbitration_block should be int type (block number)"

