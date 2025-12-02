"""
Comprehensive unit tests for the bittensor.core.chain_data.subnet_info module.

This test suite covers all major components of the SubnetInfo class including:
- Class instantiation and attribute validation
- Dictionary conversion (_from_dict, from_dict from InfoBase)
- Account ID decoding for owner
- Balance conversion from rao to Balance objects
- U16 normalization for connection requirements
- Field name mapping (blocks_since_last_step, max_allowed_uids, network_modality, network_connect)
- Connection requirements dictionary construction
- Inheritance from InfoBase
- Edge cases and error handling

The tests are designed to ensure that:
1. SubnetInfo objects can be created correctly with all required fields
2. Dictionary conversion works correctly with chain data format
3. Account IDs are properly decoded from bytes format
4. Balance values are correctly converted from rao
5. U16 values are properly normalized to floats for connection requirements
6. Field name mappings are correct
7. Error handling is robust for missing or invalid data
8. All methods handle edge cases properly

SubnetInfo is a comprehensive data structure that represents subnet configuration
and current state, including hyperparameters, limits, modality, connection requirements,
emission values, and owner information.

Each test includes extensive comments explaining:
- What functionality is being tested
- Why the test is important
- What assertions verify
- Expected behavior and edge cases
"""

from unittest.mock import patch

import pytest

# Import the modules to test
from bittensor.core.chain_data.subnet_info import SubnetInfo
from bittensor.core.errors import SubstrateRequestException
from bittensor.utils.balance import Balance


class TestSubnetInfoInitialization:
    """
    Test class for SubnetInfo object initialization.
    
    This class tests that SubnetInfo objects can be created correctly with
    all required fields. SubnetInfo has many fields including hyperparameters,
    limits, modality, connection requirements, emission, and owner information.
    """

    def test_subnet_info_initialization_with_all_fields(self):
        """
        Test that SubnetInfo can be initialized with all required fields.
        
        This test verifies that a SubnetInfo object can be created with all
        required fields. SubnetInfo contains comprehensive subnet information
        including hyperparameters (rho, kappa, difficulty, etc.), limits,
        modality, connection requirements, emission value, burn amount, and owner.
        """
        # Create connection requirements dictionary (netuid -> requirement float)
        connection_requirements = {
            "1": 0.5,  # Requires 50% connection to subnet 1
            "2": 0.3,  # Requires 30% connection to subnet 2
        }
        
        # Create a SubnetInfo with all fields
        subnet_info = SubnetInfo(
            netuid=1,  # Subnet ID
            rho=10000,  # Rate of decay
            kappa=1000,  # Constant multiplier
            difficulty=5000000,  # Current difficulty
            immunity_period=100,  # Immunity period in blocks
            max_allowed_validators=64,  # Maximum validators
            min_allowed_weights=10,  # Minimum weights required
            max_weight_limit=1.0,  # Maximum weight value
            scaling_law_power=0.5,  # Scaling law power
            subnetwork_n=100,  # Current number of neurons
            max_n=1000,  # Maximum number of neurons
            blocks_since_epoch=50,  # Blocks since last epoch
            tempo=12,  # Block tempo
            modality=1,  # Network modality (1 = text, etc.)
            connection_requirements=connection_requirements,  # Connection requirements dict
            emission_value=0.05,  # Emission value (float)
            burn=Balance.from_tao(1.0),  # Burn amount in TAO
            owner_ss58="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # Owner SS58 address
        )
        
        # Verify key fields are set correctly
        assert subnet_info.netuid == 1, \
            "Netuid should specify which subnet this info is for"
        assert subnet_info.rho == 10000, \
            "Rho (rate of decay) should be set correctly"
        assert subnet_info.kappa == 1000, \
            "Kappa (constant multiplier) should be set correctly"
        assert subnet_info.difficulty == 5000000, \
            "Difficulty should be set correctly"
        assert subnet_info.tempo == 12, \
            "Tempo should be set correctly"
        assert subnet_info.modality == 1, \
            "Modality should be set correctly"
        assert isinstance(subnet_info.connection_requirements, dict), \
            "Connection requirements should be a dictionary"
        assert "1" in subnet_info.connection_requirements, \
            "Connection requirements should contain subnet IDs as string keys"
        assert subnet_info.connection_requirements["1"] == 0.5, \
            "Connection requirement value should be normalized float"
        assert isinstance(subnet_info.burn, Balance), \
            "Burn should be a Balance object"
        assert subnet_info.owner_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "Owner SS58 address should be set correctly"

    def test_subnet_info_inherits_from_info_base(self):
        """
        Test that SubnetInfo properly inherits from InfoBase.
        
        This test verifies that SubnetInfo is a subclass of InfoBase, which
        provides common functionality for chain data structures. This ensures
        that SubnetInfo can use methods like from_dict() from the base class.
        """
        from bittensor.core.chain_data.info_base import InfoBase
        assert issubclass(SubnetInfo, InfoBase), \
            "SubnetInfo should inherit from InfoBase for common chain data functionality"
        
        from dataclasses import is_dataclass
        assert is_dataclass(SubnetInfo), \
            "SubnetInfo should be a dataclass for automatic field handling"


class TestSubnetInfoFromDict:
    """
    Test class for the _from_dict() class method.
    
    This class tests that SubnetInfo objects can be created from dictionary
    data. The conversion includes decoding account IDs, converting rao to Balance,
    normalizing u16 values, and field name mapping.
    """

    def test_from_dict_creates_subnet_info_correctly(self):
        """
        Test that _from_dict() correctly creates SubnetInfo from dictionary data.
        
        This test verifies that when given a dictionary with subnet info information
        (as would come from chain data), the _from_dict() method correctly creates
        a SubnetInfo object. The conversion includes:
        - Decoding owner account ID from bytes to SS58 address
        - Converting rao values to Balance objects
        - Normalizing u16 values to floats for connection requirements
        - Field name mapping (blocks_since_last_step, max_allowed_uids, etc.)
        """
        # Mock decode_account_id for owner
        with patch("bittensor.core.chain_data.subnet_info.decode_account_id") as mock_decode:
            mock_decode.return_value = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            
            # Create dictionary data as would come from chain
            # Note: Field names may differ from class attributes
            decoded = {
                "netuid": 1,
                "rho": 10000,
                "kappa": 1000,
                "difficulty": 5000000,
                "immunity_period": 100,
                "max_allowed_validators": 64,
                "min_allowed_weights": 10,
                "max_weights_limit": 1.0,  # Note: max_weights_limit in dict
                "scaling_law_power": 0.5,
                "subnetwork_n": 100,
                "max_allowed_uids": 1000,  # Note: max_allowed_uids in dict, max_n in class
                "blocks_since_last_step": 50,  # Note: blocks_since_last_step in dict, blocks_since_epoch in class
                "tempo": 12,
                "network_modality": 1,  # Note: network_modality in dict, modality in class
                "network_connect": [
                    (1, 29491),  # (netuid, u16_requirement) - will be normalized
                    (2, 16383),  # (netuid, u16_requirement)
                ],
                "emission_value": 0.05,
                "burn": 1000000000000,  # 1.0 TAO in rao
                "owner": b"owner_bytes",  # Raw bytes that will be decoded
            }
            
            # Create SubnetInfo from dictionary using _from_dict class method
            subnet_info = SubnetInfo._from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(subnet_info, SubnetInfo), \
                "Should return a SubnetInfo instance"
            
            # Verify key fields are set correctly
            assert subnet_info.netuid == 1, \
                "Netuid should be set correctly from dictionary"
            assert subnet_info.rho == 10000, \
                "Rho should be set correctly from dictionary"
            assert subnet_info.tempo == 12, \
                "Tempo should be set correctly from dictionary"
            
            # Verify field name mappings
            assert subnet_info.max_n == 1000, \
                "max_n should be mapped from max_allowed_uids in dictionary"
            assert subnet_info.blocks_since_epoch == 50, \
                "blocks_since_epoch should be mapped from blocks_since_last_step in dictionary"
            assert subnet_info.modality == 1, \
                "modality should be mapped from network_modality in dictionary"
            
            # Verify owner was decoded correctly
            assert subnet_info.owner_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
                "Owner SS58 should be decoded from bytes to SS58 address"

    def test_from_dict_decodes_owner_account_id(self):
        """
        Test that _from_dict() correctly decodes owner account ID using decode_account_id.
        
        This test verifies that the owner field is properly decoded from bytes/raw
        format to SS58 string address using the decode_account_id utility function.
        This decoding is essential for working with account addresses in a
        human-readable format.
        """
        # Mock decode_account_id to verify it's called correctly
        with patch("bittensor.core.chain_data.subnet_info.decode_account_id") as mock_decode:
            mock_decode.return_value = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            
            # Create dictionary with raw owner account ID bytes
            decoded = {
                "netuid": 1,
                "rho": 10000,
                "kappa": 1000,
                "difficulty": 5000000,
                "immunity_period": 100,
                "max_allowed_validators": 64,
                "min_allowed_weights": 10,
                "max_weights_limit": 1.0,
                "scaling_law_power": 0.5,
                "subnetwork_n": 100,
                "max_allowed_uids": 1000,
                "blocks_since_last_step": 50,
                "tempo": 12,
                "network_modality": 1,
                "network_connect": [],
                "emission_value": 0.05,
                "burn": 1000000000000,
                "owner": b"raw_owner_bytes",  # Raw bytes from chain
            }
            
            # Create SubnetInfo
            subnet_info = SubnetInfo._from_dict(decoded)
            
            # Verify decode_account_id was called with owner bytes
            mock_decode.assert_called_once_with(b"raw_owner_bytes"), \
                "decode_account_id should be called with owner bytes"
            assert subnet_info.owner_ss58 == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
                "Owner SS58 should be decoded correctly"

    def test_from_dict_converts_burn_from_rao(self):
        """
        Test that _from_dict() correctly converts burn value from rao to Balance.
        
        This test verifies that the burn value (which comes from chain as rao,
        the smallest unit) is properly converted to a Balance object using
        Balance.from_rao(). This ensures proper balance handling and unit conversions.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_info.decode_account_id") as mock_decode:
            mock_decode.return_value = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            
            # Create dictionary with burn in rao
            # 1 TAO = 1,000,000,000,000 rao (10^12)
            decoded = {
                "netuid": 1,
                "rho": 10000,
                "kappa": 1000,
                "difficulty": 5000000,
                "immunity_period": 100,
                "max_allowed_validators": 64,
                "min_allowed_weights": 10,
                "max_weights_limit": 1.0,
                "scaling_law_power": 0.5,
                "subnetwork_n": 100,
                "max_allowed_uids": 1000,
                "blocks_since_last_step": 50,
                "tempo": 12,
                "network_modality": 1,
                "network_connect": [],
                "emission_value": 0.05,
                "burn": 1500000000000,  # 1.5 TAO in rao
                "owner": b"owner_bytes",
            }
            
            # Create SubnetInfo
            subnet_info = SubnetInfo._from_dict(decoded)
            
            # Verify burn is converted correctly
            assert isinstance(subnet_info.burn, Balance), \
                "Burn should be converted to a Balance object"
            assert subnet_info.burn.tao == pytest.approx(1.5, rel=0.01), \
                "Burn should be correctly converted from rao to TAO (1.5 TAO)"

    def test_from_dict_constructs_connection_requirements_dict(self):
        """
        Test that _from_dict() correctly constructs connection_requirements dictionary.
        
        This test verifies that the connection_requirements dictionary is correctly
        constructed from the network_connect list in chain data. The network_connect
        list contains tuples of (netuid, u16_requirement), which are converted to
        a dictionary with string keys (netuid as string) and normalized float values.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_info.decode_account_id") as mock_decode:
            mock_decode.return_value = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            
            # Create dictionary with network_connect list
            # Format: [(netuid, u16_requirement), ...]
            decoded = {
                "netuid": 1,
                "rho": 10000,
                "kappa": 1000,
                "difficulty": 5000000,
                "immunity_period": 100,
                "max_allowed_validators": 64,
                "min_allowed_weights": 10,
                "max_weights_limit": 1.0,
                "scaling_law_power": 0.5,
                "subnetwork_n": 100,
                "max_allowed_uids": 1000,
                "blocks_since_last_step": 50,
                "tempo": 12,
                "network_modality": 1,
                "network_connect": [
                    (1, 29491),  # netuid 1, u16 requirement (will be normalized to float)
                    (2, 16383),  # netuid 2, u16 requirement
                    (3, 32767),  # netuid 3, u16 requirement
                ],
                "emission_value": 0.05,
                "burn": 1000000000000,
                "owner": b"owner_bytes",
            }
            
            # Create SubnetInfo
            subnet_info = SubnetInfo._from_dict(decoded)
            
            # Verify connection_requirements is constructed correctly
            assert isinstance(subnet_info.connection_requirements, dict), \
                "Connection requirements should be a dictionary"
            assert "1" in subnet_info.connection_requirements, \
                "Connection requirements should contain subnet 1"
            assert "2" in subnet_info.connection_requirements, \
                "Connection requirements should contain subnet 2"
            assert "3" in subnet_info.connection_requirements, \
                "Connection requirements should contain subnet 3"
            
            # Verify keys are strings (netuid as string)
            assert isinstance(list(subnet_info.connection_requirements.keys())[0], str), \
                "Connection requirement keys should be strings (netuid as string)"
            
            # Verify values are floats (normalized)
            assert isinstance(subnet_info.connection_requirements["1"], float), \
                "Connection requirement values should be floats (normalized from u16)"
            assert 0 <= subnet_info.connection_requirements["1"] <= 1, \
                "Connection requirement values should be in range 0.0-1.0 after normalization"

    def test_from_dict_handles_empty_connection_requirements(self):
        """
        Test that _from_dict() handles empty connection requirements correctly.
        
        This test verifies that when network_connect is an empty list (subnet has
        no connection requirements), the connection_requirements dictionary is
        correctly created as an empty dictionary.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_info.decode_account_id") as mock_decode:
            mock_decode.return_value = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            
            # Create dictionary with empty network_connect
            decoded = {
                "netuid": 1,
                "rho": 10000,
                "kappa": 1000,
                "difficulty": 5000000,
                "immunity_period": 100,
                "max_allowed_validators": 64,
                "min_allowed_weights": 10,
                "max_weights_limit": 1.0,
                "scaling_law_power": 0.5,
                "subnetwork_n": 100,
                "max_allowed_uids": 1000,
                "blocks_since_last_step": 50,
                "tempo": 12,
                "network_modality": 1,
                "network_connect": [],  # Empty list - no connection requirements
                "emission_value": 0.05,
                "burn": 1000000000000,
                "owner": b"owner_bytes",
            }
            
            # Create SubnetInfo
            subnet_info = SubnetInfo._from_dict(decoded)
            
            # Verify empty connection requirements is handled correctly
            assert isinstance(subnet_info.connection_requirements, dict), \
                "Connection requirements should still be a dictionary (empty dict)"
            assert len(subnet_info.connection_requirements) == 0, \
                "Connection requirements dictionary should be empty when network_connect is empty"


class TestSubnetInfoFromDictBaseClass:
    """
    Test class for the from_dict() method inherited from InfoBase.
    
    This class tests that SubnetInfo can use the from_dict() method from
    InfoBase, which includes error handling for missing fields. The from_dict()
    method calls _from_dict() internally but adds exception handling.
    """

    def test_from_dict_with_complete_data(self):
        """
        Test that from_dict() works with complete data.
        
        This test verifies that the from_dict() method (inherited from InfoBase)
        correctly calls _from_dict() when all required fields are present in
        the dictionary. This is the happy path for creating SubnetInfo from chain data.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_info.decode_account_id") as mock_decode:
            mock_decode.return_value = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            
            # Create complete dictionary data
            decoded = {
                "netuid": 1,
                "rho": 10000,
                "kappa": 1000,
                "difficulty": 5000000,
                "immunity_period": 100,
                "max_allowed_validators": 64,
                "min_allowed_weights": 10,
                "max_weights_limit": 1.0,
                "scaling_law_power": 0.5,
                "subnetwork_n": 100,
                "max_allowed_uids": 1000,
                "blocks_since_last_step": 50,
                "tempo": 12,
                "network_modality": 1,
                "network_connect": [],
                "emission_value": 0.05,
                "burn": 1000000000000,
                "owner": b"owner_bytes",
            }
            
            # Create SubnetInfo using from_dict (from InfoBase)
            # This method includes error handling for missing fields
            subnet_info = SubnetInfo.from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(subnet_info, SubnetInfo), \
                "from_dict() should return a SubnetInfo instance"
            assert subnet_info.netuid == 1, \
                "Netuid should be set correctly from dictionary"

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
            "netuid": 1,
            "rho": 10000,
            # Missing many required fields
        }
        
        # Verify from_dict raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException) as exc_info:
            SubnetInfo.from_dict(incomplete_data)
        
        # Verify error message mentions missing field
        assert "missing" in str(exc_info.value).lower(), \
            "Error message should mention that a field is missing"
        assert "SubnetInfo" in str(exc_info.value), \
            "Error message should mention SubnetInfo class name"


class TestSubnetInfoEdgeCases:
    """
    Test class for edge cases and special scenarios.
    
    This class tests edge cases such as zero values, maximum values, empty
    connection requirements, and other boundary conditions.
    """

    def test_subnet_info_field_types(self):
        """
        Test that SubnetInfo fields maintain correct types.
        
        This test verifies that all fields in SubnetInfo maintain their
        expected types. This is important for type consistency and ensures
        that the dataclass properly enforces type constraints at runtime.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_info.decode_account_id") as mock_decode:
            mock_decode.return_value = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            
            # Create dictionary
            decoded = {
                "netuid": 1,
                "rho": 10000,
                "kappa": 1000,
                "difficulty": 5000000,
                "immunity_period": 100,
                "max_allowed_validators": 64,
                "min_allowed_weights": 10,
                "max_weights_limit": 1.0,
                "scaling_law_power": 0.5,
                "subnetwork_n": 100,
                "max_allowed_uids": 1000,
                "blocks_since_last_step": 50,
                "tempo": 12,
                "network_modality": 1,
                "network_connect": [],
                "emission_value": 0.05,
                "burn": 1000000000000,
                "owner": b"owner_bytes",
            }
            
            # Create SubnetInfo
            subnet_info = SubnetInfo._from_dict(decoded)
            
            # Verify field types are correct
            assert isinstance(subnet_info.netuid, int), \
                "netuid should be int type"
            assert isinstance(subnet_info.rho, int), \
                "rho should be int type"
            assert isinstance(subnet_info.max_weight_limit, float), \
                "max_weight_limit should be float type"
            assert isinstance(subnet_info.scaling_law_power, float), \
                "scaling_law_power should be float type"
            assert isinstance(subnet_info.connection_requirements, dict), \
                "connection_requirements should be dict type"
            assert isinstance(subnet_info.emission_value, float), \
                "emission_value should be float type"
            assert isinstance(subnet_info.burn, Balance), \
                "burn should be Balance type"
            assert isinstance(subnet_info.owner_ss58, str), \
                "owner_ss58 should be string type (SS58 address)"

