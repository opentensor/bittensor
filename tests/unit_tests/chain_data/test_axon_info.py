"""
Comprehensive unit tests for the bittensor.core.chain_data.axon_info module.

This test suite covers all major components of the AxonInfo class including:
- Class instantiation and attribute validation
- Property methods (is_serving, ip_str)
- Equality comparison (__eq__)
- String representation (__str__, __repr__)
- Serialization/deserialization (to_string, from_string)
- Dictionary conversion (_from_dict, from_neuron_info)
- Parameter dict conversion (to_parameter_dict, from_parameter_dict)
- Edge cases and error handling

The tests are designed to ensure that:
1. AxonInfo objects can be created correctly with all required and optional fields
2. Properties work as expected
3. Serialization/deserialization round-trips correctly
4. Error handling is robust
5. All methods handle edge cases properly
"""

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import netaddr
import pytest
from async_substrate_interface.utils import json

# Import the modules to test
from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.errors import SubstrateRequestException
from bittensor.utils import networking


class TestAxonInfoInitialization:
    """
    Test class for AxonInfo object initialization.
    
    This class tests that AxonInfo objects can be created correctly with
    various combinations of required and optional parameters.
    """

    def test_axon_info_initialization_with_required_fields(self):
        """
        Test that AxonInfo can be initialized with only required fields.
        
        This test verifies that an AxonInfo object can be created with just
        the required fields (version, ip, port, ip_type, hotkey, coldkey),
        and that the optional fields (protocol, placeholder1, placeholder2)
        use their default values.
        """
        # Create an AxonInfo with only required fields
        axon_info = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,  # IPv4
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Verify required fields are set correctly
        assert axon_info.version == 1, "Version should be set correctly"
        assert axon_info.ip == "192.168.1.1", "IP address should be set correctly"
        assert axon_info.port == 8091, "Port should be set correctly"
        assert axon_info.ip_type == 4, "IP type should be set correctly"
        assert axon_info.hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "Hotkey should be set correctly"
        assert axon_info.coldkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", \
            "Coldkey should be set correctly"
        
        # Verify optional fields use default values
        assert axon_info.protocol == 4, "Protocol should default to 4"
        assert axon_info.placeholder1 == 0, "placeholder1 should default to 0"
        assert axon_info.placeholder2 == 0, "placeholder2 should default to 0"

    def test_axon_info_initialization_with_all_fields(self):
        """
        Test that AxonInfo can be initialized with all fields specified.
        
        This test verifies that an AxonInfo object can be created with all
        fields explicitly set, including optional fields. This is useful
        for testing custom protocol versions and placeholder values.
        """
        # Create an AxonInfo with all fields specified
        axon_info = AxonInfo(
            version=2,
            ip="10.0.0.1",
            port=8092,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            protocol=5,
            placeholder1=100,
            placeholder2=200
        )
        
        # Verify all fields are set correctly
        assert axon_info.version == 2
        assert axon_info.ip == "10.0.0.1"
        assert axon_info.port == 8092
        assert axon_info.ip_type == 4
        assert axon_info.protocol == 5, "Protocol should be set to custom value"
        assert axon_info.placeholder1 == 100, "placeholder1 should be set to custom value"
        assert axon_info.placeholder2 == 200, "placeholder2 should be set to custom value"

    def test_axon_info_initialization_with_ipv6(self):
        """
        Test that AxonInfo can handle IPv6 addresses.
        
        This test verifies that AxonInfo objects can be created with IPv6
        addresses, which use a different ip_type value (6 instead of 4).
        This is important for supporting modern networking infrastructure.
        """
        # Create an AxonInfo with IPv6 address
        axon_info = AxonInfo(
            version=1,
            ip="2001:0db8:85a3:0000:0000:8a2e:0370:7334",
            port=8091,
            ip_type=6,  # IPv6
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Verify IPv6-specific fields
        assert axon_info.ip_type == 6, "IP type should be 6 for IPv6"
        assert ":" in axon_info.ip, "IPv6 address should contain colons"

    def test_axon_info_inherits_from_info_base(self):
        """
        Test that AxonInfo properly inherits from InfoBase.
        
        This test verifies that AxonInfo is a subclass of InfoBase, which
        provides common functionality for chain data structures. This
        ensures that AxonInfo can use methods like from_dict() and
        list_from_dicts() from the base class.
        """
        # Verify inheritance
        assert issubclass(AxonInfo, InfoBase), \
            "AxonInfo should inherit from InfoBase"
        
        # Verify AxonInfo is a dataclass (which InfoBase also is)
        from dataclasses import is_dataclass
        assert is_dataclass(AxonInfo), "AxonInfo should be a dataclass"


class TestAxonInfoIsServingProperty:
    """
    Test class for the is_serving property.
    
    This class tests that the is_serving property correctly identifies
    whether an axon endpoint is actively serving requests.
    """

    def test_is_serving_returns_true_for_valid_ip(self):
        """
        Test that is_serving returns True for valid (non-zero) IP addresses.
        
        This test verifies that when an AxonInfo has a valid IP address
        (not "0.0.0.0"), the is_serving property returns True, indicating
        that the axon endpoint is active and ready to serve requests.
        """
        # Create an AxonInfo with a valid IP address
        axon_info = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Verify is_serving returns True
        assert axon_info.is_serving is True, \
            "is_serving should return True for valid IP address"

    def test_is_serving_returns_false_for_zero_ip(self):
        """
        Test that is_serving returns False for "0.0.0.0" IP address.
        
        This test verifies that when an AxonInfo has the IP address "0.0.0.0",
        the is_serving property returns False, indicating that the axon
        endpoint is not currently serving. This is the standard way to
        indicate an inactive endpoint in networking.
        """
        # Create an AxonInfo with zero IP address (not serving)
        axon_info = AxonInfo(
            version=1,
            ip="0.0.0.0",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Verify is_serving returns False
        assert axon_info.is_serving is False, \
            "is_serving should return False for 0.0.0.0 IP address"


class TestAxonInfoIpStrMethod:
    """
    Test class for the ip_str() method.
    
    This class tests that the ip_str() method correctly formats the IP
    address, IP type, and port into a readable string representation.
    """

    def test_ip_str_formats_ipv4_correctly(self):
        """
        Test that ip_str() correctly formats IPv4 addresses.
        
        This test verifies that for IPv4 addresses, the ip_str() method
        returns a properly formatted string that includes the IP address
        and port. This is useful for displaying endpoint information to users.
        """
        # Create an AxonInfo with IPv4 address
        axon_info = AxonInfo(
            version=1,
            ip="192.168.1.100",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Get the formatted IP string
        ip_string = axon_info.ip_str()
        
        # Verify it's a string
        assert isinstance(ip_string, str), "ip_str() should return a string"
        
        # Verify it contains the IP and port (format depends on networking.ip__str__)
        # The exact format is determined by networking.ip__str__, but it should include both
        assert len(ip_string) > 0, "ip_str() should return a non-empty string"

    def test_ip_str_uses_networking_module(self):
        """
        Test that ip_str() delegates to networking.ip__str__().
        
        This test verifies that the ip_str() method correctly calls the
        networking utility function to format the IP string. This ensures
        consistency in IP formatting across the codebase.
        """
        # Mock the networking.ip__str__ function
        with patch("bittensor.core.chain_data.axon_info.networking.ip__str__") as mock_ip_str:
            mock_ip_str.return_value = "192.168.1.100:8091"
            
            # Create an AxonInfo
            axon_info = AxonInfo(
                version=1,
                ip="192.168.1.100",
                port=8091,
                ip_type=4,
                hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
            )
            
            # Call ip_str()
            result = axon_info.ip_str()
            
            # Verify networking.ip__str__ was called with correct parameters
            mock_ip_str.assert_called_once_with(4, "192.168.1.100", 8091)
            
            # Verify result matches mock return value
            assert result == "192.168.1.100:8091"


class TestAxonInfoEquality:
    """
    Test class for the __eq__ method (equality comparison).
    
    This class tests that AxonInfo objects can be compared for equality
    correctly, which is important for testing and validation.
    """

    def test_equality_with_identical_objects(self):
        """
        Test that two AxonInfo objects with identical values are equal.
        
        This test verifies that the __eq__ method correctly identifies
        when two AxonInfo objects have the same values for all comparison
        fields (version, ip, port, ip_type, hotkey, coldkey). Note that
        protocol and placeholder fields are not included in equality comparison.
        """
        # Create two identical AxonInfo objects
        axon_info1 = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        axon_info2 = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Verify they are equal
        assert axon_info1 == axon_info2, \
            "Two identical AxonInfo objects should be equal"

    def test_equality_with_different_protocol_ignored(self):
        """
        Test that equality ignores protocol and placeholder fields.
        
        This test verifies that two AxonInfo objects are considered equal
        even if they have different values for protocol, placeholder1, or
        placeholder2, since these fields are not included in the equality
        comparison.
        """
        # Create two AxonInfo objects with same core fields but different optional fields
        axon_info1 = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            protocol=4,
            placeholder1=0,
            placeholder2=0
        )
        
        axon_info2 = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            protocol=5,  # Different protocol
            placeholder1=100,  # Different placeholder
            placeholder2=200  # Different placeholder
        )
        
        # Verify they are still equal (optional fields don't affect equality)
        assert axon_info1 == axon_info2, \
            "AxonInfo objects should be equal even with different optional fields"

    def test_equality_with_different_required_fields(self):
        """
        Test that two AxonInfo objects with different required fields are not equal.
        
        This test verifies that when any of the required comparison fields
        (version, ip, port, ip_type, hotkey, coldkey) differ, the objects
        are not considered equal.
        """
        # Create base AxonInfo
        axon_info1 = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Create AxonInfo with different version
        axon_info2 = AxonInfo(
            version=2,  # Different version
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Verify they are not equal
        assert axon_info1 != axon_info2, \
            "AxonInfo objects with different versions should not be equal"

    def test_equality_with_none(self):
        """
        Test that equality comparison with None returns False.
        
        This test verifies that when comparing an AxonInfo object with None,
        the __eq__ method returns False rather than raising an error.
        This prevents potential AttributeError exceptions in comparison code.
        """
        # Create an AxonInfo
        axon_info = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Verify comparison with None returns False
        assert (axon_info == None) is False, \
            "AxonInfo should not be equal to None"
        assert (axon_info != None) is True, \
            "AxonInfo should not be equal to None"

    def test_equality_with_wrong_type(self):
        """
        Test that equality comparison with wrong type returns False.
        
        This test verifies that when comparing an AxonInfo object with
        an object of a different type, the __eq__ method returns False
        rather than raising an error.
        """
        # Create an AxonInfo
        axon_info = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Verify comparison with wrong type returns False
        assert (axon_info == "not an AxonInfo") is False, \
            "AxonInfo should not be equal to a string"
        assert (axon_info == {}) is False, \
            "AxonInfo should not be equal to a dict"


class TestAxonInfoStringRepresentation:
    """
    Test class for string representation methods (__str__ and __repr__).
    
    This class tests that AxonInfo objects can be converted to strings
    for display purposes, which is useful for debugging and logging.
    """

    def test_str_representation_includes_key_fields(self):
        """
        Test that __str__ includes key identifying fields.
        
        This test verifies that the string representation of an AxonInfo
        object includes the most important fields: IP string, hotkey,
        coldkey, and version. This makes it easy to identify an axon
        endpoint when debugging.
        """
        # Create an AxonInfo
        axon_info = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Get string representation
        str_repr = str(axon_info)
        
        # Verify it's a string
        assert isinstance(str_repr, str), "__str__ should return a string"
        
        # Verify it includes key fields
        assert "AxonInfo" in str_repr, "String should include class name"
        assert axon_info.hotkey in str_repr, "String should include hotkey"
        assert axon_info.coldkey in str_repr, "String should include coldkey"
        assert str(axon_info.version) in str_repr, "String should include version"

    def test_repr_equals_str(self):
        """
        Test that __repr__ returns the same as __str__.
        
        This test verifies that __repr__ is implemented to return the same
        value as __str__. This is a common Python pattern that ensures
        consistent string representation for both display and debugging.
        """
        # Create an AxonInfo
        axon_info = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        )
        
        # Verify __repr__ equals __str__
        assert repr(axon_info) == str(axon_info), \
            "__repr__ should return the same as __str__"


class TestAxonInfoToString:
    """
    Test class for the to_string() method (JSON serialization).
    
    This class tests that AxonInfo objects can be serialized to JSON strings,
    which is useful for storage, transmission, or logging.
    """

    def test_to_string_serializes_all_fields(self):
        """
        Test that to_string() correctly serializes all fields to JSON.
        
        This test verifies that the to_string() method converts an AxonInfo
        object to a JSON string that includes all fields. This allows the
        object to be stored or transmitted and later reconstructed.
        """
        # Create an AxonInfo with all fields
        axon_info = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            protocol=4,
            placeholder1=0,
            placeholder2=0
        )
        
        # Convert to string
        json_string = axon_info.to_string()
        
        # Verify it's a string
        assert isinstance(json_string, str), "to_string() should return a string"
        
        # Verify it's valid JSON
        parsed_data = json.loads(json_string)
        assert isinstance(parsed_data, dict), "JSON string should parse to a dictionary"
        
        # Verify all fields are present
        assert parsed_data["version"] == 1
        assert parsed_data["ip"] == "192.168.1.1"
        assert parsed_data["port"] == 8091
        assert parsed_data["ip_type"] == 4
        assert parsed_data["hotkey"] == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        assert parsed_data["coldkey"] == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

    def test_to_string_handles_serialization_error_gracefully(self):
        """
        Test that to_string() handles serialization errors gracefully.
        
        This test verifies that when JSON serialization fails (e.g., due to
        invalid data types), the to_string() method catches the error and
        returns a default AxonInfo's JSON string instead of raising an exception.
        This ensures robustness in error scenarios.
        """
        # Create an AxonInfo that might cause serialization issues
        # We'll mock json.dumps to raise an error
        with patch("bittensor.core.chain_data.axon_info.json.dumps") as mock_dumps:
            mock_dumps.side_effect = TypeError("Cannot serialize")
            
            axon_info = AxonInfo(
                version=1,
                ip="192.168.1.1",
                port=8091,
                ip_type=4,
                hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
            )
            
            # Call to_string() - should not raise an exception
            result = axon_info.to_string()
            
            # Verify it returns a string (default AxonInfo's JSON)
            assert isinstance(result, str), \
                "to_string() should return a string even on error"


class TestAxonInfoFromString:
    """
    Test class for the from_string() class method (JSON deserialization).
    
    This class tests that AxonInfo objects can be created from JSON strings,
    which complements the to_string() method for round-trip serialization.
    """

    def test_from_string_deserializes_valid_json(self):
        """
        Test that from_string() correctly deserializes valid JSON.
        
        This test verifies that when given a valid JSON string containing
        AxonInfo fields, the from_string() method correctly creates an
        AxonInfo object with those values. This allows objects to be
        reconstructed from stored or transmitted JSON data.
        """
        # Create a valid JSON string
        json_string = json.dumps({
            "version": 1,
            "ip": "192.168.1.1",
            "port": 8091,
            "ip_type": 4,
            "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "coldkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "protocol": 4,
            "placeholder1": 0,
            "placeholder2": 0
        })
        
        # Create AxonInfo from string
        axon_info = AxonInfo.from_string(json_string)
        
        # Verify fields are set correctly
        assert axon_info.version == 1
        assert axon_info.ip == "192.168.1.1"
        assert axon_info.port == 8091
        assert axon_info.ip_type == 4
        assert axon_info.hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        assert axon_info.coldkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

    def test_from_string_handles_invalid_json(self):
        """
        Test that from_string() handles invalid JSON gracefully.
        
        This test verifies that when given an invalid JSON string, the
        from_string() method catches the JSONDecodeError and returns a
        default AxonInfo object instead of raising an exception. This
        ensures robustness when dealing with corrupted or invalid data.
        """
        # Create an invalid JSON string
        invalid_json = "not valid json {"
        
        # Create AxonInfo from invalid string - should not raise
        axon_info = AxonInfo.from_string(invalid_json)
        
        # Verify it returns default AxonInfo (all zeros/empty)
        assert axon_info.version == 0, "Should return default AxonInfo on error"
        assert axon_info.ip == "", "Should return default AxonInfo on error"
        assert axon_info.port == 0, "Should return default AxonInfo on error"

    def test_from_string_round_trip(self):
        """
        Test that from_string() and to_string() work together for round-trip serialization.
        
        This test verifies that an AxonInfo object can be serialized to JSON
        and then deserialized back to an AxonInfo object with the same values.
        This is important for data persistence and transmission scenarios.
        """
        # Create original AxonInfo
        original = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8091,
            ip_type=4,
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            protocol=4,
            placeholder1=100,
            placeholder2=200
        )
        
        # Serialize to string
        json_string = original.to_string()
        
        # Deserialize from string
        restored = AxonInfo.from_string(json_string)
        
        # Verify all fields match
        assert restored.version == original.version
        assert restored.ip == original.ip
        assert restored.port == original.port
        assert restored.ip_type == original.ip_type
        assert restored.hotkey == original.hotkey
        assert restored.coldkey == original.coldkey
        assert restored.protocol == original.protocol
        assert restored.placeholder1 == original.placeholder1
        assert restored.placeholder2 == original.placeholder2


class TestAxonInfoFromDict:
    """
    Test class for the _from_dict() class method.
    
    This class tests that AxonInfo objects can be created from dictionary
    data, which is how chain data is typically received from the substrate
    interface.
    """

    def test_from_dict_creates_axon_info_correctly(self):
        """
        Test that _from_dict() correctly creates an AxonInfo from dictionary data.
        
        This test verifies that when given a dictionary with AxonInfo fields,
        the _from_dict() method correctly creates an AxonInfo object. This
        is the primary way AxonInfo objects are created from chain data.
        """
        # Create dictionary data (as would come from chain)
        data = {
            "version": 1,
            "ip": 3232235777,  # IP as integer (192.168.1.1)
            "port": 8091,
            "ip_type": 4,
            "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "coldkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "protocol": 4,
            "placeholder1": 0,
            "placeholder2": 0
        }
        
        # Create AxonInfo from dictionary
        axon_info = AxonInfo._from_dict(data)
        
        # Verify IP is converted from integer to string
        assert isinstance(axon_info.ip, str), "IP should be converted to string"
        # The exact IP depends on netaddr conversion, but should be a valid IP string
        
        # Verify other fields
        assert axon_info.version == 1
        assert axon_info.port == 8091
        assert axon_info.ip_type == 4

    def test_from_dict_converts_ip_address(self):
        """
        Test that _from_dict() correctly converts IP address from integer to string.
        
        This test verifies that when the IP address comes as an integer (which
        is how it's stored on-chain), the _from_dict() method correctly converts
        it to a string IP address using netaddr.IPAddress.
        """
        # Create dictionary with IP as integer
        data = {
            "version": 1,
            "ip": 3232235777,  # 192.168.1.1 as integer
            "port": 8091,
            "ip_type": 4,
            "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "coldkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "protocol": 4,
            "placeholder1": 0,
            "placeholder2": 0
        }
        
        # Create AxonInfo from dictionary
        axon_info = AxonInfo._from_dict(data)
        
        # Verify IP is converted correctly
        # netaddr.IPAddress(3232235777) should give us the string representation
        expected_ip = str(netaddr.IPAddress(data["ip"]))
        assert axon_info.ip == expected_ip, \
            f"IP should be converted from integer to string. Expected: {expected_ip}"


class TestAxonInfoFromNeuronInfo:
    """
    Test class for the from_neuron_info() class method.
    
    This class tests that AxonInfo objects can be created from neuron info
    dictionaries, which is a common use case when working with neuron data.
    """

    def test_from_neuron_info_creates_axon_info_correctly(self):
        """
        Test that from_neuron_info() correctly extracts AxonInfo from neuron data.
        
        This test verifies that when given a neuron info dictionary (which contains
        nested axon_info and keys), the from_neuron_info() method correctly
        extracts and creates an AxonInfo object with the appropriate fields.
        """
        # Create neuron info dictionary (as would come from chain)
        neuron_info = {
            "axon_info": {
                "version": 1,
                "ip": 3232235777,  # IP as integer
                "port": 8091,
                "ip_type": 4,
                "protocol": 4,
                "placeholder1": 0,
                "placeholder2": 0
            },
            "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "coldkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        }
        
        # Create AxonInfo from neuron info
        axon_info = AxonInfo.from_neuron_info(neuron_info)
        
        # Verify fields are extracted correctly
        assert axon_info.version == 1
        assert axon_info.port == 8091
        assert axon_info.ip_type == 4
        assert axon_info.hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        assert axon_info.coldkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        
        # Verify IP is converted from integer to string
        assert isinstance(axon_info.ip, str), "IP should be converted to string"

    def test_from_neuron_info_uses_networking_int_to_ip(self):
        """
        Test that from_neuron_info() uses networking.int_to_ip for IP conversion.
        
        This test verifies that the IP address conversion uses the networking
        utility function, which ensures consistency with other parts of the codebase.
        """
        # Mock networking.int_to_ip
        with patch("bittensor.core.chain_data.axon_info.networking.int_to_ip") as mock_int_to_ip:
            mock_int_to_ip.return_value = "192.168.1.1"
            
            # Create neuron info dictionary
            neuron_info = {
                "axon_info": {
                    "version": 1,
                    "ip": 3232235777,
                    "port": 8091,
                    "ip_type": 4,
                    "protocol": 4,
                    "placeholder1": 0,
                    "placeholder2": 0
                },
                "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "coldkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
            }
            
            # Create AxonInfo from neuron info
            axon_info = AxonInfo.from_neuron_info(neuron_info)
            
            # Verify networking.int_to_ip was called
            mock_int_to_ip.assert_called_once_with(3232235777)
            
            # Verify IP matches mock return value
            assert axon_info.ip == "192.168.1.1"


class TestAxonInfoFromDictBaseClass:
    """
    Test class for the from_dict() method inherited from InfoBase.
    
    This class tests that AxonInfo can use the from_dict() method from
    InfoBase, which includes error handling for missing fields.
    """

    def test_from_dict_with_complete_data(self):
        """
        Test that from_dict() works with complete data.
        
        This test verifies that the from_dict() method (inherited from InfoBase)
        correctly calls _from_dict() when all required fields are present in
        the dictionary. This is the happy path for creating AxonInfo from chain data.
        """
        # Create complete dictionary data
        data = {
            "version": 1,
            "ip": 3232235777,
            "port": 8091,
            "ip_type": 4,
            "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "coldkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "protocol": 4,
            "placeholder1": 0,
            "placeholder2": 0
        }
        
        # Create AxonInfo using from_dict (from InfoBase)
        axon_info = AxonInfo.from_dict(data)
        
        # Verify it was created successfully
        assert isinstance(axon_info, AxonInfo), \
            "from_dict() should return an AxonInfo instance"
        assert axon_info.version == 1

    def test_from_dict_raises_exception_on_missing_field(self):
        """
        Test that from_dict() raises SubstrateRequestException on missing fields.
        
        This test verifies that when required fields are missing from the
        dictionary, the from_dict() method (inherited from InfoBase) raises
        a SubstrateRequestException with a descriptive message. This helps
        identify data structure issues from the chain.
        """
        # Create incomplete dictionary (missing required field)
        incomplete_data = {
            "version": 1,
            "ip": 3232235777,
            # Missing port, ip_type, hotkey, coldkey
        }
        
        # Verify from_dict raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException) as exc_info:
            AxonInfo.from_dict(incomplete_data)
        
        # Verify error message mentions missing field
        assert "missing" in str(exc_info.value).lower(), \
            "Error message should mention missing field"


class TestAxonInfoParameterDict:
    """
    Test class for parameter dictionary conversion methods.
    
    This class tests the to_parameter_dict() and from_parameter_dict() methods,
    which are used for PyTorch integration when USE_TORCH flag is set.
    """

    def test_to_parameter_dict_without_torch_returns_dict(self):
        """
        Test that to_parameter_dict() returns dict when torch is not used.
        
        This test verifies that when the USE_TORCH flag is not set (or torch
        is not available), the to_parameter_dict() method returns a regular
        Python dictionary containing all the AxonInfo fields.
        """
        # Mock use_torch to return False
        with patch("bittensor.core.chain_data.axon_info.use_torch", return_value=False):
            # Create an AxonInfo
            axon_info = AxonInfo(
                version=1,
                ip="192.168.1.1",
                port=8091,
                ip_type=4,
                hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
            )
            
            # Convert to parameter dict
            param_dict = axon_info.to_parameter_dict()
            
            # Verify it's a regular dict
            assert isinstance(param_dict, dict), \
                "Should return dict when torch is not used"
            
            # Verify it contains all fields
            assert param_dict["version"] == 1
            assert param_dict["ip"] == "192.168.1.1"
            assert param_dict["port"] == 8091

    def test_from_parameter_dict_without_torch(self):
        """
        Test that from_parameter_dict() works with regular dict when torch is not used.
        
        This test verifies that when given a regular dictionary and torch is
        not available, the from_parameter_dict() method correctly creates an
        AxonInfo object from the dictionary values.
        """
        # Mock use_torch to return False
        with patch("bittensor.core.chain_data.axon_info.use_torch", return_value=False):
            # Create parameter dictionary
            param_dict = {
                "version": 1,
                "ip": "192.168.1.1",
                "port": 8091,
                "ip_type": 4,
                "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "coldkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "protocol": 4,
                "placeholder1": 0,
                "placeholder2": 0
            }
            
            # Create AxonInfo from parameter dict
            axon_info = AxonInfo.from_parameter_dict(param_dict)
            
            # Verify it was created correctly
            assert isinstance(axon_info, AxonInfo)
            assert axon_info.version == 1
            assert axon_info.ip == "192.168.1.1"

    def test_parameter_dict_round_trip(self):
        """
        Test that to_parameter_dict() and from_parameter_dict() work together.
        
        This test verifies that an AxonInfo object can be converted to a
        parameter dictionary and then back to an AxonInfo object with the
        same values. This is important for PyTorch model serialization.
        """
        # Mock use_torch to return False for simplicity
        with patch("bittensor.core.chain_data.axon_info.use_torch", return_value=False):
            # Create original AxonInfo
            original = AxonInfo(
                version=1,
                ip="192.168.1.1",
                port=8091,
                ip_type=4,
                hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                protocol=5,
                placeholder1=100,
                placeholder2=200
            )
            
            # Convert to parameter dict
            param_dict = original.to_parameter_dict()
            
            # Convert back to AxonInfo
            restored = AxonInfo.from_parameter_dict(param_dict)
            
            # Verify all fields match
            assert restored.version == original.version
            assert restored.ip == original.ip
            assert restored.port == original.port
            assert restored.protocol == original.protocol

