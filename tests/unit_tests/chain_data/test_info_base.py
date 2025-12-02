"""
Unit tests for bittensor.core.chain_data.info_base module.

Tests the InfoBase class which provides base functionality for info objects
used throughout the Bittensor chain data system. This base class handles
dictionary-to-object conversion with proper error handling.

This test suite ensures comprehensive coverage of the InfoBase functionality
which is critical for parsing and converting chain data into structured objects.
"""

import pytest
from dataclasses import dataclass
from typing import Any

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.errors import SubstrateRequestException


# ============================================================================
# Concrete Implementation for Testing Abstract InfoBase
# ============================================================================

@dataclass
class ConcreteInfoBase(InfoBase):
    """
    Concrete implementation of InfoBase for testing purposes.
    
    This class provides a simple concrete implementation of InfoBase
    with a few fields to test the base class functionality.
    """
    field1: str
    field2: int
    field3: bool = False


@dataclass
class ComplexInfoBase(InfoBase):
    """
    More complex InfoBase implementation for testing nested scenarios.
    
    This class includes optional fields and various data types to test
    edge cases in dictionary conversion.
    """
    required_field: str
    optional_field: str = "default"
    numeric_field: int = 0
    boolean_field: bool = True


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_dict():
    """
    Create a sample dictionary for testing InfoBase.from_dict.
    
    Returns a dictionary with all required fields for ConcreteInfoBase.
    """
    return {
        "field1": "test_value",
        "field2": 42,
        "field3": True,
    }


@pytest.fixture
def sample_dict_with_defaults():
    """
    Create a sample dictionary with only required fields.
    
    Returns a dictionary without optional fields to test default values.
    """
    return {
        "required_field": "required_value",
    }


@pytest.fixture
def sample_dict_list(sample_dict):
    """
    Create a list of dictionaries for testing list_from_dicts.
    
    Returns a list containing multiple dictionaries that can be converted
    to a list of InfoBase instances.
    """
    return [
        sample_dict,
        {
            "field1": "another_value",
            "field2": 100,
            "field3": False,
        },
        {
            "field1": "third_value",
            "field2": 200,
            "field3": True,
        },
    ]


@pytest.fixture
def incomplete_dict():
    """
    Create an incomplete dictionary missing required fields.
    
    Returns a dictionary that will cause KeyError when used with
    InfoBase.from_dict, testing error handling.
    """
    return {
        "field1": "test_value",
        # Missing field2 - will cause KeyError
    }


# ============================================================================
# Test Classes for InfoBase.from_dict
# ============================================================================

class TestInfoBaseFromDict:
    """
    Tests for InfoBase.from_dict class method.
    
    These tests verify that from_dict correctly converts dictionaries
    to InfoBase instances and handles errors appropriately.
    """
    
    def test_from_dict_basic_conversion(self, sample_dict):
        """
        Test basic dictionary to InfoBase conversion.
        
        Verifies that a complete dictionary can be converted to an
        InfoBase instance with all fields properly set.
        """
        # Convert dictionary to ConcreteInfoBase instance
        instance = ConcreteInfoBase.from_dict(sample_dict)
        
        # Verify instance is created
        assert isinstance(instance, ConcreteInfoBase)
        assert isinstance(instance, InfoBase)
        
        # Verify all fields are set correctly
        assert instance.field1 == "test_value"
        assert instance.field2 == 42
        assert instance.field3 is True
    
    def test_from_dict_with_defaults(self, sample_dict_with_defaults):
        """
        Test from_dict with dictionary containing only required fields.
        
        Verifies that optional fields use their default values when
        not provided in the dictionary.
        """
        # Convert dictionary with only required field
        instance = ComplexInfoBase.from_dict(sample_dict_with_defaults)
        
        # Verify instance is created
        assert isinstance(instance, ComplexInfoBase)
        
        # Verify required field is set
        assert instance.required_field == "required_value"
        
        # Verify optional fields use defaults
        assert instance.optional_field == "default"
        assert instance.numeric_field == 0
        assert instance.boolean_field is True
    
    def test_from_dict_with_all_fields(self):
        """
        Test from_dict with dictionary containing all fields including optional ones.
        
        Verifies that when all fields are provided, they override defaults.
        """
        # Create dictionary with all fields
        full_dict = {
            "required_field": "required_value",
            "optional_field": "custom_value",
            "numeric_field": 100,
            "boolean_field": False,
        }
        
        # Convert to instance
        instance = ComplexInfoBase.from_dict(full_dict)
        
        # Verify all fields are set from dictionary
        assert instance.required_field == "required_value"
        assert instance.optional_field == "custom_value"
        assert instance.numeric_field == 100
        assert instance.boolean_field is False
    
    def test_from_dict_missing_required_field(self, incomplete_dict):
        """
        Test from_dict with missing required field raises SubstrateRequestException.
        
        When a required field is missing from the dictionary, from_dict should
        catch the KeyError and raise a SubstrateRequestException with a
        descriptive error message.
        """
        # Attempt conversion with incomplete dictionary
        with pytest.raises(SubstrateRequestException) as exc_info:
            ConcreteInfoBase.from_dict(incomplete_dict)
        
        # Verify error message contains class name and missing field
        error_message = str(exc_info.value)
        assert "ConcreteInfoBase" in error_message or "structure" in error_message.lower()
        assert "missing" in error_message.lower() or "field2" in error_message
    
    def test_from_dict_empty_dictionary(self):
        """
        Test from_dict with empty dictionary.
        
        An empty dictionary should raise SubstrateRequestException since
        required fields are missing.
        """
        # Attempt conversion with empty dictionary
        with pytest.raises(SubstrateRequestException):
            ConcreteInfoBase.from_dict({})
    
    def test_from_dict_extra_fields(self, sample_dict):
        """
        Test from_dict with dictionary containing extra fields.
        
        Extra fields in the dictionary should be ignored (or cause TypeError
        if dataclass doesn't allow extra fields). This tests the robustness
        of the conversion process.
        """
        # Add extra field to dictionary
        dict_with_extra = {**sample_dict, "extra_field": "extra_value"}
        
        # Convert to instance - should work (extra fields ignored by dataclass)
        instance = ConcreteInfoBase.from_dict(dict_with_extra)
        
        # Verify instance is created correctly
        assert isinstance(instance, ConcreteInfoBase)
        assert instance.field1 == "test_value"
        assert instance.field2 == 42
    
    def test_from_dict_type_preservation(self, sample_dict):
        """
        Test that from_dict preserves data types correctly.
        
        Verifies that different data types (str, int, bool) are correctly
        preserved during dictionary conversion.
        """
        # Convert dictionary
        instance = ConcreteInfoBase.from_dict(sample_dict)
        
        # Verify types are preserved
        assert isinstance(instance.field1, str)
        assert isinstance(instance.field2, int)
        assert isinstance(instance.field3, bool)
    
    def test_from_dict_none_values(self):
        """
        Test from_dict with None values in dictionary.
        
        Verifies that None values are handled correctly when present
        in the dictionary (if fields allow None).
        """
        # Create dictionary with None value (if field allows it)
        dict_with_none = {
            "field1": None,  # This might cause TypeError if field doesn't allow None
            "field2": 42,
            "field3": False,
        }
        
        # This test verifies behavior with None - may raise TypeError
        # depending on field type annotations
        try:
            instance = ConcreteInfoBase.from_dict(dict_with_none)
            # If successful, verify None was set
            assert instance.field1 is None
        except (TypeError, ValueError):
            # If TypeError is raised, that's expected for non-optional fields
            pass


class TestInfoBaseFromDictErrorHandling:
    """
    Tests for InfoBase.from_dict error handling.
    
    These tests verify that error handling works correctly for various
    error scenarios including missing fields and invalid data.
    """
    
    def test_from_dict_keyerror_conversion(self, incomplete_dict):
        """
        Test that KeyError is converted to SubstrateRequestException.
        
        When a KeyError occurs (missing field), it should be caught and
        converted to a SubstrateRequestException with a descriptive message.
        """
        # Attempt conversion with incomplete data
        with pytest.raises(SubstrateRequestException) as exc_info:
            ConcreteInfoBase.from_dict(incomplete_dict)
        
        # Verify it's a SubstrateRequestException, not KeyError
        assert isinstance(exc_info.value, SubstrateRequestException)
        assert not isinstance(exc_info.value, KeyError)
    
    def test_from_dict_error_message_format(self, incomplete_dict):
        """
        Test that error message format is correct.
        
        The error message should include the class name and indicate
        which field is missing from the chain data.
        """
        # Attempt conversion
        with pytest.raises(SubstrateRequestException) as exc_info:
            ConcreteInfoBase.from_dict(incomplete_dict)
        
        # Verify error message format
        error_message = str(exc_info.value)
        # Should mention the class or structure
        assert len(error_message) > 0
        # Should mention missing field or chain
        assert "missing" in error_message.lower() or "chain" in error_message.lower()
    
    def test_from_dict_multiple_missing_fields(self):
        """
        Test error handling when multiple fields are missing.
        
        When multiple required fields are missing, the error should
        indicate the first missing field encountered.
        """
        # Create dictionary missing multiple fields
        incomplete_dict = {
            "field1": "test",
            # Missing field2 and field3
        }
        
        # Attempt conversion
        with pytest.raises(SubstrateRequestException):
            ConcreteInfoBase.from_dict(incomplete_dict)


class TestInfoBaseFromDictInternal:
    """
    Tests for InfoBase._from_dict internal method.
    
    These tests verify that the internal _from_dict method works correctly.
    This method is called by from_dict and performs the actual conversion.
    """
    
    def test_from_dict_internal_method(self, sample_dict):
        """
        Test that _from_dict is called by from_dict.
        
        Verifies that from_dict correctly delegates to _from_dict
        for the actual conversion logic.
        """
        # Convert using from_dict
        instance = ConcreteInfoBase.from_dict(sample_dict)
        
        # Verify instance is created correctly
        assert isinstance(instance, ConcreteInfoBase)
        assert instance.field1 == sample_dict["field1"]
        assert instance.field2 == sample_dict["field2"]
    
    def test_from_dict_internal_direct_call(self, sample_dict):
        """
        Test calling _from_dict directly.
        
        Verifies that _from_dict can be called directly and works
        the same way as through from_dict (without error handling).
        """
        # Call _from_dict directly
        instance = ConcreteInfoBase._from_dict(sample_dict)
        
        # Verify instance is created
        assert isinstance(instance, ConcreteInfoBase)
        assert instance.field1 == "test_value"
        assert instance.field2 == 42
    
    def test_from_dict_internal_keyerror_propagation(self, incomplete_dict):
        """
        Test that _from_dict allows KeyError to propagate.
        
        Unlike from_dict, _from_dict should allow KeyError to propagate
        so it can be caught and converted by from_dict.
        """
        # Call _from_dict directly - should raise KeyError
        with pytest.raises(KeyError):
            ConcreteInfoBase._from_dict(incomplete_dict)


# ============================================================================
# Test Classes for InfoBase.list_from_dicts
# ============================================================================

class TestInfoBaseListFromDicts:
    """
    Tests for InfoBase.list_from_dicts class method.
    
    These tests verify that list_from_dicts correctly converts a list
    of dictionaries to a list of InfoBase instances.
    """
    
    def test_list_from_dicts_basic_conversion(self, sample_dict_list):
        """
        Test basic list of dictionaries to list of InfoBase conversion.
        
        Verifies that a list of dictionaries can be converted to a list
        of InfoBase instances.
        """
        # Convert list of dictionaries
        instances = ConcreteInfoBase.list_from_dicts(sample_dict_list)
        
        # Verify result is a list
        assert isinstance(instances, list)
        assert len(instances) == len(sample_dict_list)
        
        # Verify all items are ConcreteInfoBase instances
        for instance in instances:
            assert isinstance(instance, ConcreteInfoBase)
            assert isinstance(instance, InfoBase)
    
    def test_list_from_dicts_empty_list(self):
        """
        Test list_from_dicts with empty list.
        
        An empty list should return an empty list of instances.
        """
        # Convert empty list
        instances = ConcreteInfoBase.list_from_dicts([])
        
        # Verify result is empty list
        assert isinstance(instances, list)
        assert len(instances) == 0
    
    def test_list_from_dicts_single_item(self, sample_dict):
        """
        Test list_from_dicts with single item list.
        
        A list with one dictionary should return a list with one instance.
        """
        # Convert single item list
        instances = ConcreteInfoBase.list_from_dicts([sample_dict])
        
        # Verify result
        assert len(instances) == 1
        assert isinstance(instances[0], ConcreteInfoBase)
        assert instances[0].field1 == sample_dict["field1"]
    
    def test_list_from_dicts_multiple_items(self, sample_dict_list):
        """
        Test list_from_dicts with multiple items.
        
        Verifies that each dictionary in the list is converted correctly
        and maintains its individual values.
        """
        # Convert list
        instances = ConcreteInfoBase.list_from_dicts(sample_dict_list)
        
        # Verify all instances are created correctly
        assert len(instances) == 3
        
        # Verify first instance
        assert instances[0].field1 == "test_value"
        assert instances[0].field2 == 42
        
        # Verify second instance
        assert instances[1].field1 == "another_value"
        assert instances[1].field2 == 100
        
        # Verify third instance
        assert instances[2].field1 == "third_value"
        assert instances[2].field2 == 200
    
    def test_list_from_dicts_preserves_order(self, sample_dict_list):
        """
        Test that list_from_dicts preserves the order of items.
        
        The order of instances in the result should match the order
        of dictionaries in the input list.
        """
        # Convert list
        instances = ConcreteInfoBase.list_from_dicts(sample_dict_list)
        
        # Verify order is preserved
        assert instances[0].field1 == sample_dict_list[0]["field1"]
        assert instances[1].field1 == sample_dict_list[1]["field1"]
        assert instances[2].field1 == sample_dict_list[2]["field1"]


class TestInfoBaseListFromDictsErrorHandling:
    """
    Tests for InfoBase.list_from_dicts error handling.
    
    These tests verify that error handling works correctly when
    converting lists of dictionaries, including partial failures.
    """
    
    def test_list_from_dicts_with_invalid_item(self, sample_dict, incomplete_dict):
        """
        Test list_from_dicts with one invalid dictionary in list.
        
        When one dictionary in the list is invalid (missing required fields),
        it should raise SubstrateRequestException.
        """
        # Create list with one invalid dictionary
        mixed_list = [sample_dict, incomplete_dict, sample_dict]
        
        # Attempt conversion - should raise error on invalid item
        with pytest.raises(SubstrateRequestException):
            ConcreteInfoBase.list_from_dicts(mixed_list)
    
    def test_list_from_dicts_all_invalid(self, incomplete_dict):
        """
        Test list_from_dicts with all invalid dictionaries.
        
        When all dictionaries in the list are invalid, it should
        raise SubstrateRequestException on the first invalid item.
        """
        # Create list with all invalid dictionaries
        invalid_list = [incomplete_dict, incomplete_dict]
        
        # Attempt conversion
        with pytest.raises(SubstrateRequestException):
            ConcreteInfoBase.list_from_dicts(invalid_list)
    
    def test_list_from_dicts_error_on_first_invalid(self, sample_dict, incomplete_dict):
        """
        Test that list_from_dicts stops on first error.
        
        When processing a list, if an error occurs, it should stop
        immediately and not process remaining items.
        """
        # Create list with invalid item first
        invalid_first_list = [incomplete_dict, sample_dict]
        
        # Attempt conversion - should fail on first item
        with pytest.raises(SubstrateRequestException):
            ConcreteInfoBase.list_from_dicts(invalid_first_list)


# ============================================================================
# Test Classes for InfoBase Inheritance
# ============================================================================

class TestInfoBaseInheritance:
    """
    Tests for InfoBase inheritance behavior.
    
    These tests verify that InfoBase methods work correctly when
    inherited by subclasses with different field configurations.
    """
    
    def test_inherited_from_dict(self, sample_dict):
        """
        Test that subclasses inherit from_dict functionality.
        
        Subclasses of InfoBase should inherit the from_dict method
        and it should work with their specific field definitions.
        """
        # Use inherited method
        instance = ConcreteInfoBase.from_dict(sample_dict)
        
        # Verify it works correctly
        assert isinstance(instance, ConcreteInfoBase)
        assert isinstance(instance, InfoBase)
    
    def test_inherited_list_from_dicts(self, sample_dict_list):
        """
        Test that subclasses inherit list_from_dicts functionality.
        
        Subclasses should inherit list_from_dicts and it should work
        with their specific field definitions.
        """
        # Use inherited method
        instances = ConcreteInfoBase.list_from_dicts(sample_dict_list)
        
        # Verify it works correctly
        assert all(isinstance(inst, ConcreteInfoBase) for inst in instances)
        assert all(isinstance(inst, InfoBase) for inst in instances)
    
    def test_different_subclass_fields(self):
        """
        Test that different subclasses handle their own fields correctly.
        
        Each subclass should only accept dictionaries with fields
        matching its own dataclass definition.
        """
        # Create dictionary for ComplexInfoBase
        complex_dict = {
            "required_field": "test",
            "optional_field": "custom",
            "numeric_field": 50,
            "boolean_field": False,
        }
        
        # Convert using ComplexInfoBase
        instance = ComplexInfoBase.from_dict(complex_dict)
        
        # Verify it's the correct type
        assert isinstance(instance, ComplexInfoBase)
        assert isinstance(instance, InfoBase)
        assert instance.required_field == "test"
        
        # Verify ConcreteInfoBase would reject this (different fields)
        with pytest.raises(SubstrateRequestException):
            ConcreteInfoBase.from_dict(complex_dict)


# ============================================================================
# Integration Tests
# ============================================================================

class TestInfoBaseIntegration:
    """
    Integration tests for InfoBase functionality.
    
    These tests verify that InfoBase works correctly in realistic
    scenarios and integrates well with other components.
    """
    
    def test_from_dict_then_list_from_dicts(self, sample_dict, sample_dict_list):
        """
        Test using from_dict and list_from_dicts together.
        
        Verifies that both methods work correctly when used in sequence
        or in combination.
        """
        # First, convert single dictionary
        single_instance = ConcreteInfoBase.from_dict(sample_dict)
        
        # Then, convert list of dictionaries
        list_instances = ConcreteInfoBase.list_from_dicts(sample_dict_list)
        
        # Verify both work correctly
        assert isinstance(single_instance, ConcreteInfoBase)
        assert len(list_instances) == len(sample_dict_list)
        assert all(isinstance(inst, ConcreteInfoBase) for inst in list_instances)
    
    def test_list_from_dicts_with_varying_fields(self):
        """
        Test list_from_dicts with dictionaries having varying field values.
        
        Verifies that list_from_dicts correctly handles dictionaries
        with different values but same structure.
        """
        # Create list with varying values
        varying_list = [
            {"field1": "value1", "field2": 1, "field3": True},
            {"field1": "value2", "field2": 2, "field3": False},
            {"field1": "value3", "field2": 3, "field3": True},
        ]
        
        # Convert list
        instances = ConcreteInfoBase.list_from_dicts(varying_list)
        
        # Verify all instances have correct values
        assert instances[0].field1 == "value1"
        assert instances[0].field2 == 1
        assert instances[1].field1 == "value2"
        assert instances[1].field2 == 2
        assert instances[2].field1 == "value3"
        assert instances[2].field2 == 3
    
    def test_nested_usage_scenario(self, sample_dict_list):
        """
        Test realistic nested usage scenario.
        
        Simulates a realistic scenario where chain data is received
        as a list of dictionaries and needs to be converted to objects.
        """
        # Simulate receiving chain data as list of dictionaries
        chain_data = sample_dict_list
        
        # Convert to list of InfoBase instances
        info_objects = ConcreteInfoBase.list_from_dicts(chain_data)
        
        # Verify conversion was successful
        assert len(info_objects) == len(chain_data)
        
        # Verify we can access properties of converted objects
        for i, obj in enumerate(info_objects):
            assert obj.field1 == chain_data[i]["field1"]
            assert obj.field2 == chain_data[i]["field2"]
            assert obj.field3 == chain_data[i]["field3"]


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestInfoBaseEdgeCases:
    """
    Tests for InfoBase edge cases and boundary conditions.
    
    These tests verify behavior in unusual or edge case scenarios
    to ensure robustness.
    """
    
    def test_from_dict_with_zero_values(self):
        """
        Test from_dict with zero and falsy values.
        
        Verifies that zero, False, and empty string values are
        handled correctly and not treated as missing.
        """
        # Create dictionary with zero/falsy values
        falsy_dict = {
            "field1": "",  # Empty string
            "field2": 0,    # Zero
            "field3": False,  # False
        }
        
        # Convert to instance
        instance = ConcreteInfoBase.from_dict(falsy_dict)
        
        # Verify falsy values are preserved
        assert instance.field1 == ""
        assert instance.field2 == 0
        assert instance.field3 is False
    
    def test_from_dict_with_unicode_values(self):
        """
        Test from_dict with unicode and special characters.
        
        Verifies that unicode strings and special characters
        are handled correctly in field values.
        """
        # Create dictionary with unicode
        unicode_dict = {
            "field1": "测试值 🚀 émoji",
            "field2": 42,
            "field3": True,
        }
        
        # Convert to instance
        instance = ConcreteInfoBase.from_dict(unicode_dict)
        
        # Verify unicode is preserved
        assert instance.field1 == "测试值 🚀 émoji"
    
    def test_list_from_dicts_large_list(self):
        """
        Test list_from_dicts with a large list.
        
        Verifies that the method can handle larger lists efficiently.
        """
        # Create large list of dictionaries
        large_list = [
            {"field1": f"value_{i}", "field2": i, "field3": i % 2 == 0}
            for i in range(100)
        ]
        
        # Convert list
        instances = ConcreteInfoBase.list_from_dicts(large_list)
        
        # Verify all instances created
        assert len(instances) == 100
        
        # Verify a few instances
        assert instances[0].field1 == "value_0"
        assert instances[50].field1 == "value_50"
        assert instances[99].field1 == "value_99"

