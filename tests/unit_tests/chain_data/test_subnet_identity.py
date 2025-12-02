"""
Comprehensive unit tests for the bittensor.core.chain_data.subnet_identity module.

This test suite covers all major components of the SubnetIdentity class including:
- Class instantiation and attribute validation
- Dictionary conversion (_from_dict method)
- String field handling (subnet_name, github_repo, etc.)
- Empty string handling
- Edge cases

The tests are designed to ensure that:
1. SubnetIdentity objects can be created correctly with all required fields
2. Dictionary conversion works correctly
3. String fields are properly handled
4. Empty strings are allowed (valid edge case)
5. All methods handle edge cases properly

Note: SubnetIdentity does NOT inherit from InfoBase, unlike most other chain data classes.
It's a simple dataclass with 8 string fields representing subnet metadata.

Each test includes extensive comments explaining:
- What functionality is being tested
- Why the test is important
- What assertions verify
- Expected behavior and edge cases
"""

import pytest

# Import the module to test
from bittensor.core.chain_data.subnet_identity import SubnetIdentity


class TestSubnetIdentityInitialization:
    """
    Test class for SubnetIdentity object initialization.
    
    This class tests that SubnetIdentity objects can be created correctly with
    all required fields. SubnetIdentity has 8 string fields representing subnet
    metadata (name, GitHub repo, contact, URL, logo, Discord, description, etc.).
    """

    def test_subnet_identity_initialization_with_all_fields(self):
        """
        Test that SubnetIdentity can be initialized with all required fields.
        
        This test verifies that a SubnetIdentity object can be created with all
        required string fields. SubnetIdentity is a simple dataclass containing
        metadata about a subnet such as its name, GitHub repository, contact
        information, URLs, and description.
        """
        # Create a SubnetIdentity with all fields
        subnet_identity = SubnetIdentity(
            subnet_name="Test Subnet",
            github_repo="https://github.com/test/subnet",
            subnet_contact="contact@subnet.test",
            subnet_url="https://subnet.test",
            logo_url="https://subnet.test/logo.png",
            discord="https://discord.gg/test",
            description="This is a test subnet for unit testing purposes.",
            additional="Additional information about the subnet."
        )
        
        # Verify all fields are set correctly
        assert subnet_identity.subnet_name == "Test Subnet", \
            "Subnet name should be set correctly"
        assert subnet_identity.github_repo == "https://github.com/test/subnet", \
            "GitHub repository URL should be set correctly"
        assert subnet_identity.subnet_contact == "contact@subnet.test", \
            "Subnet contact information should be set correctly"
        assert subnet_identity.subnet_url == "https://subnet.test", \
            "Subnet URL should be set correctly"
        assert subnet_identity.logo_url == "https://subnet.test/logo.png", \
            "Logo URL should be set correctly"
        assert subnet_identity.discord == "https://discord.gg/test", \
            "Discord link should be set correctly"
        assert subnet_identity.description == "This is a test subnet for unit testing purposes.", \
            "Description should be set correctly"
        assert subnet_identity.additional == "Additional information about the subnet.", \
            "Additional information should be set correctly"

    def test_subnet_identity_initialization_with_empty_strings(self):
        """
        Test that SubnetIdentity can be initialized with empty strings.
        
        This test verifies that empty strings are allowed for all fields.
        This is a valid edge case when subnet metadata hasn't been set yet
        or when certain fields are intentionally left blank.
        """
        # Create a SubnetIdentity with empty strings
        subnet_identity = SubnetIdentity(
            subnet_name="",
            github_repo="",
            subnet_contact="",
            subnet_url="",
            logo_url="",
            discord="",
            description="",
            additional=""
        )
        
        # Verify empty strings are handled correctly
        assert subnet_identity.subnet_name == "", \
            "Empty subnet name should be valid"
        assert subnet_identity.github_repo == "", \
            "Empty GitHub repo should be valid"
        assert subnet_identity.subnet_contact == "", \
            "Empty subnet contact should be valid"
        assert subnet_identity.subnet_url == "", \
            "Empty subnet URL should be valid"
        assert subnet_identity.logo_url == "", \
            "Empty logo URL should be valid"
        assert subnet_identity.discord == "", \
            "Empty Discord link should be valid"
        assert subnet_identity.description == "", \
            "Empty description should be valid"
        assert subnet_identity.additional == "", \
            "Empty additional information should be valid"

    def test_subnet_identity_is_dataclass(self):
        """
        Test that SubnetIdentity is a dataclass.
        
        This test verifies that SubnetIdentity is properly defined as a dataclass,
        which provides automatic field handling, equality comparison, and other
        dataclass features. Note that SubnetIdentity does NOT inherit from InfoBase.
        """
        from dataclasses import is_dataclass
        assert is_dataclass(SubnetIdentity), \
            "SubnetIdentity should be a dataclass for automatic field handling"
        
        # Verify it does NOT inherit from InfoBase (unlike other chain data classes)
        from bittensor.core.chain_data.info_base import InfoBase
        assert not issubclass(SubnetIdentity, InfoBase), \
            "SubnetIdentity should NOT inherit from InfoBase (unlike most chain data classes)"


class TestSubnetIdentityFromDict:
    """
    Test class for the _from_dict() class method.
    
    This class tests that SubnetIdentity objects can be created from dictionary
    data. The conversion is straightforward as all fields are strings that map
    directly from the dictionary.
    """

    def test_from_dict_creates_subnet_identity_correctly(self):
        """
        Test that _from_dict() correctly creates SubnetIdentity from dictionary data.
        
        This test verifies that when given a dictionary with subnet identity information
        (as would come from chain data), the _from_dict() method correctly creates a
        SubnetIdentity object. All fields are strings and map directly from the dictionary.
        """
        # Create dictionary data as would come from chain
        decoded = {
            "subnet_name": "Dict Subnet",
            "github_repo": "https://github.com/dict/subnet",
            "subnet_contact": "contact@dict.test",
            "subnet_url": "https://dict.test",
            "logo_url": "https://dict.test/logo.png",
            "discord": "https://discord.gg/dict",
            "description": "Subnet created from dictionary",
            "additional": "Dict additional info",
        }
        
        # Create SubnetIdentity from dictionary using _from_dict class method
        subnet_identity = SubnetIdentity._from_dict(decoded)
        
        # Verify it was created successfully
        assert isinstance(subnet_identity, SubnetIdentity), \
            "Should return a SubnetIdentity instance"
        
        # Verify all fields are set correctly
        assert subnet_identity.subnet_name == "Dict Subnet", \
            "Subnet name should be set correctly from dictionary"
        assert subnet_identity.github_repo == "https://github.com/dict/subnet", \
            "GitHub repository should be set correctly from dictionary"
        assert subnet_identity.subnet_contact == "contact@dict.test", \
            "Subnet contact should be set correctly from dictionary"
        assert subnet_identity.subnet_url == "https://dict.test", \
            "Subnet URL should be set correctly from dictionary"
        assert subnet_identity.logo_url == "https://dict.test/logo.png", \
            "Logo URL should be set correctly from dictionary"
        assert subnet_identity.discord == "https://discord.gg/dict", \
            "Discord link should be set correctly from dictionary"
        assert subnet_identity.description == "Subnet created from dictionary", \
            "Description should be set correctly from dictionary"
        assert subnet_identity.additional == "Dict additional info", \
            "Additional information should be set correctly from dictionary"

    def test_from_dict_handles_empty_strings(self):
        """
        Test that _from_dict() handles empty strings correctly.
        
        This test verifies that when dictionary values are empty strings,
        the _from_dict() method correctly creates a SubnetIdentity object
        with empty strings. This is a valid edge case.
        """
        # Create dictionary with empty strings
        decoded = {
            "subnet_name": "",
            "github_repo": "",
            "subnet_contact": "",
            "subnet_url": "",
            "logo_url": "",
            "discord": "",
            "description": "",
            "additional": "",
        }
        
        # Create SubnetIdentity
        subnet_identity = SubnetIdentity._from_dict(decoded)
        
        # Verify empty strings are handled correctly
        assert subnet_identity.subnet_name == "", \
            "Empty subnet name should be handled correctly"
        assert subnet_identity.github_repo == "", \
            "Empty GitHub repo should be handled correctly"
        assert all(field == "" for field in [
            subnet_identity.subnet_contact,
            subnet_identity.subnet_url,
            subnet_identity.logo_url,
            subnet_identity.discord,
            subnet_identity.description,
            subnet_identity.additional,
        ]), "All other fields should also be empty strings"


class TestSubnetIdentityEdgeCases:
    """
    Test class for edge cases and special scenarios.
    
    This class tests edge cases such as long strings, special characters,
    URL formats, and other boundary conditions.
    """

    def test_subnet_identity_with_long_strings(self):
        """
        Test that SubnetIdentity handles long strings correctly.
        
        This test verifies that SubnetIdentity can handle long string values
        without issues. This is important for descriptions and additional
        information that might be lengthy.
        """
        # Create SubnetIdentity with long strings
        long_description = "A" * 1000  # 1000 character string
        long_additional = "B" * 500  # 500 character string
        
        subnet_identity = SubnetIdentity(
            subnet_name="Long String Test",
            github_repo="https://github.com/test/subnet",
            subnet_contact="contact@test.com",
            subnet_url="https://test.com",
            logo_url="https://test.com/logo.png",
            discord="https://discord.gg/test",
            description=long_description,
            additional=long_additional,
        )
        
        # Verify long strings are handled correctly
        assert len(subnet_identity.description) == 1000, \
            "Long description should be stored correctly"
        assert len(subnet_identity.additional) == 500, \
            "Long additional information should be stored correctly"
        assert subnet_identity.description == long_description, \
            "Long description content should match"

    def test_subnet_identity_with_special_characters(self):
        """
        Test that SubnetIdentity handles special characters correctly.
        
        This test verifies that SubnetIdentity can handle special characters
        in string fields, including Unicode characters, URLs with query parameters,
        and other special formatting.
        """
        # Create SubnetIdentity with special characters
        subnet_identity = SubnetIdentity(
            subnet_name="Test Subnet 🚀",
            github_repo="https://github.com/test/subnet?ref=main&branch=dev",
            subnet_contact="contact+test@subnet.example.com",
            subnet_url="https://subnet.test/path/to/page?param=value#section",
            logo_url="https://subnet.test/logo.png?v=1.0&size=large",
            discord="https://discord.gg/test-channel",
            description="Description with special chars: @#$%^&*()",
            additional="Additional info with unicode: 中文 🎉",
        )
        
        # Verify special characters are handled correctly
        assert "🚀" in subnet_identity.subnet_name, \
            "Unicode characters should be preserved"
        assert "?" in subnet_identity.github_repo, \
            "URL query parameters should be preserved"
        assert "+" in subnet_identity.subnet_contact, \
            "Email plus addressing should be preserved"
        assert "#" in subnet_identity.subnet_url, \
            "URL fragments should be preserved"
        assert "中文" in subnet_identity.additional, \
            "Non-ASCII characters should be preserved"

    def test_subnet_identity_field_types(self):
        """
        Test that SubnetIdentity fields maintain correct types.
        
        This test verifies that all fields in SubnetIdentity maintain their
        expected types (all should be strings). This is important for type
        consistency and ensures that the dataclass properly enforces type
        constraints at runtime.
        """
        # Create SubnetIdentity
        subnet_identity = SubnetIdentity(
            subnet_name="Test",
            github_repo="https://github.com/test",
            subnet_contact="contact@test.com",
            subnet_url="https://test.com",
            logo_url="https://test.com/logo.png",
            discord="https://discord.gg/test",
            description="Test description",
            additional="Test additional",
        )
        
        # Verify all field types are strings
        assert isinstance(subnet_identity.subnet_name, str), \
            "subnet_name should be string type"
        assert isinstance(subnet_identity.github_repo, str), \
            "github_repo should be string type"
        assert isinstance(subnet_identity.subnet_contact, str), \
            "subnet_contact should be string type"
        assert isinstance(subnet_identity.subnet_url, str), \
            "subnet_url should be string type"
        assert isinstance(subnet_identity.logo_url, str), \
            "logo_url should be string type"
        assert isinstance(subnet_identity.discord, str), \
            "discord should be string type"
        assert isinstance(subnet_identity.description, str), \
            "description should be string type"
        assert isinstance(subnet_identity.additional, str), \
            "additional should be string type"

    def test_subnet_identity_equality_comparison(self):
        """
        Test that SubnetIdentity objects can be compared for equality.
        
        This test verifies that SubnetIdentity objects (being dataclasses) support
        equality comparison. Two SubnetIdentity objects with the same field values
        should be considered equal.
        """
        # Create two SubnetIdentity objects with same values
        identity1 = SubnetIdentity(
            subnet_name="Test",
            github_repo="https://github.com/test",
            subnet_contact="contact@test.com",
            subnet_url="https://test.com",
            logo_url="https://test.com/logo.png",
            discord="https://discord.gg/test",
            description="Test description",
            additional="Test additional",
        )
        
        identity2 = SubnetIdentity(
            subnet_name="Test",
            github_repo="https://github.com/test",
            subnet_contact="contact@test.com",
            subnet_url="https://test.com",
            logo_url="https://test.com/logo.png",
            discord="https://discord.gg/test",
            description="Test description",
            additional="Test additional",
        )
        
        # Verify equality comparison works
        assert identity1 == identity2, \
            "Two SubnetIdentity objects with same values should be equal"
        
        # Create a different SubnetIdentity
        identity3 = SubnetIdentity(
            subnet_name="Different",
            github_repo="https://github.com/test",
            subnet_contact="contact@test.com",
            subnet_url="https://test.com",
            logo_url="https://test.com/logo.png",
            discord="https://discord.gg/test",
            description="Test description",
            additional="Test additional",
        )
        
        # Verify inequality
        assert identity1 != identity3, \
            "Two SubnetIdentity objects with different values should not be equal"

