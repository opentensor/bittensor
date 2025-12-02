"""
Comprehensive unit tests for the bittensor.core.chain_data.chain_identity module.

This test suite covers all major components of the ChainIdentity class including:
- Class instantiation and attribute validation
- Dictionary conversion (_from_dict, from_dict)
- Inheritance from InfoBase
- Edge cases and error handling

The tests are designed to ensure that:
1. ChainIdentity objects can be created correctly with all required fields
2. Dictionary conversion works correctly
3. Error handling is robust for missing fields
4. All methods handle edge cases properly
"""

import pytest

# Import the modules to test
from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.errors import SubstrateRequestException


class TestChainIdentityInitialization:
    """
    Test class for ChainIdentity object initialization.
    
    This class tests that ChainIdentity objects can be created correctly with
    all required string fields.
    """

    def test_chain_identity_initialization_with_all_fields(self):
        """
        Test that ChainIdentity can be initialized with all required fields.
        
        This test verifies that a ChainIdentity object can be created with all
        required string fields (name, url, github, image, discord, description,
        additional). All fields are required and should be strings.
        """
        # Create a ChainIdentity with all fields
        chain_identity = ChainIdentity(
            name="Test Network",
            url="https://example.com",
            github="https://github.com/example/repo",
            image="https://example.com/image.png",
            discord="https://discord.gg/example",
            description="This is a test network description",
            additional="Additional information"
        )
        
        # Verify all fields are set correctly
        assert chain_identity.name == "Test Network", "Name should be set correctly"
        assert chain_identity.url == "https://example.com", "URL should be set correctly"
        assert chain_identity.github == "https://github.com/example/repo", \
            "GitHub should be set correctly"
        assert chain_identity.image == "https://example.com/image.png", \
            "Image should be set correctly"
        assert chain_identity.discord == "https://discord.gg/example", \
            "Discord should be set correctly"
        assert chain_identity.description == "This is a test network description", \
            "Description should be set correctly"
        assert chain_identity.additional == "Additional information", \
            "Additional should be set correctly"

    def test_chain_identity_with_empty_strings(self):
        """
        Test that ChainIdentity can be initialized with empty strings.
        
        This test verifies that ChainIdentity objects can be created with empty
        strings for all fields. This is valid as empty strings are still valid
        string values, and some chain identities might not have all fields filled.
        """
        # Create a ChainIdentity with empty strings
        chain_identity = ChainIdentity(
            name="",
            url="",
            github="",
            image="",
            discord="",
            description="",
            additional=""
        )
        
        # Verify all fields are empty strings
        assert chain_identity.name == "", "Name can be empty string"
        assert chain_identity.url == "", "URL can be empty string"
        assert chain_identity.github == "", "GitHub can be empty string"
        assert chain_identity.image == "", "Image can be empty string"
        assert chain_identity.discord == "", "Discord can be empty string"
        assert chain_identity.description == "", "Description can be empty string"
        assert chain_identity.additional == "", "Additional can be empty string"

    def test_chain_identity_inherits_from_info_base(self):
        """
        Test that ChainIdentity properly inherits from InfoBase.
        
        This test verifies that ChainIdentity is a subclass of InfoBase, which
        provides common functionality for chain data structures. This ensures
        that ChainIdentity can use methods like from_dict() and list_from_dicts()
        from the base class.
        """
        # Verify inheritance
        from bittensor.core.chain_data.info_base import InfoBase
        assert issubclass(ChainIdentity, InfoBase), \
            "ChainIdentity should inherit from InfoBase"
        
        # Verify ChainIdentity is a dataclass (which InfoBase also is)
        from dataclasses import is_dataclass
        assert is_dataclass(ChainIdentity), "ChainIdentity should be a dataclass"


class TestChainIdentityFromDict:
    """
    Test class for the _from_dict() class method.
    
    This class tests that ChainIdentity objects can be created from dictionary
    data, which is how chain data is typically received from the substrate
    interface. The _from_dict method handles the field name mapping from
    chain data format to the ChainIdentity attributes.
    """

    def test_from_dict_creates_chain_identity_correctly(self):
        """
        Test that _from_dict() correctly creates a ChainIdentity from dictionary data.
        
        This test verifies that when given a dictionary with chain identity fields
        (using the chain data format with "github_repo" instead of "github"),
        the _from_dict() method correctly creates a ChainIdentity object with
        all fields mapped correctly.
        """
        # Create dictionary data (as would come from chain)
        # Note: chain data uses "github_repo" instead of "github"
        decoded = {
            "name": "Bittensor Network",
            "url": "https://bittensor.com",
            "github_repo": "https://github.com/opentensor/bittensor",
            "image": "https://bittensor.com/logo.png",
            "discord": "https://discord.gg/bittensor",
            "description": "Decentralized intelligence network",
            "additional": "Additional chain information"
        }
        
        # Create ChainIdentity from dictionary
        chain_identity = ChainIdentity._from_dict(decoded)
        
        # Verify all fields are mapped correctly
        assert chain_identity.name == "Bittensor Network", \
            "Name should be mapped from 'name' key"
        assert chain_identity.url == "https://bittensor.com", \
            "URL should be mapped from 'url' key"
        assert chain_identity.github == "https://github.com/opentensor/bittensor", \
            "GitHub should be mapped from 'github_repo' key"
        assert chain_identity.image == "https://bittensor.com/logo.png", \
            "Image should be mapped from 'image' key"
        assert chain_identity.discord == "https://discord.gg/bittensor", \
            "Discord should be mapped from 'discord' key"
        assert chain_identity.description == "Decentralized intelligence network", \
            "Description should be mapped from 'description' key"
        assert chain_identity.additional == "Additional chain information", \
            "Additional should be mapped from 'additional' key"

    def test_from_dict_maps_github_repo_to_github(self):
        """
        Test that _from_dict() correctly maps 'github_repo' to 'github' attribute.
        
        This test specifically verifies the field name mapping between the chain
        data format (which uses "github_repo") and the ChainIdentity class
        attribute (which uses "github"). This is an important mapping to ensure
        compatibility with chain data structures.
        """
        # Create dictionary with github_repo (chain format)
        decoded = {
            "name": "Test",
            "url": "https://test.com",
            "github_repo": "https://github.com/test/repo",  # Chain uses github_repo
            "image": "https://test.com/img.png",
            "discord": "https://discord.gg/test",
            "description": "Test description",
            "additional": "Test additional"
        }
        
        # Create ChainIdentity from dictionary
        chain_identity = ChainIdentity._from_dict(decoded)
        
        # Verify github_repo is mapped to github attribute
        assert chain_identity.github == "https://github.com/test/repo", \
            "github_repo from dict should be mapped to github attribute"
        assert not hasattr(chain_identity, "github_repo"), \
            "ChainIdentity should not have github_repo attribute, only github"

    def test_from_dict_handles_empty_strings(self):
        """
        Test that _from_dict() correctly handles empty string values.
        
        This test verifies that when chain data contains empty strings, the
        _from_dict() method correctly creates a ChainIdentity object with
        those empty string values. This is valid as empty strings are still
        valid values.
        """
        # Create dictionary with empty strings
        decoded = {
            "name": "",
            "url": "",
            "github_repo": "",
            "image": "",
            "discord": "",
            "description": "",
            "additional": ""
        }
        
        # Create ChainIdentity from dictionary
        chain_identity = ChainIdentity._from_dict(decoded)
        
        # Verify all fields are empty strings
        assert chain_identity.name == "", "Empty name should be handled"
        assert chain_identity.url == "", "Empty URL should be handled"
        assert chain_identity.github == "", "Empty GitHub should be handled"
        assert chain_identity.image == "", "Empty image should be handled"
        assert chain_identity.discord == "", "Empty discord should be handled"
        assert chain_identity.description == "", "Empty description should be handled"
        assert chain_identity.additional == "", "Empty additional should be handled"


class TestChainIdentityFromDictBaseClass:
    """
    Test class for the from_dict() method inherited from InfoBase.
    
    This class tests that ChainIdentity can use the from_dict() method from
    InfoBase, which includes error handling for missing fields.
    """

    def test_from_dict_with_complete_data(self):
        """
        Test that from_dict() works with complete data.
        
        This test verifies that the from_dict() method (inherited from InfoBase)
        correctly calls _from_dict() when all required fields are present in
        the dictionary. This is the happy path for creating ChainIdentity from chain data.
        """
        # Create complete dictionary data
        decoded = {
            "name": "Complete Network",
            "url": "https://complete.com",
            "github_repo": "https://github.com/complete/repo",
            "image": "https://complete.com/image.png",
            "discord": "https://discord.gg/complete",
            "description": "Complete description",
            "additional": "Complete additional"
        }
        
        # Create ChainIdentity using from_dict (from InfoBase)
        chain_identity = ChainIdentity.from_dict(decoded)
        
        # Verify it was created successfully
        assert isinstance(chain_identity, ChainIdentity), \
            "from_dict() should return a ChainIdentity instance"
        assert chain_identity.name == "Complete Network", \
            "Name should be set correctly"
        assert chain_identity.url == "https://complete.com", \
            "URL should be set correctly"

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
            "name": "Incomplete Network",
            "url": "https://incomplete.com",
            # Missing github_repo, image, discord, description, additional
        }
        
        # Verify from_dict raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException) as exc_info:
            ChainIdentity.from_dict(incomplete_data)
        
        # Verify error message mentions missing field
        assert "missing" in str(exc_info.value).lower(), \
            "Error message should mention missing field"
        assert "ChainIdentity" in str(exc_info.value), \
            "Error message should mention ChainIdentity class"

    def test_from_dict_raises_exception_on_missing_github_repo(self):
        """
        Test that from_dict() raises exception when github_repo is missing.
        
        This test specifically verifies that when the 'github_repo' field is
        missing (which is mapped to the 'github' attribute), the from_dict()
        method raises an appropriate exception. This is important because
        github_repo is a required field in the chain data format.
        """
        # Create dictionary missing github_repo
        incomplete_data = {
            "name": "Network",
            "url": "https://network.com",
            # Missing github_repo
            "image": "https://network.com/img.png",
            "discord": "https://discord.gg/network",
            "description": "Description",
            "additional": "Additional"
        }
        
        # Verify from_dict raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException) as exc_info:
            ChainIdentity.from_dict(incomplete_data)
        
        # Verify error message mentions the missing field
        assert "missing" in str(exc_info.value).lower(), \
            "Error message should indicate missing field"

    def test_from_dict_raises_exception_on_missing_name(self):
        """
        Test that from_dict() raises exception when name is missing.
        
        This test verifies that when the 'name' field is missing, the from_dict()
        method raises an appropriate exception. Name is a required field for
        chain identity information.
        """
        # Create dictionary missing name
        incomplete_data = {
            # Missing name
            "url": "https://network.com",
            "github_repo": "https://github.com/network/repo",
            "image": "https://network.com/img.png",
            "discord": "https://discord.gg/network",
            "description": "Description",
            "additional": "Additional"
        }
        
        # Verify from_dict raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException) as exc_info:
            ChainIdentity.from_dict(incomplete_data)
        
        # Verify error message mentions missing field
        assert "missing" in str(exc_info.value).lower(), \
            "Error message should indicate missing field"


class TestChainIdentityListFromDicts:
    """
    Test class for the list_from_dicts() method inherited from InfoBase.
    
    This class tests that ChainIdentity can use the list_from_dicts() method
    to create multiple ChainIdentity objects from a list of dictionaries.
    """

    def test_list_from_dicts_creates_multiple_chain_identities(self):
        """
        Test that list_from_dicts() creates multiple ChainIdentity objects.
        
        This test verifies that when given a list of dictionaries (each containing
        chain identity data), the list_from_dicts() method correctly creates
        a list of ChainIdentity objects. This is useful when processing multiple
        chain identities from chain data.
        """
        # Create list of dictionary data
        decoded_list = [
            {
                "name": "Network 1",
                "url": "https://network1.com",
                "github_repo": "https://github.com/network1/repo",
                "image": "https://network1.com/img.png",
                "discord": "https://discord.gg/network1",
                "description": "Description 1",
                "additional": "Additional 1"
            },
            {
                "name": "Network 2",
                "url": "https://network2.com",
                "github_repo": "https://github.com/network2/repo",
                "image": "https://network2.com/img.png",
                "discord": "https://discord.gg/network2",
                "description": "Description 2",
                "additional": "Additional 2"
            }
        ]
        
        # Create list of ChainIdentity objects
        chain_identities = ChainIdentity.list_from_dicts(decoded_list)
        
        # Verify it returns a list
        assert isinstance(chain_identities, list), \
            "list_from_dicts() should return a list"
        
        # Verify correct number of objects
        assert len(chain_identities) == 2, \
            "Should create 2 ChainIdentity objects"
        
        # Verify each object is a ChainIdentity
        assert isinstance(chain_identities[0], ChainIdentity), \
            "First element should be ChainIdentity"
        assert isinstance(chain_identities[1], ChainIdentity), \
            "Second element should be ChainIdentity"
        
        # Verify values are correct
        assert chain_identities[0].name == "Network 1", \
            "First ChainIdentity should have correct name"
        assert chain_identities[1].name == "Network 2", \
            "Second ChainIdentity should have correct name"

    def test_list_from_dicts_handles_empty_list(self):
        """
        Test that list_from_dicts() handles empty list correctly.
        
        This test verifies that when given an empty list, the list_from_dicts()
        method returns an empty list rather than raising an error. This is
        the expected behavior for edge cases.
        """
        # Create empty list
        empty_list = []
        
        # Create list of ChainIdentity objects (should be empty)
        chain_identities = ChainIdentity.list_from_dicts(empty_list)
        
        # Verify it returns an empty list
        assert isinstance(chain_identities, list), \
            "Should return a list even when empty"
        assert len(chain_identities) == 0, \
            "Should return empty list for empty input"

    def test_list_from_dicts_raises_on_invalid_item(self):
        """
        Test that list_from_dicts() raises exception when an item is invalid.
        
        This test verifies that when the list contains a dictionary with missing
        required fields, the list_from_dicts() method raises a SubstrateRequestException.
        This ensures data validation even when processing multiple items.
        """
        # Create list with one valid and one invalid dictionary
        decoded_list = [
            {
                "name": "Valid Network",
                "url": "https://valid.com",
                "github_repo": "https://github.com/valid/repo",
                "image": "https://valid.com/img.png",
                "discord": "https://discord.gg/valid",
                "description": "Valid description",
                "additional": "Valid additional"
            },
            {
                "name": "Invalid Network",
                # Missing required fields
            }
        ]
        
        # Verify list_from_dicts raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException):
            ChainIdentity.list_from_dicts(decoded_list)


class TestChainIdentityEdgeCases:
    """
    Test class for edge cases and special scenarios.
    
    This class tests edge cases such as very long strings, special characters,
    and other boundary conditions.
    """

    def test_chain_identity_with_long_strings(self):
        """
        Test that ChainIdentity handles very long string values.
        
        This test verifies that ChainIdentity objects can be created with
        very long string values for all fields. This is important for
        robustness when dealing with real-world chain data that might
        contain lengthy descriptions or URLs.
        """
        # Create ChainIdentity with very long strings
        long_string = "A" * 1000  # 1000 character string
        
        chain_identity = ChainIdentity(
            name=long_string,
            url=long_string,
            github=long_string,
            image=long_string,
            discord=long_string,
            description=long_string,
            additional=long_string
        )
        
        # Verify all fields are set correctly
        assert len(chain_identity.name) == 1000, \
            "Should handle long name strings"
        assert len(chain_identity.description) == 1000, \
            "Should handle long description strings"

    def test_chain_identity_with_special_characters(self):
        """
        Test that ChainIdentity handles special characters in strings.
        
        This test verifies that ChainIdentity objects can be created with
        special characters (unicode, emojis, etc.) in the string fields.
        This is important for internationalization and modern web content.
        """
        # Create ChainIdentity with special characters
        chain_identity = ChainIdentity(
            name="Test Network 🌐",
            url="https://example.com/path?query=value&other=数据",
            github="https://github.com/user/repo#readme",
            image="https://example.com/image.png",
            discord="https://discord.gg/channel",
            description="Description with émojis 🚀 and unicode 数据",
            additional="Additional: 特殊字符"
        )
        
        # Verify special characters are preserved
        assert "🌐" in chain_identity.name, \
            "Should preserve emojis in name"
        assert "数据" in chain_identity.url, \
            "Should preserve unicode in URL"
        assert "émojis" in chain_identity.description, \
            "Should preserve special characters in description"

    def test_chain_identity_field_types(self):
        """
        Test that ChainIdentity fields maintain string type.
        
        This test verifies that all fields in ChainIdentity remain as strings
        when set. This is important for type consistency and ensures that
        the dataclass properly enforces string types.
        """
        # Create ChainIdentity
        chain_identity = ChainIdentity(
            name="Test",
            url="https://test.com",
            github="https://github.com/test/repo",
            image="https://test.com/img.png",
            discord="https://discord.gg/test",
            description="Description",
            additional="Additional"
        )
        
        # Verify all fields are strings
        assert isinstance(chain_identity.name, str), \
            "Name should be string type"
        assert isinstance(chain_identity.url, str), \
            "URL should be string type"
        assert isinstance(chain_identity.github, str), \
            "GitHub should be string type"
        assert isinstance(chain_identity.image, str), \
            "Image should be string type"
        assert isinstance(chain_identity.discord, str), \
            "Discord should be string type"
        assert isinstance(chain_identity.description, str), \
            "Description should be string type"
        assert isinstance(chain_identity.additional, str), \
            "Additional should be string type"

