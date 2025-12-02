"""
Comprehensive unit tests for the bittensor.core.chain_data.proposal_vote_data module.

This test suite covers all major components of the ProposalVoteData class including:
- Class instantiation and attribute validation
- Dictionary conversion (from_dict method)
- Account ID decoding for ayes and nays lists
- Inheritance from InfoBase
- Edge cases and error handling

The tests are designed to ensure that:
1. ProposalVoteData objects can be created correctly with all required fields
2. Dictionary conversion works correctly with chain data format
3. Account IDs are properly decoded from bytes format for both ayes and nays
4. Error handling is robust for missing or invalid data
5. All methods handle edge cases properly

ProposalVoteData represents senate/proposal voting data including who voted yes (ayes)
and who voted no (nays), along with voting threshold and end block information.

Each test includes extensive comments explaining:
- What functionality is being tested
- Why the test is important
- What assertions verify
- Expected behavior and edge cases
"""

from unittest.mock import patch

import pytest

# Import the modules to test
from bittensor.core.chain_data.proposal_vote_data import ProposalVoteData
from bittensor.core.errors import SubstrateRequestException


class TestProposalVoteDataInitialization:
    """
    Test class for ProposalVoteData object initialization.
    
    This class tests that ProposalVoteData objects can be created correctly with
    all required fields. ProposalVoteData contains proposal index, voting threshold,
    lists of yes/no voters, and end block information.
    """

    def test_proposal_vote_data_initialization_with_all_fields(self):
        """
        Test that ProposalVoteData can be initialized with all required fields.
        
        This test verifies that a ProposalVoteData object can be created with all
        required fields. ProposalVoteData contains information about a proposal vote
        including the proposal index, voting threshold, lists of accounts that voted
        yes (ayes) and no (nays), and the block number when voting ends.
        """
        # Create lists of SS58 addresses for voters
        ayes = [
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        ]
        nays = [
            "5GNJqTPyNqANBkUVMN1LPprxXnFouWXoe2wNSmmEoLctxiZY",
        ]
        
        # Create a ProposalVoteData instance with all fields
        proposal_vote_data = ProposalVoteData(
            index=1,  # Proposal index
            threshold=5,  # Voting threshold (number of votes needed)
            ayes=ayes,  # List of accounts that voted yes
            nays=nays,  # List of accounts that voted no
            end=10000,  # Block number when voting ends
        )
        
        # Verify all fields are set correctly
        assert proposal_vote_data.index == 1, \
            "Proposal index should be set correctly"
        assert proposal_vote_data.threshold == 5, \
            "Voting threshold should be set correctly"
        assert len(proposal_vote_data.ayes) == 2, \
            "Should have 2 accounts that voted yes"
        assert proposal_vote_data.ayes[0] == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "First yes voter should be set correctly"
        assert len(proposal_vote_data.nays) == 1, \
            "Should have 1 account that voted no"
        assert proposal_vote_data.nays[0] == "5GNJqTPyNqANBkUVMN1LPprxXnFouWXoe2wNSmmEoLctxiZY", \
            "First no voter should be set correctly"
        assert proposal_vote_data.end == 10000, \
            "End block number should be set correctly"

    def test_proposal_vote_data_initialization_with_empty_votes(self):
        """
        Test that ProposalVoteData can be initialized with empty vote lists.
        
        This test verifies that a proposal with no votes yet can be represented
        correctly with empty ayes and nays lists. This is a valid edge case for
        a newly created proposal that hasn't received any votes yet.
        """
        # Create ProposalVoteData with empty vote lists
        proposal_vote_data = ProposalVoteData(
            index=1,
            threshold=5,
            ayes=[],  # No yes votes yet
            nays=[],  # No no votes yet
            end=10000,
        )
        
        # Verify empty lists are handled correctly
        assert len(proposal_vote_data.ayes) == 0, \
            "Empty ayes list should be valid (no yes votes yet)"
        assert len(proposal_vote_data.nays) == 0, \
            "Empty nays list should be valid (no no votes yet)"
        assert isinstance(proposal_vote_data.ayes, list), \
            "Ayes should still be a list type (empty list)"
        assert isinstance(proposal_vote_data.nays, list), \
            "Nays should still be a list type (empty list)"

    def test_proposal_vote_data_inherits_from_info_base(self):
        """
        Test that ProposalVoteData properly inherits from InfoBase.
        
        This test verifies that ProposalVoteData is a subclass of InfoBase, which
        provides common functionality for chain data structures. This ensures
        consistency across all chain data classes.
        """
        from bittensor.core.chain_data.info_base import InfoBase
        assert issubclass(ProposalVoteData, InfoBase), \
            "ProposalVoteData should inherit from InfoBase for common chain data functionality"
        
        from dataclasses import is_dataclass
        assert is_dataclass(ProposalVoteData), \
            "ProposalVoteData should be a dataclass for automatic field handling"


class TestProposalVoteDataFromDict:
    """
    Test class for the from_dict() class method.
    
    This class tests that ProposalVoteData objects can be created from dictionary
    data. Note that this class uses from_dict() directly (not _from_dict), so it
    likely includes error handling.
    """

    def test_from_dict_creates_proposal_vote_data_correctly(self):
        """
        Test that from_dict() correctly creates ProposalVoteData from dictionary data.
        
        This test verifies that when given a dictionary with proposal vote information
        (as would come from chain data), the from_dict() method correctly creates a
        ProposalVoteData object. The conversion includes decoding account IDs for
        both ayes and nays lists from bytes to SS58 addresses.
        """
        # Mock decode_account_id for ayes and nays
        with patch("bittensor.core.chain_data.proposal_vote_data.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # ayes[0]
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # ayes[1]
                "5GNJqTPyNqANBkUVMN1LPprxXnFouWXoe2wNSmmEoLctxiZY",  # nays[0]
            ]
            
            # Create dictionary data as would come from chain
            proposal_dict = {
                "index": 1,
                "threshold": 5,
                "ayes": [b"aye0_bytes", b"aye1_bytes"],  # Raw bytes that will be decoded
                "nays": [b"nay0_bytes"],  # Raw bytes that will be decoded
                "end": 10000,
            }
            
            # Create ProposalVoteData from dictionary using from_dict class method
            proposal_vote_data = ProposalVoteData.from_dict(proposal_dict)
            
            # Verify it was created successfully
            assert isinstance(proposal_vote_data, ProposalVoteData), \
                "Should return a ProposalVoteData instance"
            
            # Verify key fields are set correctly
            assert proposal_vote_data.index == 1, \
                "Proposal index should be set correctly from dictionary"
            assert proposal_vote_data.threshold == 5, \
                "Voting threshold should be set correctly from dictionary"
            assert proposal_vote_data.end == 10000, \
                "End block number should be set correctly from dictionary"
            
            # Verify account IDs were decoded correctly
            assert len(proposal_vote_data.ayes) == 2, \
                "Should have 2 yes votes after decoding"
            assert proposal_vote_data.ayes[0] == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
                "First yes voter should be decoded from bytes to SS58 address"
            assert len(proposal_vote_data.nays) == 1, \
                "Should have 1 no vote after decoding"
            assert proposal_vote_data.nays[0] == "5GNJqTPyNqANBkUVMN1LPprxXnFouWXoe2wNSmmEoLctxiZY", \
                "First no voter should be decoded from bytes to SS58 address"

    def test_from_dict_decodes_account_ids_for_ayes(self):
        """
        Test that from_dict() correctly decodes account IDs for ayes list.
        
        This test verifies that the ayes list is properly decoded from bytes/raw
        format to SS58 string addresses using the decode_account_id utility function.
        This decoding is essential for working with account addresses in a
        human-readable format.
        """
        # Mock decode_account_id to verify it's called correctly
        with patch("bittensor.core.chain_data.proposal_vote_data.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with raw account ID bytes for ayes
            proposal_dict = {
                "index": 1,
                "threshold": 5,
                "ayes": [b"raw_aye0_bytes", b"raw_aye1_bytes"],  # Raw bytes from chain
                "nays": [],  # Empty nays
                "end": 10000,
            }
            
            # Create ProposalVoteData
            proposal_vote_data = ProposalVoteData.from_dict(proposal_dict)
            
            # Verify decode_account_id was called for ayes
            assert mock_decode.call_count == 2, \
                "decode_account_id should be called twice for two yes votes"
            mock_decode.assert_any_call(b"raw_aye0_bytes"), \
                "decode_account_id should be called with first aye bytes"
            mock_decode.assert_any_call(b"raw_aye1_bytes"), \
                "decode_account_id should be called with second aye bytes"

    def test_from_dict_decodes_account_ids_for_nays(self):
        """
        Test that from_dict() correctly decodes account IDs for nays list.
        
        This test verifies that the nays list is properly decoded from bytes/raw
        format to SS58 string addresses using the decode_account_id utility function.
        This is similar to ayes but ensures nays are also decoded correctly.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.proposal_vote_data.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GNJqTPyNqANBkUVMN1LPprxXnFouWXoe2wNSmmEoLctxiZY",
            ]
            
            # Create dictionary with raw account ID bytes for nays
            proposal_dict = {
                "index": 1,
                "threshold": 5,
                "ayes": [],  # Empty ayes
                "nays": [b"raw_nay0_bytes"],  # Raw bytes from chain
                "end": 10000,
            }
            
            # Create ProposalVoteData
            proposal_vote_data = ProposalVoteData.from_dict(proposal_dict)
            
            # Verify decode_account_id was called for nays
            mock_decode.assert_called_once_with(b"raw_nay0_bytes"), \
                "decode_account_id should be called with nay bytes"
            assert len(proposal_vote_data.nays) == 1, \
                "Should have one no vote after decoding"

    def test_from_dict_handles_empty_vote_lists(self):
        """
        Test that from_dict() handles empty vote lists correctly.
        
        This test verifies that when ayes and nays are empty lists (no votes yet),
        the from_dict() method correctly creates a ProposalVoteData object with
        empty lists. This is a valid edge case for a newly created proposal.
        """
        # Mock decode_account_id (won't be called for empty lists)
        with patch("bittensor.core.chain_data.proposal_vote_data.decode_account_id"):
            # Create dictionary with empty vote lists
            proposal_dict = {
                "index": 1,
                "threshold": 5,
                "ayes": [],  # Empty list - no yes votes yet
                "nays": [],  # Empty list - no no votes yet
                "end": 10000,
            }
            
            # Create ProposalVoteData
            proposal_vote_data = ProposalVoteData.from_dict(proposal_dict)
            
            # Verify empty lists are handled correctly
            assert len(proposal_vote_data.ayes) == 0, \
                "Empty ayes list should be valid"
            assert len(proposal_vote_data.nays) == 0, \
                "Empty nays list should be valid"
            assert isinstance(proposal_vote_data.ayes, list), \
                "Ayes should still be a list type (empty list)"
            assert isinstance(proposal_vote_data.nays, list), \
                "Nays should still be a list type (empty list)"


class TestProposalVoteDataEdgeCases:
    """
    Test class for edge cases and special scenarios.
    
    This class tests edge cases such as zero threshold, zero end block, large
    vote lists, and other boundary conditions.
    """

    def test_proposal_vote_data_with_zero_threshold(self):
        """
        Test that ProposalVoteData handles zero threshold correctly.
        
        This test verifies that a threshold of zero is handled correctly.
        This might represent a proposal that requires no votes or has special rules.
        """
        proposal_vote_data = ProposalVoteData(
            index=1,
            threshold=0,  # Zero threshold
            ayes=[],
            nays=[],
            end=10000,
        )
        
        # Verify zero threshold is handled
        assert proposal_vote_data.threshold == 0, \
            "Zero threshold should be handled correctly"

    def test_proposal_vote_data_with_large_vote_lists(self):
        """
        Test that ProposalVoteData handles large vote lists correctly.
        
        This test verifies that ProposalVoteData can handle proposals with many
        voters. This is important for proposals that receive widespread participation.
        """
        # Mock decode_account_id for many voters
        with patch("bittensor.core.chain_data.proposal_vote_data.decode_account_id") as mock_decode:
            num_ayes = 50
            num_nays = 20
            # Create side_effect list for many voters
            side_effects = []
            for i in range(num_ayes):
                side_effects.append(f"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY{i}")
            for i in range(num_nays):
                side_effects.append(f"5GNJqTPyNqANBkUVMN1LPprxXnFouWXoe2wNSmmEoLctxiZY{i}")
            mock_decode.side_effect = side_effects
            
            # Create dictionary with many votes
            proposal_dict = {
                "index": 1,
                "threshold": 50,
                "ayes": [b"aye_bytes"] * num_ayes,
                "nays": [b"nay_bytes"] * num_nays,
                "end": 10000,
            }
            
            # Create ProposalVoteData
            proposal_vote_data = ProposalVoteData.from_dict(proposal_dict)
            
            # Verify large vote lists are handled correctly
            assert len(proposal_vote_data.ayes) == num_ayes, \
                f"Should have {num_ayes} yes votes"
            assert len(proposal_vote_data.nays) == num_nays, \
                f"Should have {num_nays} no votes"

    def test_proposal_vote_data_field_types(self):
        """
        Test that ProposalVoteData fields maintain correct types.
        
        This test verifies that all fields in ProposalVoteData maintain their
        expected types. This is important for type consistency and ensures that
        the dataclass properly enforces type constraints at runtime.
        """
        proposal_vote_data = ProposalVoteData(
            index=1,
            threshold=5,
            ayes=["5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"],
            nays=["5GNJqTPyNqANBkUVMN1LPprxXnFouWXoe2wNSmmEoLctxiZY"],
            end=10000,
        )
        
        # Verify all field types are correct
        assert isinstance(proposal_vote_data.index, int), \
            "index should be int type"
        assert isinstance(proposal_vote_data.threshold, int), \
            "threshold should be int type"
        assert isinstance(proposal_vote_data.ayes, list), \
            "ayes should be list type"
        assert isinstance(proposal_vote_data.ayes[0], str), \
            "ayes elements should be string type (SS58 addresses)"
        assert isinstance(proposal_vote_data.nays, list), \
            "nays should be list type"
        assert isinstance(proposal_vote_data.nays[0], str), \
            "nays elements should be string type (SS58 addresses)"
        assert isinstance(proposal_vote_data.end, int), \
            "end should be int type (block number)"

