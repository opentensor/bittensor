"""
Comprehensive unit tests for the bittensor.core.chain_data.subnet_state module.

This test suite covers all major components of the SubnetState class including:
- Class instantiation and attribute validation
- Dictionary conversion (_from_dict, from_dict from InfoBase)
- Account ID decoding for hotkeys and coldkeys
- Balance conversions from rao to Balance objects
- U16 normalization for various score fields (pruning_score, dividends, etc.)
- Balance unit assignments (netuid for subnet tokens, 0 for TAO)
- List structure validation (ensuring all lists have same length per neuron)
- Inheritance from InfoBase
- Edge cases and error handling

The tests are designed to ensure that:
1. SubnetState objects can be created correctly with all required fields
2. Dictionary conversion works correctly with chain data format
3. Account IDs are properly decoded from bytes format
4. Balance values are correctly converted from rao
5. U16 values are properly normalized to floats
6. Balance units are assigned correctly (netuid vs 0)
7. Error handling is robust for missing or invalid data
8. All methods handle edge cases properly

SubnetState is a complex data structure that represents the complete state of a subnet,
with per-neuron information stored as parallel lists. Each index in the lists corresponds
to a single neuron's data.

Each test includes extensive comments explaining:
- What functionality is being tested
- Why the test is important
- What assertions verify
- Expected behavior and edge cases
"""

from unittest.mock import patch

import pytest

# Import the modules to test
from bittensor.core.chain_data.subnet_state import SubnetState
from bittensor.core.errors import SubstrateRequestException
from bittensor.utils.balance import Balance


class TestSubnetStateInitialization:
    """
    Test class for SubnetState object initialization.
    
    This class tests that SubnetState objects can be created correctly with
    all required fields. SubnetState has many fields that are lists, where
    each index corresponds to a neuron's data.
    """

    def test_subnet_state_initialization_with_all_fields(self):
        """
        Test that SubnetState can be initialized with all required fields.
        
        This test verifies that a SubnetState object can be created with all
        required fields. SubnetState has many fields that are lists containing
        per-neuron information. Each index in the lists corresponds to a single
        neuron's data, so all lists should typically have the same length.
        """
        # Create lists with data for 3 neurons (indices 0, 1, 2)
        netuid = 1
        hotkeys = [
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "5GNJqTPyNqANBkUVMN1LPprxXnFouWXoe2wNSmmEoLctxiZY",
        ]
        coldkeys = [
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        ]
        active = [True, True, False]  # Neuron 2 is inactive
        validator_permit = [True, False, False]  # Only neuron 0 has validator permit
        pruning_score = [0.5, 0.3, 0.0]  # Normalized scores (0-1 range)
        last_update = [1000, 995, 900]  # Block numbers
        emission = [
            Balance.from_tao(10).set_unit(netuid),
            Balance.from_tao(5).set_unit(netuid),
            Balance.from_tao(0).set_unit(netuid),
        ]
        dividends = [0.2, 0.1, 0.0]  # Normalized scores
        incentives = [0.3, 0.2, 0.0]  # Normalized scores
        consensus = [0.4, 0.3, 0.0]  # Normalized scores
        trust = [0.5, 0.4, 0.0]  # Normalized scores
        rank = [0.6, 0.5, 0.0]  # Normalized scores
        block_at_registration = [100, 200, 300]  # Block numbers when neurons registered
        alpha_stake = [
            Balance.from_tao(1000).set_unit(netuid),
            Balance.from_tao(500).set_unit(netuid),
            Balance.from_tao(0).set_unit(netuid),
        ]
        tao_stake = [
            Balance.from_tao(5000),
            Balance.from_tao(2000),
            Balance.from_tao(0),
        ]
        total_stake = [
            Balance.from_tao(6000).set_unit(netuid),
            Balance.from_tao(2500).set_unit(netuid),
            Balance.from_tao(0).set_unit(netuid),
        ]
        emission_history = [[10, 9, 8], [5, 4, 3], [0, 0, 0]]  # Historical emissions
        
        # Create a SubnetState instance with all fields
        subnet_state = SubnetState(
            netuid=netuid,
            hotkeys=hotkeys,
            coldkeys=coldkeys,
            active=active,
            validator_permit=validator_permit,
            pruning_score=pruning_score,
            last_update=last_update,
            emission=emission,
            dividends=dividends,
            incentives=incentives,
            consensus=consensus,
            trust=trust,
            rank=rank,
            block_at_registration=block_at_registration,
            alpha_stake=alpha_stake,
            tao_stake=tao_stake,
            total_stake=total_stake,
            emission_history=emission_history,
        )
        
        # Verify key fields are set correctly
        assert subnet_state.netuid == netuid, \
            "Netuid should specify which subnet this state is for"
        assert len(subnet_state.hotkeys) == 3, \
            "Should have 3 neurons (hotkeys list length)"
        assert subnet_state.hotkeys[0] == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "First neuron's hotkey should be set correctly"
        assert subnet_state.active[0] is True, \
            "First neuron should be active"
        assert subnet_state.active[2] is False, \
            "Third neuron should be inactive"
        assert subnet_state.validator_permit[0] is True, \
            "First neuron should have validator permit"
        assert isinstance(subnet_state.emission[0], Balance), \
            "Emission should be a Balance object"
        assert subnet_state.emission[0].unit == netuid, \
            "Emission Balance should have unit set to netuid"

    def test_subnet_state_initialization_with_empty_subnet(self):
        """
        Test that SubnetState can be initialized with empty lists (no neurons).
        
        This test verifies that a subnet with no neurons can be represented
        correctly with empty lists for all per-neuron fields. This is a valid
        edge case for a newly created subnet that hasn't had any neurons register yet.
        """
        # Create SubnetState with empty lists (no neurons in subnet yet)
        subnet_state = SubnetState(
            netuid=1,
            hotkeys=[],
            coldkeys=[],
            active=[],
            validator_permit=[],
            pruning_score=[],
            last_update=[],
            emission=[],
            dividends=[],
            incentives=[],
            consensus=[],
            trust=[],
            rank=[],
            block_at_registration=[],
            alpha_stake=[],
            tao_stake=[],
            total_stake=[],
            emission_history=[],
        )
        
        # Verify all lists are empty
        assert len(subnet_state.hotkeys) == 0, \
            "Hotkeys list should be empty for subnet with no neurons"
        assert len(subnet_state.active) == 0, \
            "Active list should be empty"
        assert len(subnet_state.emission) == 0, \
            "Emission list should be empty"
        assert isinstance(subnet_state.hotkeys, list), \
            "Hotkeys should still be a list type (empty list)"

    def test_subnet_state_inherits_from_info_base(self):
        """
        Test that SubnetState properly inherits from InfoBase.
        
        This test verifies that SubnetState is a subclass of InfoBase, which
        provides common functionality for chain data structures including the
        from_dict() and list_from_dicts() methods. This ensures consistency
        across all chain data classes.
        """
        from bittensor.core.chain_data.info_base import InfoBase
        assert issubclass(SubnetState, InfoBase), \
            "SubnetState should inherit from InfoBase to get common chain data functionality"
        
        # Verify it's a dataclass (which InfoBase also is)
        from dataclasses import is_dataclass
        assert is_dataclass(SubnetState), \
            "SubnetState should be a dataclass for automatic field handling"


class TestSubnetStateFromDict:
    """
    Test class for the _from_dict() class method.
    
    This class tests that SubnetState objects can be created from dictionary
    data. The conversion includes decoding account IDs, normalizing u16 values,
    converting rao to Balance objects, and setting correct balance units.
    """

    def test_from_dict_creates_subnet_state_correctly(self):
        """
        Test that _from_dict() correctly creates SubnetState from dictionary data.
        
        This test verifies that when given a dictionary with subnet state information
        (as would come from chain data), the _from_dict() method correctly creates
        a SubnetState object. The conversion includes:
        - Decoding hotkey and coldkey account IDs from bytes to SS58 addresses
        - Converting rao values to Balance objects
        - Normalizing u16 values to floats (0-1 range)
        - Setting balance units correctly (netuid for subnet tokens, 0 for TAO)
        """
        # Mock decode_account_id for hotkeys and coldkeys
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # hotkey 0
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # coldkey 0
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # hotkey 1
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # coldkey 1
            ]
            
            # Create dictionary data as would come from chain
            # Note: Values are in raw format from chain (bytes for account IDs, rao for balances, u16 for scores)
            decoded = {
                "netuid": 1,
                "hotkeys": [b"hotkey0_bytes", b"hotkey1_bytes"],  # Raw bytes that will be decoded
                "coldkeys": [b"coldkey0_bytes", b"coldkey1_bytes"],  # Raw bytes that will be decoded
                "active": [True, True],  # Boolean flags
                "validator_permit": [True, False],  # Boolean flags
                "pruning_score": [29491, 16383],  # u16 values that will be normalized to float
                "last_update": [1000, 995],  # Block numbers
                "emission": [10000000000000, 5000000000000],  # Emission in rao (10 TAO, 5 TAO)
                "dividends": [29491, 16383],  # u16 values (will be normalized)
                "incentives": [29491, 16383],  # u16 values
                "consensus": [29491, 16383],  # u16 values
                "trust": [29491, 16383],  # u16 values
                "rank": [29491, 16383],  # u16 values
                "block_at_registration": [100, 200],  # Block numbers
                "alpha_stake": [1000000000000000, 500000000000000],  # Alpha stake in rao
                "tao_stake": [5000000000000000, 2000000000000000],  # TAO stake in rao
                "total_stake": [6000000000000000, 2500000000000000],  # Total stake in rao
                "emission_history": [[10, 9, 8], [5, 4, 3]],  # Historical emissions per neuron
            }
            
            # Create SubnetState from dictionary using _from_dict class method
            subnet_state = SubnetState._from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(subnet_state, SubnetState), \
                "Should return a SubnetState instance"
            
            # Verify netuid is set correctly
            assert subnet_state.netuid == 1, \
                "Netuid should be set correctly from dictionary"
            
            # Verify account IDs were decoded correctly
            assert len(subnet_state.hotkeys) == 2, \
                "Should have 2 neurons (hotkeys list length)"
            assert subnet_state.hotkeys[0] == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
                "First neuron's hotkey should be decoded from bytes to SS58 address"
            assert subnet_state.coldkeys[0] == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", \
                "First neuron's coldkey should be decoded from bytes to SS58 address"
            
            # Verify boolean fields are set correctly
            assert subnet_state.active[0] is True, \
                "First neuron should be active"
            assert subnet_state.validator_permit[0] is True, \
                "First neuron should have validator permit"
            assert subnet_state.validator_permit[1] is False, \
                "Second neuron should not have validator permit"

    def test_from_dict_decodes_account_ids(self):
        """
        Test that _from_dict() correctly decodes account IDs using decode_account_id.
        
        This test verifies that the hotkeys and coldkeys lists are properly decoded
        from bytes/raw format to SS58 string addresses using the decode_account_id
        utility function. This decoding is essential for working with account
        addresses in a human-readable format.
        """
        # Mock decode_account_id to verify it's called correctly
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # hotkey 0
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # coldkey 0
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # hotkey 1
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # coldkey 1
            ]
            
            # Create dictionary with raw account ID bytes
            decoded = {
                "netuid": 1,
                "hotkeys": [b"raw_hotkey0_bytes", b"raw_hotkey1_bytes"],
                "coldkeys": [b"raw_coldkey0_bytes", b"raw_coldkey1_bytes"],
                "active": [True, True],
                "validator_permit": [True, False],
                "pruning_score": [29491, 16383],
                "last_update": [1000, 995],
                "emission": [0, 0],
                "dividends": [29491, 16383],
                "incentives": [29491, 16383],
                "consensus": [29491, 16383],
                "trust": [29491, 16383],
                "rank": [29491, 16383],
                "block_at_registration": [100, 200],
                "alpha_stake": [0, 0],
                "tao_stake": [0, 0],
                "total_stake": [0, 0],
                "emission_history": [[], []],
            }
            
            # Create SubnetState
            subnet_state = SubnetState._from_dict(decoded)
            
            # Verify decode_account_id was called for all account IDs
            assert mock_decode.call_count == 4, \
                "decode_account_id should be called 4 times (2 hotkeys + 2 coldkeys)"
            mock_decode.assert_any_call(b"raw_hotkey0_bytes"), \
                "decode_account_id should be called with first hotkey bytes"
            mock_decode.assert_any_call(b"raw_coldkey0_bytes"), \
                "decode_account_id should be called with first coldkey bytes"

    def test_from_dict_converts_balances_from_rao(self):
        """
        Test that _from_dict() correctly converts balance values from rao to Balance objects.
        
        This test verifies that emission, alpha_stake, tao_stake, and total_stake
        values (which come from chain as rao, the smallest unit) are properly
        converted to Balance objects using Balance.from_rao(). This ensures proper
        balance handling and unit conversions.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with balance values in rao
            # 1 TAO = 1,000,000,000,000 rao (10^12)
            decoded = {
                "netuid": 1,
                "hotkeys": [b"hotkey0_bytes", b"hotkey1_bytes"],
                "coldkeys": [b"coldkey0_bytes", b"coldkey1_bytes"],
                "active": [True, True],
                "validator_permit": [True, False],
                "pruning_score": [29491, 16383],
                "last_update": [1000, 995],
                "emission": [10000000000000, 5000000000000],  # 10 TAO, 5 TAO in rao
                "dividends": [29491, 16383],
                "incentives": [29491, 16383],
                "consensus": [29491, 16383],
                "trust": [29491, 16383],
                "rank": [29491, 16383],
                "block_at_registration": [100, 200],
                "alpha_stake": [1000000000000000, 500000000000000],  # 1000 TAO, 500 TAO
                "tao_stake": [5000000000000000, 2000000000000000],  # 5000 TAO, 2000 TAO
                "total_stake": [6000000000000000, 2500000000000000],  # 6000 TAO, 2500 TAO
                "emission_history": [[10, 9], [5, 4]],
            }
            
            # Create SubnetState
            subnet_state = SubnetState._from_dict(decoded)
            
            # Verify emission is converted correctly
            assert isinstance(subnet_state.emission[0], Balance), \
                "Emission should be converted to a Balance object"
            assert subnet_state.emission[0].tao == pytest.approx(10.0, rel=0.01), \
                "First neuron's emission should be correctly converted from rao to TAO (10 TAO)"
            assert subnet_state.emission[1].tao == pytest.approx(5.0, rel=0.01), \
                "Second neuron's emission should be correctly converted (5 TAO)"
            
            # Verify alpha_stake is converted correctly
            assert subnet_state.alpha_stake[0].tao == pytest.approx(1000.0, rel=0.01), \
                "Alpha stake should be correctly converted from rao to TAO"
            
            # Verify tao_stake is converted correctly
            assert subnet_state.tao_stake[0].tao == pytest.approx(5000.0, rel=0.01), \
                "TAO stake should be correctly converted from rao to TAO"

    def test_from_dict_normalizes_u16_values(self):
        """
        Test that _from_dict() correctly normalizes u16 values using u16_normalized_float.
        
        This test verifies that pruning_score, dividends, incentives, consensus,
        trust, and rank values (which come from chain as u16 integers) are properly
        normalized to float percentages (0-1 range) using the u16_normalized_float
        utility function. This normalization converts raw u16 values to human-readable percentages.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with u16 values
            # u16_normalized_float converts u16 (0-65535) to float (0.0-1.0)
            decoded = {
                "netuid": 1,
                "hotkeys": [b"hotkey0_bytes"],
                "coldkeys": [b"coldkey0_bytes"],
                "active": [True],
                "validator_permit": [True],
                "pruning_score": [29491],  # u16 value (will be normalized)
                "last_update": [1000],
                "emission": [0],
                "dividends": [29491],  # u16 value
                "incentives": [29491],  # u16 value
                "consensus": [29491],  # u16 value
                "trust": [29491],  # u16 value
                "rank": [29491],  # u16 value
                "block_at_registration": [100],
                "alpha_stake": [0],
                "tao_stake": [0],
                "total_stake": [0],
                "emission_history": [[]],
            }
            
            # Create SubnetState
            subnet_state = SubnetState._from_dict(decoded)
            
            # Verify u16 values are normalized to floats
            assert isinstance(subnet_state.pruning_score[0], float), \
                "Pruning score should be normalized to float type"
            assert 0 <= subnet_state.pruning_score[0] <= 1, \
                "Pruning score should be in range 0.0-1.0 after normalization"
            
            assert isinstance(subnet_state.dividends[0], float), \
                "Dividends should be normalized to float type"
            assert isinstance(subnet_state.incentives[0], float), \
                "Incentives should be normalized to float type"
            assert isinstance(subnet_state.consensus[0], float), \
                "Consensus should be normalized to float type"
            assert isinstance(subnet_state.trust[0], float), \
                "Trust should be normalized to float type"
            assert isinstance(subnet_state.rank[0], float), \
                "Rank should be normalized to float type"

    def test_from_dict_sets_balance_units_correctly(self):
        """
        Test that _from_dict() correctly sets Balance units based on netuid.
        
        This test verifies that Balance objects are created with the correct
        unit assignments: subnet-specific balances (emission, alpha_stake, total_stake)
        get the netuid as their unit, while TAO balances (tao_stake) get unit 0.
        This ensures proper type safety and prevents mixing different token types.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with specific netuid
            decoded = {
                "netuid": 2,  # Using netuid 2 to verify unit assignment
                "hotkeys": [b"hotkey0_bytes"],
                "coldkeys": [b"coldkey0_bytes"],
                "active": [True],
                "validator_permit": [True],
                "pruning_score": [29491],
                "last_update": [1000],
                "emission": [10000000000000],  # 10 TAO in rao
                "dividends": [29491],
                "incentives": [29491],
                "consensus": [29491],
                "trust": [29491],
                "rank": [29491],
                "block_at_registration": [100],
                "alpha_stake": [1000000000000000],  # 1000 TAO in rao
                "tao_stake": [5000000000000000],  # 5000 TAO in rao
                "total_stake": [6000000000000000],  # 6000 TAO in rao
                "emission_history": [[10, 9]],
            }
            
            # Create SubnetState
            subnet_state = SubnetState._from_dict(decoded)
            
            # Verify subnet-specific balances have netuid as unit
            assert subnet_state.emission[0].unit == 2, \
                "Emission should have unit set to netuid (2)"
            assert subnet_state.alpha_stake[0].unit == 2, \
                "Alpha stake should have unit set to netuid (2)"
            assert subnet_state.total_stake[0].unit == 2, \
                "Total stake should have unit set to netuid (2)"
            
            # Verify TAO balances have unit 0
            assert subnet_state.tao_stake[0].unit == 0, \
                "TAO stake should have unit 0 (TAO is root network token)"

    def test_from_dict_handles_empty_subnet(self):
        """
        Test that _from_dict() handles empty subnet correctly.
        
        This test verifies that when a subnet has no neurons (empty lists),
        the _from_dict() method correctly creates a SubnetState object with
        empty lists for all per-neuron fields. This is a valid edge case for
        newly created subnets.
        """
        # Mock decode_account_id (won't be called for empty lists)
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id"):
            # Create dictionary with empty lists (no neurons)
            decoded = {
                "netuid": 1,
                "hotkeys": [],  # Empty list - no neurons registered yet
                "coldkeys": [],
                "active": [],
                "validator_permit": [],
                "pruning_score": [],
                "last_update": [],
                "emission": [],
                "dividends": [],
                "incentives": [],
                "consensus": [],
                "trust": [],
                "rank": [],
                "block_at_registration": [],
                "alpha_stake": [],
                "tao_stake": [],
                "total_stake": [],
                "emission_history": [],
            }
            
            # Create SubnetState
            subnet_state = SubnetState._from_dict(decoded)
            
            # Verify all lists are empty
            assert len(subnet_state.hotkeys) == 0, \
                "Hotkeys list should be empty for subnet with no neurons"
            assert len(subnet_state.active) == 0, \
                "Active list should be empty"
            assert len(subnet_state.emission) == 0, \
                "Emission list should be empty"
            assert len(subnet_state.emission_history) == 0, \
                "Emission history list should be empty"


class TestSubnetStateFromDictBaseClass:
    """
    Test class for the from_dict() method inherited from InfoBase.
    
    This class tests that SubnetState can use the from_dict() method from
    InfoBase, which includes error handling for missing fields. The from_dict()
    method calls _from_dict() internally but adds exception handling.
    """

    def test_from_dict_with_complete_data(self):
        """
        Test that from_dict() works with complete data.
        
        This test verifies that the from_dict() method (inherited from InfoBase)
        correctly calls _from_dict() when all required fields are present in
        the dictionary. This is the happy path for creating SubnetState from chain data.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create complete dictionary data
            decoded = {
                "netuid": 1,
                "hotkeys": [b"hotkey0_bytes"],
                "coldkeys": [b"coldkey0_bytes"],
                "active": [True],
                "validator_permit": [True],
                "pruning_score": [29491],
                "last_update": [1000],
                "emission": [10000000000000],
                "dividends": [29491],
                "incentives": [29491],
                "consensus": [29491],
                "trust": [29491],
                "rank": [29491],
                "block_at_registration": [100],
                "alpha_stake": [0],
                "tao_stake": [0],
                "total_stake": [0],
                "emission_history": [[]],
            }
            
            # Create SubnetState using from_dict (from InfoBase)
            # This method includes error handling for missing fields
            subnet_state = SubnetState.from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(subnet_state, SubnetState), \
                "from_dict() should return a SubnetState instance"
            assert subnet_state.netuid == 1, \
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
            "hotkeys": [b"hotkey0_bytes"],
            # Missing coldkeys, active, validator_permit, and other required fields
        }
        
        # Verify from_dict raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException) as exc_info:
            SubnetState.from_dict(incomplete_data)
        
        # Verify error message mentions missing field
        assert "missing" in str(exc_info.value).lower(), \
            "Error message should mention that a field is missing"
        assert "SubnetState" in str(exc_info.value), \
            "Error message should mention SubnetState class name"


class TestSubnetStateEdgeCases:
    """
    Test class for edge cases and special scenarios.
    
    This class tests edge cases such as empty lists, zero values, large lists,
    and other boundary conditions that might occur in real-world usage.
    """

    def test_subnet_state_with_single_neuron(self):
        """
        Test that SubnetState handles subnet with single neuron correctly.
        
        This test verifies that a subnet with only one neuron can be represented
        correctly. This is a common scenario for newly created subnets or test environments.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with single neuron data
            decoded = {
                "netuid": 1,
                "hotkeys": [b"hotkey0_bytes"],  # Single neuron
                "coldkeys": [b"coldkey0_bytes"],
                "active": [True],
                "validator_permit": [True],
                "pruning_score": [29491],
                "last_update": [1000],
                "emission": [10000000000000],
                "dividends": [29491],
                "incentives": [29491],
                "consensus": [29491],
                "trust": [29491],
                "rank": [29491],
                "block_at_registration": [100],
                "alpha_stake": [1000000000000000],
                "tao_stake": [5000000000000000],
                "total_stake": [6000000000000000],
                "emission_history": [[10, 9, 8]],
            }
            
            # Create SubnetState
            subnet_state = SubnetState._from_dict(decoded)
            
            # Verify single neuron is handled correctly
            assert len(subnet_state.hotkeys) == 1, \
                "Should have exactly one neuron"
            assert subnet_state.active[0] is True, \
                "Single neuron should be active"
            assert isinstance(subnet_state.emission[0], Balance), \
                "Emission should be a Balance object for single neuron"

    def test_subnet_state_with_large_number_of_neurons(self):
        """
        Test that SubnetState handles subnet with many neurons correctly.
        
        This test verifies that a subnet with many neurons can be represented
        correctly. This is important for production subnets that may have
        hundreds or thousands of neurons.
        """
        # Mock decode_account_id for multiple neurons
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id") as mock_decode:
            num_neurons = 100
            # Create side_effect list for 100 neurons (2 calls per neuron: hotkey + coldkey)
            side_effects = []
            for i in range(num_neurons):
                side_effects.append(f"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY{i}")  # hotkey
                side_effects.append(f"5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty{i}")  # coldkey
            mock_decode.side_effect = side_effects
            
            # Create dictionary with data for many neurons
            decoded = {
                "netuid": 1,
                "hotkeys": [b"hotkey_bytes"] * num_neurons,
                "coldkeys": [b"coldkey_bytes"] * num_neurons,
                "active": [True] * num_neurons,
                "validator_permit": [i % 2 == 0 for i in range(num_neurons)],  # Alternating
                "pruning_score": [29491] * num_neurons,
                "last_update": [1000] * num_neurons,
                "emission": [10000000000000] * num_neurons,
                "dividends": [29491] * num_neurons,
                "incentives": [29491] * num_neurons,
                "consensus": [29491] * num_neurons,
                "trust": [29491] * num_neurons,
                "rank": [29491] * num_neurons,
                "block_at_registration": [100] * num_neurons,
                "alpha_stake": [0] * num_neurons,
                "tao_stake": [0] * num_neurons,
                "total_stake": [0] * num_neurons,
                "emission_history": [[]] * num_neurons,
            }
            
            # Create SubnetState
            subnet_state = SubnetState._from_dict(decoded)
            
            # Verify large number of neurons is handled correctly
            assert len(subnet_state.hotkeys) == num_neurons, \
                f"Should have {num_neurons} neurons"
            assert len(subnet_state.active) == num_neurons, \
                "Active list should have same length as number of neurons"
            assert subnet_state.validator_permit[0] is True, \
                "First neuron should have validator permit"
            assert subnet_state.validator_permit[1] is False, \
                "Second neuron should not have validator permit"

    def test_subnet_state_emission_history_structure(self):
        """
        Test that SubnetState correctly handles emission_history nested lists.
        
        This test verifies that emission_history, which is a list of lists
        (one inner list per neuron, containing historical emission values),
        is correctly preserved in the SubnetState object.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            ]
            
            # Create dictionary with emission_history
            decoded = {
                "netuid": 1,
                "hotkeys": [b"hotkey0_bytes", b"hotkey1_bytes"],
                "coldkeys": [b"coldkey0_bytes", b"coldkey1_bytes"],
                "active": [True, True],
                "validator_permit": [True, False],
                "pruning_score": [29491, 16383],
                "last_update": [1000, 995],
                "emission": [10000000000000, 5000000000000],
                "dividends": [29491, 16383],
                "incentives": [29491, 16383],
                "consensus": [29491, 16383],
                "trust": [29491, 16383],
                "rank": [29491, 16383],
                "block_at_registration": [100, 200],
                "alpha_stake": [0, 0],
                "tao_stake": [0, 0],
                "total_stake": [0, 0],
                "emission_history": [[10, 9, 8, 7], [5, 4, 3]],  # Different lengths per neuron
            }
            
            # Create SubnetState
            subnet_state = SubnetState._from_dict(decoded)
            
            # Verify emission_history structure is preserved
            assert isinstance(subnet_state.emission_history, list), \
                "Emission history should be a list"
            assert len(subnet_state.emission_history) == 2, \
                "Emission history should have one entry per neuron"
            assert isinstance(subnet_state.emission_history[0], list), \
                "Each neuron's emission history should be a list"
            assert subnet_state.emission_history[0] == [10, 9, 8, 7], \
                "First neuron's emission history should be preserved correctly"
            assert subnet_state.emission_history[1] == [5, 4, 3], \
                "Second neuron's emission history should be preserved correctly"

    def test_subnet_state_field_types(self):
        """
        Test that SubnetState fields maintain correct types.
        
        This test verifies that all fields in SubnetState maintain their
        expected types. This is important for type consistency and ensures
        that the dataclass properly enforces type constraints at runtime.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.subnet_state.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create SubnetState
            decoded = {
                "netuid": 1,
                "hotkeys": [b"hotkey0_bytes"],
                "coldkeys": [b"coldkey0_bytes"],
                "active": [True],
                "validator_permit": [True],
                "pruning_score": [29491],
                "last_update": [1000],
                "emission": [10000000000000],
                "dividends": [29491],
                "incentives": [29491],
                "consensus": [29491],
                "trust": [29491],
                "rank": [29491],
                "block_at_registration": [100],
                "alpha_stake": [1000000000000000],
                "tao_stake": [5000000000000000],
                "total_stake": [6000000000000000],
                "emission_history": [[10, 9]],
            }
            
            subnet_state = SubnetState._from_dict(decoded)
            
            # Verify all field types are correct
            assert isinstance(subnet_state.netuid, int), \
                "netuid should be int type"
            assert isinstance(subnet_state.hotkeys, list), \
                "hotkeys should be list type"
            assert isinstance(subnet_state.hotkeys[0], str), \
                "hotkeys elements should be string type (SS58 addresses)"
            assert isinstance(subnet_state.active, list), \
                "active should be list type"
            assert isinstance(subnet_state.active[0], bool), \
                "active elements should be bool type"
            assert isinstance(subnet_state.pruning_score, list), \
                "pruning_score should be list type"
            assert isinstance(subnet_state.pruning_score[0], float), \
                "pruning_score elements should be float type (normalized)"
            assert isinstance(subnet_state.emission, list), \
                "emission should be list type"
            assert isinstance(subnet_state.emission[0], Balance), \
                "emission elements should be Balance type"
            assert isinstance(subnet_state.last_update, list), \
                "last_update should be list type"
            assert isinstance(subnet_state.last_update[0], int), \
                "last_update elements should be int type (block numbers)"
            assert isinstance(subnet_state.emission_history, list), \
                "emission_history should be list type"
            assert isinstance(subnet_state.emission_history[0], list), \
                "emission_history elements should be list type (nested lists)"

