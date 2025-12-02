"""
Comprehensive unit tests for the bittensor.core.chain_data.subnet_hyperparameters module.

This test suite covers all major components of the SubnetHyperparameters class including:
- Class instantiation and attribute validation
- Dictionary conversion (_from_dict, from_dict from InfoBase)
- Fixed point to float conversion (alpha_sigmoid_steepness)
- Field name mapping (max_weights_limit to max_weight_limit)
- Inheritance from InfoBase
- Edge cases and error handling

The tests are designed to ensure that:
1. SubnetHyperparameters objects can be created correctly with all required fields
2. Dictionary conversion works correctly with chain data format
3. Fixed point values are properly converted to floats
4. Field name mappings are correct
5. Error handling is robust for missing or invalid data
6. All methods handle edge cases properly

SubnetHyperparameters contains many configuration parameters for subnet operation,
including difficulty settings, weight limits, burn values, tempo, and various
feature flags. This is a critical data structure for subnet configuration.

Each test includes extensive comments explaining:
- What functionality is being tested
- Why the test is important
- What assertions verify
- Expected behavior and edge cases
"""

import pytest

# Import the modules to test
from bittensor.core.chain_data.subnet_hyperparameters import SubnetHyperparameters
from bittensor.core.errors import SubstrateRequestException


class TestSubnetHyperparametersInitialization:
    """
    Test class for SubnetHyperparameters object initialization.
    
    This class tests that SubnetHyperparameters objects can be created correctly
    with all required fields. SubnetHyperparameters has many fields (35+ fields)
    that configure various aspects of subnet operation.
    """

    def test_subnet_hyperparameters_initialization_with_all_fields(self):
        """
        Test that SubnetHyperparameters can be initialized with all required fields.
        
        This test verifies that a SubnetHyperparameters object can be created with
        all required fields. SubnetHyperparameters has many configuration parameters
        including difficulty settings, weight limits, burn values, tempo, immunity
        period, and various feature flags. This is a comprehensive test to ensure
        all fields can be set correctly.
        """
        # Create a SubnetHyperparameters with all fields
        # Note: This class has 35+ fields, so we'll set representative values
        subnet_hyperparams = SubnetHyperparameters(
            rho=10000,  # Rate of decay (typically in thousands)
            kappa=1000,  # Constant multiplier
            immunity_period=100,  # Blocks of immunity
            min_allowed_weights=10,  # Minimum weights required
            max_weight_limit=1.0,  # Maximum weight value (float)
            tempo=12,  # Block tempo
            min_difficulty=1000000,  # Minimum difficulty
            max_difficulty=1000000000,  # Maximum difficulty
            weights_version=1,  # Weights version number
            weights_rate_limit=100,  # Rate limit for weights
            adjustment_interval=1000,  # Blocks between adjustments
            activity_cutoff=10000,  # Activity cutoff threshold
            registration_allowed=True,  # Whether registration is allowed
            target_regs_per_interval=10,  # Target registrations per interval
            min_burn=1000000000000,  # Minimum burn in rao
            max_burn=10000000000000,  # Maximum burn in rao
            bonds_moving_avg=1000,  # Moving average period for bonds
            max_regs_per_block=5,  # Maximum registrations per block
            serving_rate_limit=100,  # Serving rate limit
            max_validators=64,  # Maximum number of validators
            adjustment_alpha=100,  # Alpha for adjustments
            difficulty=5000000,  # Current difficulty
            commit_reveal_period=100,  # Period for commit-reveal
            commit_reveal_weights_enabled=True,  # Commit-reveal enabled flag
            alpha_high=1000000,  # High alpha value
            alpha_low=100000,  # Low alpha value
            liquid_alpha_enabled=True,  # Liquid alpha feature flag
            alpha_sigmoid_steepness=5.0,  # Sigmoid steepness (float)
            yuma_version=1,  # Yuma version number
            subnet_is_active=True,  # Subnet active flag
            transfers_enabled=True,  # Transfers enabled flag
            bonds_reset_enabled=False,  # Bonds reset enabled flag
            user_liquidity_enabled=True,  # User liquidity enabled flag
        )
        
        # Verify key fields are set correctly
        assert subnet_hyperparams.rho == 10000, \
            "Rho (rate of decay) should be set correctly"
        assert subnet_hyperparams.kappa == 1000, \
            "Kappa (constant multiplier) should be set correctly"
        assert subnet_hyperparams.tempo == 12, \
            "Tempo should be set correctly"
        assert subnet_hyperparams.max_weight_limit == 1.0, \
            "Max weight limit should be set correctly (float)"
        assert subnet_hyperparams.registration_allowed is True, \
            "Registration allowed flag should be set correctly"
        assert subnet_hyperparams.subnet_is_active is True, \
            "Subnet is active flag should be set correctly"
        assert isinstance(subnet_hyperparams.alpha_sigmoid_steepness, float), \
            "Alpha sigmoid steepness should be float type"

    def test_subnet_hyperparameters_inherits_from_info_base(self):
        """
        Test that SubnetHyperparameters properly inherits from InfoBase.
        
        This test verifies that SubnetHyperparameters is a subclass of InfoBase,
        which provides common functionality for chain data structures. This ensures
        that SubnetHyperparameters can use methods like from_dict() from the base class.
        """
        from bittensor.core.chain_data.info_base import InfoBase
        assert issubclass(SubnetHyperparameters, InfoBase), \
            "SubnetHyperparameters should inherit from InfoBase for common chain data functionality"
        
        from dataclasses import is_dataclass
        assert is_dataclass(SubnetHyperparameters), \
            "SubnetHyperparameters should be a dataclass for automatic field handling"


class TestSubnetHyperparametersFromDict:
    """
    Test class for the _from_dict() class method.
    
    This class tests that SubnetHyperparameters objects can be created from dictionary
    data. The conversion includes fixed point to float conversion and field name mapping.
    """

    def test_from_dict_creates_subnet_hyperparameters_correctly(self):
        """
        Test that _from_dict() correctly creates SubnetHyperparameters from dictionary data.
        
        This test verifies that when given a dictionary with subnet hyperparameter information
        (as would come from chain data), the _from_dict() method correctly creates a
        SubnetHyperparameters object. The conversion includes:
        - Converting fixed point values to floats (alpha_sigmoid_steepness)
        - Field name mapping (max_weights_limit to max_weight_limit)
        """
        # Create dictionary data as would come from chain
        # Note: alpha_sigmoid_steepness comes as fixed point integer (32 frac bits)
        decoded = {
            "rho": 10000,
            "kappa": 1000,
            "immunity_period": 100,
            "min_allowed_weights": 10,
            "max_weights_limit": 1.0,  # Note: field name in dict is "max_weights_limit"
            "tempo": 12,
            "min_difficulty": 1000000,
            "max_difficulty": 1000000000,
            "weights_version": 1,
            "weights_rate_limit": 100,
            "adjustment_interval": 1000,
            "activity_cutoff": 10000,
            "registration_allowed": True,
            "target_regs_per_interval": 10,
            "min_burn": 1000000000000,
            "max_burn": 10000000000000,
            "bonds_moving_avg": 1000,
            "max_regs_per_block": 5,
            "serving_rate_limit": 100,
            "max_validators": 64,
            "adjustment_alpha": 100,
            "difficulty": 5000000,
            "commit_reveal_period": 100,
            "commit_reveal_weights_enabled": True,
            "alpha_high": 1000000,
            "alpha_low": 100000,
            "liquid_alpha_enabled": True,
            "alpha_sigmoid_steepness": 2147483648,  # Fixed point: 5.0 with 32 frac bits
            "yuma_version": 1,
            "subnet_is_active": True,
            "transfers_enabled": True,
            "bonds_reset_enabled": False,
            "user_liquidity_enabled": True,
        }
        
        # Create SubnetHyperparameters from dictionary using _from_dict class method
        subnet_hyperparams = SubnetHyperparameters._from_dict(decoded)
        
        # Verify it was created successfully
        assert isinstance(subnet_hyperparams, SubnetHyperparameters), \
            "Should return a SubnetHyperparameters instance"
        
        # Verify key fields are set correctly
        assert subnet_hyperparams.rho == 10000, \
            "Rho should be set correctly from dictionary"
        assert subnet_hyperparams.kappa == 1000, \
            "Kappa should be set correctly from dictionary"
        assert subnet_hyperparams.tempo == 12, \
            "Tempo should be set correctly from dictionary"
        
        # Verify field name mapping (max_weights_limit -> max_weight_limit)
        assert subnet_hyperparams.max_weight_limit == 1.0, \
            "Max weight limit should be mapped from max_weights_limit in dictionary"
        
        # Verify boolean flags are set correctly
        assert subnet_hyperparams.registration_allowed is True, \
            "Registration allowed flag should be set correctly"
        assert subnet_hyperparams.subnet_is_active is True, \
            "Subnet is active flag should be set correctly"
        assert subnet_hyperparams.bonds_reset_enabled is False, \
            "Bonds reset enabled flag should be set correctly"

    def test_from_dict_converts_fixed_point_to_float(self):
        """
        Test that _from_dict() correctly converts fixed point to float for alpha_sigmoid_steepness.
        
        This test verifies that the alpha_sigmoid_steepness value (which comes from chain
        as a fixed point integer with 32 fractional bits) is properly converted to a float
        using fixed_to_float(). This conversion is necessary for mathematical operations.
        """
        # Create dictionary with fixed point value
        # Fixed point: value = integer_value / (2^frac_bits)
        # Example: 2147483648 (0x80000000) with 32 frac_bits = 5.0
        # fixed_to_float(2147483648, 32) should convert to approximately 5.0
        decoded = {
            "rho": 10000,
            "kappa": 1000,
            "immunity_period": 100,
            "min_allowed_weights": 10,
            "max_weights_limit": 1.0,
            "tempo": 12,
            "min_difficulty": 1000000,
            "max_difficulty": 1000000000,
            "weights_version": 1,
            "weights_rate_limit": 100,
            "adjustment_interval": 1000,
            "activity_cutoff": 10000,
            "registration_allowed": True,
            "target_regs_per_interval": 10,
            "min_burn": 1000000000000,
            "max_burn": 10000000000000,
            "bonds_moving_avg": 1000,
            "max_regs_per_block": 5,
            "serving_rate_limit": 100,
            "max_validators": 64,
            "adjustment_alpha": 100,
            "difficulty": 5000000,
            "commit_reveal_period": 100,
            "commit_reveal_weights_enabled": True,
            "alpha_high": 1000000,
            "alpha_low": 100000,
            "liquid_alpha_enabled": True,
            "alpha_sigmoid_steepness": 2147483648,  # Fixed point representation of 5.0
            "yuma_version": 1,
            "subnet_is_active": True,
            "transfers_enabled": True,
            "bonds_reset_enabled": False,
            "user_liquidity_enabled": True,
        }
        
        # Create SubnetHyperparameters
        subnet_hyperparams = SubnetHyperparameters._from_dict(decoded)
        
        # Verify fixed point conversion
        assert isinstance(subnet_hyperparams.alpha_sigmoid_steepness, float), \
            "Alpha sigmoid steepness should be converted to float type"
        assert subnet_hyperparams.alpha_sigmoid_steepness > 0, \
            "Alpha sigmoid steepness should be a positive float value"

    def test_from_dict_maps_field_name_correctly(self):
        """
        Test that _from_dict() correctly maps max_weights_limit to max_weight_limit.
        
        This test verifies that the field name mapping from chain data format
        (max_weights_limit) to the class attribute (max_weight_limit) works correctly.
        This is important because chain data field names may differ from class attribute names.
        """
        # Create dictionary with max_weights_limit (chain data format)
        decoded = {
            "rho": 10000,
            "kappa": 1000,
            "immunity_period": 100,
            "min_allowed_weights": 10,
            "max_weights_limit": 0.5,  # Field name in dictionary
            "tempo": 12,
            "min_difficulty": 1000000,
            "max_difficulty": 1000000000,
            "weights_version": 1,
            "weights_rate_limit": 100,
            "adjustment_interval": 1000,
            "activity_cutoff": 10000,
            "registration_allowed": True,
            "target_regs_per_interval": 10,
            "min_burn": 1000000000000,
            "max_burn": 10000000000000,
            "bonds_moving_avg": 1000,
            "max_regs_per_block": 5,
            "serving_rate_limit": 100,
            "max_validators": 64,
            "adjustment_alpha": 100,
            "difficulty": 5000000,
            "commit_reveal_period": 100,
            "commit_reveal_weights_enabled": True,
            "alpha_high": 1000000,
            "alpha_low": 100000,
            "liquid_alpha_enabled": True,
            "alpha_sigmoid_steepness": 2147483648,
            "yuma_version": 1,
            "subnet_is_active": True,
            "transfers_enabled": True,
            "bonds_reset_enabled": False,
            "user_liquidity_enabled": True,
        }
        
        # Create SubnetHyperparameters
        subnet_hyperparams = SubnetHyperparameters._from_dict(decoded)
        
        # Verify field name mapping
        assert subnet_hyperparams.max_weight_limit == 0.5, \
            "max_weight_limit should be mapped from max_weights_limit in dictionary"
        assert isinstance(subnet_hyperparams.max_weight_limit, float), \
            "max_weight_limit should be float type"


class TestSubnetHyperparametersFromDictBaseClass:
    """
    Test class for the from_dict() method inherited from InfoBase.
    
    This class tests that SubnetHyperparameters can use the from_dict() method from
    InfoBase, which includes error handling for missing fields. The from_dict()
    method calls _from_dict() internally but adds exception handling.
    """

    def test_from_dict_with_complete_data(self):
        """
        Test that from_dict() works with complete data.
        
        This test verifies that the from_dict() method (inherited from InfoBase)
        correctly calls _from_dict() when all required fields are present in
        the dictionary. This is the happy path for creating SubnetHyperparameters
        from chain data with proper error handling wrapper.
        """
        # Create complete dictionary data
        decoded = {
            "rho": 10000,
            "kappa": 1000,
            "immunity_period": 100,
            "min_allowed_weights": 10,
            "max_weights_limit": 1.0,
            "tempo": 12,
            "min_difficulty": 1000000,
            "max_difficulty": 1000000000,
            "weights_version": 1,
            "weights_rate_limit": 100,
            "adjustment_interval": 1000,
            "activity_cutoff": 10000,
            "registration_allowed": True,
            "target_regs_per_interval": 10,
            "min_burn": 1000000000000,
            "max_burn": 10000000000000,
            "bonds_moving_avg": 1000,
            "max_regs_per_block": 5,
            "serving_rate_limit": 100,
            "max_validators": 64,
            "adjustment_alpha": 100,
            "difficulty": 5000000,
            "commit_reveal_period": 100,
            "commit_reveal_weights_enabled": True,
            "alpha_high": 1000000,
            "alpha_low": 100000,
            "liquid_alpha_enabled": True,
            "alpha_sigmoid_steepness": 2147483648,
            "yuma_version": 1,
            "subnet_is_active": True,
            "transfers_enabled": True,
            "bonds_reset_enabled": False,
            "user_liquidity_enabled": True,
        }
        
        # Create SubnetHyperparameters using from_dict (from InfoBase)
        # This method includes error handling for missing fields
        subnet_hyperparams = SubnetHyperparameters.from_dict(decoded)
        
        # Verify it was created successfully
        assert isinstance(subnet_hyperparams, SubnetHyperparameters), \
            "from_dict() should return a SubnetHyperparameters instance"
        assert subnet_hyperparams.rho == 10000, \
            "Rho should be set correctly from dictionary"

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
            "rho": 10000,
            "kappa": 1000,
            # Missing many required fields
        }
        
        # Verify from_dict raises SubstrateRequestException
        with pytest.raises(SubstrateRequestException) as exc_info:
            SubnetHyperparameters.from_dict(incomplete_data)
        
        # Verify error message mentions missing field
        assert "missing" in str(exc_info.value).lower(), \
            "Error message should mention that a field is missing"
        assert "SubnetHyperparameters" in str(exc_info.value), \
            "Error message should mention SubnetHyperparameters class name"


class TestSubnetHyperparametersEdgeCases:
    """
    Test class for edge cases and special scenarios.
    
    This class tests edge cases such as zero values, maximum values, boolean
    flag combinations, and other boundary conditions.
    """

    def test_subnet_hyperparameters_with_zero_values(self):
        """
        Test that SubnetHyperparameters handles zero values correctly.
        
        This test verifies that zero values for integer fields are handled correctly.
        Some fields like min_difficulty, min_burn might legitimately be zero in
        certain configurations.
        """
        # Create dictionary with zero values for some fields
        decoded = {
            "rho": 0,
            "kappa": 0,
            "immunity_period": 0,
            "min_allowed_weights": 0,
            "max_weights_limit": 0.0,
            "tempo": 0,
            "min_difficulty": 0,
            "max_difficulty": 0,
            "weights_version": 0,
            "weights_rate_limit": 0,
            "adjustment_interval": 0,
            "activity_cutoff": 0,
            "registration_allowed": False,
            "target_regs_per_interval": 0,
            "min_burn": 0,
            "max_burn": 0,
            "bonds_moving_avg": 0,
            "max_regs_per_block": 0,
            "serving_rate_limit": 0,
            "max_validators": 0,
            "adjustment_alpha": 0,
            "difficulty": 0,
            "commit_reveal_period": 0,
            "commit_reveal_weights_enabled": False,
            "alpha_high": 0,
            "alpha_low": 0,
            "liquid_alpha_enabled": False,
            "alpha_sigmoid_steepness": 0,  # Fixed point zero
            "yuma_version": 0,
            "subnet_is_active": False,
            "transfers_enabled": False,
            "bonds_reset_enabled": False,
            "user_liquidity_enabled": False,
        }
        
        # Create SubnetHyperparameters
        subnet_hyperparams = SubnetHyperparameters._from_dict(decoded)
        
        # Verify zero values are handled correctly
        assert subnet_hyperparams.rho == 0, \
            "Zero rho should be handled correctly"
        assert subnet_hyperparams.max_weight_limit == 0.0, \
            "Zero max weight limit should be handled correctly"
        assert subnet_hyperparams.registration_allowed is False, \
            "False registration_allowed should be handled correctly"

    def test_subnet_hyperparameters_field_types(self):
        """
        Test that SubnetHyperparameters fields maintain correct types.
        
        This test verifies that all fields in SubnetHyperparameters maintain their
        expected types. This is important for type consistency and ensures that
        the dataclass properly enforces type constraints at runtime.
        """
        # Create dictionary with various field types
        decoded = {
            "rho": 10000,
            "kappa": 1000,
            "immunity_period": 100,
            "min_allowed_weights": 10,
            "max_weights_limit": 1.0,  # Float type
            "tempo": 12,
            "min_difficulty": 1000000,
            "max_difficulty": 1000000000,
            "weights_version": 1,
            "weights_rate_limit": 100,
            "adjustment_interval": 1000,
            "activity_cutoff": 10000,
            "registration_allowed": True,  # Boolean type
            "target_regs_per_interval": 10,
            "min_burn": 1000000000000,
            "max_burn": 10000000000000,
            "bonds_moving_avg": 1000,
            "max_regs_per_block": 5,
            "serving_rate_limit": 100,
            "max_validators": 64,
            "adjustment_alpha": 100,
            "difficulty": 5000000,
            "commit_reveal_period": 100,
            "commit_reveal_weights_enabled": True,  # Boolean type
            "alpha_high": 1000000,
            "alpha_low": 100000,
            "liquid_alpha_enabled": True,  # Boolean type
            "alpha_sigmoid_steepness": 2147483648,
            "yuma_version": 1,
            "subnet_is_active": True,  # Boolean type
            "transfers_enabled": True,  # Boolean type
            "bonds_reset_enabled": False,  # Boolean type
            "user_liquidity_enabled": True,  # Boolean type
        }
        
        # Create SubnetHyperparameters
        subnet_hyperparams = SubnetHyperparameters._from_dict(decoded)
        
        # Verify field types are correct
        assert isinstance(subnet_hyperparams.rho, int), \
            "rho should be int type"
        assert isinstance(subnet_hyperparams.max_weight_limit, float), \
            "max_weight_limit should be float type"
        assert isinstance(subnet_hyperparams.registration_allowed, bool), \
            "registration_allowed should be bool type"
        assert isinstance(subnet_hyperparams.alpha_sigmoid_steepness, float), \
            "alpha_sigmoid_steepness should be float type (converted from fixed point)"

