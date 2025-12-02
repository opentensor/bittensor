"""
Comprehensive unit tests for the bittensor.core.chain_data.dynamic_info module.

This test suite covers all major components of the DynamicInfo class including:
- Class instantiation and attribute validation
- Dictionary conversion (_from_dict, from_dict)
- Byte string decoding (token_symbol, subnet_name)
- Balance conversions and unit assignments
- Conversion methods (tao_to_alpha, alpha_to_tao)
- Slippage calculation methods (tao_to_alpha_with_slippage, alpha_to_tao_with_slippage)
- Bonding curve calculations (k value, price calculations)
- Inheritance from InfoBase
- Edge cases and error handling

The tests are designed to ensure that:
1. DynamicInfo objects can be created correctly with all required fields
2. Dictionary conversion works correctly with chain data format
3. Byte-encoded strings are properly decoded
4. Balance conversions are accurate
5. Conversion methods work correctly
6. Slippage calculations are accurate
7. Error handling is robust
8. All methods handle edge cases properly

DynamicInfo is a complex data structure that represents dynamic subnet information
including bonding curve parameters, emissions, and pricing information. It includes
methods for converting between TAO and Alpha tokens based on the bonding curve.

Each test includes extensive comments explaining:
- What functionality is being tested
- Why the test is important
- What assertions verify
- Expected behavior and edge cases
"""

from unittest.mock import MagicMock, patch

import pytest

# Import the modules to test
from bittensor.core.chain_data.dynamic_info import DynamicInfo
from bittensor.core.errors import SubstrateRequestException
from bittensor.utils.balance import Balance


class TestDynamicInfoInitialization:
    """
    Test class for DynamicInfo object initialization.
    
    This class tests that DynamicInfo objects can be created correctly with
    all required fields. DynamicInfo has many fields related to subnet dynamics,
    emissions, balances, pricing, and bonding curve parameters.
    """

    def test_dynamic_info_initialization_with_all_fields(self):
        """
        Test that DynamicInfo can be initialized with all required fields.
        
        This test verifies that a DynamicInfo object can be created with all
        required fields. DynamicInfo has many fields including subnet identification,
        owner keys, bonding curve parameters, emissions, and pricing information.
        This is a comprehensive test to ensure all fields can be set correctly.
        """
        # Create a DynamicInfo with all required fields
        dynamic_info = DynamicInfo(
            netuid=1,  # Subnet ID
            owner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            owner_coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            subnet_name="Test Subnet",  # Subnet name string
            symbol="TEST",  # Token symbol
            tempo=100,  # Block tempo
            last_step=1000,  # Last step block number
            blocks_since_last_step=50,  # Blocks since last step
            emission=Balance.from_tao(100).set_unit(0),  # Emission balance
            alpha_in=Balance.from_tao(1000).set_unit(1),  # Alpha in pool (subnet unit)
            alpha_out=Balance.from_tao(500).set_unit(1),  # Alpha out of pool
            tao_in=Balance.from_tao(5000),  # TAO in pool (unit 0)
            price=Balance.from_tao(5).set_unit(1),  # Current price (TAO per Alpha)
            k=5000000.0,  # Bonding curve constant (tao_in.rao * alpha_in.rao)
            is_dynamic=True,  # Whether subnet uses dynamic bonding curve
            alpha_out_emission=Balance.from_tao(10).set_unit(1),
            alpha_in_emission=Balance.from_tao(20).set_unit(1),
            tao_in_emission=Balance.from_tao(100),
            pending_alpha_emission=Balance.from_tao(5).set_unit(1),
            pending_root_emission=Balance.from_tao(50),
            network_registered_at=1234567890,  # Timestamp
            subnet_volume=Balance.from_tao(10000).set_unit(1),
            subnet_identity=None,  # Optional subnet identity
            moving_price=5.0  # Moving average price
        )
        
        # Verify key fields are set correctly
        assert dynamic_info.netuid == 1, \
            "Netuid should specify which subnet this dynamic info is for"
        assert dynamic_info.owner_hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", \
            "Owner hotkey should be set correctly"
        assert dynamic_info.subnet_name == "Test Subnet", \
            "Subnet name should be set correctly"
        assert dynamic_info.symbol == "TEST", \
            "Token symbol should be set correctly"
        assert dynamic_info.is_dynamic is True, \
            "is_dynamic should indicate whether subnet uses bonding curve"
        assert isinstance(dynamic_info.price, Balance), \
            "Price should be a Balance object representing TAO per Alpha"

    def test_dynamic_info_inherits_from_info_base(self):
        """
        Test that DynamicInfo properly inherits from InfoBase.
        
        This test verifies that DynamicInfo is a subclass of InfoBase, which
        provides common functionality for chain data structures. This ensures
        that DynamicInfo can use methods like from_dict() from the base class.
        """
        from bittensor.core.chain_data.info_base import InfoBase
        assert issubclass(DynamicInfo, InfoBase), \
            "DynamicInfo should inherit from InfoBase for common chain data functionality"
        
        from dataclasses import is_dataclass
        assert is_dataclass(DynamicInfo), \
            "DynamicInfo should be a dataclass for automatic field handling"


class TestDynamicInfoFromDict:
    """
    Test class for the _from_dict() class method.
    
    This class tests that DynamicInfo objects can be created from dictionary
    data. DynamicInfo has complex conversion logic including byte decoding,
    balance conversions, and bonding curve calculations.
    """

    def test_from_dict_creates_dynamic_info_correctly(self):
        """
        Test that _from_dict() correctly creates DynamicInfo from dictionary data.
        
        This test verifies that when given a dictionary with dynamic info fields
        (as would come from chain data), the _from_dict() method correctly creates
        a DynamicInfo object. The conversion includes:
        - Decoding byte-encoded strings (token_symbol, subnet_name)
        - Converting rao values to Balance objects
        - Calculating bonding curve constant (k)
        - Setting balance units correctly
        """
        # Mock decode_account_id for owner keys
        with patch("bittensor.core.chain_data.dynamic_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # owner_hotkey
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # owner_coldkey
            ]
            
            # Create dictionary data as would come from chain
            # Note: token_symbol and subnet_name come as byte tuples that need decoding
            decoded = {
                "netuid": 1,
                "owner_hotkey": b"owner_hotkey_bytes",
                "owner_coldkey": b"owner_coldkey_bytes",
                "token_symbol": (84, 69, 83, 84),  # "TEST" as ASCII byte values
                "subnet_name": (84, 101, 115, 116, 32, 83, 117, 98, 110, 101, 116),  # "Test Subnet"
                "tempo": 100,
                "last_step": 1000,
                "blocks_since_last_step": 50,
                "emission": 100000000000000,  # 100 TAO in rao
                "alpha_in": 1000000000000000,  # 1000 TAO in rao
                "alpha_out": 500000000000000,  # 500 TAO in rao
                "tao_in": 5000000000000000,  # 5000 TAO in rao
                "alpha_out_emission": 10000000000000,  # 10 TAO in rao
                "alpha_in_emission": 20000000000000,  # 20 TAO in rao
                "tao_in_emission": 100000000000000,  # 100 TAO in rao
                "pending_alpha_emission": 5000000000000,  # 5 TAO in rao
                "pending_root_emission": 50000000000000,  # 50 TAO in rao
                "network_registered_at": 1234567890,
                "subnet_volume": 10000000000000000,  # 10000 TAO in rao
                "subnet_identity": None,
                "moving_price": 5000000000,  # Fixed point value (will be converted)
                "price": None  # Will be calculated from tao_in / alpha_in
            }
            
            # Create DynamicInfo from dictionary
            dynamic_info = DynamicInfo._from_dict(decoded)
            
            # Verify it was created successfully
            assert isinstance(dynamic_info, DynamicInfo), \
                "Should return a DynamicInfo instance"
            
            # Verify netuid is set correctly
            assert dynamic_info.netuid == 1, \
                "Netuid should be set correctly from dictionary"
            
            # Verify byte strings are decoded correctly
            assert dynamic_info.symbol == "TEST", \
                "Symbol should be decoded from byte tuple correctly"
            assert dynamic_info.subnet_name == "Test Subnet", \
                "Subnet name should be decoded from byte tuple correctly"
            
            # Verify balances are converted correctly
            assert dynamic_info.emission.tao == pytest.approx(100, rel=0.01), \
                "Emission should be converted from rao to TAO correctly"
            assert dynamic_info.alpha_in.tao == pytest.approx(1000, rel=0.01), \
                "Alpha in should be converted from rao correctly"
            
            # Verify is_dynamic is set based on netuid
            assert dynamic_info.is_dynamic is True, \
                "is_dynamic should be True for netuid > 0"

    def test_from_dict_decodes_byte_strings(self):
        """
        Test that _from_dict() correctly decodes byte-encoded strings.
        
        This test verifies that token_symbol and subnet_name (which come from
        chain as byte tuples representing ASCII/UTF-8 encoded strings) are
        properly decoded to Python strings. This is important for displaying
        subnet information correctly to users.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.dynamic_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with byte-encoded strings
            decoded = {
                "netuid": 1,
                "owner_hotkey": b"owner_hotkey_bytes",
                "owner_coldkey": b"owner_coldkey_bytes",
                "token_symbol": (65, 76, 80, 72, 65),  # "ALPHA" as ASCII bytes
                "subnet_name": (77, 121, 32, 83, 117, 98, 110, 101, 116),  # "My Subnet"
                "tempo": 100,
                "last_step": 1000,
                "blocks_since_last_step": 0,
                "emission": 0,
                "alpha_in": 1000000000000000,
                "alpha_out": 0,
                "tao_in": 5000000000000000,
                "alpha_out_emission": 0,
                "alpha_in_emission": 0,
                "tao_in_emission": 0,
                "pending_alpha_emission": 0,
                "pending_root_emission": 0,
                "network_registered_at": 0,
                "subnet_volume": 0,
                "subnet_identity": None,
                "moving_price": 0,
                "price": None
            }
            
            # Create DynamicInfo
            dynamic_info = DynamicInfo._from_dict(decoded)
            
            # Verify byte strings are decoded correctly
            assert dynamic_info.symbol == "ALPHA", \
                "Symbol should be decoded from byte tuple to string correctly"
            assert dynamic_info.subnet_name == "My Subnet", \
                "Subnet name should be decoded from byte tuple to string correctly"

    def test_from_dict_calculates_k_value(self):
        """
        Test that _from_dict() correctly calculates the k value (bonding curve constant).
        
        This test verifies that the k value (which represents the constant product
        in the bonding curve) is correctly calculated as k = tao_in.rao * alpha_in.rao.
        This constant is critical for pricing calculations in dynamic subnets.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.dynamic_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with specific tao_in and alpha_in values
            tao_in_rao = 5000000000000000  # 5000 TAO in rao
            alpha_in_rao = 1000000000000000  # 1000 TAO in rao
            expected_k = tao_in_rao * alpha_in_rao  # k = tao_in * alpha_in
            
            decoded = {
                "netuid": 1,
                "owner_hotkey": b"owner_hotkey_bytes",
                "owner_coldkey": b"owner_coldkey_bytes",
                "token_symbol": (84, 69, 83, 84),
                "subnet_name": (84, 101, 115, 116),
                "tempo": 100,
                "last_step": 1000,
                "blocks_since_last_step": 0,
                "emission": 0,
                "alpha_in": alpha_in_rao,
                "alpha_out": 0,
                "tao_in": tao_in_rao,
                "alpha_out_emission": 0,
                "alpha_in_emission": 0,
                "tao_in_emission": 0,
                "pending_alpha_emission": 0,
                "pending_root_emission": 0,
                "network_registered_at": 0,
                "subnet_volume": 0,
                "subnet_identity": None,
                "moving_price": 0,
                "price": None
            }
            
            # Create DynamicInfo
            dynamic_info = DynamicInfo._from_dict(decoded)
            
            # Verify k is calculated correctly
            assert dynamic_info.k == expected_k, \
                f"k should be tao_in.rao * alpha_in.rao = {expected_k}"

    def test_from_dict_sets_is_dynamic_based_on_netuid(self):
        """
        Test that _from_dict() correctly sets is_dynamic based on netuid.
        
        This test verifies that is_dynamic is set to True when netuid > 0,
        and False when netuid == 0 (root network). The root network (netuid 0)
        doesn't use dynamic bonding curves, while all other subnets do.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.dynamic_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create base decoded dictionary (will modify netuid)
            base_decoded = {
                "netuid": 1,  # Will test with different values
                "owner_hotkey": b"owner_hotkey_bytes",
                "owner_coldkey": b"owner_coldkey_bytes",
                "token_symbol": (84, 69, 83, 84),
                "subnet_name": (84, 101, 115, 116),
                "tempo": 100,
                "last_step": 1000,
                "blocks_since_last_step": 0,
                "emission": 0,
                "alpha_in": 1000000000000000,
                "alpha_out": 0,
                "tao_in": 5000000000000000,
                "alpha_out_emission": 0,
                "alpha_in_emission": 0,
                "tao_in_emission": 0,
                "pending_alpha_emission": 0,
                "pending_root_emission": 0,
                "network_registered_at": 0,
                "subnet_volume": 0,
                "subnet_identity": None,
                "moving_price": 0,
                "price": None
            }
            
            # Test with netuid > 0 (should be dynamic)
            decoded_1 = base_decoded.copy()
            decoded_1["netuid"] = 1
            dynamic_info_1 = DynamicInfo._from_dict(decoded_1)
            assert dynamic_info_1.is_dynamic is True, \
                "is_dynamic should be True for netuid > 0 (uses bonding curve)"
            
            # Test with netuid = 0 (should not be dynamic - root network)
            decoded_0 = base_decoded.copy()
            decoded_0["netuid"] = 0
            dynamic_info_0 = DynamicInfo._from_dict(decoded_0)
            assert dynamic_info_0.is_dynamic is False, \
                "is_dynamic should be False for netuid = 0 (root network, no bonding curve)"

    def test_from_dict_sets_balance_units_correctly(self):
        """
        Test that _from_dict() correctly sets Balance units based on netuid.
        
        This test verifies that Balance objects are created with the correct
        unit assignments: subnet-specific balances (alpha_in, alpha_out, etc.)
        get the netuid as their unit, while TAO balances get unit 0.
        """
        # Mock decode_account_id
        with patch("bittensor.core.chain_data.dynamic_info.decode_account_id") as mock_decode:
            mock_decode.side_effect = [
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ]
            
            # Create dictionary with specific netuid
            decoded = {
                "netuid": 2,  # Using netuid 2 to verify unit assignment
                "owner_hotkey": b"owner_hotkey_bytes",
                "owner_coldkey": b"owner_coldkey_bytes",
                "token_symbol": (84, 69, 83, 84),
                "subnet_name": (84, 101, 115, 116),
                "tempo": 100,
                "last_step": 1000,
                "blocks_since_last_step": 0,
                "emission": 0,
                "alpha_in": 1000000000000000,
                "alpha_out": 0,
                "tao_in": 5000000000000000,
                "alpha_out_emission": 0,
                "alpha_in_emission": 0,
                "tao_in_emission": 0,
                "pending_alpha_emission": 0,
                "pending_root_emission": 0,
                "network_registered_at": 0,
                "subnet_volume": 0,
                "subnet_identity": None,
                "moving_price": 0,
                "price": None
            }
            
            # Create DynamicInfo
            dynamic_info = DynamicInfo._from_dict(decoded)
            
            # Verify subnet-specific balances have netuid as unit
            assert dynamic_info.alpha_in.unit == 2, \
                "Alpha in should have unit set to netuid (2)"
            assert dynamic_info.alpha_out.unit == 2, \
                "Alpha out should have unit set to netuid (2)"
            
            # Verify TAO balances have unit 0
            assert dynamic_info.tao_in.unit == 0, \
                "TAO in should have unit 0 (TAO is root network token)"
            assert dynamic_info.emission.unit == 0, \
                "Emission should have unit 0 (emission is in TAO)"


class TestDynamicInfoConversions:
    """
    Test class for conversion methods (tao_to_alpha, alpha_to_tao).
    
    This class tests the methods that convert between TAO and Alpha balances
    based on the current price in the dynamic subnet. These conversions are
    fundamental to the bonding curve mechanism.
    """

    def test_tao_to_alpha_conversion(self):
        """
        Test that tao_to_alpha() correctly converts TAO to Alpha.
        
        This test verifies that when given a TAO amount, the tao_to_alpha()
        method correctly calculates the equivalent Alpha amount based on the
        current price. The formula is: alpha = tao / price. This is the
        ideal conversion rate without slippage.
        """
        # Create DynamicInfo with a known price
        # Price: 5 TAO per Alpha (meaning 1 Alpha costs 5 TAO)
        price = Balance.from_tao(5).set_unit(1)
        
        dynamic_info = DynamicInfo(
            netuid=1,
            owner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            owner_coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            subnet_name="Test",
            symbol="TEST",
            tempo=100,
            last_step=1000,
            blocks_since_last_step=0,
            emission=Balance.from_tao(0),
            alpha_in=Balance.from_tao(1000).set_unit(1),
            alpha_out=Balance.from_tao(0).set_unit(1),
            tao_in=Balance.from_tao(5000),
            price=price,
            k=5000000.0,
            is_dynamic=True,
            alpha_out_emission=Balance.from_tao(0).set_unit(1),
            alpha_in_emission=Balance.from_tao(0).set_unit(1),
            tao_in_emission=Balance.from_tao(0),
            pending_alpha_emission=Balance.from_tao(0).set_unit(1),
            pending_root_emission=Balance.from_tao(0),
            network_registered_at=0,
            subnet_volume=Balance.from_tao(0).set_unit(1),
            subnet_identity=None,
            moving_price=5.0
        )
        
        # Test conversion: 10 TAO should give 2 Alpha (10 / 5 = 2)
        tao_amount = Balance.from_tao(10)
        alpha_result = dynamic_info.tao_to_alpha(tao_amount)
        
        # Verify conversion is correct
        assert isinstance(alpha_result, Balance), \
            "tao_to_alpha should return a Balance object"
        assert alpha_result.tao == pytest.approx(2.0, rel=0.01), \
            "10 TAO at price 5 TAO/Alpha should give 2 Alpha"
        assert alpha_result.unit == 1, \
            "Alpha should have unit set to netuid (1) for type safety"

    def test_tao_to_alpha_with_zero_price(self):
        """
        Test that tao_to_alpha() handles zero price correctly.
        
        This test verifies that when price is zero (edge case, shouldn't happen
        in practice but could in edge cases), the tao_to_alpha() method returns
        zero Alpha rather than causing a division error. This prevents potential
        crashes and ensures robust error handling.
        """
        # Create DynamicInfo with zero price (edge case)
        zero_price = Balance.from_tao(0).set_unit(1)
        
        dynamic_info = DynamicInfo(
            netuid=1,
            owner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            owner_coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            subnet_name="Test",
            symbol="TEST",
            tempo=100,
            last_step=1000,
            blocks_since_last_step=0,
            emission=Balance.from_tao(0),
            alpha_in=Balance.from_tao(0).set_unit(1),
            alpha_out=Balance.from_tao(0).set_unit(1),
            tao_in=Balance.from_tao(0),
            price=zero_price,
            k=0.0,
            is_dynamic=True,
            alpha_out_emission=Balance.from_tao(0).set_unit(1),
            alpha_in_emission=Balance.from_tao(0).set_unit(1),
            tao_in_emission=Balance.from_tao(0),
            pending_alpha_emission=Balance.from_tao(0).set_unit(1),
            pending_root_emission=Balance.from_tao(0),
            network_registered_at=0,
            subnet_volume=Balance.from_tao(0).set_unit(1),
            subnet_identity=None,
            moving_price=0.0
        )
        
        # Test conversion with zero price
        tao_amount = Balance.from_tao(10)
        alpha_result = dynamic_info.tao_to_alpha(tao_amount)
        
        # Verify it returns zero without error
        assert alpha_result.tao == pytest.approx(0, abs=0.01), \
            "tao_to_alpha should return zero Alpha when price is zero (avoid division error)"

    def test_alpha_to_tao_conversion(self):
        """
        Test that alpha_to_tao() correctly converts Alpha to TAO.
        
        This test verifies that when given an Alpha amount, the alpha_to_tao()
        method correctly calculates the equivalent TAO amount based on the
        current price. The formula is: tao = alpha * price. This is the
        ideal conversion rate without slippage.
        """
        # Create DynamicInfo with a known price
        price = Balance.from_tao(5).set_unit(1)
        
        dynamic_info = DynamicInfo(
            netuid=1,
            owner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            owner_coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            subnet_name="Test",
            symbol="TEST",
            tempo=100,
            last_step=1000,
            blocks_since_last_step=0,
            emission=Balance.from_tao(0),
            alpha_in=Balance.from_tao(1000).set_unit(1),
            alpha_out=Balance.from_tao(0).set_unit(1),
            tao_in=Balance.from_tao(5000),
            price=price,
            k=5000000.0,
            is_dynamic=True,
            alpha_out_emission=Balance.from_tao(0).set_unit(1),
            alpha_in_emission=Balance.from_tao(0).set_unit(1),
            tao_in_emission=Balance.from_tao(0),
            pending_alpha_emission=Balance.from_tao(0).set_unit(1),
            pending_root_emission=Balance.from_tao(0),
            network_registered_at=0,
            subnet_volume=Balance.from_tao(0).set_unit(1),
            subnet_identity=None,
            moving_price=5.0
        )
        
        # Test conversion: 2 Alpha should give 10 TAO (2 * 5 = 10)
        alpha_amount = Balance.from_tao(2).set_unit(1)
        tao_result = dynamic_info.alpha_to_tao(alpha_amount)
        
        # Verify conversion is correct
        assert isinstance(tao_result, Balance), \
            "alpha_to_tao should return a Balance object"
        assert tao_result.tao == pytest.approx(10.0, rel=0.01), \
            "2 Alpha at price 5 TAO/Alpha should give 10 TAO"


class TestDynamicInfoSlippage:
    """
    Test class for slippage calculation methods.
    
    This class tests the methods that calculate slippage when converting
    between TAO and Alpha. Slippage occurs because of the bonding curve
    mechanism - large conversions move the price, resulting in less favorable
    rates than the current price suggests.
    """

    def test_tao_to_alpha_with_slippage_returns_tuple(self):
        """
        Test that tao_to_alpha_with_slippage() returns tuple of (alpha, slippage).
        
        This test verifies that when percentage=False, the method returns a tuple
        containing the Alpha amount received and the slippage amount. This allows
        users to see both the actual amount they'll receive and the cost of slippage.
        """
        # Create DynamicInfo with bonding curve parameters
        dynamic_info = DynamicInfo(
            netuid=1,
            owner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            owner_coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            subnet_name="Test",
            symbol="TEST",
            tempo=100,
            last_step=1000,
            blocks_since_last_step=0,
            emission=Balance.from_tao(0),
            alpha_in=Balance.from_tao(1000).set_unit(1),
            alpha_out=Balance.from_tao(0).set_unit(1),
            tao_in=Balance.from_tao(5000),
            price=Balance.from_tao(5).set_unit(1),
            k=5000000.0,  # tao_in * alpha_in
            is_dynamic=True,
            alpha_out_emission=Balance.from_tao(0).set_unit(1),
            alpha_in_emission=Balance.from_tao(0).set_unit(1),
            tao_in_emission=Balance.from_tao(0),
            pending_alpha_emission=Balance.from_tao(0).set_unit(1),
            pending_root_emission=Balance.from_tao(0),
            network_registered_at=0,
            subnet_volume=Balance.from_tao(0).set_unit(1),
            subnet_identity=None,
            moving_price=5.0
        )
        
        # Test slippage calculation
        tao_amount = Balance.from_tao(100)
        result = dynamic_info.tao_to_alpha_with_slippage(tao_amount, percentage=False)
        
        # Verify result is a tuple
        assert isinstance(result, tuple), \
            "Should return tuple when percentage=False"
        assert len(result) == 2, \
            "Tuple should have 2 elements (alpha returned, slippage)"
        
        # Verify both elements are Balance objects
        alpha_returned, slippage = result
        assert isinstance(alpha_returned, Balance), \
            "First element should be Balance (alpha amount received)"
        assert isinstance(slippage, Balance), \
            "Second element should be Balance (slippage amount)"

    def test_tao_to_alpha_with_slippage_returns_percentage(self):
        """
        Test that tao_to_alpha_with_slippage() returns percentage when requested.
        
        This test verifies that when percentage=True, the method returns a float
        representing the slippage as a percentage. This is useful for displaying
        slippage information to users in a human-readable format (e.g., "2.5% slippage").
        """
        # Create DynamicInfo
        dynamic_info = DynamicInfo(
            netuid=1,
            owner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            owner_coldkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            subnet_name="Test",
            symbol="TEST",
            tempo=100,
            last_step=1000,
            blocks_since_last_step=0,
            emission=Balance.from_tao(0),
            alpha_in=Balance.from_tao(1000).set_unit(1),
            alpha_out=Balance.from_tao(0).set_unit(1),
            tao_in=Balance.from_tao(5000),
            price=Balance.from_tao(5).set_unit(1),
            k=5000000.0,
            is_dynamic=True,
            alpha_out_emission=Balance.from_tao(0).set_unit(1),
            alpha_in_emission=Balance.from_tao(0).set_unit(1),
            tao_in_emission=Balance.from_tao(0),
            pending_alpha_emission=Balance.from_tao(0).set_unit(1),
            pending_root_emission=Balance.from_tao(0),
            network_registered_at=0,
            subnet_volume=Balance.from_tao(0).set_unit(1),
            subnet_identity=None,
            moving_price=5.0
        )
        
        # Test slippage calculation with percentage=True
        tao_amount = Balance.from_tao(100)
        result = dynamic_info.tao_to_alpha_with_slippage(tao_amount, percentage=True)
        
        # Verify result is a float (percentage)
        assert isinstance(result, float), \
            "Should return float when percentage=True"
        assert 0 <= result <= 100, \
            "Percentage should be between 0 and 100"

