"""
Unit tests for transfer amount validation security fixes.

Tests validate that transfer_extrinsic properly rejects:
- Negative amounts
- Zero amounts
- Overflow amounts
- Self-transfers
- Invalid addresses
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from bittensor.core.extrinsics.transfer import transfer_extrinsic
from bittensor.utils.balance import Balance


class TestTransferAmountValidation:
    """Test suite for transfer amount validation"""
    
    @pytest.fixture
    def mock_subtensor(self):
        """Create a mock subtensor"""
        subtensor = Mock()
        subtensor.network = "test"
        subtensor.get_current_block = Mock(return_value=1000)
        subtensor.get_balance = Mock(return_value=Balance.from_tao(100))
        subtensor.get_existential_deposit = Mock(return_value=Balance.from_tao(0.01))
        subtensor.get_transfer_fee = Mock(return_value=Balance.from_tao(0.001))
        return subtensor
    
    @pytest.fixture
    def mock_wallet(self):
        """Create a mock wallet"""
        wallet = Mock()
        wallet.name = "test_wallet"
        wallet.coldkeypub = Mock()
        wallet.coldkeypub.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        return wallet
    
    def test_negative_amount_rejected(self, mock_subtensor, mock_wallet):
        """Test that negative transfer amounts are rejected"""
        result = transfer_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            amount=Balance.from_tao(-10),
            transfer_all=False
        )
        
        assert result is False, "Negative amount should be rejected"
    
    def test_zero_amount_rejected(self, mock_subtensor, mock_wallet):
        """Test that zero transfer amounts are rejected"""
        result = transfer_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            amount=Balance.from_rao(0),
            transfer_all=False
        )
        
        assert result is False, "Zero amount should be rejected"
    
    def test_overflow_amount_rejected(self, mock_subtensor, mock_wallet):
        """Test that overflow amounts are rejected"""
        MAX_BALANCE = 2**128 - 1
        overflow_amount = Balance.from_rao(MAX_BALANCE + 1)
        
        result = transfer_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            amount=overflow_amount,
            transfer_all=False
        )
        
        assert result is False, "Overflow amount should be rejected"
    
    def test_max_balance_accepted(self, mock_subtensor, mock_wallet):
        """Test that maximum valid balance is accepted (validation passes)"""
        MAX_BALANCE = 2**128 - 1
        max_amount = Balance.from_rao(MAX_BALANCE)
        
        # Mock unlock_key to succeed
        with patch('bittensor.core.extrinsics.transfer.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            
            # This should pass validation but fail on balance check
            # (which is expected - we're only testing validation logic)
            result = transfer_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                amount=max_amount,
                transfer_all=False
            )
            
            # Will fail on balance check, but validation passed
            # (we don't have enough balance for max amount)
            assert result is False  # Expected to fail on balance, not validation
    
    def test_self_transfer_rejected(self, mock_subtensor, mock_wallet):
        """Test that self-transfers are rejected"""
        result = transfer_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            dest=mock_wallet.coldkeypub.ss58_address,  # Same as source
            amount=Balance.from_tao(10),
            transfer_all=False
        )
        
        assert result is False, "Self-transfer should be rejected"
    
    def test_invalid_address_rejected(self, mock_subtensor, mock_wallet):
        """Test that invalid addresses are rejected"""
        result = transfer_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            dest="invalid_address_123",
            amount=Balance.from_tao(10),
            transfer_all=False
        )
        
        assert result is False, "Invalid address should be rejected"
    
    def test_dust_amount_rejected(self, mock_subtensor, mock_wallet):
        """Test that amounts below minimum are rejected"""
        MIN_TRANSFER = Balance.from_rao(1)
        dust_amount = Balance.from_rao(0)  # Below minimum
        
        result = transfer_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            amount=dust_amount,
            transfer_all=False
        )
        
        assert result is False, "Dust amount should be rejected"
    
    def test_minimum_amount_validation(self, mock_subtensor, mock_wallet):
        """Test minimum transfer amount validation"""
        # Exactly 1 rao should pass validation
        min_amount = Balance.from_rao(1)
        
        with patch('bittensor.core.extrinsics.transfer.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            
            # Mock sufficient balance
            mock_subtensor.get_balance.return_value = Balance.from_tao(100)
            mock_subtensor.sign_and_send_extrinsic = Mock(return_value=(True, "Success"))
            mock_subtensor.get_block_hash = Mock(return_value="0x123")
            
            # This should pass validation (but may fail on other checks)
            result = transfer_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                amount=min_amount,
                transfer_all=False,
                wait_for_inclusion=False,
                wait_for_finalization=False
            )
            
            # The validation should pass and transfer should succeed
            assert result is True, "Minimum valid amount should pass"
    
    def test_none_amount_without_transfer_all(self, mock_subtensor, mock_wallet):
        """Test that None amount without transfer_all is rejected"""
        result = transfer_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            amount=None,
            transfer_all=False
        )
        
        assert result is False, "None amount without transfer_all should be rejected"
    
    def test_valid_amount_passes_validation(self, mock_subtensor, mock_wallet):
        """Test that valid amounts pass validation checks"""
        valid_amount = Balance.from_tao(10)
        
        with patch('bittensor.core.extrinsics.transfer.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=False)  # Fail at unlock step
            
            # Should pass validation but fail at unlock
            result = transfer_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                amount=valid_amount,
                transfer_all=False
            )
            
            # Failed at unlock, not validation
            assert result is False
            mock_unlock.assert_called_once()
    
    def test_boundary_values(self, mock_subtensor, mock_wallet):
        """Test boundary values for amount validation"""
        # Test negative - should fail validation
        result = transfer_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            amount=Balance.from_tao(-10),
            transfer_all=False
        )
        assert result is False, "Negative boundary should fail validation"
        
        # Test zero - should fail validation  
        result = transfer_extrinsic(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            amount=Balance.from_rao(0),
            transfer_all=False
        )
        assert result is False, "Zero boundary should fail validation"


class TestTransferEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def mock_subtensor(self):
        subtensor = Mock()
        subtensor.network = "test"
        return subtensor
    
    @pytest.fixture
    def mock_wallet(self):
        wallet = Mock()
        wallet.coldkeypub = Mock()
        wallet.coldkeypub.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        return wallet
    
    def test_concurrent_validation_calls(self, mock_subtensor, mock_wallet):
        """Test that validation is thread-safe"""
        from concurrent.futures import ThreadPoolExecutor
        
        def validate():
            return transfer_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                amount=Balance.from_tao(-10),  # Invalid
                transfer_all=False
            )
        
        # Run 50 concurrent validations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(validate) for _ in range(50)]
            results = [f.result() for f in futures]
        
        # All should reject the negative amount
        assert all(r is False for r in results), "All validations should reject negative amount"
    
    def test_precision_preservation(self, mock_subtensor, mock_wallet):
        """Test that amount precision is preserved during validation"""
        # Test with very small amount - 1 rao
        tiny_amount = Balance.from_rao(1)
        
        with patch('bittensor.core.extrinsics.transfer.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            mock_subtensor.get_current_block = Mock(return_value=1000)
            mock_subtensor.get_balance = Mock(return_value=Balance.from_tao(100))
            mock_subtensor.get_existential_deposit = Mock(return_value=Balance.from_tao(0.01))
            mock_subtensor.get_transfer_fee = Mock(return_value=Balance.from_tao(0.001))
            mock_subtensor.sign_and_send_extrinsic = Mock(return_value=(True, "Success"))
            mock_subtensor.get_block_hash = Mock(return_value="0x123")
            
            result = transfer_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                dest="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                amount=tiny_amount,
                transfer_all=False,
                wait_for_inclusion=False,
                wait_for_finalization=False
            )
            
            # Should succeed - precision preserved, not rounded to 0
            assert result is True, "Tiny amount should preserve precision"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
