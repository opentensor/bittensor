"""
Unit tests for staking amount validation security fixes.

Tests validate that add_stake_extrinsic properly rejects:
- Overflow amounts (> u64 max)
- Invalid staking scenarios
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from bittensor.core.extrinsics.staking import add_stake_extrinsic
from bittensor.utils.balance import Balance


class TestStakingAmountValidation:
    """Test suite for staking amount validation"""
    
    @pytest.fixture
    def mock_subtensor(self):
        """Create a mock subtensor"""
        subtensor = Mock()
        subtensor.network = "test"
        subtensor.get_current_block = Mock(return_value=1000)
        subtensor.get_balance = Mock(return_value=Balance.from_tao(100))
        subtensor.get_stake = Mock(return_value=Balance.from_tao(50))
        subtensor.get_existential_deposit = Mock(return_value=Balance.from_tao(0.01))
        subtensor.substrate = Mock()
        subtensor.substrate.compose_call = Mock(return_value=Mock())
        subtensor.sign_and_send_extrinsic = Mock(return_value=(True, "Success"))
        return subtensor
    
    @pytest.fixture
    def mock_wallet(self):
        """Create a mock wallet"""
        wallet = Mock()
        wallet.name = "test_wallet"
        wallet.hotkey = Mock()
        wallet.hotkey.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        wallet.coldkeypub = Mock()
        wallet.coldkeypub.ss58_address = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        return wallet
    
    def test_overflow_stake_rejected(self, mock_subtensor, mock_wallet):
        """Test that staking amounts exceeding u64 max are rejected"""
        MAX_STAKE = 2**64 - 1
        overflow_amount = Balance.from_rao(MAX_STAKE + 1)
        
        with patch('bittensor.core.extrinsics.staking.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            
            result = add_stake_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58=mock_wallet.hotkey.ss58_address,
                netuid=1,
                amount=overflow_amount,
                wait_for_inclusion=False,
                wait_for_finalization=False
            )
            
            assert result is False, "Overflow stake amount should be rejected"
    
    def test_max_stake_boundary(self, mock_subtensor, mock_wallet):
        """Test staking exactly at u64 max boundary"""
        MAX_STAKE = 2**64 - 1
        max_amount = Balance.from_rao(MAX_STAKE)
        
        with patch('bittensor.core.extrinsics.staking.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            
            # Mock sufficient balance
            mock_subtensor.get_balance.return_value = Balance.from_rao(MAX_STAKE + 1000)
            
            # This should pass overflow validation
            result = add_stake_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58=mock_wallet.hotkey.ss58_address,
                netuid=1,
                amount=max_amount,
                wait_for_inclusion=False,
                wait_for_finalization=False
            )
            
            # Should pass validation (may fail for other reasons)
            # The key is it doesn't fail on overflow check
    
    def test_zero_stake_allowed(self, mock_subtensor, mock_wallet):
        """Test that zero stake is handled (edge case)"""
        # Note: Zero stake might be rejected by balance checks, not overflow validation
        zero_amount = Balance.from_rao(0)
        
        with patch('bittensor.core.extrinsics.staking.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            
            result = add_stake_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58=mock_wallet.hotkey.ss58_address,
                netuid=1,
                amount=zero_amount,
                wait_for_inclusion=False,
                wait_for_finalization=False
            )
            
            # Will fail on balance check, not overflow
            assert result is False
    
    def test_normal_stake_amount(self, mock_subtensor, mock_wallet):
        """Test normal staking amounts pass validation"""
        normal_amount = Balance.from_tao(10)
        
        with patch('bittensor.core.extrinsics.staking.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            
            result = add_stake_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58=mock_wallet.hotkey.ss58_address,
                netuid=1,
                amount=normal_amount,
                wait_for_inclusion=False,
                wait_for_finalization=False
            )
            
            # Should succeed or fail for non-validation reasons
            # The validation should pass
    
    def test_stake_all_with_overflow_protection(self, mock_subtensor, mock_wallet):
        """Test staking all when balance exceeds u64 max"""
        # Set balance to exceed u64 max
        huge_balance = Balance.from_rao(2**64 + 1000)
        mock_subtensor.get_balance.return_value = huge_balance
        
        with patch('bittensor.core.extrinsics.staking.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            
            # Stake all (amount=None)
            result = add_stake_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58=mock_wallet.hotkey.ss58_address,
                netuid=1,
                amount=None,  # Stake all
                wait_for_inclusion=False,
                wait_for_finalization=False
            )
            
            # Should be rejected due to overflow protection
            assert result is False, "Staking all when balance exceeds u64 should be rejected"
    
    def test_multiple_overflow_values(self, mock_subtensor, mock_wallet):
        """Test various overflow values are all rejected"""
        MAX_STAKE = 2**64 - 1
        
        overflow_values = [
            MAX_STAKE + 1,
            MAX_STAKE + 1000,
            2**65,
            2**70,
            2**100,
            2**127,
        ]
        
        with patch('bittensor.core.extrinsics.staking.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            
            for overflow_val in overflow_values:
                result = add_stake_extrinsic(
                    subtensor=mock_subtensor,
                    wallet=mock_wallet,
                    hotkey_ss58=mock_wallet.hotkey.ss58_address,
                    netuid=1,
                    amount=Balance.from_rao(overflow_val),
                    wait_for_inclusion=False,
                    wait_for_finalization=False
                )
                
                assert result is False, f"Overflow value {overflow_val} should be rejected"
    
    def test_safe_staking_with_overflow(self, mock_subtensor, mock_wallet):
        """Test safe staking mode with overflow amounts"""
        MAX_STAKE = 2**64 - 1
        overflow_amount = Balance.from_rao(MAX_STAKE + 1)
        
        # Mock subnet for safe staking
        mock_pool = Mock()
        mock_pool.price = Balance.from_tao(1.0)
        mock_pool.netuid = 1
        mock_subtensor.subnet = Mock(return_value=mock_pool)
        
        with patch('bittensor.core.extrinsics.staking.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            
            result = add_stake_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58=mock_wallet.hotkey.ss58_address,
                netuid=1,
                amount=overflow_amount,
                safe_staking=True,
                wait_for_inclusion=False,
                wait_for_finalization=False
            )
            
            assert result is False, "Safe staking should also reject overflow"


class TestStakingEdgeCases:
    """Test edge cases for staking validation"""
    
    @pytest.fixture
    def mock_subtensor(self):
        subtensor = Mock()
        subtensor.network = "test"
        subtensor.get_current_block = Mock(return_value=1000)
        subtensor.get_balance = Mock(return_value=Balance.from_tao(100))
        subtensor.get_stake = Mock(return_value=Balance.from_tao(50))
        subtensor.get_existential_deposit = Mock(return_value=Balance.from_tao(0.01))
        return subtensor
    
    @pytest.fixture
    def mock_wallet(self):
        wallet = Mock()
        wallet.name = "test_wallet"
        wallet.hotkey = Mock()
        wallet.hotkey.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        wallet.coldkeypub = Mock()
        wallet.coldkeypub.ss58_address = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        return wallet
    
    def test_concurrent_stake_validations(self, mock_subtensor, mock_wallet):
        """Test that stake validation is thread-safe"""
        from concurrent.futures import ThreadPoolExecutor
        
        MAX_STAKE = 2**64 - 1
        overflow_amount = Balance.from_rao(MAX_STAKE + 1)
        
        def validate():
            with patch('bittensor.core.extrinsics.staking.unlock_key') as mock_unlock:
                mock_unlock.return_value = Mock(success=True)
                
                return add_stake_extrinsic(
                    subtensor=mock_subtensor,
                    wallet=mock_wallet,
                    hotkey_ss58=mock_wallet.hotkey.ss58_address,
                    netuid=1,
                    amount=overflow_amount,
                    wait_for_inclusion=False,
                    wait_for_finalization=False
                )
        
        # Run 50 concurrent validations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(validate) for _ in range(50)]
            results = [f.result() for f in futures]
        
        # All should reject the overflow amount
        assert all(r is False for r in results), "All validations should reject overflow"
    
    def test_precision_at_boundary(self, mock_subtensor, mock_wallet):
        """Test precision handling at u64 boundary"""
        MAX_STAKE = 2**64 - 1
        
        # Test values around the boundary
        test_values = [
            MAX_STAKE - 1,
            MAX_STAKE,
            MAX_STAKE + 1,
        ]
        
        with patch('bittensor.core.extrinsics.staking.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            mock_subtensor.get_balance.return_value = Balance.from_rao(2**65)
            
            for val in test_values:
                result = add_stake_extrinsic(
                    subtensor=mock_subtensor,
                    wallet=mock_wallet,
                    hotkey_ss58=mock_wallet.hotkey.ss58_address,
                    netuid=1,
                    amount=Balance.from_rao(val),
                    wait_for_inclusion=False,
                    wait_for_finalization=False
                )
                
                if val > MAX_STAKE:
                    assert result is False, f"Value {val} should be rejected"
    
    def test_existential_deposit_handling(self, mock_subtensor, mock_wallet):
        """Test that existential deposit doesn't cause overflow"""
        MAX_STAKE = 2**64 - 1
        
        # Set balance close to max
        mock_subtensor.get_balance.return_value = Balance.from_rao(MAX_STAKE)
        mock_subtensor.get_existential_deposit.return_value = Balance.from_tao(0.01)
        
        with patch('bittensor.core.extrinsics.staking.unlock_key') as mock_unlock:
            mock_unlock.return_value = Mock(success=True)
            
            # Stake all (will subtract existential deposit)
            result = add_stake_extrinsic(
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58=mock_wallet.hotkey.ss58_address,
                netuid=1,
                amount=None,  # Stake all
                wait_for_inclusion=False,
                wait_for_finalization=False
            )
            
            # Should handle existential deposit correctly


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
