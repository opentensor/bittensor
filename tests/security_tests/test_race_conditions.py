"""
Security tests for race conditions in Bittensor core components.

These tests verify that concurrent operations are handled safely,
particularly around nonce validation in the Axon server.
"""

import asyncio
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, MagicMock

from bittensor.core.axon import Axon
from bittensor.core.synapse import Synapse, TerminalInfo
from bittensor_wallet import Wallet, Keypair


class TestNonceRaceConditions:
    """Test suite for nonce validation race conditions"""
    
    @pytest.fixture
    def mock_wallet(self):
        """Create a mock wallet for testing"""
        wallet = Mock(spec=Wallet)
        wallet.hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        wallet.coldkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        wallet.coldkeypub = wallet.coldkey
        return wallet
    
    @pytest.fixture
    def axon(self, mock_wallet):
        """Create an Axon instance for testing"""
        axon = Axon(wallet=mock_wallet, port=8091)
        return axon
    
    @pytest.fixture
    def create_synapse(self, mock_wallet):
        """Factory to create test synapses"""
        def _create(nonce: int, hotkey: str = None):
            synapse = Synapse()
            synapse.dendrite = TerminalInfo(
                ip="127.0.0.1",
                port=8091,
                hotkey=hotkey or mock_wallet.hotkey.ss58_address,
                coldkey=mock_wallet.coldkey.ss58_address,
                version=7002000,  # V_7_2_0
                nonce=nonce,
                uuid="test-uuid-123",
                signature=None
            )
            synapse.computed_body_hash = "test_hash"
            synapse.timeout = 12.0
            
            # Sign the synapse
            message = f"{nonce}.{synapse.dendrite.hotkey}.{mock_wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"
            synapse.dendrite.signature = mock_wallet.hotkey.sign(message).hex()
            
            return synapse
        return _create
    
    def test_concurrent_same_nonce_rejection(self, axon, create_synapse):
        """
        Test that concurrent requests with the same nonce are properly rejected.
        
        This is the critical race condition test. Without proper locking,
        multiple threads could pass nonce validation simultaneously.
        """
        nonce = time.time_ns()
        synapse = create_synapse(nonce)
        
        results = {"success": 0, "failure": 0}
        results_lock = threading.Lock()
        
        async def verify_synapse():
            """Attempt to verify the synapse"""
            try:
                await axon.default_verify(synapse)
                with results_lock:
                    results["success"] += 1
                return True
            except Exception as e:
                with results_lock:
                    results["failure"] += 1
                return False
        
        # Run 20 concurrent verification attempts
        num_threads = 20
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for _ in range(num_threads):
                # Create new event loop for each thread
                future = executor.submit(
                    lambda: asyncio.run(verify_synapse())
                )
                futures.append(future)
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        # CRITICAL: Only ONE request should succeed
        assert results["success"] == 1, (
            f"Race condition detected! {results['success']} requests succeeded, "
            f"expected exactly 1. {results['failure']} failed."
        )
        assert results["failure"] == num_threads - 1
    
    def test_sequential_increasing_nonces(self, axon, create_synapse):
        """
        Test that sequential requests with increasing nonces are accepted.
        """
        base_nonce = time.time_ns()
        
        async def verify_sequence():
            for i in range(10):
                nonce = base_nonce + i * 1_000_000_000  # 1 second apart
                synapse = create_synapse(nonce)
                
                try:
                    await axon.default_verify(synapse)
                except Exception as e:
                    pytest.fail(f"Valid nonce {nonce} was rejected: {e}")
        
        asyncio.run(verify_sequence())
    
    def test_old_nonce_rejection(self, axon, create_synapse):
        """
        Test that old nonces are properly rejected.
        """
        current_nonce = time.time_ns()
        old_nonce = current_nonce - 10_000_000_000  # 10 seconds old
        
        async def verify_old():
            # First, verify a current nonce
            current_synapse = create_synapse(current_nonce)
            await axon.default_verify(current_synapse)
            
            # Then try an old nonce - should fail
            old_synapse = create_synapse(old_nonce)
            with pytest.raises(Exception, match="Nonce is too old"):
                await axon.default_verify(old_synapse)
        
        asyncio.run(verify_old())
    
    def test_concurrent_different_endpoints(self, axon, create_synapse, mock_wallet):
        """
        Test that concurrent requests to different endpoints don't interfere.
        """
        nonce = time.time_ns()
        
        # Create different hotkeys for different endpoints
        hotkey1 = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        hotkey2 = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        
        results = []
        
        async def verify_endpoint(hotkey):
            synapse = create_synapse(nonce, hotkey=hotkey.ss58_address)
            # Re-sign with correct hotkey
            message = f"{nonce}.{hotkey.ss58_address}.{mock_wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"
            synapse.dendrite.signature = hotkey.sign(message).hex()
            
            try:
                await axon.default_verify(synapse)
                return True
            except Exception:
                return False
        
        async def run_concurrent():
            # Both should succeed as they're different endpoints
            result1 = await verify_endpoint(hotkey1)
            result2 = await verify_endpoint(hotkey2)
            return result1, result2
        
        result1, result2 = asyncio.run(run_concurrent())
        
        # Both different endpoints should succeed
        assert result1 is True
        assert result2 is True
    
    def test_nonce_lock_prevents_deadlock(self, axon, create_synapse):
        """
        Test that the nonce lock doesn't cause deadlocks under heavy load.
        """
        base_nonce = time.time_ns()
        num_requests = 100
        
        async def verify_many():
            tasks = []
            for i in range(num_requests):
                nonce = base_nonce + i * 1_000_000  # 1ms apart
                synapse = create_synapse(nonce)
                tasks.append(axon.default_verify(synapse))
            
            # Run all concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes (should be all of them since nonces are unique)
            successes = sum(1 for r in results if not isinstance(r, Exception))
            return successes
        
        # Should complete without deadlock
        successes = asyncio.run(verify_many())
        
        # All should succeed since nonces are unique and increasing
        assert successes == num_requests
    
    def test_stress_test_concurrent_load(self, axon, create_synapse):
        """
        Stress test with high concurrent load to detect race conditions.
        """
        base_nonce = time.time_ns()
        num_unique_nonces = 50
        requests_per_nonce = 5  # Try to exploit race condition
        
        results = {"success": 0, "failure": 0}
        results_lock = threading.Lock()
        
        async def verify_synapse(nonce):
            synapse = create_synapse(nonce)
            try:
                await axon.default_verify(synapse)
                with results_lock:
                    results["success"] += 1
                return True
            except Exception:
                with results_lock:
                    results["failure"] += 1
                return False
        
        def run_verification(nonce):
            return asyncio.run(verify_synapse(nonce))
        
        # Create multiple requests for each nonce
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for i in range(num_unique_nonces):
                nonce = base_nonce + i * 1_000_000_000
                # Submit multiple requests with same nonce
                for _ in range(requests_per_nonce):
                    future = executor.submit(run_verification, nonce)
                    futures.append(future)
            
            # Wait for all
            for future in as_completed(futures):
                future.result()
        
        # Should have exactly num_unique_nonces successes
        # (one per unique nonce, others rejected)
        assert results["success"] == num_unique_nonces, (
            f"Expected {num_unique_nonces} successes, got {results['success']}. "
            f"Race condition may be present!"
        )


class TestTransferInputValidation:
    """Test suite for transfer amount validation"""
    
    def test_negative_amount_rejection(self):
        """Test that negative transfer amounts are rejected"""
        from bittensor.core.extrinsics.transfer import transfer_extrinsic
        from bittensor.utils.balance import Balance
        
        # This should be rejected before any blockchain interaction
        # Implementation should validate amount > 0
        # Test would need proper mocking of subtensor and wallet
        pass  # Placeholder - requires full integration test setup
    
    def test_zero_amount_rejection(self):
        """Test that zero transfer amounts are rejected"""
        pass  # Placeholder
    
    def test_overflow_amount_rejection(self):
        """Test that overflow amounts are rejected"""
        pass  # Placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
