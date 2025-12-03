"""
Unit tests for nonce validation security and thread safety.

Tests focus on:
- Thread-safe nonce updates
- Replay attack prevention
- Nonce freshness validation
- Concurrent request handling
"""

import pytest
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, MagicMock, patch

from bittensor.core.axon import Axon
from bittensor.core.synapse import Synapse, TerminalInfo
from bittensor_wallet import Wallet, Keypair


class TestNonceThreadSafety:
    """Test thread safety of nonce validation"""
    
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
        def _create(nonce: int, hotkey: str = None, uuid: str = None):
            synapse = Synapse()
            synapse.dendrite = TerminalInfo(
                ip="127.0.0.1",
                port=8091,
                hotkey=hotkey or mock_wallet.hotkey.ss58_address,
                coldkey=mock_wallet.coldkey.ss58_address,
                version=7002000,  # V_7_2_0
                nonce=nonce,
                uuid=uuid or "test-uuid-123",
                signature=None
            )
            synapse.computed_body_hash = "test_hash"
            synapse.timeout = 12.0
            
            # Sign the synapse
            message = f"{nonce}.{synapse.dendrite.hotkey}.{mock_wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"
            synapse.dendrite.signature = mock_wallet.hotkey.sign(message).hex()
            
            return synapse
        return _create
    
    def test_lock_exists(self, axon):
        """Test that nonce lock is initialized"""
        assert hasattr(axon, '_nonce_lock'), "Axon should have _nonce_lock attribute"
        assert axon._nonce_lock is not None, "Nonce lock should be initialized"
    
    def test_lock_is_reentrant(self, axon):
        """Test that the lock is reentrant (RLock)"""
        from threading import RLock
        assert isinstance(axon._nonce_lock, RLock), "Nonce lock should be RLock"
    
    def test_single_nonce_single_thread(self, axon, create_synapse):
        """Test basic nonce validation in single thread"""
        nonce = time.time_ns()
        synapse = create_synapse(nonce)
        
        async def verify():
            await axon.default_verify(synapse)
        
        # First verification should succeed
        asyncio.run(verify())
        
        # Second verification with same nonce should fail
        with pytest.raises(Exception, match="Nonce is too old"):
            asyncio.run(verify())
    
    def test_increasing_nonces_accepted(self, axon, create_synapse):
        """Test that increasing nonces are accepted"""
        base_nonce = time.time_ns()
        
        async def verify_sequence():
            for i in range(5):
                nonce = base_nonce + i * 1_000_000_000  # 1 second apart
                synapse = create_synapse(nonce)
                await axon.default_verify(synapse)
        
        # All should succeed
        asyncio.run(verify_sequence())
    
    def test_decreasing_nonces_rejected(self, axon, create_synapse):
        """Test that decreasing nonces are rejected"""
        base_nonce = time.time_ns()
        
        async def verify_sequence():
            # First, verify a high nonce
            high_nonce = base_nonce + 10_000_000_000
            synapse1 = create_synapse(high_nonce)
            await axon.default_verify(synapse1)
            
            # Then try a lower nonce - should fail
            low_nonce = base_nonce
            synapse2 = create_synapse(low_nonce)
            with pytest.raises(Exception, match="Nonce is too old"):
                await axon.default_verify(synapse2)
        
        asyncio.run(verify_sequence())
    
    def test_concurrent_same_nonce_only_one_succeeds(self, axon, create_synapse):
        """Test that only one of concurrent requests with same nonce succeeds"""
        nonce = time.time_ns()
        synapse = create_synapse(nonce)
        
        results = {"success": 0, "failure": 0}
        results_lock = threading.Lock()
        
        async def verify_synapse():
            try:
                await axon.default_verify(synapse)
                with results_lock:
                    results["success"] += 1
                return True
            except Exception:
                with results_lock:
                    results["failure"] += 1
                return False
        
        # Run 20 concurrent verification attempts
        num_threads = 20
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(lambda: asyncio.run(verify_synapse()))
                for _ in range(num_threads)
            ]
            
            for future in as_completed(futures):
                future.result()
        
        # CRITICAL: Only ONE request should succeed
        assert results["success"] == 1, (
            f"Expected exactly 1 success, got {results['success']}"
        )
        assert results["failure"] == num_threads - 1
    
    def test_different_uuids_different_nonce_tracking(self, axon, create_synapse, mock_wallet):
        """Test that different UUIDs maintain separate nonce tracking"""
        nonce = time.time_ns()
        
        # Same nonce, different UUIDs
        synapse1 = create_synapse(nonce, uuid="uuid-1")
        synapse2 = create_synapse(nonce, uuid="uuid-2")
        
        async def verify_both():
            # Both should succeed as they have different endpoint keys
            await axon.default_verify(synapse1)
            await axon.default_verify(synapse2)
        
        asyncio.run(verify_both())
    
    def test_different_hotkeys_different_nonce_tracking(self, axon, create_synapse):
        """Test that different hotkeys maintain separate nonce tracking"""
        nonce = time.time_ns()
        
        hotkey1 = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        hotkey2 = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        
        async def verify_both():
            # Create synapses with different hotkeys
            synapse1 = create_synapse(nonce, hotkey=hotkey1.ss58_address)
            synapse2 = create_synapse(nonce, hotkey=hotkey2.ss58_address)
            
            # Re-sign with correct hotkeys
            for synapse, hotkey in [(synapse1, hotkey1), (synapse2, hotkey2)]:
                message = f"{nonce}.{hotkey.ss58_address}.{axon.wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"
                synapse.dendrite.signature = hotkey.sign(message).hex()
            
            # Both should succeed
            await axon.default_verify(synapse1)
            await axon.default_verify(synapse2)
        
        asyncio.run(verify_both())
    
    def test_nonce_storage_isolation(self, axon, create_synapse):
        """Test that nonce storage is properly isolated per endpoint"""
        base_nonce = time.time_ns()
        
        async def verify_isolated():
            # Verify nonce for endpoint 1
            synapse1 = create_synapse(base_nonce, uuid="endpoint-1")
            await axon.default_verify(synapse1)
            
            # Verify same nonce for endpoint 2 - should succeed (different endpoint)
            synapse2 = create_synapse(base_nonce, uuid="endpoint-2")
            await axon.default_verify(synapse2)
            
            # Verify higher nonce for endpoint 1 - should succeed
            synapse3 = create_synapse(base_nonce + 1_000_000_000, uuid="endpoint-1")
            await axon.default_verify(synapse3)
            
            # Verify same nonce again for endpoint 1 - should fail
            synapse4 = create_synapse(base_nonce, uuid="endpoint-1")
            with pytest.raises(Exception, match="Nonce is too old"):
                await axon.default_verify(synapse4)
        
        asyncio.run(verify_isolated())


class TestNonceFreshnessValidation:
    """Test nonce freshness and time-based validation"""
    
    @pytest.fixture
    def mock_wallet(self):
        wallet = Mock(spec=Wallet)
        wallet.hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        wallet.coldkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        wallet.coldkeypub = wallet.coldkey
        return wallet
    
    @pytest.fixture
    def axon(self, mock_wallet):
        return Axon(wallet=mock_wallet, port=8091)
    
    @pytest.fixture
    def create_synapse(self, mock_wallet):
        def _create(nonce: int):
            synapse = Synapse()
            synapse.dendrite = TerminalInfo(
                ip="127.0.0.1",
                port=8091,
                hotkey=mock_wallet.hotkey.ss58_address,
                coldkey=mock_wallet.coldkey.ss58_address,
                version=7002000,
                nonce=nonce,
                uuid="test-uuid-123",
                signature=None
            )
            synapse.computed_body_hash = "test_hash"
            synapse.timeout = 12.0
            
            message = f"{nonce}.{synapse.dendrite.hotkey}.{mock_wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"
            synapse.dendrite.signature = mock_wallet.hotkey.sign(message).hex()
            
            return synapse
        return _create
    
    def test_fresh_nonce_accepted(self, axon, create_synapse):
        """Test that fresh nonces are accepted"""
        fresh_nonce = time.time_ns()
        synapse = create_synapse(fresh_nonce)
        
        async def verify():
            await axon.default_verify(synapse)
        
        asyncio.run(verify())
    
    def test_very_old_nonce_rejected(self, axon, create_synapse):
        """Test that very old nonces are rejected"""
        # Nonce from 1 hour ago
        old_nonce = time.time_ns() - (3600 * 1_000_000_000)
        synapse = create_synapse(old_nonce)
        
        async def verify():
            with pytest.raises(Exception, match="Nonce is too old"):
                await axon.default_verify(synapse)
        
        asyncio.run(verify())
    
    def test_future_nonce_accepted(self, axon, create_synapse):
        """Test that slightly future nonces are accepted (clock skew)"""
        # Nonce 1 second in future
        future_nonce = time.time_ns() + 1_000_000_000
        synapse = create_synapse(future_nonce)
        
        async def verify():
            await axon.default_verify(synapse)
        
        asyncio.run(verify())
    
    def test_nonce_window_validation(self, axon, create_synapse):
        """Test nonce validation within allowed window"""
        # Test nonces at various time offsets
        current_time = time.time_ns()
        
        test_cases = [
            (current_time, True, "Current time"),
            (current_time - 1_000_000_000, True, "1 second ago"),
            (current_time - 3_000_000_000, True, "3 seconds ago"),
            (current_time + 1_000_000_000, True, "1 second future"),
        ]
        
        for nonce, should_succeed, description in test_cases:
            synapse = create_synapse(nonce)
            
            async def verify():
                try:
                    await axon.default_verify(synapse)
                    return True
                except Exception:
                    return False
            
            result = asyncio.run(verify())
            if should_succeed:
                assert result, f"{description} should succeed"


class TestNonceStressTests:
    """Stress tests for nonce validation under load"""
    
    @pytest.fixture
    def mock_wallet(self):
        wallet = Mock(spec=Wallet)
        wallet.hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        wallet.coldkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        wallet.coldkeypub = wallet.coldkey
        return wallet
    
    @pytest.fixture
    def axon(self, mock_wallet):
        return Axon(wallet=mock_wallet, port=8091)
    
    @pytest.fixture
    def create_synapse(self, mock_wallet):
        def _create(nonce: int, uuid: str = None):
            synapse = Synapse()
            synapse.dendrite = TerminalInfo(
                ip="127.0.0.1",
                port=8091,
                hotkey=mock_wallet.hotkey.ss58_address,
                coldkey=mock_wallet.coldkey.ss58_address,
                version=7002000,
                nonce=nonce,
                uuid=uuid or f"uuid-{nonce}",
                signature=None
            )
            synapse.computed_body_hash = "test_hash"
            synapse.timeout = 12.0
            
            message = f"{nonce}.{synapse.dendrite.hotkey}.{mock_wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"
            synapse.dendrite.signature = mock_wallet.hotkey.sign(message).hex()
            
            return synapse
        return _create
    
    def test_high_volume_sequential(self, axon, create_synapse):
        """Test high volume of sequential requests"""
        base_nonce = time.time_ns()
        num_requests = 1000
        
        async def verify_many():
            for i in range(num_requests):
                nonce = base_nonce + i * 1_000_000  # 1ms apart
                synapse = create_synapse(nonce)
                await axon.default_verify(synapse)
        
        # Should complete without errors
        asyncio.run(verify_many())
    
    def test_high_volume_concurrent(self, axon, create_synapse):
        """Test high volume of concurrent requests with unique nonces"""
        base_nonce = time.time_ns()
        num_requests = 100
        
        async def verify_one(i):
            nonce = base_nonce + i * 1_000_000
            synapse = create_synapse(nonce)
            await axon.default_verify(synapse)
        
        async def verify_many():
            tasks = [verify_one(i) for i in range(num_requests)]
            await asyncio.gather(*tasks)
        
        # All should succeed
        asyncio.run(verify_many())
    
    def test_lock_contention_measurement(self, axon, create_synapse):
        """Test and measure lock contention under load"""
        base_nonce = time.time_ns()
        num_threads = 50
        requests_per_thread = 10
        
        timings = []
        timings_lock = threading.Lock()
        
        def worker(thread_id):
            for i in range(requests_per_thread):
                nonce = base_nonce + (thread_id * requests_per_thread + i) * 1_000_000
                synapse = create_synapse(nonce, uuid=f"thread-{thread_id}-req-{i}")
                
                start = time.time()
                asyncio.run(axon.default_verify(synapse))
                elapsed = time.time() - start
                
                with timings_lock:
                    timings.append(elapsed)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()
        
        # Analyze timings
        avg_time = sum(timings) / len(timings)
        max_time = max(timings)
        
        # Lock overhead should be minimal
        assert avg_time < 0.01, f"Average time {avg_time}s too high (lock contention?)"
        assert max_time < 0.1, f"Max time {max_time}s too high (deadlock risk?)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
