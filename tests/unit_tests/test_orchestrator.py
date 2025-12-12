
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import types

# Helper to mock a module in sys.modules
def force_mock_module(name):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
    return m

# Mock importlib behavior for version
with patch("importlib.metadata.version", return_value="8.0.0"):
    
    # 1. Mock External Dependencies
    mw = force_mock_module("bittensor_wallet")
    mw_utils = force_mock_module("bittensor_wallet.utils")
    mw_utils.SS58_FORMAT = 42
    mw.utils = mw_utils
    
    # 2. Mock Internal Dependencies that are heavy/broken
    # We carefully DO NOT mock "bittensor" or "bittensor.core" or "bittensor.core.orchestrator"
    # so that the real orchestrator code loads.
    
    # Mock AsyncSubtensor module
    mas = force_mock_module("bittensor.core.async_subtensor")
    class MockAsyncSubtensor:
        def __init__(self, *args, **kwargs):
            self.substrate = AsyncMock()
        async def initialize(self): pass
        async def get_account_nonce(self, addr): return 0
    mas.AsyncSubtensor = MockAsyncSubtensor
    
    # Mock Utils
    mu = force_mock_module("bittensor.utils")
    mbtl = force_mock_module("bittensor.utils.btlogging")
    mbtl.logging = MagicMock()
    mu.btlogging = mbtl
    
    force_mock_module("bittensor.utils.easy_imports")
    force_mock_module("bittensor.errors")
    
    # 3. Import the Target
    # We need to make sure bittensor.__init__ doesn't explode if it's loaded.
    # If bittensor/__init__.py imports things we mocked, it should be fine.
    
    from bittensor.core.orchestrator.orchestrator import (
        TransactionOrchestrator, 
        TransactionMetadata, 
        TransactionStatus,
        RBFPolicy
    )

class TestTransactionOrchestrator(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_wallet = MagicMock()
        self.mock_wallet.hotkey.ss58_address = "5DummyAddress"

        self.mock_subtensor = MagicMock()
        self.mock_subtensor.substrate = AsyncMock()
        self.mock_subtensor.get_account_nonce = AsyncMock(return_value=100)
        self.mock_subtensor.initialize = AsyncMock()
        
        self.mock_subtensor.substrate.compose_call = AsyncMock(return_value="0xCallData")
        self.mock_subtensor.substrate.create_signed_extrinsic = AsyncMock(return_value="0xSignedExtrinsic")
        self.mock_subtensor.substrate.submit_extrinsic = AsyncMock(return_value="0xHash123")

    async def test_nonce_manager(self):
        orch = TransactionOrchestrator(self.mock_wallet, self.mock_subtensor)
        await orch.start()
        
        nonce1 = await orch.nonce_manager.get_next_nonce()
        self.assertEqual(nonce1, 100)
        
        nonce2 = await orch.nonce_manager.get_next_nonce()
        self.assertEqual(nonce2, 101)
        
        await orch.stop()

    async def test_submission_flow(self):
        orch = TransactionOrchestrator(self.mock_wallet, self.mock_subtensor)
        await orch.start()
        
        tx = await orch.submit_extrinsic("Module", "func", {"arg": 1})
        
        # Wait briefly for worker to process
        await asyncio.sleep(0.1)
        
        self.assertEqual(tx.status, TransactionStatus.SUBMITTED)
        self.assertIn("0xHash123", tx.extrinsic_hashes)
        self.assertEqual(tx.assigned_nonce, 100)
        
        self.mock_subtensor.substrate.compose_call.assert_called_with(
            call_module="Module", 
            call_function="func", 
            call_params={"arg": 1}
        )
        
        await orch.stop()

    async def test_rbf_trigger(self):
        orch = TransactionOrchestrator(self.mock_wallet, self.mock_subtensor)
        await orch.start()
        
        # Mock submit failure with 1014
        self.mock_subtensor.substrate.submit_extrinsic.side_effect = [
            Exception("Priority is too low: (1014)"),
            "0xHashNew" # Success on second try
        ]
        
        tx = await orch.submit_extrinsic("Module", "func", {})
        
        await asyncio.sleep(0.2)
        
        self.assertEqual(tx.status, TransactionStatus.SUBMITTED)
        self.assertEqual(tx.retry_count, 1)
        self.assertGreater(tx.current_tip, 0)
        self.assertIn("0xHashNew", tx.extrinsic_hashes)
        
        await orch.stop()

    async def test_stuck_transaction(self):
        # Short timeout for testing
        config = RBFPolicy(stuck_timeout=0.1)
        orch = TransactionOrchestrator(self.mock_wallet, self.mock_subtensor, config=config)
        await orch.start()
        
        self.mock_subtensor.substrate.submit_extrinsic.return_value = "0xHash1"
        
        tx = await orch.submit_extrinsic("Module", "func", {})
        await asyncio.sleep(0.05) # Let it submit
        self.assertEqual(tx.status, TransactionStatus.SUBMITTED)
        
        # Manually age the transaction
        tx.last_submit_at = 0 # Epoch 0
        
        # Update mock to expect another submission (RBF)
        self.mock_subtensor.substrate.submit_extrinsic.return_value = "0xHash2"
    
        # Manually trigger RBF because we can't wait for monitor loop timing in unit test reliably
        await orch._trigger_rbf(tx)
        
        # Wait for worker to pick up re-queued tx
        await asyncio.sleep(0.1)
        
        self.assertEqual(tx.retry_count, 1)
        self.assertIn("0xHash2", tx.extrinsic_hashes)
        
        await orch.stop()

if __name__ == '__main__':
    unittest.main()

