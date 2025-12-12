
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.utils.btlogging import logging

class TransactionStatus(Enum):
    QUEUED = "QUEUED"
    SUBMITTED = "SUBMITTED"
    IN_BLOCK = "IN_BLOCK"
    FINALIZED = "FINALIZED" 
    STUCK = "STUCK"
    FAILED = "FAILED"
    REPLACED = "REPLACED"

def generate_uuid() -> str:
    return str(uuid.uuid4())

@dataclass
class TransactionMetadata:
    """
    Immutable tracking object for a specific user intent.
    """
    call_module: str
    call_function: str
    call_params: Dict[str, Any]
    
    # Internal State
    id: str = field(default_factory=generate_uuid)
    assigned_nonce: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    
    # RBF Tracking
    current_tip: int = 0
    extrinsic_hashes: List[str] = field(default_factory=list)
    status: TransactionStatus = TransactionStatus.QUEUED
    last_submit_at: float = 0.0
    retry_count: int = 0
    
    # Error tracking
    last_error: Optional[str] = None
    expiration_block: Optional[int] = None

class OptimisticNonceManager:
    """
    Manages the local nonce state to allow for high-throughput submission
    without querying the chain for every transaction.
    """
    def __init__(self, subtensor: AsyncSubtensor, wallet_address: str):
        self.subtensor = subtensor
        self.wallet_address = wallet_address
        self.local_nonce: Optional[int] = None
        self._lock = asyncio.Lock()

    async def sync(self):
        """
        Force synchronization with the chain state.
        """
        async with self._lock:
            # We use the underlying substrate interface to get the nonce directly
            # This avoids any caching that might be in higher layers if they exist
            # though async_subtensor usually wraps this directly.
            self.local_nonce = await self.subtensor.get_account_nonce(self.wallet_address)
            logging.debug(f"Nonce synced to: {self.local_nonce}")

    async def get_next_nonce(self) -> int:
        """
        Returns the next nonce and increments the local counter.
        """
        async with self._lock:
            if self.local_nonce is None:
                await self.sync()
            
            nonce = self.local_nonce
            self.local_nonce += 1
            return nonce

    async def reset(self):
        """
        Resets the local nonce to None, forcing a sync on next request.
        """
        async with self._lock:
            self.local_nonce = None

@dataclass
class RBFPolicy:
    """
    Configuration for Replace-By-Fee logic.
    """
    base_tip: int = 0
    increment_type: str = "percentage" # "linear" or "percentage"
    increment_value: float = 0.15 # 15% increase or fixed RAO amount
    epsilon: int = 1000 # Minimum increment in RAO
    max_tip: int = 1_000_000_000 # 1 TAO safety limit (example)
    stuck_timeout: float = 24.0 # Seconds before considering a tx stuck

    def calculate_next_tip(self, current_tip: int) -> int:
        if self.increment_type == "percentage":
            next_tip = int(current_tip * (1 + self.increment_value))
            # If current tip is 0, percentage won't work, so apply epsilon or base
            if next_tip == current_tip:
                 next_tip += self.epsilon
        else:
            next_tip = current_tip + int(self.increment_value)
        
        # Ensure we always bump by at least epsilon if the calculation resulted in less
        if next_tip < current_tip + self.epsilon:
            next_tip = current_tip + self.epsilon
            
        return min(next_tip, self.max_tip)

class TransactionOrchestrator:
    """
    Stateful engine to manage transaction lifecycles, nonces, and RBF.
    """
    def __init__(
        self, 
        wallet: "bittensor_wallet.Wallet",
        subtensor: Optional[AsyncSubtensor] = None,
        config: Optional[RBFPolicy] = None
    ):
        self.wallet = wallet
        self.subtensor = subtensor or AsyncSubtensor()
        self.config = config or RBFPolicy()
        
        self.nonce_manager = OptimisticNonceManager(self.subtensor, self.wallet.hotkey.ss58_address)
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue() # Stores TransactionMetadata
        self.active_transactions: Dict[str, TransactionMetadata] = {} # id -> metadata
        
        self._worker_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """
        Initializes the connection and starts the worker loops.
        """
        if self._running:
            return

        if not self.subtensor.substrate:
             await self.subtensor.initialize()
             
        await self.nonce_manager.sync()
        
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        self._monitor_task = asyncio.create_task(self._monitor_pool())
        logging.info("TransactionOrchestrator started.")

    async def stop(self):
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()
        try:
            await self._worker_task
            await self._monitor_task
        except asyncio.CancelledError:
            pass

    async def submit_extrinsic(
        self,
        call_module: str,
        call_function: str,
        call_params: Dict[str, Any],
        wait_for_finalization: bool = False, # Kept for API compatibility, but usually False for high throughput
        wait_for_inclusion: bool = False
    ) -> TransactionMetadata:
        """
        Submits an intent to the orchestrator.
        """
        tx = TransactionMetadata(
            call_module=call_module,
            call_function=call_function,
            call_params=call_params,
            current_tip=self.config.base_tip
        )
        
        # Assign nonce immediately to preserve order of python calls
        tx.assigned_nonce = await self.nonce_manager.get_next_nonce()
        
        self.active_transactions[tx.id] = tx
        await self.queue.put((tx.assigned_nonce, tx)) # Priority queue by nonce
        
        logging.info(f"Queued transaction {call_module}.{call_function} with nonce {tx.assigned_nonce}")
        
        if wait_for_inclusion or wait_for_finalization:
             await self._wait_for_status(tx, wait_for_finalization)
             
        return tx

    async def get_transaction_status(self, tx_id: str) -> Optional[TransactionStatus]:
        if tx_id in self.active_transactions:
            return self.active_transactions[tx_id].status
        return None

    async def _wait_for_status(self, tx: TransactionMetadata, finalization: bool):
        """
        Helper to block until inclusion or finalization.
        """
        while True:
            if tx.status == TransactionStatus.FAILED:
                raise Exception(f"Transaction failed: {tx.last_error}")
            
            if finalization:
                if tx.status == TransactionStatus.FINALIZED:
                    return
            else:
                 if tx.status in [TransactionStatus.IN_BLOCK, TransactionStatus.FINALIZED]:
                     return
                     
            await asyncio.sleep(1)

    async def _process_queue(self):
        """
        Worker loop that signs and submits transactions.
        """
        while self._running:
            try:
                _, tx = await self.queue.get()
                
                if tx.status in [TransactionStatus.FAILED, TransactionStatus.FINALIZED]:
                    self.queue.task_done()
                    continue

                try:
                    # Compose call
                    call = await self.subtensor.substrate.compose_call(
                        call_module=tx.call_module,
                        call_function=tx.call_function,
                        call_params=tx.call_params
                    )
                    
                    # Sign extrinsic
                    # accessing internal substrate interface for explicit nonce/tip control
                    extrinsic = await self.subtensor.substrate.create_signed_extrinsic(
                        call=call,
                        keypair=self.wallet.hotkey,
                        nonce=tx.assigned_nonce,
                        tip=tx.current_tip
                    )
                    
                    # Submit
                    # We don't wait for inclusion here, we just fire it off.
                    try:
                        receipt = await self.subtensor.substrate.submit_extrinsic(
                            extrinsic,
                            wait_for_inclusion=False,
                            wait_for_finalization=False
                        )
                        # Handle varied return types from different substrate interface versions
                        tx_hash = receipt.extrinsic_hash if hasattr(receipt, 'extrinsic_hash') else str(receipt)
                        
                        tx.extrinsic_hashes.append(tx_hash)
                        tx.status = TransactionStatus.SUBMITTED
                        tx.last_submit_at = time.time()
                        logging.info(f"Submitted {tx.id} (Nonce: {tx.assigned_nonce}) Hash: {tx_hash}")

                    except Exception as e:
                        # Handle immediate submission errors (e.g. priority too low, invalid nonce)
                        error_str = str(e)
                        if "1014" in error_str or "Priority is too low" in error_str:
                             logging.warning(f"Priority too low for {tx.id}. Triggering RBF.")
                             await self._trigger_rbf(tx)
                        elif "Invalid Nonce" in error_str or "Transaction is outdated" in error_str:
                             logging.error(f"Nonce misalignment for {tx.id}. Triggering re-sync.")
                             await self.nonce_manager.reset()
                             # Re-queue? If nonce is invalid, we might need to re-assign nonce.
                             # But that breaks the sequence. 
                             # Simpler approach: Fail this, and let user retry? 
                             # Or drift detection: sync nonce, and if actual nonce > tx.nonce, then this tx is already done or dead.
                             # If actual nonce < tx.nonce, we are in future.
                             # For now, mark FAILED to stop loop.
                             tx.status = TransactionStatus.FAILED
                             tx.last_error = error_str
                        else:
                             logging.error(f"Submission error for {tx.id}: {e}")
                             tx.status = TransactionStatus.FAILED
                             tx.last_error = str(e)

                except Exception as e:
                    logging.error(f"Fatal error processing {tx.id}: {e}")
                    tx.status = TransactionStatus.FAILED
                    tx.last_error = str(e)
                
                finally:
                    self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Worker loop error: {e}")
                await asyncio.sleep(1)

    async def _monitor_pool(self):
        """
        Monitors submitted transactions for stuckness.
        """
        while self._running:
            try:
                # Iterate over copy of values to avoid modification issues
                current_time = time.time()
                for tx in list(self.active_transactions.values()):
                    if tx.status == TransactionStatus.SUBMITTED:
                        # Check timeout
                        if current_time - tx.last_submit_at > self.config.stuck_timeout:
                            logging.warning(f"Transaction {tx.id} stuck. Triggering RBF.")
                            await self._trigger_rbf(tx)
                        else:
                            # Optional: Check chain status if we want faster feedback than just timeout
                            # But that's expensive. relying on timeout is standard for RBF.
                            pass
                            
                await asyncio.sleep(5) # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)

    async def _trigger_rbf(self, tx: TransactionMetadata):
        """
        Calculates new tip and re-queues the transaction.
        """
        new_tip = self.config.calculate_next_tip(tx.current_tip)
        if new_tip > self.config.max_tip:
            logging.error(f"Max tip reached for {tx.id}. Cannot RBF.")
            # Don't fail it yet, maybe it will eventually pass.
            # Or mark as STUCK
            tx.status = TransactionStatus.STUCK
            return

        tx.current_tip = new_tip
        tx.retry_count += 1
        logging.info(f"RBF: Increasing tip to {tx.current_tip} for {tx.id}")
        
        # Push back to queue. 
        # Since queue is priority queue by nonce, and we use the same nonce,
        # it will be processed in correct order relative to other nonces,
        # but we want it processed ASAP.
        await self.queue.put((tx.assigned_nonce, tx))
