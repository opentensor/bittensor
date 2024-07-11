from dataclasses import dataclass

import binascii
import hashlib
import math
import multiprocessing
import multiprocessing.queues  # this must be imported separately, or could break type annotations
import os
import random
import time
from datetime import timedelta
from queue import Empty, Full
import typing
from typing import List, Optional, Tuple, Union

import backoff

import bittensor
from Crypto.Hash import keccak
from rich import console as rich_console
from rich import status as rich_status

from . import ss58_address_to_bytes
from .formatting import get_human_readable, millify
from ._register_cuda import solve_cuda
from .registration import POWSolution, LazyLoadedTorch


if typing.TYPE_CHECKING:
    import torch
else:
    torch = LazyLoadedTorch()


@dataclass
class ColdkeySwapStatistics:
    difficulty: int
    block_number: int
    block_hash: str
    time_spent_total: float = 0.0
    rounds_total: int = 0
    time_average: float = 0.0
    time_spent: float = 0.0
    hash_rate_perpetual: float = 0.0
    hash_rate: float = 0.0


class SwapPOWSolution(POWSolution):
    """A solution to the Coldkey Swap PoW problem."""

    def is_stale(self, _: "bittensor.subtensor") -> bool:
        """Returns True if the POW is stale.
        This means the block the POW is solved for is
        too old and/or no longer valid.
        """
        False  # No age criteria for coldkey swap POW


class ColdkeySwapStatisticsLogger:
    """Logs statistics for a coldkey swap."""

    console: rich_console.Console
    status: Optional[rich_status.Status]

    def __init__(
        self, console: rich_console.Console, output_in_place: bool = True
    ) -> None:
        self.console = console

        if output_in_place:
            self.status = self.console.status("Solving")
        else:
            self.status = None

    def start(self) -> None:
        if self.status is not None:
            self.status.start()

    def stop(self) -> None:
        if self.status is not None:
            self.status.stop()

    def get_status_message(
        self, stats: ColdkeySwapStatistics, verbose: bool = False
    ) -> str:
        message = (
            "Solving\n"
            + f"Time Spent (total): [bold white]{timedelta(seconds=stats.time_spent_total)}[/bold white]\n"
            + (
                f"Time Spent This Round: {timedelta(seconds=stats.time_spent)}\n"
                + f"Time Spent Average: {timedelta(seconds=stats.time_average)}\n"
                if verbose
                else ""
            )
            + f"Coldkey Swap Difficulty: [bold white]{millify(stats.difficulty)}[/bold white]\n"
            + f"Iters (Inst/Perp): [bold white]{get_human_readable(stats.hash_rate, 'H')}/s / "
            + f"{get_human_readable(stats.hash_rate_perpetual, 'H')}/s[/bold white]\n"
            + f"Block Number: [bold white]{stats.block_number}[/bold white]\n"
            + f"Block Hash: [bold white]{stats.block_hash[:10]}...[/bold white]\n"
        )
        return message

    def update(self, stats: ColdkeySwapStatistics, verbose: bool = False) -> None:
        if self.status is not None:
            self.status.update(self.get_status_message(stats, verbose=verbose))
        else:
            self.console.log(self.get_status_message(stats, verbose=verbose))


def _calculate_difficulty(
    base_difficulty: int,
    swap_attempts: int,
) -> int:
    return base_difficulty * (2**swap_attempts)


def _hex_bytes_to_u8_list(hex_bytes: bytes):
    hex_chunks = [int(hex_bytes[i : i + 2], 16) for i in range(0, len(hex_bytes), 2)]
    return hex_chunks


def _create_seal_hash(block_and_hotkey_hash_bytes: bytes, nonce: int) -> bytes:
    nonce_bytes = binascii.hexlify(nonce.to_bytes(8, "little"))
    pre_seal = nonce_bytes + binascii.hexlify(block_and_hotkey_hash_bytes)[:64]
    seal_sh256 = hashlib.sha256(bytearray(_hex_bytes_to_u8_list(pre_seal))).digest()
    kec = keccak.new(digest_bits=256)
    seal = kec.update(seal_sh256).digest()
    return seal


def _seal_meets_difficulty(seal: bytes, difficulty: int, limit: int):
    seal_number = int.from_bytes(seal, "big")
    product = seal_number * difficulty
    return product < limit


class _SolverBase(multiprocessing.Process):
    """
    A process that solves the registration PoW problem.

    Args:
        proc_num: int
            The number of the process being created.
        num_proc: int
            The total number of processes running.
        update_interval: int
            The number of nonces to try to solve before checking for a new block.
        finished_queue: multiprocessing.Queue
            The queue to put the process number when a process finishes each update_interval.
            Used for calculating the average time per update_interval across all processes.
        solution_queue: multiprocessing.Queue
            The queue to put the solution the process has found during the pow solve.
        newBlockEvent: multiprocessing.Event
            The event to set by the main process when a new block is finalized in the network.
            The solver process will check for the event after each update_interval.
            The solver process will get the new block hash and difficulty and start solving for a new nonce.
        stopEvent: multiprocessing.Event
            The event to set by the main process when all the solver processes should stop.
            The solver process will check for the event after each update_interval.
            The solver process will stop when the event is set.
            Used to stop the solver processes when a solution is found.
        curr_block: multiprocessing.Array
            The array containing this process's current block hash.
            The main process will set the array to the new block hash when a new block is finalized in the network.
            The solver process will get the new block hash from this array when newBlockEvent is set.
        curr_block_num: multiprocessing.Value
            The value containing this process's current block number.
            The main process will set the value to the new block number when a new block is finalized in the network.
            The solver process will get the new block number from this value when newBlockEvent is set.
        curr_diff: multiprocessing.Array
            The array containing this process's current difficulty.
            The main process will set the array to the new difficulty when a new block is finalized in the network.
            The solver process will get the new difficulty from this array when newBlockEvent is set.
        check_block: multiprocessing.Lock
            The lock to prevent this process from getting the new block data while the main process is updating the data.
        limit: int
            The limit of the pow solve for a valid solution.
    """

    proc_num: int
    num_proc: int
    update_interval: int
    finished_queue: multiprocessing.Queue
    solution_queue: multiprocessing.Queue
    newBlockEvent: multiprocessing.Event
    stopEvent: multiprocessing.Event
    coldkey_bytes: bytes
    curr_block: multiprocessing.Array
    curr_block_num: multiprocessing.Value
    curr_diff: multiprocessing.Array
    check_block: multiprocessing.Lock
    limit: int

    def __init__(
        self,
        proc_num,
        num_proc,
        update_interval,
        finished_queue,
        solution_queue,
        stopEvent,
        curr_block,
        curr_block_num,
        curr_diff,
        check_block,
        limit,
    ):
        multiprocessing.Process.__init__(self, daemon=True)
        self.proc_num = proc_num
        self.num_proc = num_proc
        self.update_interval = update_interval
        self.finished_queue = finished_queue
        self.solution_queue = solution_queue
        self.newBlockEvent = multiprocessing.Event()
        self.newBlockEvent.clear()
        self.curr_block = curr_block
        self.curr_block_num = curr_block_num
        self.curr_diff = curr_diff
        self.check_block = check_block
        self.stopEvent = stopEvent
        self.limit = limit

    def run(self):
        raise NotImplementedError("_SolverBase is an abstract class")

    @staticmethod
    def create_shared_memory() -> (
        Tuple[multiprocessing.Array, multiprocessing.Value, multiprocessing.Array]
    ):
        """Creates shared memory for the solver processes to use."""
        curr_block = multiprocessing.Array("h", 32, lock=True)  # byte array
        curr_block_num = multiprocessing.Value("i", 0, lock=True)  # int
        curr_diff = multiprocessing.Array("Q", [0, 0], lock=True)  # [high, low]

        return curr_block, curr_block_num, curr_diff


class _Solver(_SolverBase):
    def run(self):
        block_number: int
        block_and_hotkey_hash_bytes: bytes
        block_difficulty: int
        nonce_limit = int(math.pow(2, 64)) - 1

        # Start at random nonce
        nonce_start = random.randint(0, nonce_limit)
        nonce_end = nonce_start + self.update_interval
        while not self.stopEvent.is_set():
            if self.newBlockEvent.is_set():
                with self.check_block:
                    block_number = self.curr_block_num.value
                    block_and_hotkey_hash_bytes = bytes(self.curr_block)
                    block_difficulty = _registration_diff_unpack(self.curr_diff)

                self.newBlockEvent.clear()

            # Do a block of nonces
            solution = _solve_for_nonce_block(
                nonce_start,
                nonce_end,
                block_and_hotkey_hash_bytes,
                block_difficulty,
                self.limit,
                block_number,
            )
            if solution is not None:
                self.solution_queue.put(solution)

            try:
                # Send time
                self.finished_queue.put_nowait(self.proc_num)
            except Full:
                pass

            nonce_start = random.randint(0, nonce_limit)
            nonce_start = nonce_start % nonce_limit
            nonce_end = nonce_start + self.update_interval


class _CUDASolver(_SolverBase):
    dev_id: int
    tpb: int

    def __init__(
        self,
        proc_num,
        num_proc,
        update_interval,
        finished_queue,
        solution_queue,
        stopEvent,
        curr_block,
        curr_block_num,
        curr_diff,
        check_block,
        limit,
        dev_id: int,
        tpb: int,
    ):
        super().__init__(
            proc_num,
            num_proc,
            update_interval,
            finished_queue,
            solution_queue,
            stopEvent,
            curr_block,
            curr_block_num,
            curr_diff,
            check_block,
            limit,
        )
        self.dev_id = dev_id
        self.tpb = tpb

    def run(self):
        block_number: int = 0  # dummy value
        block_and_hotkey_hash_bytes: bytes = b"0" * 32  # dummy value
        block_difficulty: int = int(math.pow(2, 64)) - 1  # dummy value
        nonce_limit = int(math.pow(2, 64)) - 1  # U64MAX

        # Start at random nonce
        nonce_start = random.randint(0, nonce_limit)
        while not self.stopEvent.is_set():
            if self.newBlockEvent.is_set():
                with self.check_block:
                    block_number = self.curr_block_num.value
                    block_and_hotkey_hash_bytes = bytes(self.curr_block)
                    block_difficulty = _registration_diff_unpack(self.curr_diff)

                self.newBlockEvent.clear()

            # Do a block of nonces
            solution = _solve_for_nonce_block_cuda(
                nonce_start,
                self.update_interval,
                block_and_hotkey_hash_bytes,
                block_difficulty,
                self.limit,
                block_number,
                self.dev_id,
                self.tpb,
            )
            if solution is not None:
                self.solution_queue.put(solution)

            try:
                # Signal that a nonce_block was finished using queue
                # send our proc_num
                self.finished_queue.put(self.proc_num)
            except Full:
                pass

            # increase nonce by number of nonces processed
            nonce_start += self.update_interval * self.tpb
            nonce_start = nonce_start % nonce_limit


def _solve_for_nonce_block_cuda(
    nonce_start: int,
    update_interval: int,
    block_and_hotkey_hash_bytes: bytes,
    difficulty: int,
    limit: int,
    block_number: int,
    dev_id: int,
    tpb: int,
) -> Optional[SwapPOWSolution]:
    """Tries to solve the POW on a CUDA device for a block of nonces (nonce_start, nonce_start + update_interval * tpb"""
    solution, seal = solve_cuda(
        nonce_start,
        update_interval,
        tpb,
        block_and_hotkey_hash_bytes,
        difficulty,
        limit,
        dev_id,
    )

    if solution != -1:
        # Check if solution is valid (i.e. not -1)
        return POWSolution(solution, block_number, difficulty, seal)

    return None


def _solve_for_nonce_block(
    nonce_start: int,
    nonce_end: int,
    block_and_hotkey_hash_bytes: bytes,
    difficulty: int,
    limit: int,
    block_number: int,
) -> Optional[SwapPOWSolution]:
    """Tries to solve the POW for a block of nonces (nonce_start, nonce_end)"""
    for nonce in range(nonce_start, nonce_end):
        # Create seal.
        seal = _create_seal_hash(block_and_hotkey_hash_bytes, nonce)

        # Check if seal meets difficulty
        if _seal_meets_difficulty(seal, difficulty, limit):
            # Found a solution, save it.
            return SwapPOWSolution(nonce, block_number, difficulty, seal)

    return None


def _registration_diff_unpack(packed_diff: multiprocessing.Array) -> int:
    """Unpacks the packed two 32-bit integers into one 64-bit integer. Little endian."""
    return int(packed_diff[0] << 32 | packed_diff[1])


def _registration_diff_pack(diff: int, packed_diff: multiprocessing.Array):
    """Packs the difficulty into two 32-bit integers. Little endian."""
    packed_diff[0] = diff >> 32
    packed_diff[1] = diff & 0xFFFFFFFF  # low 32 bits


def _hash_block_with_coldkey(block_bytes: bytes, coldkey_bytes: bytes) -> bytes:
    """Hashes the block with the hotkey using Keccak-256 to get 32 bytes"""
    kec = keccak.new(digest_bits=256)
    kec = kec.update(bytearray(block_bytes + coldkey_bytes))
    block_and_hotkey_hash_bytes = kec.digest()
    return block_and_hotkey_hash_bytes


def _update_curr_block(
    curr_diff: multiprocessing.Array,
    curr_block: multiprocessing.Array,
    curr_block_num: multiprocessing.Value,
    block_number: int,
    block_bytes: bytes,
    diff: int,
    coldkey_bytes: bytes,
    lock: multiprocessing.Lock,
):
    with lock:
        curr_block_num.value = block_number
        # Hash the block with the hotkey
        block_and_hotkey_hash_bytes = _hash_block_with_coldkey(
            block_bytes, coldkey_bytes
        )
        for i in range(32):
            curr_block[i] = block_and_hotkey_hash_bytes[i]
        _registration_diff_pack(diff, curr_diff)


def get_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # OSX does not have sched_getaffinity
        return os.cpu_count()


def _solve_for_coldkey_swap_difficulty_cpu(
    subtensor,
    wallet: "bittensor.wallet",
    old_coldkey: str,
    output_in_place: bool = True,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    n_samples: int = 10,
    alpha_: float = 0.80,
    log_verbose: bool = False,
) -> Optional[SwapPOWSolution]:
    """
    Solves the POW for a coldkey-swap using multiprocessing.
    Args:
        subtensor
            Subtensor to connect to for block information and to submit.
        wallet:
            wallet to use for coldkey-swap.
        old_coldkey: str
            The old coldkey to swap from.
        output_in_place: bool
            If true, prints the status in place. Otherwise, prints the status on a new line.
        num_processes: int
            Number of processes to use.
        update_interval: int
            Number of nonces to solve before updating block information.
        n_samples: int
            The number of samples of the hash_rate to keep for the EWMA
        alpha_: float
            The alpha for the EWMA for the hash_rate calculation
        log_verbose: bool
            If true, prints more verbose logging of the POW metrics.
    Note: The hash rate is calculated as an exponentially weighted moving average in order to make the measure more robust.
    """
    if num_processes == None:
        # get the number of allowed processes for this process
        num_processes = min(1, get_cpu_count())

    if update_interval is None:
        update_interval = 1_000_000  # Should be high because we have no age criteria

    limit = int(math.pow(2, 256)) - 1

    curr_block, curr_block_num, curr_diff = _Solver.create_shared_memory()

    # Establish communication queues
    stopEvent = multiprocessing.Event()
    stopEvent.clear()

    solution_queue = multiprocessing.Queue()
    finished_queues = [multiprocessing.Queue() for _ in range(num_processes)]
    check_block = multiprocessing.Lock()

    coldkey_bytes = ss58_address_to_bytes(old_coldkey)

    # Start consumers
    solvers = [
        _Solver(
            i,
            num_processes,
            update_interval,
            finished_queues[i],
            solution_queue,
            stopEvent,
            curr_block,
            curr_block_num,
            curr_diff,
            check_block,
            limit,
        )
        for i in range(num_processes)
    ]

    # Get first block
    block_number, block_hash = _get_block_with_retry(subtensor)
    base_difficulty, swap_attempts = _get_swap_difficulty_with_retry(
        subtensor, old_coldkey
    )

    # Calculate the (current) actual difficulty
    difficulty = _calculate_difficulty(base_difficulty, swap_attempts)
    old_difficulty = difficulty

    block_bytes = bytes.fromhex(block_hash[2:])
    # Set to current block
    _update_curr_block(
        curr_diff,
        curr_block,
        curr_block_num,
        block_number,
        block_bytes,
        difficulty,
        coldkey_bytes,
        check_block,
    )

    # Set new block events for each solver to start at the initial block
    for worker in solvers:
        worker.newBlockEvent.set()

    for worker in solvers:
        worker.start()  # start the solver processes

    start_time = time.time()  # time that the registration started
    time_last = start_time  # time that the last work blocks completed

    curr_stats = ColdkeySwapStatistics(
        time_spent_total=0.0,
        time_average=0.0,
        rounds_total=0,
        time_spent=0.0,
        hash_rate_perpetual=0.0,
        hash_rate=0.0,
        difficulty=difficulty,
        block_number=block_number,
        block_hash=block_hash,
    )

    start_time_perpetual = time.time()

    console = bittensor.__console__
    logger = ColdkeySwapStatisticsLogger(console, output_in_place)
    logger.start()

    solution = None

    hash_rates = [0] * n_samples  # The last n true hash_rates
    weights = [
        alpha_**i + alpha_ for i in range(n_samples)
    ]  # weights decay by alpha, but never reach zero

    while True:
        # Wait until a solver finds a solution
        try:
            solution = solution_queue.get(block=True, timeout=0.25)
            if solution is not None:
                break
        except Empty:
            # No solution found, try again
            pass

        # No need to check for new blocks, we don't have any age criteria
        # BUT, we should check for new difficulty, as it can change
        old_difficulty = _check_for_newest_difficulty_and_update(
            subtensor=subtensor,
            old_coldkey=old_coldkey,
            old_difficulty=old_difficulty,
            curr_diff=curr_diff,
            curr_block=curr_block,
            curr_block_num=curr_block_num,
            curr_stats=curr_stats,
            update_curr_block=_update_curr_block,
            check_block=check_block,
            solvers=solvers,
        )

        num_time = 0
        for finished_queue in finished_queues:
            try:
                proc_num = finished_queue.get(timeout=0.1)
                num_time += 1

            except Empty:
                continue

        time_now = time.time()  # get current time
        time_since_last = time_now - time_last  # get time since last work block(s)
        if num_time > 0 and time_since_last > 0.0:
            # create EWMA of the hash_rate to make measure more robust

            hash_rate_ = (num_time * update_interval) / time_since_last
            hash_rates.append(hash_rate_)
            hash_rates.pop(0)  # remove the 0th data point

            # Calculate weighted average, avoiding division by zero
            weighted_sum = sum([hash_rates[i] * weights[i] for i in range(n_samples)])
            weight_sum = sum(weights)
            curr_stats.hash_rate = weighted_sum / weight_sum if weight_sum > 0 else 0

            # update time last to now
            time_last = time_now

            curr_stats.time_average = (
                curr_stats.time_average * curr_stats.rounds_total
                + curr_stats.time_spent
            ) / (curr_stats.rounds_total + num_time)
            curr_stats.rounds_total += num_time

        # Update stats
        curr_stats.time_spent = time_since_last
        new_time_spent_total = time_now - start_time_perpetual
        curr_stats.hash_rate_perpetual = (
            curr_stats.rounds_total * update_interval
        ) / new_time_spent_total
        curr_stats.time_spent_total = new_time_spent_total

        # Update the logger
        logger.update(curr_stats, verbose=log_verbose)

    # exited while, solution contains the nonce or wallet is registered
    stopEvent.set()  # stop all other processes
    logger.stop()

    # terminate and wait for all solvers to exit
    _terminate_workers_and_wait_for_exit(solvers)

    return solution


@backoff.on_exception(backoff.constant, Exception, interval=1, max_tries=3)
def _get_block_with_retry(subtensor: "bittensor.subtensor") -> Tuple[int, bytes]:
    """
    Gets the current block number, and block hash from the substrate node.

    Args:
        subtensor (:obj:`bittensor.subtensor`, `required`):
            The subtensor object to use to get the block number, difficulty, and block hash.

    Returns:
        block_number (:obj:`int`):
            The current block number.

        block_hash (:obj:`bytes`):
            The current block hash.

    Raises:
        Exception: If the block hash is None.
        ValueError: If the difficulty is None.
    """
    block_number = subtensor.get_current_block()
    block_hash = subtensor.get_block_hash(block_number)
    if block_hash is None:
        raise Exception(
            "Network error. Could not connect to substrate to get block hash"
        )
    return block_number, block_hash


@backoff.on_exception(backoff.constant, Exception, interval=1, max_tries=3)
def _get_swap_difficulty_with_retry(subtensor, coldkey_address) -> Tuple[int, int]:
    max_retries = 3
    for _ in range(max_retries):
        try:
            base_difficulty = subtensor.get_base_difficulty()
            swap_attempts = len(
                subtensor.get_coldkey_swap_destinations(coldkey_address)
            )
            return base_difficulty, swap_attempts
        except Exception as e:
            bittensor.logging.warning(f"Failed to get swap difficulty: {e}")
            time.sleep(1)
    raise Exception("Failed to get swap difficulty after multiple attempts")


class _UsingSpawnStartMethod:
    def __init__(self, force: bool = False):
        self._old_start_method = None
        self._force = force

    def __enter__(self):
        self._old_start_method = multiprocessing.get_start_method(allow_none=True)
        if self._old_start_method == None:
            self._old_start_method = "spawn"  # default to spawn

        multiprocessing.set_start_method("spawn", force=self._force)

    def __exit__(self, *args):
        # restore the old start method
        multiprocessing.set_start_method(self._old_start_method, force=True)


def _check_for_newest_difficulty_and_update(
    subtensor,
    old_coldkey,
    old_difficulty,
    curr_diff,
    curr_block,
    curr_block_num,
    curr_stats: ColdkeySwapStatistics,
    update_curr_block,
    check_block,
    solvers,
):
    base_difficulty, swap_attempts = _get_swap_difficulty_with_retry(
        subtensor, old_coldkey
    )
    difficulty = _calculate_difficulty(base_difficulty, swap_attempts)
    if difficulty != old_difficulty:
        block_number, block_hash = _get_block_with_retry(subtensor)
        block_bytes = bytes.fromhex(block_hash[2:])

        coldkey_bytes = ss58_address_to_bytes(old_coldkey)
        update_curr_block(
            curr_diff,
            curr_block,
            curr_block_num,
            block_number,
            block_bytes,
            coldkey_bytes,
            check_block,
            difficulty,
        )
        curr_stats.block_number = block_number
        curr_stats.block_hash = block_hash
        curr_stats.difficulty = difficulty

        for s in solvers:
            s.newBlockEvent.set()
        return difficulty
    return old_difficulty


def _solve_for_coldkey_swap_difficulty_cuda(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    old_coldkey: str,
    output_in_place: bool = True,
    update_interval: Optional[int] = None,
    tpb: int = 512,
    dev_id: Union[List[int], int] = 0,
    n_samples: int = 10,
    alpha_: float = 0.80,
    log_verbose: bool = False,
) -> Optional[SwapPOWSolution]:
    """
    Solves the registration fast using CUDA
    Args:
        subtensor: bittensor.subtensor
            The subtensor node to grab blocks
        wallet: bittensor.wallet
            The wallet to register
        old_coldkey: str
            The old coldkey to swap from.
        output_in_place: bool
            If true, prints the output in place, otherwise prints to new lines
        update_interval: Optional[int] (default: 1_000_000)
            The number of nonces to try before checking for more blocks
        tpb: int
            The number of threads per block. CUDA param that should match the GPU capability
        dev_id: Union[List[int], int]
            The CUDA device IDs to execute the registration on, either a single device or a list of devices
        n_samples: int
            The number of samples of the hash_rate to keep for the EWMA
        alpha_: float
            The alpha for the EWMA for the hash_rate calculation
        log_verbose: bool
            If true, prints more verbose logging of the registration metrics.
    Note: The hash rate is calculated as an exponentially weighted moving average in order to make the measure more robust.
    """
    if isinstance(dev_id, int):
        dev_id = [dev_id]
    elif dev_id is None:
        dev_id = [0]

    if update_interval is None:
        # Can be high because we have no age requirements
        update_interval = 1_000_000

    if not torch.cuda.is_available():
        raise Exception("CUDA not available")

    limit = int(math.pow(2, 256)) - 1

    # Set mp start to use spawn so CUDA doesn't complain
    with _UsingSpawnStartMethod(force=True):
        curr_block, curr_block_num, curr_diff = _CUDASolver.create_shared_memory()

        ## Create a worker per CUDA device
        num_processes = len(dev_id)

        # Establish communication queues
        stopEvent = multiprocessing.Event()
        stopEvent.clear()
        solution_queue = multiprocessing.Queue()
        finished_queues = [multiprocessing.Queue() for _ in range(num_processes)]
        check_block = multiprocessing.Lock()

        coldkey_bytes = ss58_address_to_bytes(old_coldkey)
        # Start workers
        solvers = [
            _CUDASolver(
                i,
                num_processes,
                update_interval,
                finished_queues[i],
                solution_queue,
                stopEvent,
                curr_block,
                curr_block_num,
                curr_diff,
                check_block,
                limit,
                dev_id[i],
                tpb,
            )
            for i in range(num_processes)
        ]

        # Get first block
        block_number, block_hash = _get_block_with_retry(subtensor)
        base_difficulty, swap_attempts = _get_swap_difficulty_with_retry(
            subtensor, old_coldkey
        )

        # Calculate the (current) actual difficulty
        difficulty = _calculate_difficulty(base_difficulty, swap_attempts)
        old_difficulty = difficulty

        block_bytes = bytes.fromhex(block_hash[2:])

        # Set to current block
        _update_curr_block(
            curr_diff,
            curr_block,
            curr_block_num,
            block_number,
            block_bytes,
            difficulty,
            coldkey_bytes,
            check_block,
        )

        # Set new block events for each solver to start at the initial block
        for worker in solvers:
            worker.newBlockEvent.set()

        for worker in solvers:
            worker.start()  # start the solver processes

        start_time = time.time()  # time that the registration started
        time_last = start_time  # time that the last work blocks completed

        curr_stats = ColdkeySwapStatistics(
            time_spent_total=0.0,
            time_average=0.0,
            rounds_total=0,
            time_spent=0.0,
            hash_rate_perpetual=0.0,
            hash_rate=0.0,  # EWMA hash_rate (H/s)
            difficulty=difficulty,
            block_number=block_number,
            block_hash=block_hash,
        )

        start_time_perpetual = time.time()

        console = bittensor.__console__
        logger = ColdkeySwapStatisticsLogger(console, output_in_place)
        logger.start()

        hash_rates = [0] * n_samples  # The last n true hash_rates
        weights = [alpha_**i for i in range(n_samples)]  # weights decay by alpha

        solution = None
        while True:  # loop until solution is found
            # Wait until a solver finds a solution
            try:
                solution = solution_queue.get(block=True, timeout=0.15)
                if solution is not None:
                    break
            except Empty:
                # No solution found, try again
                pass

            # No need to check for new blocks, we don't have any age criteria
            # BUT, we should check for new difficulty, as it can change
            old_difficulty = _check_for_newest_difficulty_and_update(
                subtensor=subtensor,
                old_coldkey=old_coldkey,
                old_difficulty=old_difficulty,
                curr_diff=curr_diff,
                curr_block=curr_block,
                curr_block_num=curr_block_num,
                curr_stats=curr_stats,
                update_curr_block=_update_curr_block,
                check_block=check_block,
                solvers=solvers,
            )

            num_time = 0
            # Get times for each solver
            for finished_queue in finished_queues:
                try:
                    proc_num = finished_queue.get(timeout=0.1)
                    num_time += 1

                except Empty:
                    continue

            time_now = time.time()  # get current time
            time_since_last = time_now - time_last  # get time since last work block(s)
            if num_time > 0 and time_since_last > 0.0:
                # create EWMA of the hash_rate to make measure more robust

                hash_rate_ = (num_time * tpb * update_interval) / time_since_last
                hash_rates.append(hash_rate_)
                hash_rates.pop(0)  # remove the 0th data point
                curr_stats.hash_rate = sum(
                    [hash_rates[i] * weights[i] for i in range(n_samples)]
                ) / (sum(weights))

                # update time last to now
                time_last = time_now

                curr_stats.time_average = (
                    curr_stats.time_average * curr_stats.rounds_total
                    + curr_stats.time_spent
                ) / (curr_stats.rounds_total + num_time)
                curr_stats.rounds_total += num_time

            # Update stats
            curr_stats.time_spent = time_since_last
            new_time_spent_total = time_now - start_time_perpetual
            curr_stats.hash_rate_perpetual = (
                curr_stats.rounds_total * (tpb * update_interval)
            ) / new_time_spent_total
            curr_stats.time_spent_total = new_time_spent_total

            # Update the logger
            logger.update(curr_stats, verbose=log_verbose)

        # exited while, found_solution contains the nonce or wallet is registered

        stopEvent.set()  # stop all other processes
        logger.stop()

        # terminate and wait for all solvers to exit
        _terminate_workers_and_wait_for_exit(solvers)

        return solution


def _terminate_workers_and_wait_for_exit(
    workers: List[Union[multiprocessing.Process, multiprocessing.queues.Queue]],
) -> None:
    for worker in workers:
        if isinstance(worker, multiprocessing.queues.Queue):
            worker.join_thread()
        else:
            worker.join()
        worker.close()


def create_pow_for_coldkey_swap(
    subtensor,
    wallet,
    old_coldkey: str,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
    max_iterations: int = 1000000,
) -> Optional[SwapPOWSolution]:
    """
    Creates a proof of work for coldkey swap.

    Args:
        subtensor (bittensor.subtensor): Subtensor object for blockchain interaction.
        wallet (bittensor.wallet): Wallet object containing keys.
        old_coldkey (str): SS58 address of the old coldkey.
        output_in_place (bool, optional): Whether to print output in place. Defaults to True.
        cuda (bool, optional): Whether to use CUDA for PoW calculation. Defaults to False.
        dev_id (Union[List[int], int], optional): CUDA device ID(s). Defaults to 0.
        tpb (int, optional): Threads per block for CUDA. Defaults to 256.
        num_processes (int, optional): Number of CPU processes to use. Defaults to None.
        update_interval (int, optional): Interval to update block information. Defaults to None.
        log_verbose (bool, optional): Whether to log verbose output. Defaults to False.
        max_iterations (int, optional): Maximum number of iterations for CUDA solver. Defaults to 1000000.

    Returns:
        Optional[SwapPOWSolution]: The solved PoW solution, or None if not found.

    Raises:
        RuntimeError: If CUDA is not available when cuda=True.
    """
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return _solve_for_coldkey_swap_difficulty_cuda(
            subtensor,
            wallet,
            old_coldkey,
            output_in_place,
            update_interval,
            tpb,
            dev_id,
            log_verbose=log_verbose,
        )
    else:
        return _solve_for_coldkey_swap_difficulty_cpu(
            subtensor,
            wallet,
            old_coldkey,
            output_in_place,
            num_processes,
            update_interval,
            log_verbose=log_verbose,
        )
