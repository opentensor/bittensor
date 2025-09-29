"""This module provides utilities for solving Proof-of-Work (PoW) challenges in Bittensor network."""

import binascii
import functools
import hashlib
import math
import multiprocessing as mp
import os
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing.queues import Queue as QueueType
from queue import Empty, Full
from typing import Callable, Optional, Union, TYPE_CHECKING

import numpy
from Crypto.Hash import keccak

from bittensor.utils.btlogging import logging
from bittensor.utils.formatting import get_human_readable, millify
from bittensor.utils.registration.register_cuda import solve_cuda


def use_torch() -> bool:
    """Force the use of torch over numpy for certain operations."""
    return True if os.getenv("USE_TORCH") == "1" else False


def legacy_torch_api_compat(func):
    """
    Convert function operating on numpy Input&Output to legacy torch Input&Output API if `use_torch()` is True.

    Parameters:
        func: Function with numpy Input/Output to be decorated.

    Returns:
        decorated: Decorated function.
    """

    @functools.wraps(func)
    def decorated(*args, **kwargs):
        if use_torch():
            # if argument is a Torch tensor, convert it to numpy
            args = [
                arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ]
            kwargs = {
                key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
        ret = func(*args, **kwargs)
        if use_torch():
            # if return value is a numpy array, convert it to Torch tensor
            if isinstance(ret, numpy.ndarray):
                ret = torch.from_numpy(ret)
        return ret

    return decorated


@functools.cache
def _get_real_torch():
    try:
        import torch as _real_torch
    except ImportError:
        _real_torch = None
    return _real_torch


def log_no_torch_error():
    logging.error(
        "This command requires torch. You can install torch for bittensor"
        ' with `pip install bittensor[torch]` or `pip install ".[torch]"`'
        " if installing from source, and then run the command with USE_TORCH=1 {command}"
    )


class LazyLoadedTorch:
    """A lazy-loading proxy for the torch module."""

    def __bool__(self):
        return bool(_get_real_torch())

    def __getattr__(self, name):
        if real_torch := _get_real_torch():
            return getattr(real_torch, name)
        else:
            log_no_torch_error()
            raise ImportError("torch not installed")


if TYPE_CHECKING:
    import torch
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor_wallet import Wallet
else:
    torch = LazyLoadedTorch()


def _hex_bytes_to_u8_list(hex_bytes: bytes) -> list[int]:
    """ """
    return [int(hex_bytes[i : i + 2], 16) for i in range(0, len(hex_bytes), 2)]


def _create_seal_hash(block_and_hotkey_hash_bytes: bytes, nonce: int) -> bytes:
    """
    Create a cryptographic seal hash from the given block and hotkey hash bytes and nonce.

    This function generates a seal hash by combining the given block and hotkey hash bytes with a nonce.
    It first converts the nonce to a byte representation, then concatenates it with the first 64 hex characters of the
    block and hotkey hash bytes. The result is then hashed using SHA-256 followed by the Keccak-256 algorithm to produce
    the final seal hash.

    Parameters:
        block_and_hotkey_hash_bytes: The combined hash bytes of the block and hotkey.
        nonce: The nonce value used for hashing.

    Returns:
        The resulting seal hash.
    """
    nonce_bytes = binascii.hexlify(nonce.to_bytes(8, "little"))
    pre_seal = nonce_bytes + binascii.hexlify(block_and_hotkey_hash_bytes)[:64]
    seal_sh256 = hashlib.sha256(bytearray(_hex_bytes_to_u8_list(pre_seal))).digest()
    kec = keccak.new(digest_bits=256)
    seal = kec.update(seal_sh256).digest()
    return seal


def _seal_meets_difficulty(seal: bytes, difficulty: int, limit: int) -> bool:
    """Determines if a seal meets the specified difficulty."""
    seal_number = int.from_bytes(seal, "big")
    product = seal_number * difficulty
    return product < limit


@dataclass
class POWSolution:
    """A solution to the registration PoW problem."""

    nonce: int
    block_number: int
    difficulty: int
    seal: bytes

    def is_stale(self, subtensor: "Subtensor") -> bool:
        """
        Synchronous implementation. Returns True if the POW is stale.

        This means the block the POW is solved for is within 3 blocks of the current block.
        """
        return self.block_number < subtensor.get_current_block() - 3

    async def is_stale_async(self, subtensor: "AsyncSubtensor") -> bool:
        """
        Asynchronous implementation. Returns True if the POW is stale.

        This means the block the POW is solved for is within 3 blocks of the current block.
        """
        current_block = await subtensor.substrate.get_block_number(None)
        return self.block_number < current_block - 3


class UsingSpawnStartMethod:
    def __init__(self, force: bool = False):
        self._old_start_method = None
        self._force = force

    def __enter__(self):
        self._old_start_method = mp.get_start_method(allow_none=True)
        if self._old_start_method is None:
            self._old_start_method = "spawn"  # default to spawn

        mp.set_start_method("spawn", force=self._force)

    def __exit__(self, *args):
        # restore the old start method
        mp.set_start_method(self._old_start_method, force=True)


class _SolverBase(mp.Process):
    """
    A process that solves the registration PoW problem.

    Parameters:
        proc_num: The number of the process being created.
        num_proc: The total number of processes running.
        update_interval: The number of nonces to try to solve before checking for a new block.
        finished_queue: The queue to put the process number when a process finishes each update_interval. Used for
            calculating the average time per update_interval across all processes.
        solution_queue: The queue to put the solution the process has found during the pow solve.
        stopEvent: The event to set by the main process when all the solver processes should stop. The solver process
            will check for the event after each update_interval. The solver process will stop when the event is set.
            Used to stop the solver processes when a solution is found.
        curr_block: The array containing this process's current block hash. The main process will set the array to the
            new block hash when a new block is finalized in the network. The solver process will get the new block hash
            from this array when newBlockEvent is set.
        curr_block_num: The value containing this process's current block number. The main process will set the value to
            the new block number when a new block is finalized in the network. The solver process will get the new block
            number from this value when newBlockEvent is set.
        curr_diff: The array containing this process's current difficulty. The main process will set the array to the
            new difficulty when a new block is finalized in the network. The solver process will get the new difficulty
            from this array when newBlockEvent is set.
        check_block: The lock to prevent this process from getting the new block data while the main process is updating
            the data.
        limit: The limit of the pow solve for a valid solution.
    """

    proc_num: int
    num_proc: int
    update_interval: int
    finished_queue: "mp.Queue"
    solution_queue: "mp.Queue"
    # newBlockEvent: "mp.Event"
    newBlockEvent: "mp.Event"
    stopEvent: "mp.Event"
    hotkey_bytes: bytes
    curr_block: "mp.Array"
    curr_block_num: "mp.Value"
    curr_diff: "mp.Array"
    check_block: "mp.Lock"
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
        mp.Process.__init__(self, daemon=True)
        self.proc_num = proc_num
        self.num_proc = num_proc
        self.update_interval = update_interval
        self.finished_queue = finished_queue
        self.solution_queue = solution_queue
        self.newBlockEvent = mp.Event()
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
    def create_shared_memory() -> tuple["mp.Array", "mp.Value", "mp.Array"]:
        """Creates shared memory for the solver processes to use."""
        curr_block = mp.Array("h", 32, lock=True)  # byte array
        curr_block_num = mp.Value("i", 0, lock=True)  # int
        curr_diff = mp.Array("Q", [0, 0], lock=True)  # [high, low]

        return curr_block, curr_block_num, curr_diff


class Solver(_SolverBase):
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


class CUDASolver(_SolverBase):
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
) -> Optional["POWSolution"]:
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
        # Check if solution is valid (i.e., not -1)
        return POWSolution(solution, block_number, difficulty, seal)

    return None


def _solve_for_nonce_block(
    nonce_start: int,
    nonce_end: int,
    block_and_hotkey_hash_bytes: bytes,
    difficulty: int,
    limit: int,
    block_number: int,
) -> Optional["POWSolution"]:
    """Tries to solve the POW for a block of nonces (nonce_start, nonce_end)"""
    for nonce in range(nonce_start, nonce_end):
        # Create seal.
        seal = _create_seal_hash(block_and_hotkey_hash_bytes, nonce)

        # Check if seal meets difficulty
        if _seal_meets_difficulty(seal, difficulty, limit):
            # Found a solution, save it.
            return POWSolution(nonce, block_number, difficulty, seal)

    return None


def _registration_diff_unpack(packed_diff: "mp.Array") -> int:
    """Unpacks the packed two 32-bit integers into one 64-bit integer. Little endian."""
    return int(packed_diff[0] << 32 | packed_diff[1])


def _registration_diff_pack(diff: int, packed_diff: "mp.Array"):
    """Packs the difficulty into two 32-bit integers. Little endian."""
    packed_diff[0] = diff >> 32
    packed_diff[1] = diff & 0xFFFFFFFF  # low 32 bits


def _hash_block_with_hotkey(block_bytes: bytes, hotkey_bytes: bytes) -> bytes:
    """Hashes the block with the hotkey using Keccak-256 to get 32 bytes"""
    kec = keccak.new(digest_bits=256)
    kec = kec.update(bytearray(block_bytes + hotkey_bytes))
    block_and_hotkey_hash_bytes = kec.digest()
    return block_and_hotkey_hash_bytes


def update_curr_block(
    curr_diff: "mp.Array",
    curr_block: "mp.Array",
    curr_block_num: "mp.Value",
    block_number: int,
    block_bytes: bytes,
    diff: int,
    hotkey_bytes: bytes,
    lock: "mp.Lock",
):
    """
    Update the current block data with the provided block information and difficulty.

    This function updates the current block and its difficulty in a thread-safe manner. It sets the current block
    number, hashes the block with the hotkey, updates the current block bytes, and packs the difficulty.

    Parameters:
        curr_diff: Shared array to store the current difficulty.
        curr_block: Shared array to store the current block data.
        curr_block_num: Shared value to store the current block number.
        block_number: The block number to set as the current block number.
        block_bytes: The block data bytes to be hashed with the hotkey.
        diff: The difficulty value to be packed into the current difficulty array.
        hotkey_bytes: The hotkey bytes used for hashing the block.
        lock: A lock to ensure thread-safe updates.
    """
    with lock:
        curr_block_num.value = block_number
        # Hash the block with the hotkey
        block_and_hotkey_hash_bytes = _hash_block_with_hotkey(block_bytes, hotkey_bytes)
        for i in range(32):
            curr_block[i] = block_and_hotkey_hash_bytes[i]
        _registration_diff_pack(diff, curr_diff)


def get_cpu_count() -> int:
    """Returns the number of CPUs in the system."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # macOS does not have sched_getaffinity
        return os.cpu_count()


@dataclass
class RegistrationStatistics:
    """Statistics for a registration."""

    time_spent_total: float
    rounds_total: int
    time_average: float
    time_spent: float
    hash_rate_perpetual: float
    hash_rate: float
    difficulty: int
    block_number: int
    block_hash: str


class Status:
    def __init__(self, status: str):
        self._status = status

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, status: str):
        self._status = status


class Console:
    @staticmethod
    def status(status: str):
        return Status(status)

    @staticmethod
    def log(text: str):
        print(text)


class RegistrationStatisticsLogger:
    """Logs statistics for a registration."""

    status: Optional["Status"]

    def __init__(
        self,
        console: Optional["Console"] = None,
        output_in_place: bool = True,
    ) -> None:
        if console is None:
            console = Console()

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

    @classmethod
    def get_status_message(
        cls, stats: "RegistrationStatistics", verbose: bool = False
    ) -> str:
        """Generates the status message based on registration statistics."""
        message = (
            "Solving\n"
            + f"Time Spent (total): [bold white]{timedelta(seconds=stats.time_spent_total)}[/bold white]\n"
            + (
                f"Time Spent This Round: {timedelta(seconds=stats.time_spent)}\n"
                + f"Time Spent Average: {timedelta(seconds=stats.time_average)}\n"
                if verbose
                else ""
            )
            + f"Registration Difficulty: [bold white]{millify(stats.difficulty)}[/bold white]\n"
            + f"Iters (Inst/Perp): [bold white]{get_human_readable(stats.hash_rate, 'H')}/s / "
            + f"{get_human_readable(stats.hash_rate_perpetual, 'H')}/s[/bold white]\n"
            + f"Block Number: [bold white]{stats.block_number}[/bold white]\n"
            + f"Block Hash: [bold white]{stats.block_hash.encode('utf-8')}[/bold white]\n"
        )
        return message

    def update(self, stats: "RegistrationStatistics", verbose: bool = False) -> None:
        if self.status is not None:
            self.status.update(self.get_status_message(stats, verbose=verbose))
        else:
            self.console.log(self.get_status_message(stats, verbose=verbose))


def _solve_for_difficulty_fast(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    output_in_place: bool = True,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    n_samples: int = 10,
    alpha_: float = 0.80,
    log_verbose: bool = False,
) -> Optional[POWSolution]:
    """
    Solves the POW for registration using multiprocessing.

    Parameters:
        subtensor: Subtensor instance.
        wallet: wallet to use for registration.
        netuid: The netuid of the subnet to register to.
        output_in_place: If true, prints the status in place. Otherwise, prints the status on a new line.
        num_processes: Number of processes to use.
        update_interval: Number of nonces to solve before updating block information.
        n_samples: The number of samples of the hash_rate to keep for the EWMA.
        alpha_: The alpha for the EWMA for the hash_rate calculation.
        log_verbose: If true, prints more verbose logging of the registration metrics.

    Note:
        The hash rate is calculated as an exponentially weighted moving average in order to make the measure more robust.
        We can also modify the update interval to do smaller blocks of work, while still updating the block information
        after a different number of nonces, to increase the transparency of the process while still keeping the speed.
    """
    if num_processes is None:
        # get the number of allowed processes for this process
        num_processes = min(1, get_cpu_count())

    if update_interval is None:
        update_interval = 50_000

    limit = int(math.pow(2, 256)) - 1

    curr_block, curr_block_num, curr_diff = Solver.create_shared_memory()

    # Establish communication queues
    # See the Solver class for more information on the queues.
    stopEvent = mp.Event()
    stopEvent.clear()

    solution_queue = mp.Queue()
    finished_queues = [mp.Queue() for _ in range(num_processes)]
    check_block = mp.Lock()

    hotkey_bytes = (
        wallet.coldkeypub.public_key if netuid == -1 else wallet.hotkey.public_key
    )
    # Start consumers
    solvers = [
        Solver(
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
    block_number, difficulty, block_hash = _get_block_with_retry(
        subtensor=subtensor, netuid=netuid
    )

    block_bytes = bytes.fromhex(block_hash[2:])
    old_block_number = block_number
    # Set to current block
    update_curr_block(
        curr_diff,
        curr_block,
        curr_block_num,
        block_number,
        block_bytes,
        difficulty,
        hotkey_bytes,
        check_block,
    )

    # Set new block events for each solver to start at the initial block
    for worker in solvers:
        worker.newBlockEvent.set()

    for worker in solvers:
        worker.start()  # start the solver processes

    start_time = time.time()  # time that the registration started
    time_last = start_time  # time that the last work blocks completed

    curr_stats = RegistrationStatistics(
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

    logger = RegistrationStatisticsLogger(output_in_place=output_in_place)
    logger.start()

    solution = None

    hash_rates = [0] * n_samples  # The last n true hash_rates
    weights = [alpha_**i for i in range(n_samples)]  # weights decay by alpha

    while netuid == -1 or not subtensor.is_hotkey_registered(
        netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
    ):
        # Wait until a solver finds a solution
        try:
            solution = solution_queue.get(block=True, timeout=0.25)
            if solution is not None:
                break
        except Empty:
            # No solution found, try again
            pass

        # check for new block
        old_block_number = _check_for_newest_block_and_update(
            subtensor=subtensor,
            netuid=netuid,
            hotkey_bytes=hotkey_bytes,
            old_block_number=old_block_number,
            curr_diff=curr_diff,
            curr_block=curr_block,
            curr_block_num=curr_block_num,
            curr_stats=curr_stats,
            update_curr_block_=update_curr_block,
            check_block=check_block,
            solvers=solvers,
        )

        num_time = 0
        for finished_queue in finished_queues:
            try:
                finished_queue.get(timeout=0.1)
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
            curr_stats.rounds_total * update_interval
        ) / new_time_spent_total
        curr_stats.time_spent_total = new_time_spent_total

        # Update the logger
        logger.update(curr_stats, verbose=log_verbose)

    # exited while, solution contains the nonce or wallet is registered
    stopEvent.set()  # stop all other processes
    logger.stop()

    # terminate and wait for all solvers to exit
    terminate_workers_and_wait_for_exit(solvers)

    return solution


def _get_block_with_retry(subtensor: "Subtensor", netuid: int) -> tuple[int, int, str]:
    """
    Gets the current block number, difficulty, and block hash from the substrate node.

    Parameters:
        subtensor: The subtensor instance.
        netuid: The netuid of the network to get the block number, difficulty, and block hash from.

    Returns:
        tuple[int, int, bytes]
            - block_number: The current block number.
            - difficulty: The current difficulty of the subnet.
            - block_hash: The current block hash.

    Raises:
        Exception: If the block hash is None.
        ValueError: If the difficulty is None.
    """
    block_number = subtensor.get_current_block()
    difficulty = 1_000_000 if netuid == -1 else subtensor.difficulty(netuid=netuid)
    block_hash = subtensor.get_block_hash(block_number)
    if block_hash is None:
        raise Exception(
            "Network error. Could not connect to substrate to get block hash"
        )
    if difficulty is None:
        raise ValueError("Chain error. Difficulty is None")
    return block_number, difficulty, block_hash


def _check_for_newest_block_and_update(
    subtensor: "Subtensor",
    netuid: int,
    old_block_number: int,
    hotkey_bytes: bytes,
    curr_diff: "mp.Array",
    curr_block: "mp.Array",
    curr_block_num: "mp.Value",
    update_curr_block_: "Callable",
    check_block: "mp.Lock",
    solvers: Union[list["Solver"], list["CUDASolver"]],
    curr_stats: "RegistrationStatistics",
) -> int:
    """
    Checks for a new block and updates the current block information if a new block is found.

    Parameters:
        subtensor: Subtensor instance.
        netuid: The netuid to use for retrieving the difficulty.
        old_block_number: The old block number to check against.
        hotkey_bytes: The bytes of the hotkey's pubkey.
        curr_diff: The current difficulty as a multiprocessing array.
        curr_block: Where the current block is stored as a multiprocessing array.
        curr_block_num: Where the current block number is stored as a multiprocessing value.
        update_curr_block_: A function that updates the current block.
        check_block: A mp lock that is used to check for a new block.
        solvers: A list of solvers to update the current block for.
        curr_stats: The current registration statistics to update.

    Returns:
        The current block number.
    """
    block_number = subtensor.get_current_block()
    if block_number != old_block_number:
        old_block_number = block_number
        # update block information
        block_number, difficulty, block_hash = _get_block_with_retry(
            subtensor=subtensor, netuid=netuid
        )
        block_bytes = bytes.fromhex(block_hash[2:])

        update_curr_block_(
            curr_diff,
            curr_block,
            curr_block_num,
            block_number,
            block_bytes,
            difficulty,
            hotkey_bytes,
            check_block,
        )
        # Set new block events for each solver

        for worker in solvers:
            worker.newBlockEvent.set()

        # update stats
        curr_stats.block_number = block_number
        curr_stats.block_hash = block_hash
        curr_stats.difficulty = difficulty

    return old_block_number


def _solve_for_difficulty_fast_cuda(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    output_in_place: bool = True,
    update_interval: int = 50_000,
    tpb: int = 512,
    dev_id: Union[list[int], int] = 0,
    n_samples: int = 10,
    alpha_: float = 0.80,
    log_verbose: bool = False,
) -> Optional["POWSolution"]:
    """
    Solves the registration fast using CUDA.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor Wallet instance.
        netuid: The netuid of the subnet to register to.
        output_in_place: If true, prints the output in place, otherwise prints to new lines.
        update_interval: The number of nonces to try before checking for more blocks.
        tpb: The number of threads per block. CUDA param that should match the GPU capability
        dev_id: The CUDA device IDs to execute the registration on, either a single device or a list of devices.
        n_samples: The number of samples of the hash_rate to keep for the EWMA.
        alpha_: The alpha for the EWMA for the hash_rate calculation.
        log_verbose: If true, prints more verbose logging of the registration metrics.

    Note:
        The hash rate is calculated as an exponentially weighted moving average in order to make the measure more robust.
    """
    if isinstance(dev_id, int):
        dev_id = [dev_id]
    elif dev_id is None:
        dev_id = [0]

    if update_interval is None:
        update_interval = 50_000

    if not torch.cuda.is_available():
        raise Exception("CUDA not available")

    limit = int(math.pow(2, 256)) - 1

    # Set mp start to use spawn so CUDA doesn't complain
    with UsingSpawnStartMethod(force=True):
        curr_block, curr_block_num, curr_diff = CUDASolver.create_shared_memory()

        # Create a worker per CUDA device
        num_processes = len(dev_id)

        # Establish communication queues
        stopEvent = mp.Event()
        stopEvent.clear()
        solution_queue = mp.Queue()
        finished_queues = [mp.Queue() for _ in range(num_processes)]
        check_block = mp.Lock()

        hotkey_bytes = wallet.hotkey.public_key
        # Start workers
        solvers = [
            CUDASolver(
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
        block_number, difficulty, block_hash = _get_block_with_retry(
            subtensor=subtensor, netuid=netuid
        )

        block_bytes = bytes.fromhex(block_hash[2:])
        old_block_number = block_number

        # Set to current block
        update_curr_block(
            curr_diff,
            curr_block,
            curr_block_num,
            block_number,
            block_bytes,
            difficulty,
            hotkey_bytes,
            check_block,
        )

        # Set new block events for each solver to start at the initial block
        for worker in solvers:
            worker.newBlockEvent.set()

        for worker in solvers:
            worker.start()  # start the solver processes

        start_time = time.time()  # time that the registration started
        time_last = start_time  # time that the last work blocks completed

        curr_stats = RegistrationStatistics(
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

        logger = RegistrationStatisticsLogger(output_in_place=output_in_place)
        logger.start()

        hash_rates = [0] * n_samples  # The last n true hash_rates
        weights = [alpha_**i for i in range(n_samples)]  # weights decay by alpha

        solution = None
        while netuid == -1 or not subtensor.is_hotkey_registered(
            netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
        ):
            # Wait until a solver finds a solution
            try:
                solution = solution_queue.get(block=True, timeout=0.15)
                if solution is not None:
                    break
            except Empty:
                # No solution found, try again
                pass

            # check for new block
            old_block_number = _check_for_newest_block_and_update(
                subtensor=subtensor,
                netuid=netuid,
                hotkey_bytes=hotkey_bytes,
                curr_diff=curr_diff,
                curr_block=curr_block,
                curr_block_num=curr_block_num,
                old_block_number=old_block_number,
                curr_stats=curr_stats,
                update_curr_block_=update_curr_block,
                check_block=check_block,
                solvers=solvers,
            )

            num_time = 0
            # Get times for each solver
            for finished_queue in finished_queues:
                try:
                    finished_queue.get(timeout=0.1)
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
        terminate_workers_and_wait_for_exit(solvers)

        return solution


def terminate_workers_and_wait_for_exit(
    workers: list[Union[mp.Process, QueueType]],
) -> None:
    for worker in workers:
        if isinstance(worker, QueueType):
            worker.join_thread()
        else:
            try:
                worker.join(3.0)
            except subprocess.TimeoutExpired:
                worker.terminate()
        try:
            worker.close()
        except ValueError:
            worker.terminate()


def create_pow(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[list[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> Optional["POWSolution"]:
    """
    Creates a proof of work for the given subtensor and wallet.

    Parameters:
        subtensor: The Subtensor instance.
        wallet: The Bittensor Wallet instance.
        netuid: The netuid for the subnet to create a proof of work for.
        output_in_place: If true, prints the progress of the proof of work to the console in-place. Meaning the progress
            is printed on the same lines.
        cuda: If true, uses CUDA to solve the proof of work.
        dev_id: The CUDA device id(s) to use. If cuda is true and dev_id is a list, then multiple CUDA devices will be
            used to solve the proof of work.
        tpb: The number of threads per block to use when solving the proof of work. Should be a multiple of 32.
        num_processes: The number of processes to use when solving the proof of work. If None, then the number of
            processes is equal to the number of CPU cores.
        update_interval: The number of nonces to run before checking for a new block.
        log_verbose: If true, prints the progress of the proof of work more verbosely.

    Returns:
        The proof of work solution or None if the wallet is already registered or there is a different error.

    Raises:
       ValueError: If the subnet does not exist.
    """
    if netuid != -1:
        if not subtensor.subnet_exists(netuid=netuid):
            raise ValueError(f"Subnet {netuid} does not exist.")

    if cuda:
        logging.debug("Solve difficulty with CUDA.")
        solution: Optional[POWSolution] = _solve_for_difficulty_fast_cuda(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            output_in_place=output_in_place,
            dev_id=dev_id,
            tpb=tpb,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )
    else:
        logging.debug("Solve difficulty.")
        solution: Optional[POWSolution] = _solve_for_difficulty_fast(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            output_in_place=output_in_place,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )
    return solution
