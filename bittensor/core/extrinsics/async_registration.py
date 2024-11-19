import asyncio
import binascii
import functools
import hashlib
import io
import math
import multiprocessing as mp
import os
import random
import subprocess
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import Process, Event, Lock, Array, Value, Queue
from multiprocessing.queues import Queue as Queue_Type
from queue import Empty, Full
from typing import Optional, Union, TYPE_CHECKING, Callable, Any

import backoff
import numpy as np
from Crypto.Hash import keccak
from bittensor_wallet import Wallet
from rich.console import Console
from rich.status import Status
from substrateinterface.exceptions import SubstrateRequestException

from bittensor.utils import format_error_message, unlock_key
from bittensor.utils.btlogging import logging
from bittensor.utils.formatting import millify, get_human_readable

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor


# TODO: compair and remove existing code (bittensor.utils.registration)


def use_torch() -> bool:
    """Force the use of torch over numpy for certain operations."""
    return True if os.getenv("USE_TORCH") == "1" else False


def legacy_torch_api_compat(func: Callable):
    """
    Convert function operating on numpy Input&Output to legacy torch Input&Output API if `use_torch()` is True.

    Args:
        func: Function with numpy Input/Output to be decorated.

    Returns:
        Decorated function
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
            if isinstance(ret, np.ndarray):
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
    logging.info(
        "This command requires torch. You can install torch with `pip install torch` and run the command again."
    )


@dataclass
class POWSolution:
    """A solution to the registration PoW problem."""

    nonce: int
    block_number: int
    difficulty: int
    seal: bytes

    async def is_stale(self, subtensor: "AsyncSubtensor") -> bool:
        """
        Returns True if the POW is stale.

        This means the block the POW is solved for is within 3 blocks of the current block.
        """
        current_block = await subtensor.substrate.get_block_number(None)
        return self.block_number < current_block - 3


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


class RegistrationStatisticsLogger:
    """Logs statistics for a registration."""

    console: Console
    status: Optional[Status]

    def __init__(
        self, console_: Optional["Console"] = None, output_in_place: bool = True
    ) -> None:
        if console_ is None:
            console_ = Console()
        self.console = console_

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
        cls, stats: RegistrationStatistics, verbose: bool = False
    ) -> str:
        """Provides a message of the current status of the block solving as a str for a logger or stdout."""
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

    def update(self, stats: RegistrationStatistics, verbose: bool = False) -> None:
        """Passes the current status to the logger."""
        if self.status is not None:
            self.status.update(self.get_status_message(stats, verbose=verbose))
        else:
            self.console.log(self.get_status_message(stats, verbose=verbose))


class _SolverBase(Process):
    """
    A process that solves the registration PoW problem.

    Args:
        proc_num: The number of the process being created.
        num_proc: The total number of processes running.
        update_interval: The number of nonces to try to solve before checking for a new block.
        finished_queue: The queue to put the process number when a process finishes each update_interval. Used for calculating the average time per update_interval across all processes.
        solution_queue: The queue to put the solution the process has found during the pow solve.
        stop_event: The event to set by the main process when all the solver processes should stop. The solver process will check for the event after each update_interval. The solver process will stop when the event is set. Used to stop the solver processes when a solution is found.
        curr_block: The array containing this process's current block hash. The main process will set the array to the new block hash when a new block is finalized in the network. The solver process will get the new block hash from this array when newBlockEvent is set
        curr_block_num: The value containing this process's current block number. The main process will set the value to the new block number when a new block is finalized in the network. The solver process will get the new block number from this value when new_block_event is set.
        curr_diff: The array containing this process's current difficulty. The main process will set the array to the new difficulty when a new block is finalized in the network. The solver process will get the new difficulty from this array when newBlockEvent is set.
        check_block: The lock to prevent this process from getting the new block data while the main process is updating the data.
        limit: The limit of the pow solve for a valid solution.

    Returns:
        new_block_event: The event to set by the main process when a new block is finalized in the network. The solver process will check for the event after each update_interval. The solver process will get the new block hash and difficulty and start solving for a new nonce.
    """

    proc_num: int
    num_proc: int
    update_interval: int
    finished_queue: Queue_Type
    solution_queue: Queue_Type
    new_block_event: Event
    stop_event: Event
    hotkey_bytes: bytes
    curr_block: Array
    curr_block_num: Value
    curr_diff: Array
    check_block: Lock
    limit: int

    def __init__(
        self,
        proc_num,
        num_proc,
        update_interval,
        finished_queue,
        solution_queue,
        stop_event,
        curr_block,
        curr_block_num,
        curr_diff,
        check_block,
        limit,
    ):
        Process.__init__(self, daemon=True)
        self.proc_num = proc_num
        self.num_proc = num_proc
        self.update_interval = update_interval
        self.finished_queue = finished_queue
        self.solution_queue = solution_queue
        self.new_block_event = Event()
        self.new_block_event.clear()
        self.curr_block = curr_block
        self.curr_block_num = curr_block_num
        self.curr_diff = curr_diff
        self.check_block = check_block
        self.stop_event = stop_event
        self.limit = limit

    def run(self):
        raise NotImplementedError("_SolverBase is an abstract class")

    @staticmethod
    def create_shared_memory() -> tuple[Array, Value, Array]:
        """Creates shared memory for the solver processes to use."""
        curr_block = Array("h", 32, lock=True)  # byte array
        curr_block_num = Value("i", 0, lock=True)  # int
        curr_diff = Array("Q", [0, 0], lock=True)  # [high, low]

        return curr_block, curr_block_num, curr_diff


class _Solver(_SolverBase):
    """Performs POW Solution."""

    def run(self):
        block_number: int
        block_and_hotkey_hash_bytes: bytes
        block_difficulty: int
        nonce_limit = int(math.pow(2, 64)) - 1

        # Start at random nonce
        nonce_start = random.randint(0, nonce_limit)
        nonce_end = nonce_start + self.update_interval
        while not self.stop_event.is_set():
            if self.new_block_event.is_set():
                with self.check_block:
                    block_number = self.curr_block_num.value
                    block_and_hotkey_hash_bytes = bytes(self.curr_block)
                    block_difficulty = _registration_diff_unpack(self.curr_diff)

                self.new_block_event.clear()

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
    """Performs POW Solution using CUDA."""

    dev_id: int
    tpb: int

    def __init__(
        self,
        proc_num,
        num_proc,
        update_interval,
        finished_queue,
        solution_queue,
        stop_event,
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
            stop_event,
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
        while not self.stop_event.is_set():
            if self.new_block_event.is_set():
                with self.check_block:
                    block_number = self.curr_block_num.value
                    block_and_hotkey_hash_bytes = bytes(self.curr_block)
                    block_difficulty = _registration_diff_unpack(self.curr_diff)

                self.new_block_event.clear()

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


class LazyLoadedTorch:
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
else:
    torch = LazyLoadedTorch()


class MaxSuccessException(Exception):
    """Raised when the POW Solver has reached the max number of successful solutions."""


class MaxAttemptsException(Exception):
    """Raised when the POW Solver has reached the max number of attempts."""


async def is_hotkey_registered(
    subtensor: "AsyncSubtensor", netuid: int, hotkey_ss58: str
) -> bool:
    """Checks to see if the hotkey is registered on a given netuid"""
    _result = await subtensor.substrate.query(
        module="SubtensorModule",
        storage_function="Uids",
        params=[netuid, hotkey_ss58],
    )
    if _result is not None:
        return True
    else:
        return False


async def _check_for_newest_block_and_update(
    subtensor: "AsyncSubtensor",
    netuid: int,
    old_block_number: int,
    hotkey_bytes: bytes,
    curr_diff: Array,
    curr_block: Array,
    curr_block_num: Value,
    update_curr_block: "Callable",
    check_block: Lock,
    solvers: list[_Solver],
    curr_stats: "RegistrationStatistics",
) -> int:
    """
    Checks for a new block and updates the current block information if a new block is found.

    Args:
        subtensor: The subtensor object to use for getting the current block.
        netuid: The netuid to use for retrieving the difficulty.
        old_block_number: The old block number to check against.
        hotkey_bytes: The bytes of the hotkey's pubkey.
        curr_diff: The current difficulty as a multiprocessing array.
        curr_block: Where the current block is stored as a multiprocessing array.
        curr_block_num: Where the current block number is stored as a multiprocessing value.
        update_curr_block: A function that updates the current block.
        check_block: A mp lock that is used to check for a new block.
        solvers: A list of solvers to update the current block for.
        curr_stats: The current registration statistics to update.

    Returns:
        The current block number.
    """
    block_number = await subtensor.substrate.get_block_number(None)
    if block_number != old_block_number:
        old_block_number = block_number
        # update block information
        block_number, difficulty, block_hash = await _get_block_with_retry(
            subtensor=subtensor, netuid=netuid
        )
        block_bytes = bytes.fromhex(block_hash[2:])

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
        # Set new block events for each solver

        for worker in solvers:
            worker.new_block_event.set()

        # update stats
        curr_stats.block_number = block_number
        curr_stats.block_hash = block_hash
        curr_stats.difficulty = difficulty

    return old_block_number


async def _block_solver(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    num_processes: int,
    netuid: int,
    dev_id: list[int],
    tpb: int,
    update_interval: int,
    curr_block,
    curr_block_num,
    curr_diff,
    n_samples,
    alpha_,
    output_in_place,
    log_verbose,
    cuda: bool,
):
    """Shared code used by the Solvers to solve the POW solution."""
    limit = int(math.pow(2, 256)) - 1

    # Establish communication queues
    # See the _Solver class for more information on the queues.
    stop_event = Event()
    stop_event.clear()

    solution_queue = Queue()
    finished_queues = [Queue() for _ in range(num_processes)]
    check_block = Lock()

    hotkey_bytes = (
        wallet.coldkeypub.public_key if netuid == -1 else wallet.hotkey.public_key
    )

    if cuda:
        # Create a worker per CUDA device
        num_processes = len(dev_id)
        solvers = [
            _CUDASolver(
                i,
                num_processes,
                update_interval,
                finished_queues[i],
                solution_queue,
                stop_event,
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
    else:
        # Start consumers
        solvers = [
            _Solver(
                i,
                num_processes,
                update_interval,
                finished_queues[i],
                solution_queue,
                stop_event,
                curr_block,
                curr_block_num,
                curr_diff,
                check_block,
                limit,
            )
            for i in range(num_processes)
        ]

    # Get first block
    block_number, difficulty, block_hash = await _get_block_with_retry(
        subtensor=subtensor, netuid=netuid
    )

    block_bytes = bytes.fromhex(block_hash[2:])
    old_block_number = block_number
    # Set to current block
    _update_curr_block(
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
        worker.new_block_event.set()

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

    timeout = 0.15 if cuda else 0.15
    while netuid == -1 or not await is_hotkey_registered(
        subtensor, netuid, wallet.hotkey.ss58_address
    ):
        # Wait until a solver finds a solution
        try:
            solution = solution_queue.get(block=True, timeout=timeout)
            if solution is not None:
                break
        except Empty:
            # No solution found, try again
            pass

        # check for new block
        old_block_number = await _check_for_newest_block_and_update(
            subtensor=subtensor,
            netuid=netuid,
            hotkey_bytes=hotkey_bytes,
            old_block_number=old_block_number,
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
                finished_queue.get(timeout=0.1)
                num_time += 1

            except Empty:
                continue

        time_now = time.time()  # get current time
        time_since_last = time_now - time_last  # get time since last work block(s)
        if num_time > 0 and time_since_last > 0.0:
            # create EWMA of the hash_rate to make measure more robust

            if cuda:
                hash_rate_ = (num_time * tpb * update_interval) / time_since_last
            else:
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
        if cuda:
            curr_stats.hash_rate_perpetual = (
                curr_stats.rounds_total * (tpb * update_interval)
            ) / new_time_spent_total
        else:
            curr_stats.hash_rate_perpetual = (
                curr_stats.rounds_total * update_interval
            ) / new_time_spent_total
        curr_stats.time_spent_total = new_time_spent_total

        # Update the logger
        logger.update(curr_stats, verbose=log_verbose)

    # exited while, solution contains the nonce or wallet is registered
    stop_event.set()  # stop all other processes
    logger.stop()

    # terminate and wait for all solvers to exit
    _terminate_workers_and_wait_for_exit(solvers)

    return solution


async def _solve_for_difficulty_fast_cuda(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
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
    Solves the registration fast using CUDA

    Args:
        subtensor: The subtensor node to grab blocks
        wallet: The wallet to register
        netuid: The netuid of the subnet to register to.
        output_in_place: If true, prints the output in place, otherwise prints to new lines
        update_interval: The number of nonces to try before checking for more blocks
        tpb: The number of threads per block. CUDA param that should match the GPU capability
        dev_id: The CUDA device IDs to execute the registration on, either a single device or a list of devices
        n_samples: The number of samples of the hash_rate to keep for the EWMA
        alpha_: The alpha for the EWMA for the hash_rate calculation
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

    # Set mp start to use spawn so CUDA doesn't complain
    with _UsingSpawnStartMethod(force=True):
        curr_block, curr_block_num, curr_diff = _CUDASolver.create_shared_memory()

        solution = await _block_solver(
            subtensor=subtensor,
            wallet=wallet,
            num_processes=None,
            netuid=netuid,
            dev_id=dev_id,
            tpb=tpb,
            update_interval=update_interval,
            curr_block=curr_block,
            curr_block_num=curr_block_num,
            curr_diff=curr_diff,
            n_samples=n_samples,
            alpha_=alpha_,
            output_in_place=output_in_place,
            log_verbose=log_verbose,
            cuda=True,
        )

        return solution


async def _solve_for_difficulty_fast(
    subtensor,
    wallet: Wallet,
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

    Args:
        subtensor: Subtensor to connect to for block information and to submit.
        wallet: wallet to use for registration.
        netuid: The netuid of the subnet to register to.
        output_in_place: If true, prints the status in place. Otherwise, prints the status on a new line.
        num_processes: Number of processes to use.
        update_interval: Number of nonces to solve before updating block information.
        n_samples: The number of samples of the hash_rate to keep for the EWMA
        alpha_: The alpha for the EWMA for the hash_rate calculation
        log_verbose: If true, prints more verbose logging of the registration metrics.

    Notes:
        The hash rate is calculated as an exponentially weighted moving average in order to make the measure more robust.
        We can also modify the update interval to do smaller blocks of work, while still updating the block information after a different number of nonces, to increase the transparency of the process while still keeping the speed.
    """
    if not num_processes:
        # get the number of allowed processes for this process
        num_processes = min(1, get_cpu_count())

    if update_interval is None:
        update_interval = 50_000

    curr_block, curr_block_num, curr_diff = _Solver.create_shared_memory()

    solution = await _block_solver(
        subtensor=subtensor,
        wallet=wallet,
        num_processes=num_processes,
        netuid=netuid,
        dev_id=None,
        tpb=None,
        update_interval=update_interval,
        curr_block=curr_block,
        curr_block_num=curr_block_num,
        curr_diff=curr_diff,
        n_samples=n_samples,
        alpha_=alpha_,
        output_in_place=output_in_place,
        log_verbose=log_verbose,
        cuda=False,
    )

    return solution


def _terminate_workers_and_wait_for_exit(
    workers: list[Union[Process, Queue_Type]],
) -> None:
    for worker in workers:
        if isinstance(worker, Queue_Type):
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


# TODO verify this works with async
@backoff.on_exception(backoff.constant, Exception, interval=1, max_tries=3)
async def _get_block_with_retry(
    subtensor: "AsyncSubtensor", netuid: int
) -> tuple[int, int, bytes]:
    """
    Gets the current block number, difficulty, and block hash from the substrate node.

    Args:
        subtensor: The subtensor object to use to get the block number, difficulty, and block hash.
        netuid: The netuid of the network to get the block number, difficulty, and block hash from.

    Returns:
        The current block number, difficulty of the subnet, block hash

    Raises:
        Exception: If the block hash is None.
        ValueError: If the difficulty is None.
    """
    block_number = await subtensor.substrate.get_block_number(None)
    block_hash = await subtensor.substrate.get_block_hash(
        block_number
    )  # TODO check if I need to do all this
    try:
        difficulty = (
            1_000_000
            if netuid == -1
            else int(
                await subtensor.get_hyperparameter(
                    param_name="Difficulty", netuid=netuid, block_hash=block_hash
                )
            )
        )
    except TypeError:
        raise ValueError("Chain error. Difficulty is None")
    except SubstrateRequestException:
        raise Exception(
            "Network error. Could not connect to substrate to get block hash"
        )
    return block_number, difficulty, block_hash


def _registration_diff_unpack(packed_diff: Array) -> int:
    """Unpacks the packed two 32-bit integers into one 64-bit integer. Little endian."""
    return int(packed_diff[0] << 32 | packed_diff[1])


def _registration_diff_pack(diff: int, packed_diff: Array):
    """Packs the difficulty into two 32-bit integers. Little endian."""
    packed_diff[0] = diff >> 32
    packed_diff[1] = diff & 0xFFFFFFFF  # low 32 bits


class _UsingSpawnStartMethod:
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


async def create_pow(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    netuid: int,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[list[int], int] = 0,
    tpb: int = 256,
    num_processes: int = None,
    update_interval: int = None,
    log_verbose: bool = False,
) -> Optional[dict[str, Any]]:
    """
    Creates a proof of work for the given subtensor and wallet.

    Args:
        subtensor: The subtensor to create a proof of work for.
        wallet: The wallet to create a proof of work for.
        netuid: The netuid for the subnet to create a proof of work for.
        output_in_place: If true, prints the progress of the proof of work to the console in-place. Meaning the progress is printed on the same lines.
        cuda: If true, uses CUDA to solve the proof of work.
        dev_id: The CUDA device id(s) to use. If cuda is true and dev_id is a list, then multiple CUDA devices will be used to solve the proof of work.
        tpb: The number of threads per block to use when solving the proof of work. Should be a multiple of 32.
        num_processes: The number of processes to use when solving the proof of work. If None, then the number of processes is equal to the number of CPU cores.
        update_interval: The number of nonces to run before checking for a new block.
        log_verbose: If true, prints the progress of the proof of work more verbosely.

    Returns:
        The proof of work solution or None if the wallet is already registered or there is a different error.

    Raises:
        ValueError: If the subnet does not exist.
    """
    if netuid != -1:
        if not await subtensor.subnet_exists(netuid=netuid):
            raise ValueError(f"Subnet {netuid} does not exist")

    if cuda:
        solution: Optional[POWSolution] = await _solve_for_difficulty_fast_cuda(
            subtensor,
            wallet,
            netuid=netuid,
            output_in_place=output_in_place,
            dev_id=dev_id,
            tpb=tpb,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )
    else:
        solution: Optional[POWSolution] = await _solve_for_difficulty_fast(
            subtensor,
            wallet,
            netuid=netuid,
            output_in_place=output_in_place,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )

    return solution


def _solve_for_nonce_block_cuda(
    nonce_start: int,
    update_interval: int,
    block_and_hotkey_hash_bytes: bytes,
    difficulty: int,
    limit: int,
    block_number: int,
    dev_id: int,
    tpb: int,
) -> Optional[POWSolution]:
    """
    Tries to solve the POW on a CUDA device for a block of nonces (nonce_start, nonce_start + update_interval * tpb
    """
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
) -> Optional[POWSolution]:
    """
    Tries to solve the POW for a block of nonces (nonce_start, nonce_end)
    """
    for nonce in range(nonce_start, nonce_end):
        # Create seal.
        seal = _create_seal_hash(block_and_hotkey_hash_bytes, nonce)

        # Check if seal meets difficulty
        if _seal_meets_difficulty(seal, difficulty, limit):
            # Found a solution, save it.
            return POWSolution(nonce, block_number, difficulty, seal)

    return None


class CUDAException(Exception):
    """An exception raised when an error occurs in the CUDA environment."""


def _hex_bytes_to_u8_list(hex_bytes: bytes):
    hex_chunks = [int(hex_bytes[i : i + 2], 16) for i in range(0, len(hex_bytes), 2)]
    return hex_chunks


def _create_seal_hash(block_and_hotkey_hash_bytes: bytes, nonce: int) -> bytes:
    """
    Create a cryptographic seal hash from the given block and hotkey hash bytes and nonce.

    This function generates a seal hash by combining the given block and hotkey hash bytes with a nonce.
    It first converts the nonce to a byte representation, then concatenates it with the first 64 hex characters of the block and hotkey hash bytes. The result is then hashed using SHA-256 followed by the Keccak-256 algorithm to produce the final seal hash.

    Args:
        block_and_hotkey_hash_bytes (bytes): The combined hash bytes of the block and hotkey.
        nonce (int): The nonce value used for hashing.

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
    """Determines if a seal meets the specified difficulty"""
    seal_number = int.from_bytes(seal, "big")
    product = seal_number * difficulty
    return product < limit


def _hash_block_with_hotkey(block_bytes: bytes, hotkey_bytes: bytes) -> bytes:
    """Hashes the block with the hotkey using Keccak-256 to get 32 bytes"""
    kec = keccak.new(digest_bits=256)
    kec = kec.update(bytearray(block_bytes + hotkey_bytes))
    block_and_hotkey_hash_bytes = kec.digest()
    return block_and_hotkey_hash_bytes


def _update_curr_block(
    curr_diff: Array,
    curr_block: Array,
    curr_block_num: Value,
    block_number: int,
    block_bytes: bytes,
    diff: int,
    hotkey_bytes: bytes,
    lock: Lock,
):
    """
    Update the current block data with the provided block information and difficulty.

    This function updates the current block and its difficulty in a thread-safe manner. It sets the current block
    number, hashes the block with the hotkey, updates the current block bytes, and packs the difficulty.

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
    block_hash: bytes


def solve_cuda(
    nonce_start: np.int64,
    update_interval: np.int64,
    tpb: int,
    block_and_hotkey_hash_bytes: bytes,
    difficulty: int,
    limit: int,
    dev_id: int = 0,
) -> tuple[np.int64, bytes]:
    """
    Solves the PoW problem using CUDA.

    nonce_start: Starting nonce.
    update_interval: Number of nonces to solve before updating block information.
    tpb: Threads per block.
    block_and_hotkey_hash_bytes: Keccak(Bytes of the block hash + bytes of the hotkey) 64 bytes.
    difficulty: Difficulty of the PoW problem.
    limit: Upper limit of the nonce.
    dev_id: The CUDA device ID

    :return: (nonce, seal) corresponding to the solution. Returns -1 for nonce if no solution is found.
    """

    try:
        import cubit
    except ImportError:
        raise ImportError("Please install cubit")

    upper = int(limit // difficulty)

    upper_bytes = upper.to_bytes(32, byteorder="little", signed=False)

    # Call cython function
    # int blockSize, uint64 nonce_start, uint64 update_interval, const unsigned char[:] limit,
    # const unsigned char[:] block_bytes, int dev_id
    block_and_hotkey_hash_hex = binascii.hexlify(block_and_hotkey_hash_bytes)[:64]

    solution = cubit.solve_cuda(
        tpb,
        nonce_start,
        update_interval,
        upper_bytes,
        block_and_hotkey_hash_hex,
        dev_id,
    )  # 0 is first GPU
    seal = None
    if solution != -1:
        seal = _create_seal_hash(block_and_hotkey_hash_hex, solution)
        if _seal_meets_difficulty(seal, difficulty, limit):
            return solution, seal
        else:
            return -1, b"\x00" * 32

    return solution, seal


def reset_cuda():
    """
    Resets the CUDA environment.
    """
    try:
        import cubit
    except ImportError:
        raise ImportError("Please install cubit")

    cubit.reset_cuda()


def log_cuda_errors() -> str:
    """
    Logs any CUDA errors.
    """
    try:
        import cubit
    except ImportError:
        raise ImportError("Please install cubit")

    f = io.StringIO()
    with redirect_stdout(f):
        cubit.log_cuda_errors()

    s = f.getvalue()

    return s


async def register_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[list[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> bool:
    """Registers the wallet to the chain.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): initialized AsyncSubtensor object to use for chain interactions
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        netuid (int): The ``netuid`` of the subnet to register on.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        max_allowed_attempts (int): Maximum number of attempts to register the wallet.
        output_in_place (bool): Whether the POW solving should be outputted to the console as it goes along.
        cuda (bool): If `True`, the wallet should be registered using CUDA device(s).
        dev_id: The CUDA device id to use, or a list of device ids.
        tpb: The number of threads per block (CUDA).
        num_processes: The number of processes to use to register.
        update_interval: The number of nonces to solve between updates.
        log_verbose: If `True`, the registration process will log more information.

    Returns:
        `True` if extrinsic was finalized or included in the block. If we did not wait for finalization/inclusion, the response is `True`.
    """

    logging.debug("Checking subnet status")
    if not await subtensor.subnet_exists(netuid):
        logging.error(
            f":cross_mark: <red>Failed error:</red> subnet <blue>{netuid}</blue> does not exist."
        )
        return False

    logging.info(
        f":satellite: <magenta>Checking Account on subnet</magenta> <blue>{netuid}</blue> <magenta>...</magenta>"
    )
    neuron = await subtensor.get_neuron_for_pubkey_and_subnet(
        hotkey_ss58=wallet.hotkey.ss58_address,
        netuid=netuid,
    )
    if not neuron.is_null:
        logging.debug(
            f"Wallet <green>{wallet}</green> is already registered on subnet <blue>{neuron.netuid}</blue> with uid<blue>{neuron.uid}</blue>."
        )
        return True

    if not torch:
        log_no_torch_error()
        return False

    # Attempt rolling registration.
    attempts = 1
    pow_result: Optional[POWSolution]
    while True:
        logging.info(
            f":satellite: <magenta>Registering...</magenta> <blue>({attempts}/{max_allowed_attempts})</blue>"
        )
        # Solve latest POW.
        if cuda:
            if not torch.cuda.is_available():
                return False
            pow_result = await create_pow(
                subtensor,
                wallet,
                netuid,
                output_in_place,
                cuda=cuda,
                dev_id=dev_id,
                tpb=tpb,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )
        else:
            pow_result = await create_pow(
                subtensor,
                wallet,
                netuid,
                output_in_place,
                cuda=cuda,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )

        # pow failed
        if not pow_result:
            # might be registered already on this subnet
            is_registered = await is_hotkey_registered(
                subtensor, netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
            )
            if is_registered:
                logging.error(
                    f":white_heavy_check_mark: <green>Already registered on netuid:</green> <blue>{netuid}</blue>"
                )
                return True

        # pow successful, proceed to submit pow to chain for registration
        else:
            logging.info(":satellite: <magenta>Submitting POW...</magenta>")
            # check if pow result is still valid
            while not await pow_result.is_stale(subtensor=subtensor):
                call = await subtensor.substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="register",
                    call_params={
                        "netuid": netuid,
                        "block_number": pow_result.block_number,
                        "nonce": pow_result.nonce,
                        "work": [int(byte_) for byte_ in pow_result.seal],
                        "hotkey": wallet.hotkey.ss58_address,
                        "coldkey": wallet.coldkeypub.ss58_address,
                    },
                )
                extrinsic = await subtensor.substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.hotkey
                )
                response = await subtensor.substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                if not wait_for_finalization and not wait_for_inclusion:
                    success, err_msg = True, ""
                else:
                    await response.process_events()
                    success = await response.is_success
                    if not success:
                        success, err_msg = (
                            False,
                            format_error_message(
                                await response.error_message,
                                substrate=subtensor.substrate,
                            ),
                        )
                        # Look error here
                        # https://github.com/opentensor/subtensor/blob/development/pallets/subtensor/src/errors.rs

                        if "HotKeyAlreadyRegisteredInSubNet" in err_msg:
                            logging.info(
                                f":white_heavy_check_mark: <green>Already Registered on subnet:</green> <blue>{netuid}</blue>."
                            )
                            return True
                        logging.error(f":cross_mark: <red>Failed</red>: {err_msg}")
                        await asyncio.sleep(0.5)

                # Successful registration, final check for neuron and pubkey
                if success:
                    logging.info(":satellite: Checking Registration status...")
                    is_registered = await is_hotkey_registered(
                        subtensor,
                        netuid=netuid,
                        hotkey_ss58=wallet.hotkey.ss58_address,
                    )
                    if is_registered:
                        logging.success(
                            ":white_heavy_check_mark: <green>Registered</green>"
                        )
                        return True
                    else:
                        # neuron not found, try again
                        logging.error(
                            ":cross_mark: <red>Unknown error. Neuron not found.</red>"
                        )
                        continue
            else:
                # Exited loop because pow is no longer valid.
                logging.error("<red>POW is stale.</red>")
                # Try again.
                continue

        if attempts < max_allowed_attempts:
            # Failed registration, retry pow
            attempts += 1
            logging.error(
                f":satellite: <magenta>Failed registration, retrying pow ...</magenta> <blue>({attempts}/{max_allowed_attempts})</blue>"
            )
        else:
            # Failed to register after max attempts.
            logging.error("<red>No more attempts.</red>")
            return False


async def run_faucet_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: int = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
    max_successes: int = 3,
) -> tuple[bool, str]:
    """Runs a continual POW to get a faucet of TAO on the test net.

    Args:
        subtensor: The subtensor interface object used to run the extrinsic
        wallet: Bittensor wallet object.
        wait_for_inclusion: If set, waits for the extrinsic to enter a block before returning `True`, or returns `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization: If set, waits for the extrinsic to be finalized on the chain before returning `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        max_allowed_attempts: Maximum number of attempts to register the wallet.
        output_in_place: Whether to output logging data as the process runs.
        cuda: If `True`, the wallet should be registered using CUDA device(s).
        dev_id: The CUDA device id to use
        tpb: The number of threads per block (CUDA).
        num_processes: The number of processes to use to register.
        update_interval: The number of nonces to solve between updates.
        log_verbose: If `True`, the registration process will log more information.
        max_successes: The maximum number of successful faucet runs for the wallet.

    Returns:
        `True` if extrinsic was finalized or included in the block. If we did not wait for finalization/inclusion, the response is also `True`
    """

    if not torch:
        log_no_torch_error()
        return False, "Requires torch"

    # Unlock coldkey
    if not (unlock := unlock_key(wallet)).success:
        return False, unlock.message

    # Get previous balance.
    old_balance = await subtensor.get_balance(wallet.coldkeypub.ss58_address)

    # Attempt rolling registration.
    attempts = 1
    successes = 1
    while True:
        try:
            pow_result = None
            while pow_result is None or await pow_result.is_stale(subtensor=subtensor):
                # Solve latest POW.
                if cuda:
                    if not torch.cuda.is_available():
                        return False, "CUDA is not available."
                    pow_result: Optional[POWSolution] = await create_pow(
                        subtensor,
                        wallet,
                        -1,
                        output_in_place,
                        cuda=cuda,
                        dev_id=dev_id,
                        tpb=tpb,
                        num_processes=num_processes,
                        update_interval=update_interval,
                        log_verbose=log_verbose,
                    )
                else:
                    pow_result: Optional[POWSolution] = await create_pow(
                        subtensor,
                        wallet,
                        -1,
                        output_in_place,
                        cuda=cuda,
                        num_processes=num_processes,
                        update_interval=update_interval,
                        log_verbose=log_verbose,
                    )
            call = await subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="faucet",
                call_params={
                    "block_number": pow_result.block_number,
                    "nonce": pow_result.nonce,
                    "work": [int(byte_) for byte_ in pow_result.seal],
                },
            )
            extrinsic = await subtensor.substrate.create_signed_extrinsic(
                call=call, keypair=wallet.coldkey
            )
            response = await subtensor.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # process if registration successful, try again if pow is still valid
            await response.process_events()
            if not await response.is_success:
                logging.error(
                    f":cross_mark: <red>Failed</red>: {format_error_message(await response.error_message, subtensor.substrate)}"
                )
                if attempts == max_allowed_attempts:
                    raise MaxAttemptsException
                attempts += 1
                # Wait a bit before trying again
                time.sleep(1)

            # Successful registration
            else:
                new_balance = await subtensor.get_balance(
                    wallet.coldkeypub.ss58_address
                )
                logging.info(
                    f"Balance: <blue>{old_balance[wallet.coldkeypub.ss58_address]}</blue> :arrow_right: <green>{new_balance[wallet.coldkeypub.ss58_address]}</green>"
                )
                old_balance = new_balance

                if successes == max_successes:
                    raise MaxSuccessException

                attempts = 1  # Reset attempts on success
                successes += 1

        except KeyboardInterrupt:
            return True, "Done"

        except MaxSuccessException:
            return True, f"Max successes reached: {3}"

        except MaxAttemptsException:
            return False, f"Max attempts reached: {max_allowed_attempts}"
