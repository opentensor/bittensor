import bittensor
from bittensor.utils.registration import (
    POWSolution,
    _create_seal_hash,
    _seal_meets_difficulty,
)
import multiprocessing
import time
import math
from typing import Optional, Union, List
from queue import Empty
import torch

from rich import console as rich_console
from rich import status as rich_status
from datetime import timedelta
from .formatting import get_human_readable, millify

# import cubit
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


def _solve_for_coldkey_swap_difficulty_cpu(
    subtensor,
    wallet,
    base_difficulty: int,
    swap_attempts: int,
    output_in_place: bool = True,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> Optional[POWSolution]:
    if num_processes is None:
        num_processes = min(1, multiprocessing.cpu_count())

    if update_interval is None:
        update_interval = 50_000

    limit = int(math.pow(2, 256)) - 1

    curr_block, curr_block_num = _create_shared_memory()

    stopEvent = multiprocessing.Event()
    stopEvent.clear()

    solution_queue = multiprocessing.Queue()
    finished_queues = [multiprocessing.Queue() for _ in range(num_processes)]
    check_block = multiprocessing.Lock()

    hotkey_bytes = wallet.hotkey.public_key

    # Calculate the actual difficulty
    difficulty = base_difficulty * (2**swap_attempts)

    solvers = [
        _ColdkeySwapSolver(
            i,
            num_processes,
            update_interval,
            finished_queues[i],
            solution_queue,
            stopEvent,
            curr_block,
            curr_block_num,
            difficulty,
            check_block,
            limit,
        )
        for i in range(num_processes)
    ]

    block_number, block_hash = _get_block_with_retry(subtensor)

    block_bytes = bytes.fromhex(block_hash[2:])
    old_block_number = block_number
    _update_curr_block(
        curr_block,
        curr_block_num,
        block_number,
        block_bytes,
        hotkey_bytes,
        check_block,
    )

    for worker in solvers:
        worker.newBlockEvent.set()
        worker.start()

    start_time = time.time()
    time_last = start_time

    curr_stats = ColdkeySwapStatistics(difficulty, block_number, block_hash)

    console = bittensor.__console__
    logger = ColdkeySwapStatisticsLogger(console, output_in_place)
    logger.start()

    solution = None

    while True:
        try:
            solution = solution_queue.get(block=True, timeout=0.25)
            if solution is not None:
                break
        except Empty:
            pass

        old_block_number = _check_for_newest_block_and_update(
            subtensor=subtensor,
            hotkey_bytes=hotkey_bytes,
            old_block_number=old_block_number,
            curr_block=curr_block,
            curr_block_num=curr_block_num,
            curr_stats=curr_stats,
            update_curr_block=_update_curr_block,
            check_block=check_block,
            solvers=solvers,
        )

        num_time = sum(1 for q in finished_queues if not q.empty())

        time_now = time.time()
        time_since_last = time_now - time_last
        if num_time > 0 and time_since_last > 0.0:
            hash_rate = (num_time * update_interval) / time_since_last
            curr_stats.hash_rate = hash_rate
            time_last = time_now

            curr_stats.time_average = (
                curr_stats.time_average * curr_stats.rounds_total
                + curr_stats.time_spent
            ) / (curr_stats.rounds_total + num_time)
            curr_stats.rounds_total += num_time

        curr_stats.time_spent = time_since_last
        new_time_spent_total = time_now - start_time
        curr_stats.hash_rate_perpetual = (
            curr_stats.rounds_total * update_interval
        ) / new_time_spent_total
        curr_stats.time_spent_total = new_time_spent_total

        logger.update(curr_stats, verbose=log_verbose)

    stopEvent.set()
    logger.stop()

    for worker in solvers:
        worker.join()

    return solution


def _solve_for_coldkey_swap_difficulty_cuda(
    subtensor,
    wallet,
    base_difficulty: int,
    swap_attempts: int,
    output_in_place: bool,
    dev_id: Union[List[int], int],
    tpb: int,
    num_processes: Optional[int],
    update_interval: Optional[int],
    log_verbose: bool,
    max_iterations: int = 1000000,  # Add a maximum number of iterations
) -> Optional[POWSolution]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if update_interval is None:
        update_interval = 50_000

    limit = int(math.pow(2, 256)) - 1

    hotkey_bytes = wallet.hotkey.public_key

    # Calculate the actual difficulty
    difficulty = base_difficulty * (2**swap_attempts)

    block_number, block_hash = _get_block_with_retry(subtensor)
    block_bytes = bytes.fromhex(block_hash[2:])

    block_and_hotkey_hash = _create_block_and_hotkey_hash(block_bytes, hotkey_bytes)

    nonce_start = 0
    console = bittensor.__console__
    logger = ColdkeySwapStatisticsLogger(console, output_in_place)
    logger.start()

    start_time = time.time()
    last_update_time = start_time

    iteration_count = 0

    try:
        while iteration_count < max_iterations:
            solution, seal = cubit.solve_cuda(
                tpb,
                nonce_start,
                update_interval,
                limit.to_bytes(32, "big"),
                block_and_hotkey_hash,
                dev_id,
            )

            current_time = time.time()
            elapsed_time = current_time - last_update_time
            total_time = current_time - start_time

            if solution != -1:
                logger.stop()
                return POWSolution(solution, block_number, difficulty, seal)

            hash_rate = update_interval / elapsed_time
            total_hashes = nonce_start + update_interval

            curr_stats = ColdkeySwapStatistics(difficulty, block_number, block_hash)
            curr_stats.time_spent_total = total_time
            curr_stats.time_average = total_time
            curr_stats.rounds_total = total_hashes // update_interval
            curr_stats.time_spent = elapsed_time
            curr_stats.hash_rate_perpetual = total_hashes / total_time
            curr_stats.hash_rate = hash_rate

            logger.update(curr_stats, verbose=log_verbose)

            nonce_start += update_interval
            last_update_time = current_time

            new_block_number, new_block_hash = _get_block_with_retry(subtensor)
            if new_block_number != block_number:
                block_number = new_block_number
                block_hash = new_block_hash
                block_bytes = bytes.fromhex(block_hash[2:])
                block_and_hotkey_hash = _create_block_and_hotkey_hash(
                    block_bytes, hotkey_bytes
                )
                nonce_start = 0

            iteration_count += 1

    except Exception as e:
        bittensor.logging.error(f"Error during CUDA PoW solving: {str(e)}")
    finally:
        logger.stop()

    bittensor.logging.info(
        "Maximum iterations reached or error occurred. No solution found."
    )
    return None


def _create_shared_memory():
    curr_block = multiprocessing.Array("B", 40)
    curr_block_num = multiprocessing.Value("i", 0)
    curr_diff = multiprocessing.Value("I", 0)
    return curr_block, curr_block_num, curr_diff


def _get_block_with_retry(subtensor):
    max_retries = 3
    for _ in range(max_retries):
        try:
            block_number = subtensor.get_current_block()
            block_hash = subtensor.get_block_hash(block_number)
            return block_number, block_hash
        except Exception as e:
            bittensor.logging.warning(f"Failed to get block info: {e}")
            time.sleep(1)
    raise Exception("Failed to get block info after multiple attempts")


def _update_curr_block(
    curr_block, curr_block_num, block_number, block_bytes, hotkey_bytes, check_block
):
    with check_block:
        curr_block_num.value = block_number
        curr_block[:32] = block_bytes
        curr_block[32:] = hotkey_bytes


def _check_for_newest_block_and_update(
    subtensor,
    hotkey_bytes,
    old_block_number,
    curr_block,
    curr_block_num,
    curr_stats,
    update_curr_block,
    check_block,
    solvers,
):
    block_number, block_hash = _get_block_with_retry(subtensor)
    if block_number != old_block_number:
        block_bytes = bytes.fromhex(block_hash[2:])
        update_curr_block(
            curr_block,
            curr_block_num,
            block_number,
            block_bytes,
            hotkey_bytes,
            check_block,
        )
        curr_stats.block_number = block_number
        curr_stats.block_hash = block_hash
        for s in solvers:
            s.newBlockEvent.set()
        return block_number
    return old_block_number


def _create_block_and_hotkey_hash(block_bytes: bytes, hotkey_bytes: bytes) -> bytes:
    return block_bytes + hotkey_bytes


class _ColdkeySwapSolver(multiprocessing.Process):
    def __init__(
        self,
        process_id,
        num_processes,
        update_interval,
        finished_queue,
        solution_queue,
        stopEvent,
        curr_block,
        curr_block_num,
        difficulty,
        check_block,
        limit,
    ):
        super().__init__()
        self.process_id = process_id
        self.num_processes = num_processes
        self.update_interval = update_interval
        self.finished_queue = finished_queue
        self.solution_queue = solution_queue
        self.stopEvent = stopEvent
        self.curr_block = curr_block
        self.curr_block_num = curr_block_num
        self.difficulty = difficulty
        self.check_block = check_block
        self.limit = limit
        self.newBlockEvent = multiprocessing.Event()

    def run(self):
        while not self.stopEvent.is_set():
            self.newBlockEvent.wait()
            self.newBlockEvent.clear()

            with self.check_block:
                block_bytes = bytes(self.curr_block[:32])
                hotkey_bytes = bytes(self.curr_block[32:])
                block_number = self.curr_block_num.value

            nonce_start = self.process_id
            while not self.stopEvent.is_set() and not self.newBlockEvent.is_set():
                solution = self._solve_for_nonce_block(
                    nonce_start,
                    nonce_start + self.update_interval,
                    block_bytes + hotkey_bytes,
                    self.difficulty,
                    self.limit,
                    block_number,
                )
                if solution:
                    self.solution_queue.put(solution)
                    return
                self.finished_queue.put(self.process_id)
                nonce_start += self.update_interval * self.num_processes

    def _solve_for_nonce_block(
        self,
        nonce_start,
        nonce_end,
        block_and_hotkey_hash_bytes,
        difficulty,
        limit,
        block_number,
    ):
        for nonce in range(nonce_start, nonce_end, self.num_processes):
            seal = _create_seal_hash(block_and_hotkey_hash_bytes, nonce)
            if _seal_meets_difficulty(seal, difficulty, limit):
                return POWSolution(nonce, block_number, difficulty, seal)
        return None


def create_pow_for_coldkey_swap(
    subtensor,
    wallet,
    base_difficulty: int,
    swap_attempts: int,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
    max_iterations: int = 1000000,
) -> Optional[POWSolution]:
    """
    Creates a proof of work for coldkey swap.

    Args:
        subtensor (bittensor.subtensor): Subtensor object for blockchain interaction.
        wallet (bittensor.wallet): Wallet object containing keys.
        base_difficulty (int): Base difficulty for coldkey swap PoW.
        swap_attempts (int): Number of swap attempts for coldkey swap PoW.
        output_in_place (bool, optional): Whether to print output in place. Defaults to True.
        cuda (bool, optional): Whether to use CUDA for PoW calculation. Defaults to False.
        dev_id (Union[List[int], int], optional): CUDA device ID(s). Defaults to 0.
        tpb (int, optional): Threads per block for CUDA. Defaults to 256.
        num_processes (int, optional): Number of CPU processes to use. Defaults to None.
        update_interval (int, optional): Interval to update block information. Defaults to None.
        log_verbose (bool, optional): Whether to log verbose output. Defaults to False.
        max_iterations (int, optional): Maximum number of iterations for CUDA solver. Defaults to 1000000.

    Returns:
        Optional[POWSolution]: The solved PoW solution, or None if not found.

    Raises:
        RuntimeError: If CUDA is not available when cuda=True.
    """
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return _solve_for_coldkey_swap_difficulty_cuda(
            subtensor,
            wallet,
            base_difficulty,
            swap_attempts,
            output_in_place,
            dev_id,
            tpb,
            num_processes,
            update_interval,
            log_verbose,
            max_iterations,
        )
    else:
        return _solve_for_coldkey_swap_difficulty_cpu(
            subtensor,
            wallet,
            base_difficulty,
            swap_attempts,
            output_in_place,
            num_processes,
            update_interval,
            log_verbose,
        )


# Additional helper functions


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, byteorder="big")


def bytes_to_int(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, byteorder="big")
