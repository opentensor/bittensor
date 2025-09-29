"""This module provides async utilities for solving Proof-of-Work (PoW) challenges in Bittensor network."""

import math
import time
from multiprocessing import Event, Lock, Array, Value, Queue
from queue import Empty
from typing import Callable, Union, Optional, TYPE_CHECKING

from bittensor.core.errors import SubstrateRequestException
from bittensor.utils.btlogging import logging
from bittensor.utils.registration.pow import (
    get_cpu_count,
    update_curr_block,
    terminate_workers_and_wait_for_exit,
    CUDASolver,
    torch,
    RegistrationStatistics,
    RegistrationStatisticsLogger,
    Solver,
    UsingSpawnStartMethod,
)

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor_wallet import Wallet
    from bittensor.utils.registration import POWSolution


async def _get_block_with_retry(
    subtensor: "AsyncSubtensor", netuid: int
) -> tuple[int, int, str]:
    """
    Gets the current block number, difficulty, and block hash from the substrate node.

    Parameters:
        subtensor: The subtensor object to use to get the block number, difficulty, and block hash.
        netuid: The netuid of the network to get the block number, difficulty, and block hash from.

    Returns:
        The current block number, difficulty of the subnet, block hash

    Raises:
        Exception: If the block hash is None.
        ValueError: If the difficulty is None.
    """
    block = await subtensor.substrate.get_block()
    block_hash = block["header"]["hash"]
    block_number = block["header"]["number"]
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


async def _check_for_newest_block_and_update(
    subtensor: "AsyncSubtensor",
    netuid: int,
    old_block_number: int,
    hotkey_bytes: bytes,
    curr_diff: Array,
    curr_block: Array,
    curr_block_num: Value,
    update_curr_block_: "Callable",
    check_block: Lock,
    solvers: list[Solver],
    curr_stats: "RegistrationStatistics",
) -> int:
    """
    Check for the newest block and update block-related information and states across solvers if a new block is detected.

    Parameters:
        subtensor: The subtensor instance interface.
        netuid: The network UID for the blockchain.
        old_block_number: The previously known block number.
        hotkey_bytes: The bytes representation of the hotkey.
        curr_diff: The current difficulty level.
        curr_block: The current block information.
        curr_block_num: The current block number.
        update_curr_block_: Function to update current block information.
        check_block: Lock object for synchronizing block checking.
        solvers: List of solvers to notify of new blocks.
        curr_stats: Current registration statistics to update.

    Returns:
        int: The updated block number which is the same as the new block
             number if it was detected, otherwise the old block number.
    """
    block_number = await subtensor.substrate.get_block_number(None)
    if block_number != old_block_number:
        old_block_number = block_number
        # update block information
        block_number, difficulty, block_hash = await _get_block_with_retry(
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


async def _block_solver(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
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

    if cuda:
        num_processes = len(dev_id)

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
        solvers = [
            CUDASolver(
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
            Solver(
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

    timeout = 0.15 if cuda else 0.15
    while netuid == -1 or not await subtensor.is_hotkey_registered(
        wallet.hotkey.ss58_address, netuid
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
    terminate_workers_and_wait_for_exit(solvers)

    return solution


async def _solve_for_difficulty_fast_cuda(
    subtensor: "AsyncSubtensor",
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
    Solves the registration fast using CUDA

    Parameters:
        subtensor: The subtensor object to use to get the block number, difficulty, and block hash.
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

    num_processes = min(1, get_cpu_count())

    if update_interval is None:
        update_interval = 50_000

    if not torch.cuda.is_available():
        raise Exception("CUDA not available")

    # Set mp start to use spawn so CUDA doesn't complain
    with UsingSpawnStartMethod(force=True):
        curr_block, curr_block_num, curr_diff = CUDASolver.create_shared_memory()

        solution = await _block_solver(
            subtensor=subtensor,
            wallet=wallet,
            num_processes=num_processes,
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
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    output_in_place: bool = True,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    n_samples: int = 10,
    alpha_: float = 0.80,
    log_verbose: bool = False,
) -> Optional["POWSolution"]:
    """
    Solves the POW for registration using multiprocessing.

    Parameters:
        subtensor: The subtensor object to use to get the block number, difficulty, and block hash.
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
        We can also modify the update interval to do smaller blocks of work, while still updating the block information
        after a different number of nonces, to increase the transparency of the process while still keeping the speed.
    """
    if not num_processes:
        # get the number of allowed processes for this process
        num_processes = min(1, get_cpu_count())

    if update_interval is None:
        update_interval = 50_000

    curr_block, curr_block_num, curr_diff = Solver.create_shared_memory()

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


async def create_pow_async(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[list[int], int] = 0,
    tpb: int = 256,
    num_processes: int = None,
    update_interval: int = None,
    log_verbose: bool = False,
) -> "POWSolution":
    """
    Creates a proof of work for the given subtensor and wallet.

    Parameters:
        subtensor: The subtensor instance.
        wallet: The wallet to create a proof of work for.
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
        if not await subtensor.subnet_exists(netuid=netuid):
            raise ValueError(f"Subnet {netuid} does not exist")
    solution: Optional[POWSolution]
    if cuda:
        logging.debug("Solve difficulty with CUDA.")
        solution = await _solve_for_difficulty_fast_cuda(
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
        solution = await _solve_for_difficulty_fast(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            output_in_place=output_in_place,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )

    return solution
