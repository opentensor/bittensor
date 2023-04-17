import binascii
import hashlib
import math
import multiprocessing
import os
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from queue import Empty, Full
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import backoff
import bittensor
import torch
from Crypto.Hash import keccak
from rich import console as rich_console
from rich import status as rich_status

from ._register_cuda import solve_cuda


class CUDAException(Exception):
    """An exception raised when an error occurs in the CUDA environment."""
    pass


def hex_bytes_to_u8_list( hex_bytes: bytes ):
    hex_chunks = [int(hex_bytes[i:i+2], 16) for i in range(0, len(hex_bytes), 2)]
    return hex_chunks


def u8_list_to_hex( values: list ):
    total = 0
    for val in reversed(values):
        total = (total << 8) + val
    return total 


def create_seal_hash( block_hash:bytes, nonce:int ) -> bytes:
    block_bytes = block_hash.encode('utf-8')[2:]
    nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
    pre_seal = nonce_bytes + block_bytes
    seal_sh256 = hashlib.sha256( bytearray(hex_bytes_to_u8_list(pre_seal)) ).digest()
    kec = keccak.new(digest_bits=256)
    seal = kec.update( seal_sh256 ).digest()
    return seal


def seal_meets_difficulty( seal:bytes, difficulty:int ):
    seal_number = int.from_bytes(seal, "big")
    product = seal_number * difficulty
    limit = int(math.pow(2,256))- 1
    if product > limit:
        return False
    else:
        return True
    

def solve_for_difficulty( block_hash, difficulty ):
    meets = False
    nonce = -1
    while not meets:
        nonce += 1 
        seal = create_seal_hash( block_hash, nonce )
        meets = seal_meets_difficulty( seal, difficulty )
        if nonce > 1:
            break
    return nonce, seal


def get_human_readable(num, suffix="H"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Y{suffix}"


def millify(n: int):
    millnames = ['',' K',' M',' B',' T']
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.2f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


def POWNotStale(subtensor: 'bittensor.Subtensor', pow_result: Dict) -> bool:
    """Returns True if the POW is not stale.
    This means the block the POW is solved for is within 3 blocks of the current block.
    """
    return pow_result['block_number'] >= subtensor.get_current_block() - 3


@dataclass
class POWSolution:
    """A solution to the registration PoW problem."""
    nonce: int
    block_number: int
    difficulty: int
    seal: bytes


class SolverBase(multiprocessing.Process):
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
    curr_block: multiprocessing.Array
    curr_block_num: multiprocessing.Value
    curr_diff: multiprocessing.Array
    check_block: multiprocessing.Lock
    limit: int

    def __init__(self, proc_num, num_proc, update_interval, finished_queue, solution_queue, stopEvent, curr_block, curr_block_num, curr_diff, check_block, limit):
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
        raise NotImplementedError("SolverBase is an abstract class")


class Solver(SolverBase):
    def run(self):
        block_number: int
        block_bytes: bytes
        block_difficulty: int
        nonce_limit = int(math.pow(2,64)) - 1

        # Start at random nonce
        nonce_start = random.randint( 0, nonce_limit )
        nonce_end = nonce_start + self.update_interval
        while not self.stopEvent.is_set():
            if self.newBlockEvent.is_set():
                with self.check_block:
                    block_number = self.curr_block_num.value
                    block_bytes = bytes(self.curr_block)
                    block_difficulty = registration_diff_unpack(self.curr_diff)

                self.newBlockEvent.clear()
                
            # Do a block of nonces
            solution = solve_for_nonce_block(self, nonce_start, nonce_end, block_bytes, block_difficulty, self.limit, block_number)
            if solution is not None:
                self.solution_queue.put(solution)

            try:
                # Send time
                self.finished_queue.put_nowait(self.proc_num)
            except Full:
                pass
                
            nonce_start = random.randint( 0, nonce_limit )
            nonce_start = nonce_start % nonce_limit
            nonce_end = nonce_start + self.update_interval


class CUDASolver(SolverBase):
    dev_id: int
    TPB: int

    def __init__(self, proc_num, num_proc, update_interval, finished_queue, solution_queue, stopEvent, curr_block, curr_block_num, curr_diff, check_block, limit, dev_id: int, TPB: int):
        super().__init__(proc_num, num_proc, update_interval, finished_queue, solution_queue, stopEvent, curr_block, curr_block_num, curr_diff, check_block, limit)
        self.dev_id = dev_id
        self.TPB = TPB

    def run(self):
        block_number: int = 0 # dummy value
        block_bytes: bytes = b'0' * 32 # dummy value
        block_difficulty: int = int(math.pow(2,64)) - 1 # dummy value
        nonce_limit = int(math.pow(2,64)) - 1 # U64MAX

        # Start at random nonce
        nonce_start = random.randint( 0, nonce_limit )
        while not self.stopEvent.is_set():
            if self.newBlockEvent.is_set():
                with self.check_block:
                    block_number = self.curr_block_num.value
                    block_bytes = bytes(self.curr_block)
                    block_difficulty = registration_diff_unpack(self.curr_diff)

                self.newBlockEvent.clear()
                
            # Do a block of nonces
            solution = solve_for_nonce_block_cuda(self, nonce_start, self.update_interval, block_bytes, block_difficulty, self.limit, block_number, self.dev_id, self.TPB)
            if solution is not None:
                self.solution_queue.put(solution)

            try:
                # Signal that a nonce_block was finished using queue
                # send our proc_num
                self.finished_queue.put(self.proc_num)
            except Full:
                pass
            
            # increase nonce by number of nonces processed
            nonce_start += self.update_interval * self.TPB 
            nonce_start = nonce_start % nonce_limit


def solve_for_nonce_block_cuda(solver: CUDASolver, nonce_start: int, update_interval: int, block_bytes: bytes, difficulty: int, limit: int, block_number: int, dev_id: int, TPB: int) -> Optional[POWSolution]:
    """Tries to solve the POW on a CUDA device for a block of nonces (nonce_start, nonce_start + update_interval * TPB"""
    solution, seal = solve_cuda(
                    nonce_start,
                    update_interval,
                    TPB,
                    block_bytes, 
                    difficulty, 
                    limit,
                    dev_id)

    if (solution != -1):
        # Check if solution is valid (i.e. not -1)
        return POWSolution(solution, block_number, difficulty, seal)

    return None


def solve_for_nonce_block(solver: Solver, nonce_start: int, nonce_end: int, block_bytes: bytes, difficulty: int, limit: int, block_number: int) -> Optional[POWSolution]:
    """Tries to solve the POW for a block of nonces (nonce_start, nonce_end)""" 
    for nonce in range(nonce_start, nonce_end):
        # Create seal.
        nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
        pre_seal = nonce_bytes + block_bytes
        seal_sh256 = hashlib.sha256( bytearray(hex_bytes_to_u8_list(pre_seal)) ).digest()
        kec = keccak.new(digest_bits=256)
        seal = kec.update( seal_sh256 ).digest()
        seal_number = int.from_bytes(seal, "big")

        # Check if seal meets difficulty
        product = seal_number * difficulty
        if product < limit:
            # Found a solution, save it.
            return POWSolution(nonce, block_number, difficulty, seal)

    return None


def registration_diff_unpack(packed_diff: multiprocessing.Array) -> int:
    """Unpacks the packed two 32-bit integers into one 64-bit integer. Little endian."""
    return int(packed_diff[0] << 32 | packed_diff[1])


def registration_diff_pack(diff: int, packed_diff: multiprocessing.Array):
    """Packs the difficulty into two 32-bit integers. Little endian."""
    packed_diff[0] = diff >> 32
    packed_diff[1] = diff & 0xFFFFFFFF # low 32 bits


def update_curr_block(curr_diff: multiprocessing.Array, curr_block: multiprocessing.Array, curr_block_num: multiprocessing.Value, block_number: int, block_bytes: bytes, diff: int, lock: multiprocessing.Lock):
    with lock:
        curr_block_num.value = block_number
        for i in range(64):
            curr_block[i] = block_bytes[i]
        registration_diff_pack(diff, curr_diff)


def get_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # OSX does not have sched_getaffinity
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
    

class RegistrationStatisticsLogger:
    """Logs statistics for a registration."""
    console: rich_console.Console
    status: Optional[rich_status.Status] 

    def __init__( self, console: rich_console.Console, output_in_place: bool = True) -> None:
        self.console = console
        
        if output_in_place:
            self.status = self.console.status("Solving")
        else:
            self.status = None
        
    def start( self ) -> None:
        if self.status is not None:
            self.status.start()

    def stop( self ) -> None:
        if self.status is not None:
            self.status.stop()


    def get_status_message(cls, stats: RegistrationStatistics, verbose: bool = False) -> str:
        message = \
        "Solving\n" + \
        f"Time Spent (total): [bold white]{timedelta(seconds=stats.time_spent_total)}[/bold white]\n" + \
        (
            f"Time Spent This Round: {timedelta(seconds=stats.time_spent)}\n" + \
            f"Time Spent Average: {timedelta(seconds=stats.time_average)}\n" if verbose else ""
        ) + \
        f"Registration Difficulty: [bold white]{millify(stats.difficulty)}[/bold white]\n" + \
        f"Iters (Inst/Perp): [bold white]{get_human_readable(stats.hash_rate, 'H')}/s / " + \
            f"{get_human_readable(stats.hash_rate_perpetual, 'H')}/s[/bold white]\n" + \
        f"Block Number: [bold white]{stats.block_number}[/bold white]\n" + \
        f"Block Hash: [bold white]{stats.block_hash.encode('utf-8')}[/bold white]\n"
        return message


    def update( self, stats: RegistrationStatistics, verbose: bool = False ) -> None:
        if self.status is not None:
            self.status.update( self.get_status_message(stats, verbose=verbose) )
        else:
            self.console.log( self.get_status_message(stats, verbose=verbose), )


def solve_for_difficulty_fast( subtensor, wallet, output_in_place: bool = True, num_processes: Optional[int] = None, update_interval: Optional[int] = None,  n_samples: int = 10, alpha_: float = 0.80, log_verbose: bool = False ) -> Optional[POWSolution]:
    """
    Solves the POW for registration using multiprocessing.
    Args:
        subtensor
            Subtensor to connect to for block information and to submit.
        wallet:
            Wallet to use for registration.
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
            If true, prints more verbose logging of the registration metrics.
    Note: The hash rate is calculated as an exponentially weighted moving average in order to make the measure more robust.
    Note: 
    - We can also modify the update interval to do smaller blocks of work,
        while still updating the block information after a different number of nonces,
        to increase the transparency of the process while still keeping the speed.
    """
    if num_processes == None:
        # get the number of allowed processes for this process
        num_processes = min(1, get_cpu_count())

    if update_interval is None:
        update_interval = 50_000
        
    limit = int(math.pow(2,256)) - 1

    curr_block = multiprocessing.Array('h', 64, lock=True) # byte array
    curr_block_num = multiprocessing.Value('i', 0, lock=True) # int
    curr_diff = multiprocessing.Array('Q', [0, 0], lock=True) # [high, low]

    # Establish communication queues
    ## See the Solver class for more information on the queues.
    stopEvent = multiprocessing.Event()
    stopEvent.clear()

    solution_queue = multiprocessing.Queue()
    finished_queues = [multiprocessing.Queue() for _ in range(num_processes)]
    check_block = multiprocessing.Lock()
    
    # Start consumers
    solvers = [ Solver(i, num_processes, update_interval, finished_queues[i], solution_queue, stopEvent, curr_block, curr_block_num, curr_diff, check_block, limit)
                for i in range(num_processes) ]

    # Get first block
    block_number = subtensor.get_current_block()
    difficulty = subtensor.difficulty
    block_hash = subtensor.substrate.get_block_hash( block_number )
    while block_hash == None:
        block_hash = subtensor.substrate.get_block_hash( block_number )
    block_bytes = block_hash.encode('utf-8')[2:]
    old_block_number = block_number
    # Set to current block
    update_curr_block(curr_diff, curr_block, curr_block_num, block_number, block_bytes, difficulty, check_block)

    # Set new block events for each solver to start at the initial block
    for worker in solvers:
        worker.newBlockEvent.set()
    
    for worker in solvers:
        worker.start() # start the solver processes

    start_time = time.time() # time that the registration started
    time_last = start_time # time that the last work blocks completed
    
    curr_stats = RegistrationStatistics(
        time_spent_total = 0.0,
        time_average = 0.0,
        rounds_total = 0,
        time_spent = 0.0,
        hash_rate_perpetual = 0.0,
        hash_rate = 0.0,
        difficulty = difficulty,
        block_number = block_number,
        block_hash = block_hash
    )

    start_time_perpetual = time.time()
    

    console = bittensor.__console__
    logger = RegistrationStatisticsLogger(console, output_in_place)
    logger.start()

    solution = None

    hash_rates = [0] * n_samples # The last n true hash_rates
    weights = [alpha_ ** i for i in range(n_samples)] # weights decay by alpha
    
    while not wallet.is_registered(subtensor):
        # Wait until a solver finds a solution
        try:
            solution = solution_queue.get(block=True, timeout=0.25)
            if solution is not None:
                break
        except Empty:
            # No solution found, try again
            pass

        # check for new block
        old_block_number = check_for_newest_block_and_update(
            subtensor = subtensor,
            old_block_number=old_block_number,
            curr_diff=curr_diff,
            curr_block=curr_block,
            curr_block_num=curr_block_num,
            curr_stats=curr_stats,
            update_curr_block=update_curr_block,
            check_block=check_block,
            solvers=solvers
        )
                
        num_time = 0
        for finished_queue in finished_queues:
            try:
                proc_num = finished_queue.get(timeout=0.1)
                num_time += 1

            except Empty:
                continue
        
        time_now = time.time() # get current time
        time_since_last = time_now - time_last # get time since last work block(s)
        if num_time > 0 and time_since_last > 0.0:
            # create EWMA of the hash_rate to make measure more robust
        
            hash_rate_ = (num_time * update_interval) / time_since_last
            hash_rates.append(hash_rate_)
            hash_rates.pop(0) # remove the 0th data point
            curr_stats.hash_rate = sum([hash_rates[i]*weights[i] for i in range(n_samples)])/(sum(weights))

            # update time last to now
            time_last = time_now

            curr_stats.time_average = (curr_stats.time_average*curr_stats.rounds_total + curr_stats.time_spent)/(curr_stats.rounds_total+num_time)
            curr_stats.rounds_total += num_time

        # Update stats
        curr_stats.time_spent = time_since_last
        new_time_spent_total = time_now - start_time_perpetual
        curr_stats.hash_rate_perpetual = (curr_stats.rounds_total*update_interval)/ new_time_spent_total
        curr_stats.time_spent_total = new_time_spent_total

        # Update the logger
        logger.update(curr_stats, verbose=log_verbose)

    # exited while, solution contains the nonce or wallet is registered
    stopEvent.set() # stop all other processes
    logger.stop()

    # terminate and wait for all solvers to exit
    terminate_workers_and_wait_for_exit(solvers)

    return solution


@backoff.on_exception(backoff.constant,
                            Exception,
                            interval=1,
                            max_tries=3)
def get_block_with_retry(subtensor: 'bittensor.Subtensor') -> Tuple[int, int, bytes]:
    block_number = subtensor.get_current_block()
    difficulty = subtensor.difficulty
    block_hash = subtensor.substrate.get_block_hash( block_number )
    if block_hash is None:
        raise Exception("Network error. Could not connect to substrate to get block hash")
    return block_number, difficulty, block_hash


class UsingSpawnStartMethod():
    def __init__(self, force: bool = False):
        self._old_start_method = None
        self._force = force

    def __enter__(self):
        self._old_start_method = multiprocessing.get_start_method(allow_none=True)
        if self._old_start_method == None:
            self._old_start_method = 'spawn' # default to spawn

        multiprocessing.set_start_method('spawn', force=self._force)

    def __exit__(self, *args):
        # restore the old start method
        multiprocessing.set_start_method(self._old_start_method, force=True)


def check_for_newest_block_and_update(
    subtensor: 'bittensor.Subtensor',
    old_block_number: int,
    curr_diff: multiprocessing.Array,
    curr_block: multiprocessing.Array,
    curr_block_num: multiprocessing.Value,
    update_curr_block: Callable,
    check_block: 'multiprocessing.Lock',
    solvers: List[Solver],
    curr_stats: RegistrationStatistics
    ) -> int:
    """
    Checks for a new block and updates the current block information if a new block is found.
    Args:
        subtensor (:obj:`bittensor.Subtensor`, `required`):
            The subtensor object to use for getting the current block.
        old_block_number (:obj:`int`, `required`):
            The old block number to check against.
        curr_diff (:obj:`multiprocessing.Array`, `required`):
            The current difficulty as a multiprocessing array.
        curr_block (:obj:`multiprocessing.Array`, `required`):
            Where the current block is stored as a multiprocessing array.
        curr_block_num (:obj:`multiprocessing.Value`, `required`):
            Where the current block number is stored as a multiprocessing value.
        update_curr_block (:obj:`Callable`, `required`):
            A function that updates the current block.
        check_block (:obj:`multiprocessing.Lock`, `required`):
            A mp lock that is used to check for a new block.
        solvers (:obj:`List[Solver]`, `required`):
            A list of solvers to update the current block for.
        curr_stats (:obj:`RegistrationStatistics`, `required`):
            The current registration statistics to update.
    Returns:
        (int) The current block number.
    """
    block_number = subtensor.get_current_block()
    if block_number != old_block_number:
        old_block_number = block_number
        # update block information
        block_hash = subtensor.substrate.get_block_hash( block_number)
        while block_hash == None:
            block_hash = subtensor.substrate.get_block_hash( block_number)
        block_bytes = block_hash.encode('utf-8')[2:]
        difficulty = subtensor.difficulty

        update_curr_block(curr_diff, curr_block, curr_block_num, block_number, block_bytes, difficulty, check_block)
        # Set new block events for each solver

        for worker in solvers:
            worker.newBlockEvent.set()

        # update stats
        curr_stats.block_number = block_number
        curr_stats.block_hash = block_hash
        curr_stats.difficulty = difficulty

    return old_block_number


def solve_for_difficulty_fast_cuda( subtensor: 'bittensor.Subtensor', wallet: 'bittensor.Wallet', output_in_place: bool = True, update_interval: int = 50_000, TPB: int = 512, dev_id: Union[List[int], int] = 0, n_samples: int = 10, alpha_: float = 0.80, log_verbose: bool = False ) -> Optional[POWSolution]:
    """
    Solves the registration fast using CUDA
    Args:
        subtensor: bittensor.Subtensor
            The subtensor node to grab blocks
        wallet: bittensor.Wallet
            The wallet to register
        output_in_place: bool
            If true, prints the output in place, otherwise prints to new lines
        update_interval: int
            The number of nonces to try before checking for more blocks
        TPB: int
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
        update_interval = 50_000

    if not torch.cuda.is_available():
        raise Exception("CUDA not available")
        
    limit = int(math.pow(2,256)) - 1

    # Set mp start to use spawn so CUDA doesn't complain
    with UsingSpawnStartMethod(force=True):
        curr_block = multiprocessing.Array('h', 64, lock=True) # byte array
        curr_block_num = multiprocessing.Value('i', 0, lock=True) # int
        curr_diff = multiprocessing.Array('Q', [0, 0], lock=True) # [high, low]

        ## Create a worker per CUDA device
        num_processes = len(dev_id)

        # Establish communication queues
        stopEvent = multiprocessing.Event()
        stopEvent.clear()
        solution_queue = multiprocessing.Queue()
        finished_queues = [multiprocessing.Queue() for _ in range(num_processes)]
        check_block = multiprocessing.Lock()
        
        # Start workers
        solvers = [ CUDASolver(i, num_processes, update_interval, finished_queues[i], solution_queue, stopEvent, curr_block, curr_block_num, curr_diff, check_block, limit, dev_id[i], TPB)
                    for i in range(num_processes) ]


        # Get first block
        block_number = subtensor.get_current_block()
        difficulty = subtensor.difficulty
        block_hash = subtensor.substrate.get_block_hash( block_number )
        while block_hash == None:
            block_hash = subtensor.substrate.get_block_hash( block_number )
        block_bytes = block_hash.encode('utf-8')[2:]
        old_block_number = block_number
        
        # Set to current block
        update_curr_block(curr_diff, curr_block, curr_block_num, block_number, block_bytes, difficulty, check_block)

        # Set new block events for each solver to start at the initial block
        for worker in solvers:
            worker.newBlockEvent.set()
        
        for worker in solvers:
            worker.start() # start the solver processes
        
        start_time = time.time() # time that the registration started
        time_last = start_time # time that the last work blocks completed
        
        curr_stats = RegistrationStatistics(
            time_spent_total = 0.0,
            time_average = 0.0,
            rounds_total = 0,
            time_spent = 0.0,
            hash_rate_perpetual = 0.0,
            hash_rate = 0.0, # EWMA hash_rate (H/s)
            difficulty = difficulty,
            block_number = block_number,
            block_hash = block_hash
        )

        start_time_perpetual = time.time()

        console = bittensor.__console__
        logger = RegistrationStatisticsLogger(console, output_in_place)
        logger.start()

        hash_rates = [0] * n_samples # The last n true hash_rates
        weights = [alpha_ ** i for i in range(n_samples)] # weights decay by alpha

        solution = None
        while not wallet.is_registered(subtensor):
            # Wait until a solver finds a solution
            try:
                solution = solution_queue.get(block=True, timeout=0.15)
                if solution is not None:
                    break
            except Empty:
                # No solution found, try again
                pass
            
            # check for new block
            old_block_number = check_for_newest_block_and_update(
                subtensor = subtensor,
                curr_diff=curr_diff,
                curr_block=curr_block,
                curr_block_num=curr_block_num,
                old_block_number=old_block_number,
                curr_stats=curr_stats,
                update_curr_block=update_curr_block,
                check_block=check_block,
                solvers=solvers
            )
                    
            num_time = 0
            # Get times for each solver
            for finished_queue in finished_queues:
                try:
                    proc_num = finished_queue.get(timeout=0.1)
                    num_time += 1
            
                except Empty:
                    continue
            
            time_now = time.time() # get current time
            time_since_last = time_now - time_last # get time since last work block(s)
            if num_time > 0 and time_since_last > 0.0:
                # create EWMA of the hash_rate to make measure more robust
            
                hash_rate_ = (num_time * TPB * update_interval) / time_since_last
                hash_rates.append(hash_rate_)
                hash_rates.pop(0) # remove the 0th data point
                curr_stats.hash_rate = sum([hash_rates[i]*weights[i] for i in range(n_samples)])/(sum(weights))

                # update time last to now
                time_last = time_now

                curr_stats.time_average = (curr_stats.time_average*curr_stats.rounds_total + curr_stats.time_spent)/(curr_stats.rounds_total+num_time)
                curr_stats.rounds_total += num_time
            
            # Update stats
            curr_stats.time_spent = time_since_last
            new_time_spent_total = time_now - start_time_perpetual
            curr_stats.hash_rate_perpetual = (curr_stats.rounds_total * (TPB * update_interval))/ new_time_spent_total
            curr_stats.time_spent_total = new_time_spent_total

            # Update the logger
            logger.update(curr_stats, verbose=log_verbose)
        
        # exited while, found_solution contains the nonce or wallet is registered
        
        stopEvent.set() # stop all other processes
        logger.stop()

        # terminate and wait for all solvers to exit
        terminate_workers_and_wait_for_exit(solvers)

        return solution


def terminate_workers_and_wait_for_exit(workers: List[multiprocessing.Process]) -> None:
    for worker in workers:
        worker.terminate()
        worker.join()


def create_pow(
    subtensor,
    wallet,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    tpb: int = 256,
    num_processes: int = None,
    update_interval: int = None,
    log_verbose: bool = False
    ) -> Optional[Dict[str, Any]]:
    if cuda:
        solution: POWSolution = solve_for_difficulty_fast_cuda( subtensor, wallet, output_in_place=output_in_place, \
            dev_id=dev_id, TPB=tpb, update_interval=update_interval, log_verbose=log_verbose
        )
    else:
        solution: POWSolution = solve_for_difficulty_fast( subtensor, wallet, output_in_place=output_in_place, \
            num_processes=num_processes, update_interval=update_interval, log_verbose=log_verbose
        )

    return None if solution is None else {
        'nonce': solution.nonce, 
        'difficulty': solution.difficulty,
        'block_number': solution.block_number, 
        'work': binascii.hexlify(solution.seal)
    }
