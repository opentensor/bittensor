import binascii
import datetime
import hashlib
import math
import multiprocessing
import numbers
import random
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any, Dict, Optional, Tuple, Union

import bittensor
import pandas
import requests
import torch
from Crypto.Hash import keccak
from substrateinterface import Keypair, KeypairType
from substrateinterface.utils import ss58

from .register_cuda import reset_cuda, solve_cuda


def indexed_values_to_dataframe ( 
        prefix: Union[str, int],
        index: Union[list, torch.LongTensor], 
        values: Union[list, torch.Tensor],
        filter_zeros: bool = False
    ) -> 'pandas.DataFrame':
    # Type checking.
    if not isinstance(prefix, str) and not isinstance(prefix, numbers.Number):
        raise ValueError('Passed prefix must have type str or Number')
    if isinstance(prefix, numbers.Number):
        prefix = str(prefix)
    if not isinstance(index, list) and not isinstance(index, torch.Tensor):
        raise ValueError('Passed uids must have type list or torch.Tensor')
    if not isinstance(values, list) and not isinstance(values, torch.Tensor):
        raise ValueError('Passed values must have type list or torch.Tensor')
    if not isinstance(index, list):
        index = index.tolist()
    if not isinstance(values, list):
        values = values.tolist()

    index = [ idx_i for idx_i in index if idx_i < len(values) and idx_i >= 0 ]
    dataframe = pandas.DataFrame(columns=[prefix], index = index )
    for idx_i in index:
        value_i = values[ idx_i ]
        if value_i > 0 or not filter_zeros:
            dataframe.loc[idx_i] = pandas.Series( { str(prefix): value_i } )
    return dataframe

def unbiased_topk( values, k, dim=0, sorted = True, largest = True):
    r""" Selects topk as in torch.topk but does not bias lower indices when values are equal.
        Args:
            values: (torch.Tensor)
                Values to index into.
            k: (int):
                Number to take.
            
        Return:
            topk: (torch.Tensor):
                topk k values.
            indices: (torch.LongTensor)
                indices of the topk values.
    """
    permutation = torch.randperm(values.shape[ dim ])
    permuted_values = values[ permutation ]
    topk, indices = torch.topk( permuted_values,  k, dim = dim, sorted=sorted, largest=largest )
    return topk, permutation[ indices ]

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

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

@dataclass
class POWSolution:
    """A solution to the registration PoW problem."""
    nonce: int
    block_number: int
    difficulty: int
    seal: bytes

class Solver(multiprocessing.Process):
    proc_num: int
    num_proc: int
    update_interval: int
    best_queue: multiprocessing.Queue
    time_queue: multiprocessing.Queue
    solution_queue: multiprocessing.Queue
    newBlockEvent: multiprocessing.Event
    stopEvent: multiprocessing.Event
    curr_block: multiprocessing.Array
    curr_block_num: multiprocessing.Value
    curr_diff: multiprocessing.Array
    check_block: multiprocessing.Lock
    limit: int

    def __init__(self, proc_num, num_proc, update_interval, best_queue, time_queue, solution_queue, stopEvent, curr_block, curr_block_num, curr_diff, check_block, limit):
        multiprocessing.Process.__init__(self)
        self.proc_num = proc_num
        self.num_proc = num_proc
        self.update_interval = update_interval
        self.best_queue = best_queue
        self.time_queue = time_queue
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
        block_number: int
        block_bytes: bytes
        block_difficulty: int
        nonce_limit = int(math.pow(2,64)) - 1

        # Start at random nonce
        nonce_start = self.update_interval * self.proc_num + random.randint( 0, nonce_limit )
        nonce_end = nonce_start + self.update_interval
        while not self.stopEvent.is_set():
            if self.newBlockEvent.is_set():
                with self.check_block:
                    block_number = self.curr_block_num.value
                    block_bytes = bytes(self.curr_block)
                    block_difficulty = int(self.curr_diff[0] >> 32 | self.curr_diff[1])

                self.newBlockEvent.clear()
                # reset nonces to start from random point
                nonce_start = self.update_interval * self.proc_num + random.randint( 0, nonce_limit )
                nonce_end = nonce_start + self.update_interval
                
            # Do a block of nonces
            solution, time = solve_for_nonce_block(self, nonce_start, nonce_end, block_bytes, block_difficulty, self.limit, block_number)
            if solution is not None:
                self.solution_queue.put(solution)

            # Send time
            self.time_queue.put_nowait(time)
                
            nonce_start += self.update_interval * self.num_proc
            nonce_end += self.update_interval * self.num_proc


def solve_for_nonce_block(solver: Solver, nonce_start: int, nonce_end: int, block_bytes: bytes, difficulty: int, limit: int, block_number: int) -> Tuple[Optional[POWSolution], int]:
    best_local = float('inf')
    best_seal_local = [0]*32
    start = time.time()
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
            print(f"{solver.proc_num} found a solution: {nonce}, {block_number}, {str(block_bytes)}, {str(seal)}, {difficulty}")
            # Found a solution, save it.
            return POWSolution(nonce, block_number, difficulty, seal), time.time() - start

        if (product - limit) < best_local: 
            best_local = product - limit
            best_seal_local = seal

    # Send best solution to best queue.
    solver.best_queue.put((best_local, best_seal_local))
    return None, time.time() - start


def solve_for_difficulty_fast( subtensor, wallet, num_processes: int = None, update_interval: int = 50_000 ) -> Optional[POWSolution]:
    """
    Solves the POW for registration using multiprocessing.
    Args:
        subtensor
            Subtensor to connect to for block information and to submit.
        wallet:
            Wallet to use for registration.
        num_processes: int
            Number of processes to use.
        update_interval: int
            Number of nonces to solve before updating block information.
    Note: 
    - We can also modify the update interval to do smaller blocks of work,
        while still updating the block information after a different number of nonces,
        to increase the transparency of the process while still keeping the speed.
    """
    if num_processes == None:
        num_processes = multiprocessing.cpu_count()

    if update_interval is None:
        update_interval = 50_000
        
    limit = int(math.pow(2,256)) - 1

    console = bittensor.__console__
    status = console.status("Solving")

    best_seal: bytes
    best_number: int
    best_number = float('inf')

    curr_block = multiprocessing.Array('h', 64, lock=True) # byte array
    curr_block_num = multiprocessing.Value('i', 0, lock=True) # int
    curr_diff = multiprocessing.Array('Q', [0, 0], lock=True) # [high, low]

    def update_curr_block(block_number: int, block_bytes: bytes, diff: int, lock: multiprocessing.Lock):
        with lock:
            curr_block_num.value = block_number
            for i in range(64):
                curr_block[i] = block_bytes[i]
            curr_diff[0] = diff >> 32
            curr_diff[1] = diff & 0xFFFFFFFF # low 32 bits

    status.start()

    # Establish communication queues
    stopEvent = multiprocessing.Event()
    stopEvent.clear()
    best_queue = multiprocessing.Queue()
    solution_queue = multiprocessing.Queue()
    time_queue = multiprocessing.Queue()
    check_block = multiprocessing.Lock()
    
    # Start consumers
    solvers = [ Solver(i, num_processes, update_interval, best_queue, time_queue, solution_queue, stopEvent, curr_block, curr_block_num, curr_diff, check_block, limit)
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
    update_curr_block(block_number, block_bytes, difficulty, check_block)

    # Set new block events for each solver to start
    for w in solvers:
        w.newBlockEvent.set()
    
    for w in solvers:
        w.start() # start the solver processes
    
    start_time = time.time()
    solution = None
    best_seal = None
    itrs_per_sec = 0
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
        block_number = subtensor.get_current_block()
        if block_number != old_block_number:
            old_block_number = block_number
            # update block information
            block_hash = subtensor.substrate.get_block_hash( block_number)
            while block_hash == None:
                block_hash = subtensor.substrate.get_block_hash( block_number)
            block_bytes = block_hash.encode('utf-8')[2:]
            difficulty = subtensor.difficulty

            update_curr_block(block_number, block_bytes, difficulty, check_block)
            # Set new block events for each solver
            for w in solvers:
                w.newBlockEvent.set()
                
        # Get times for each solver
        time_total = 0
        num_time = 0
        while time_queue.qsize() > 0:
            try:
                time_ = time_queue.get_nowait()
                time_total += time_
                num_time += 1

            except Empty:
                break
        
        if num_time > 0:
            time_avg = time_total / num_time
            itrs_per_sec = update_interval*num_processes / time_avg
            
        #times = [ time_queue.get() for _ in solvers ]
        #time_avg = average(times)

       

        # get best solution
        while best_queue.qsize() > 0:
            try:
                num, seal = best_queue.get_nowait()
                if num < best_number:
                    best_number = num
                    best_seal = seal

            except Empty:
                break
        
        message = f"""Solving 
            time spent: {time.time() - start_time}
            Difficulty: [bold white]{millify(difficulty)}[/bold white]
            Iters: [bold white]{get_human_readable(int(itrs_per_sec), 'H')}/s[/bold white]
            Block: [bold white]{block_number}[/bold white]
            Block_hash: [bold white]{block_hash.encode('utf-8')}[/bold white]
            Best: [bold white]{binascii.hexlify(bytes(best_seal) if best_seal else bytes(0))}[/bold white]"""
        status.update(message.replace(" ", ""))
    
    # exited while, found_solution contains the nonce or wallet is registered
    if solution is not None:
        stopEvent.set() # stop all other processes
        status.stop()

        return solution

    status.stop()
    return None
    
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

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def solve_for_difficulty_fast_cuda( subtensor: 'bittensor.Subtensor', wallet: 'bittensor.Wallet', update_interval: int = 50_000, TPB: int = 512, dev_id: int = 0 ) -> Optional[POWSolution]:
    """
    Solves the registration fast using CUDA
    Args:
        subtensor: bittensor.Subtensor
            The subtensor node to grab blocks
        wallet: bittensor.Wallet
            The wallet to register
        update_interval: int
            The number of nonces to try before checking for more blocks
        TPB: int
            The number of threads per block. CUDA param that should match the GPU capability
        dev_id: int
            The CUDA device ID to execute the registration on
    """
    if not torch.cuda.is_available():
        raise Exception("CUDA not available")

    if update_interval is None:
        update_interval = 50_000
        
    block_number = subtensor.get_current_block()
    difficulty = subtensor.difficulty
    block_hash = subtensor.substrate.get_block_hash( block_number )
    while block_hash == None:
        block_hash = subtensor.substrate.get_block_hash( block_number )
    block_bytes = block_hash.encode('utf-8')[2:]
    
    nonce = 0
    limit = int(math.pow(2,256)) - 1
    start_time = time.time()

    console = bittensor.__console__
    status = console.status("Solving")
    
    solution = -1
    start_time = time.time()
    interval_time = start_time

    status.start()
    while solution == -1 and not wallet.is_registered(subtensor):
        solution, seal = solve_cuda(nonce,
                        update_interval,
                        TPB,
                        block_bytes, 
                        block_number,
                        difficulty, 
                        limit,
                        dev_id)

        if (solution != -1):
            # Attempt to reset CUDA device
            reset_cuda()
            status.stop()
            new_bn = subtensor.get_current_block()
            print(f"Found solution for bn: {block_number}; Newest: {new_bn}")            
            return POWSolution(solution, block_number, difficulty, seal)

        nonce += (TPB * update_interval)
        if (nonce >= int(math.pow(2,63))):
            nonce = 0
        itrs_per_sec = (TPB * update_interval) / (time.time() - interval_time)
        interval_time = time.time()
        difficulty = subtensor.difficulty
        block_number = subtensor.get_current_block()
        block_hash = subtensor.substrate.get_block_hash( block_number)
        while block_hash == None:
            block_hash = subtensor.substrate.get_block_hash( block_number)
        block_bytes = block_hash.encode('utf-8')[2:]

        message = f"""Solving 
            time spent: {datetime.timedelta(seconds=time.time() - start_time)}
            Nonce: [bold white]{nonce}[/bold white]
            Difficulty: [bold white]{millify(difficulty)}[/bold white]
            Iters: [bold white]{get_human_readable(int(itrs_per_sec), "H")}/s[/bold white]
            Block: [bold white]{block_number}[/bold white]
            Block_hash: [bold white]{block_hash.encode('utf-8')}[/bold white]"""
        status.update(message.replace(" ", ""))
        
    # exited while, found_solution contains the nonce or wallet is registered
    if solution == -1: # didn't find solution
        reset_cuda()
        status.stop()
        return None
    
    else:
        reset_cuda()
        # Shouldn't get here
        status.stop()
        return None

def create_pow( subtensor, wallet, cuda: bool = False, dev_id: int = 0, tpb: int = 256, num_processes: int = None, update_interval: int = None ) -> Optional[Dict[str, Any]]:
    if cuda:
        solution: POWSolution = solve_for_difficulty_fast_cuda( subtensor, wallet, dev_id=dev_id, TPB=tpb, update_interval=update_interval )
    else:
        solution: POWSolution = solve_for_difficulty_fast( subtensor, wallet, num_processes=num_processes, update_interval=update_interval )
    nonce, block_number, difficulty, seal = solution.nonce, solution.block_number, solution.difficulty, solution.seal
    return None if nonce is None else {
        'nonce': nonce, 
        'difficulty': difficulty,
        'block_number': block_number, 
        'work': binascii.hexlify(seal)
    }

def version_checking():
    response = requests.get(bittensor.__pipaddress__)
    latest_version = response.json()['info']['version']
    version_split = latest_version.split(".")
    latest_version_as_int = (100 * int(version_split[0])) + (10 * int(version_split[1])) + (1 * int(version_split[2]))

    if latest_version_as_int > bittensor.__version_as_int__:
        print('\u001b[31m Current Bittensor Version: {}, Latest Bittensor Version {} \n Please update to the latest version'.format(bittensor.__version__,latest_version))

def is_valid_ss58_address( address: str ) -> bool:
    """
    Checks if the given address is a valid ss58 address.

    Args:
        address(str): The address to check.

    Returns:
        True if the address is a valid ss58 address for Bittensor, False otherwise.
    """
    try:
        return ss58.is_valid_ss58_address( address, valid_ss58_format=bittensor.__ss58_format__ )
    except (IndexError):
        return False

def is_valid_ed25519_pubkey( public_key: Union[str, bytes] ) -> bool:
    """
    Checks if the given public_key is a valid ed25519 key.

    Args:
        public_key(Union[str, bytes]): The public_key to check.

    Returns:    
        True if the public_key is a valid ed25519 key, False otherwise.
    
    """
    try:
        if isinstance( public_key, str ):
            if len(public_key) != 64 and len(public_key) != 66:
                raise ValueError( "a public_key should be 64 or 66 characters" )
        elif isinstance( public_key, bytes ):
            if len(public_key) != 32:
                raise ValueError( "a public_key should be 32 bytes" )
        else:
            raise ValueError( "public_key must be a string or bytes" )

        keypair = Keypair(
            public_key=public_key,
            ss58_format=bittensor.__ss58_format__
        )

        ss58_addr = keypair.ss58_address
        return ss58_addr is not None

    except (ValueError, IndexError):
        return False

def is_valid_destination_address( address: Union[str, bytes] ) -> bool:
    """
    Checks if the given address is a valid destination address.

    Args:
        address(Union[str, bytes]): The address to check.

    Returns:
        True if the address is a valid destination address, False otherwise.
    """
    if isinstance( address, str ):
        # Check if ed25519
        if address.startswith('0x'):
            if not is_valid_ed25519_pubkey( address ):
                bittensor.__console__.print(":cross_mark: [red]Invalid Destination Public Key[/red]: {}".format( address ))
                return False
        # Assume ss58 address
        else:
            if not is_valid_ss58_address( address ):
                bittensor.__console__.print(":cross_mark: [red]Invalid Destination Address[/red]: {}".format( address ))
                return False
    elif isinstance( address, bytes ):
        # Check if ed25519
        if not is_valid_ed25519_pubkey( address ):
            bittensor.__console__.print(":cross_mark: [red]Invalid Destination Public Key[/red]: {}".format( address ))
            return False
    else:
        bittensor.__console__.print(":cross_mark: [red]Invalid Destination[/red]: {}".format( address ))
        return False
        
    return True
