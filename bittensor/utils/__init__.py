import binascii
import multiprocessing
import ctypes
import struct
import hashlib
from Crypto.Hash import keccak
import math
import bittensor
import random
import rich
import time
import torch
import numbers
import pandas
import requests
from substrateinterface.utils import ss58
from substrateinterface import Keypair, KeypairType
from typing import Any, Tuple, List, Union, Optional


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
    nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
    block_bytes = block_hash.encode('utf-8')[2:]
    pre_seal = nonce_bytes + block_bytes
    seal = hashlib.sha256( bytearray(hex_bytes_to_u8_list(pre_seal)) ).digest()
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

def solve_for_difficulty_fast( subtensor, wallet, num_processes: int = None, update_interval: int = 500000 ) -> Tuple[int, int, Any, int, Any]:
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
    - We should modify the number of processes based on user input.
    - We can also modify the update interval to do smaller blocks of work,
        while still updating the block information after a different number of nonces,
        to increase the transparency of the process while still keeping the speed.
    """
    if num_processes == None:
        num_processes = multiprocessing.cpu_count()
        
    block_number = subtensor.get_current_block()
    difficulty = subtensor.difficulty
    block_hash = subtensor.substrate.get_block_hash( block_number )
    while block_hash == None:
        block_hash = subtensor.substrate.get_block_hash( block_number )
    block_bytes = block_hash.encode('utf-8')[2:]
    
    limit = int(math.pow(2,256)) - 1
    nonce_limit = int(math.pow(2,64)) - 1
    nonce = random.randint( 0, nonce_limit )
    start_time = time.time()

    console = bittensor.__console__
    status = console.status("Solving")

    #found_solution = multiprocessing.Value('q', -1, lock=False) # int
    found_solution = multiprocessing.Array('Q', [0, 0, 0], lock=True) # [valid, nonce_high, nonce_low]
    best_raw = struct.pack("d", float('inf'))
    best = multiprocessing.Array(ctypes.c_char, best_raw, lock=True) # byte array to get around int size of ctypes
    best_seal = multiprocessing.Array('h', 32, lock=True) # short array should hold bytes (0, 256)
    
    with multiprocessing.Pool(processes=num_processes, initializer=initProcess_, initargs=(solve_, found_solution, best, best_seal)) as pool:
        status.start()
        while found_solution[0] == 0 and not wallet.is_registered(subtensor):
            iterable = [( nonce_start, 
                            nonce_start + update_interval , 
                            block_bytes, 
                            difficulty, 
                            block_hash, 
                            block_number, 
                            limit) for nonce_start in list(range(nonce, nonce + update_interval*num_processes, update_interval))]
            result = pool.starmap(solve_, iterable=iterable)
            old_nonce = nonce
            nonce += update_interval*num_processes
            nonce = nonce % nonce_limit
            itrs_per_sec = update_interval*num_processes / (time.time() - start_time)
            start_time = time.time()
            difficulty = subtensor.difficulty
            block_number = subtensor.get_current_block()
            block_hash = subtensor.substrate.get_block_hash( block_number)
            while block_hash == None:
                block_hash = subtensor.substrate.get_block_hash( block_number)
            block_bytes = block_hash.encode('utf-8')[2:]
            with best_seal.get_lock():
                message = f"""Solving 
                    time spent: {time.time() - start_time}
                    Nonce: [bold white]{nonce}[/bold white]
                    Difficulty: [bold white]{difficulty}[/bold white]
                    Iters: [bold white]{int(itrs_per_sec)}/s[/bold white]
                    Block: [bold white]{block_number}[/bold white]
                    Block_hash: [bold white]{block_hash.encode('utf-8')}[/bold white]
                    Best: [bold white]{binascii.hexlify(bytes(best_seal) or bytes(0))}[/bold white]"""
                status.update(message.replace(" ", ""))
        
        # exited while, found_solution contains the nonce or wallet is registered
        if found_solution[0] == 0: # didn't find solution
            status.stop()
            return None, None, None, None, None
        
        found_unpacked: int = found_solution[1] << 32 | found_solution[2]
        nonce, block_number, block_hash, difficulty, seal = result[ math.floor( (found_unpacked-old_nonce) / update_interval) ]
        status.stop()
        return nonce, block_number, block_hash, difficulty, seal

def initProcess_(f, found_solution, best, best_seal):
    f.found = found_solution
    f.best = best 
    f.best_seal = best_seal

def solve_(nonce_start, nonce_end, block_bytes, difficulty, block_hash, block_number, limit):
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
        product = seal_number * difficulty

        if product < limit:
            with solve_.found.get_lock():
                solve_.found[0] = 1;
                solve_.found[1] = nonce >> 32
                solve_.found[2] = nonce & 0xFFFFFFFF # low 32 bits
            return (nonce, block_number, block_hash, difficulty, seal)

        if (product - limit) < best_local: 
            best_local = product - limit
            best_seal_local = seal

    with solve_.best.get_lock():
        best_value_as_d = struct.unpack('d', solve_.best.raw)[0]
        
        if best_local < best_value_as_d:    
            with solve_.best_seal.get_lock():
                solve_.best.raw = struct.pack('d', best_local)
                for i in range(32):
                    solve_.best_seal[i] = best_seal_local[i]

    return None

def create_pow( subtensor, wallet ):
    nonce, block_number, block_hash, difficulty, seal = solve_for_difficulty_fast( subtensor, wallet )
    return None if nonce is None else {
        'nonce': nonce, 
        'difficulty': difficulty,
        'block_number': block_number, 
        'block_hash': block_hash, 
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

def is_valid_bittensor_address_or_public_key( address: Union[str, bytes] ) -> bool:
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
            return is_valid_ed25519_pubkey( address )
        else:
            # Assume ss58 address
            return is_valid_ss58_address( address )
    elif isinstance( address, bytes ):
        # Check if ed25519
        return is_valid_ed25519_pubkey( address )
    else:
        # Invalid address type
        return False
