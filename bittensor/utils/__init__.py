import binascii
import struct
import hashlib
import math
import bittensor
import rich
import time
import torch
import numbers
import pandas
from typing import Tuple, List, Union, Optional


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

def indexed_values_to_wandb( 
        wandb_data: dict, 
        prefix: Union[str, int],
        index: Union[list, torch.LongTensor], 
        values: Union[list, torch.Tensor],
    ):
    r""" Adds Values to wandb data grouped by index.
        Args:
            wandb_data: dict:
                Wandb dictionary to fill

            prefix str:
                Prefix name given to wandb log value.

            index: Union[list(int), torch.LongTensor]:
                Index into values which act as group.

            values: Union[list, torch.Tensor]:
                Values to index into. If index is non existend value is not added.
    """
    # Type checking and converion.
    if not isinstance(wandb_data, dict):
        raise ValueError('Passed wandb_data must have type dict')
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

    for idx_i in index:
        # Convert index to integer or fail.
        try:
            int_idx = int( idx_i )
        except Exception as e:
            bittensor.logging.error('Failed to convert index: {} to integer with error'.format(idx_i, e))

        # Add value to wandb data dict.
        if int_idx < len(values) and int_idx > 0:
            log_value = values[ int_idx ]
            wandb_data[ '{}/{}'.format( int_idx, prefix ) ] = log_value


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
    limit = int(math.pow(2,256) - 1)
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


def solve_for_difficulty_fast( subtensor ):
    block_number = subtensor.get_current_block()
    difficulty = subtensor.difficulty
    block_hash = subtensor.substrate.get_block_hash( block_number )
    while block_hash == None:
        block_hash = subtensor.substrate.get_block_hash( block_number )
    block_bytes = block_hash.encode('utf-8')[2:]
    
    meets = False
    nonce = -1
    limit = int(math.pow(2,256) - 1)
    best = math.inf
    update_interval = 100000
    start_time = time.time()

    console = bittensor.__console__
    with console.status("Solving") as status:
        while not meets:
            nonce += 1 

            # Create seal.
            nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
            pre_seal = nonce_bytes + block_bytes
            seal = hashlib.sha256( bytearray(hex_bytes_to_u8_list(pre_seal)) ).digest()

            seal_number = int.from_bytes(seal, "big")
            product = seal_number * difficulty
            if product - limit < best:
                best = product - limit
                best_seal = seal

            if product < limit:
                return nonce, block_number, block_hash, difficulty, seal

            if nonce % update_interval == 0:
                itrs_per_sec = update_interval / (time.time() - start_time)
                start_time = time.time()
                difficulty = subtensor.difficulty
                block_number = subtensor.get_current_block()
                block_hash = subtensor.substrate.get_block_hash( block_number)
                while block_hash == None:
                    block_hash = subtensor.substrate.get_block_hash( block_number)
                block_bytes = block_hash.encode('utf-8')[2:]
                status.update("Solving\n  Nonce: [bold white]{}[/bold white]\n  Iters: [bold white]{}/s[/bold white]\n  Difficulty: [bold white]{}[/bold white]\n  Block: [bold white]{}[/bold white]\n  Best: [bold white]{}[/bold white]".format( nonce, int(itrs_per_sec), difficulty, block_hash.encode('utf-8'), binascii.hexlify(best_seal) ))
      

def create_pow( subtensor ):
    nonce, block_number, block_hash, difficulty, seal = solve_for_difficulty_fast( subtensor )
    return {
        'nonce': nonce, 
        'difficulty': difficulty,
        'block_number': block_number, 
        'block_hash': block_hash, 
        'work': binascii.hexlify(seal)
    }
