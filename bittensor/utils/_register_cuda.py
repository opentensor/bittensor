import binascii
import hashlib
import math
from typing import Tuple

import numpy as np
from Crypto.Hash import keccak

from contextlib import redirect_stdout
import io


def solve_cuda(nonce_start: np.int64, update_interval: np.int64, TPB: int, block_and_hotkey_hash_bytes: bytes, difficulty: int, limit: int, dev_id: int = 0) -> Tuple[np.int64, bytes]:
    """
    Solves the PoW problem using CUDA.
    Args:
        nonce_start: int64
            Starting nonce.
        update_interval: int64
            Number of nonces to solve before updating block information.
        TPB: int
            Threads per block.
        block_and_hotkey_hash_bytes: bytes
            Keccak(Bytes of the block hash + bytes of the hotkey) 64 bytes.
        difficulty: int256
            Difficulty of the PoW problem.
        limit: int256
            Upper limit of the nonce.
        dev_id: int (default=0)
            The CUDA device ID
    Returns:
        Tuple[int64, bytes]
            Tuple of the nonce and the seal corresponding to the solution.  
            Returns -1 for nonce if no solution is found.     
    """ 

    try:
        import cubit
    except ImportError:
        raise ImportError("Please install cubit")


    upper = int(limit // difficulty)

    upper_bytes = upper.to_bytes(32, byteorder='little', signed=False)

    def _hex_bytes_to_u8_list( hex_bytes: bytes ):
        hex_chunks = [int(hex_bytes[i:i+2], 16) for i in range(0, len(hex_bytes), 2)]
        return hex_chunks

    def _create_seal_hash( block_and_hotkey_hash_hex: bytes, nonce:int ) -> bytes:
        nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
        pre_seal = nonce_bytes + block_and_hotkey_hash_hex
        seal_sh256 = hashlib.sha256( bytearray(_hex_bytes_to_u8_list(pre_seal)) ).digest()
        kec = keccak.new(digest_bits=256)
        seal = kec.update( seal_sh256 ).digest()
        return seal

    def _seal_meets_difficulty( seal:bytes, difficulty:int ):
        seal_number = int.from_bytes(seal, "big")
        product = seal_number * difficulty
        limit = int(math.pow(2,256)) - 1  

        return product < limit

    # Call cython function
    # int blockSize, uint64 nonce_start, uint64 update_interval, const unsigned char[:] limit,
    # const unsigned char[:] block_bytes, int dev_id
    block_and_hotkey_hash_hex = binascii.hexlify(block_and_hotkey_hash_bytes)[:64]

    solution = cubit.solve_cuda(TPB, nonce_start, update_interval, upper_bytes, block_and_hotkey_hash_hex, dev_id) # 0 is first GPU
    seal = None
    if solution != -1:
        seal = _create_seal_hash(block_and_hotkey_hash_hex, solution)
        if _seal_meets_difficulty(seal, difficulty):
            return solution, seal
        else:
            return -1, b'\x00' * 32

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
        
    


