from typing import Tuple
import math

import hashlib
import binascii
from bittensor_register_cuda import solve_cuda as solve_cuda_c, reset_cuda as reset_cuda_c

import numpy as np

def solve_cuda(nonce_start: np.int64, update_interval: np.int64, TPB: int, block_bytes: bytes, difficulty: np.int64, limit: np.int64) -> Tuple[np.int64, bytes]:
    """
    Solves the PoW problem using CUDA.
    Args:
        nonce_start: int32
            Starting nonce.
        update_interval: int32
            Number of nonces to solve before updating block information.
        TPB: int
            Threads per block.
        block_bytes: bytes
            Bytes of the block hash. 64 bytes.
        difficulty: int32
            Difficulty of the PoW problem.
        limit: int32
            Upper limit of the nonce.
    Returns:
        Tuple[int32, bytes]
            Tuple of the nonce and the seal corresponding to the solution.  
            Returns -1 for nonce if no solution is found.     
    """  
    upper = int(limit // difficulty)

    upper_bytes = upper.to_bytes(32, byteorder='little', signed=False)

    def seal_meets_difficulty( seal:bytes, difficulty:int ):
        seal_number = int.from_bytes(seal, "big")
        product = seal_number * difficulty
        limit = int(math.pow(2,256))- 1
        upper = int(limit // difficulty)        
        if product > limit:
            return False
        else:
            return True
    def hex_bytes_to_u8_list( hex_bytes: bytes ):
        hex_chunks = [int(hex_bytes[i:i+2], 16) for i in range(0, len(hex_bytes), 2)]
        return hex_chunks

    def create_seal_hash( block_bytes:bytes, nonce:int ) -> bytes:
        nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
        pre_seal = nonce_bytes + block_bytes
        seal = hashlib.sha256( bytearray(hex_bytes_to_u8_list(pre_seal)) ).digest()
        return seal

    # Call cython function
    # int blockSize, int64 nonce_start, uint64 update_interval,
    #  uint64 difficulty, const unsigned char[:] limit, const unsigned char[:] block_bytes
    solution = solve_cuda_c(TPB, nonce_start, update_interval, upper_bytes, block_bytes)
    seal = None
    if solution != -1:
        seal = create_seal_hash(block_bytes, solution)
        if seal_meets_difficulty(seal, difficulty):
            return solution, seal
        else:
            return -1, b'\x00' * 32

    return solution, seal

def reset_cuda():
    """
    Resets the CUDA environment.
    """
    reset_cuda_c()