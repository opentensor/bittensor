import binascii
from contextlib import redirect_stdout
import hashlib
import io
import math
from typing import Tuple, List, Any, Dict

import numpy as np
from Crypto.Hash import keccak

try:
    import cubit
except ImportError:
    raise ImportError("Please install cubit")


def solve_cuda(
    nonce_start: np.int64,
    update_interval: np.int64,
    tpb: int,
    block_and_hotkey_hash_bytes: bytes,
    difficulty: int,
    limit: int,
    dev_id: int = 0,
) -> Tuple[np.int64, bytes]:
    """
    Solves the PoW problem using CUDA.
    Args:
        nonce_start: int64
            Starting nonce.
        update_interval: int64
            Number of nonces to solve before updating block information.
        tpb: int
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
    upper = int(limit // difficulty)
    upper_bytes = upper.to_bytes(32, byteorder="little", signed=False)

    def _hex_bytes_to_u8_list(hex_bytes: bytes) -> List[int]:
        """Convert hex bytes to a list of unsigned 8-bit integers."""
        return [int(hex_bytes[i: i + 2], 16) for i in range(0, len(hex_bytes), 2)]

    def _create_seal_hash(block_hotkey_hash_hex: bytes, nonce: np.int64) -> bytes:
        """Create a seal hash."""
        nonce_bytes = binascii.hexlify(int(nonce).to_bytes(8, "little"))
        pre_seal = nonce_bytes + block_hotkey_hash_hex
        seal_sha256 = hashlib.sha256(bytearray(_hex_bytes_to_u8_list(pre_seal))).digest()
        kec = keccak.new(digest_bits=256)
        created_seal = kec.update(seal_sha256).digest()
        return created_seal

    def _seal_meets_difficulty(check_seal: bytes, seal_difficulty: int) -> bool:
        """Check if seal meets the difficulty."""
        seal_number = int.from_bytes(check_seal, "big")
        product = seal_number * seal_difficulty
        check_limit = int(math.pow(2, 256)) - 1
        return product < check_limit

    block_and_hotkey_hash_hex: bytes = binascii.hexlify(block_and_hotkey_hash_bytes)[:64]

    solution: np.int64 = cubit.solve_cuda(
        tpb,
        nonce_start,
        update_interval,
        upper_bytes,
        block_and_hotkey_hash_hex,
        dev_id,
    )

    seal: bytes = b""
    if solution != np.int64(-1):
        seal = _create_seal_hash(block_and_hotkey_hash_hex, solution)
        if _seal_meets_difficulty(seal, difficulty):
            return solution, seal

    return np.int64(-1), b"\x00" * 32


def solve_cuda(
    nonce_start: np.int64,
    update_interval: np.int64,
    tpb: int,
    block_and_hotkey_hash_bytes: bytes,
    difficulty: int,
    limit: int,
    dev_id: int = 0,
) -> Tuple[np.int64, bytes]:
    """
    Solves the PoW problem using CUDA.
    Args:
        nonce_start: int64
            Starting nonce.
        update_interval: int64
            Number of nonces to solve before updating block information.
        tpb: int
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

    upper_bytes = upper.to_bytes(32, byteorder="little", signed=False)

    def _hex_bytes_to_u8_list(hex_bytes: bytes) -> List[int]:
        hex_chunks = [
            int(hex_bytes[i: i + 2], 16) for i in range(0, len(hex_bytes), 2)
        ]
        return hex_chunks

    def _create_seal_hash(block_hotkey_hash_hex: bytes, nonce: int) -> bytes:
        nonce_bytes = binascii.hexlify(nonce.to_bytes(8, "little"))
        pre_seal = nonce_bytes + block_hotkey_hash_hex
        seal_sh256 = hashlib.sha256(bytearray(_hex_bytes_to_u8_list(pre_seal))).digest()
        kec = keccak.new(digest_bits=256)
        created_seal = kec.update(seal_sh256).digest()
        return created_seal

    def _seal_meets_difficulty(check_seal: bytes, difficulty: int):
        seal_number = int.from_bytes(check_seal, "big")
        product = seal_number * difficulty
        check_limit = int(math.pow(2, 256)) - 1

        return product < check_limit

    # Call cython function
    # int blockSize, uint64 nonce_start, uint64 update_interval, const unsigned char[:] limit,
    # const unsigned char[:] block_bytes, int dev_id
    block_and_hotkey_hash_hex: bytes = binascii.hexlify(block_and_hotkey_hash_bytes)[:64]

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
        if _seal_meets_difficulty(seal, difficulty):
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
