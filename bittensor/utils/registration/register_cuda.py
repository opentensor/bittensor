"""This module provides functions for solving Proof of Work (PoW) problems using CUDA."""

import binascii
import hashlib
import io
from contextlib import redirect_stdout
from typing import Any, Union

import numpy as np
from Crypto.Hash import keccak


def _hex_bytes_to_u8_list(hex_bytes: bytes) -> list[int]:
    """
    Convert a sequence of bytes in hexadecimal format to a list of
    unsigned 8-bit integers.

    Parameters:
        hex_bytes: A sequence of bytes in hexadecimal format.

    Returns:
        A list of unsigned 8-bit integers.

    """
    return [int(hex_bytes[i : i + 2], 16) for i in range(0, len(hex_bytes), 2)]


def _create_seal_hash(block_and_hotkey_hash_hex_: bytes, nonce: int) -> bytes:
    """Creates a seal hash from the block and hotkey hash and nonce."""
    nonce_bytes = binascii.hexlify(nonce.to_bytes(8, "little"))
    pre_seal = nonce_bytes + block_and_hotkey_hash_hex_
    seal_sh256 = hashlib.sha256(bytearray(_hex_bytes_to_u8_list(pre_seal))).digest()
    kec = keccak.new(digest_bits=256)
    return kec.update(seal_sh256).digest()


def _seal_meets_difficulty(seal_: bytes, difficulty: int, limit: int) -> bool:
    """Checks if the seal meets the given difficulty."""
    seal_number = int.from_bytes(seal_, "big")
    product = seal_number * difficulty
    # limit = int(math.pow(2, 256)) - 1
    return product < limit


def solve_cuda(
    nonce_start: "np.int64",
    update_interval: "np.int64",
    tpb: int,
    block_and_hotkey_hash_bytes: bytes,
    difficulty: int,
    limit: int,
    dev_id: int = 0,
) -> Union[tuple[Any, bytes], tuple[int, bytes], tuple[Any, None]]:
    """
    Solves the PoW problem using CUDA.

    Parameters:
        nonce_start:  Starting nonce.
        update_interval: Number of nonces to solve before updating block information.
        tpb: Threads per block.
        block_and_hotkey_hash_bytes: Keccak(Bytes of the block hash + bytes of the hotkey) 64 bytes.
        difficulty: Difficulty of the PoW problem.
        limit: Upper limit of the nonce.
        dev_id: The CUDA device ID.

    Returns:
        Tuple of the nonce and the seal corresponding to the solution. Returns -1 for nonce if no solution is found.
    """

    try:
        import cubit
    except ImportError:
        raise ImportError(
            "Please install cubit. See the instruction https://github.com/opentensor/cubit?tab=readme-ov-file#install."
        )

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
    """Resets the CUDA environment."""
    try:
        import cubit
    except ImportError:
        raise ImportError("Please install cubit")
    cubit.reset_cuda()


def log_cuda_errors() -> str:
    """Logs any CUDA errors."""
    try:
        import cubit
    except ImportError:
        raise ImportError("Please install cubit")

    file = io.StringIO()
    with redirect_stdout(file):
        cubit.log_cuda_errors()
    return file.getvalue()
