# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import binascii
import hashlib
import io
import math
from contextlib import redirect_stdout
from typing import Any, Union

import numpy as np
from Crypto.Hash import keccak


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

    Args:
        nonce_start (numpy.int64):  Starting nonce.
        update_interval (numpy.int64): Number of nonces to solve before updating block information.
        tpb (int): Threads per block.
        block_and_hotkey_hash_bytes (bytes): Keccak(Bytes of the block hash + bytes of the hotkey) 64 bytes.
        difficulty (int): Difficulty of the PoW problem.
        limit (int): Upper limit of the nonce.
        dev_id (int): The CUDA device ID. Defaults to ``0``.

    Returns:
        (Union[tuple[Any, bytes], tuple[int, bytes], tuple[Any, None]]): Tuple of the nonce and the seal corresponding to the solution. Returns -1 for nonce if no solution is found.
    """

    try:
        import cubit
    except ImportError:
        raise ImportError(
            "Please install cubit. See the instruction https://github.com/opentensor/cubit?tab=readme-ov-file#install."
        )

    upper = int(limit // difficulty)

    upper_bytes = upper.to_bytes(32, byteorder="little", signed=False)

    def _hex_bytes_to_u8_list(hex_bytes: bytes):
        """Converts a sequence of hex bytes to a list of unsigned 8-bit integers."""
        hex_chunks = [
            int(hex_bytes[i : i + 2], 16) for i in range(0, len(hex_bytes), 2)
        ]
        return hex_chunks

    def _create_seal_hash(block_and_hotkey_hash_hex_: bytes, nonce: int) -> bytes:
        """Creates a seal hash from the block and hotkey hash and nonce."""
        nonce_bytes = binascii.hexlify(nonce.to_bytes(8, "little"))
        pre_seal = nonce_bytes + block_and_hotkey_hash_hex_
        seal_sh256 = hashlib.sha256(bytearray(_hex_bytes_to_u8_list(pre_seal))).digest()
        kec = keccak.new(digest_bits=256)
        return kec.update(seal_sh256).digest()

    def _seal_meets_difficulty(seal_: bytes, difficulty_: int):
        """Checks if the seal meets the given difficulty."""
        seal_number = int.from_bytes(seal_, "big")
        product = seal_number * difficulty_
        limit_ = int(math.pow(2, 256)) - 1

        return product < limit_

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
        if _seal_meets_difficulty(seal, difficulty):
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
