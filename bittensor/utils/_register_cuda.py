import binascii
from contextlib import redirect_stdout
import hashlib
import io
import math
from typing import Tuple, List, Any, Dict

import numpy as np
from Crypto.Hash import keccak


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
    """
    upper = int(limit // difficulty)
    upper_bytes = upper.to_bytes(32, byteorder="little", signed=False)

    def _hex_bytes_to_u8_list(hex_bytes: bytes) -> List[int]:
        """Convert hex bytes to a list of unsigned 8-bit integers."""
        return [int(hex_bytes[i: i + 2], 16) for i in range(0, len(hex_bytes), 2)]

    def _create_seal_hash(block_hotkey_hash_hex: bytes, nonce: np.int64) -> bytes:
        """Create a seal hash."""
        nonce_bytes = binascii.hexlify(nonce.to_bytes(8, "little"))
        pre_seal = nonce_bytes + block_hotkey_hash_hex
        seal_sha256 = hashlib.sha256(bytearray(_hex_bytes_to_u8_list(pre_seal))).digest()
        kec = keccak.new(digest_bits=256)
        created_seal = kec.update(seal_sha256).digest()
        return created_seal

    def _seal_meets_difficulty(check_seal: bytes, difficulty: int) -> bool:
        """Check if seal meets the difficulty."""
        seal_number = int.from_bytes(check_seal, "big")
        product = seal_number * difficulty
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
