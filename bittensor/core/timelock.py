import struct
import time
from typing import Optional, Union

from bittensor_commit_reveal import (
    encrypt as _btr_encrypt,
    decrypt as _btr_decrypt,
    get_latest_round,
)

TLE_ENCRYPTED_DATA_SUFFIX = b"AES_GCM_"


def encrypt(
    data: Union[bytes, str], n_blocks: int, block_time: Union[int, float] = 12.0
) -> tuple[bytes, int]:
    """Encrypts data using TimeLock Encryption

    Arguments:
        data: Any bytes data to be encrypted.
        n_blocks: Number of blocks to encrypt.
        block_time: Time in seconds for each block. Default is `12.0` seconds.

    Returns:
        tuple: A tuple containing the encrypted data and reveal TimeLock reveal round.

    Raises:
        PyValueError: If failed to encrypt data.

    Usage:
        data = "From Cortex to Bittensor"

        # default usage
        encrypted_data, reveal_round = encrypt(data, 10)

        # passing block_time for fast-blocks node
        encrypted_data, reveal_round = encrypt(data, 15, block_time=0.25)

        encrypted_data, reveal_round = encrypt(data, 5)


    Note:
        For using this function with fast-blocks node you need to set block_time to 0.25 seconds.
        data, round = encrypt(data, n_blocks, block_time=0.25)
    """
    if isinstance(data, str):
        data = data.encode()
    return _btr_encrypt(data, n_blocks, block_time)


def decrypt(
    encrypted_data: bytes, no_errors: bool = True, return_str: bool = False
) -> Optional[Union[bytes, str]]:
    """Decrypts encrypted data using TimeLock Decryption

    Arguments:
        encrypted_data: Encrypted data to be decrypted.
        no_errors: If True, no errors will be raised during decryption.
        return_str: convert decrypted data to string if `True`. Default is `False`.

    Returns:
        decrypted_data: Decrypted data, when reveled round is reached.

    Usage:
        # default usage
        decrypted_data = decrypt(encrypted_data)

        # passing no_errors=False for raising errors during decryption
        decrypted_data = decrypt(encrypted_data, no_errors=False)

        # passing return_str=True for returning decrypted data as string
        decrypted_data = decrypt(encrypted_data, return_str=True)
    """
    result = _btr_decrypt(encrypted_data, no_errors)
    if result is None:
        return None
    if return_str:
        return result.decode()
    return result


def wait_reveal_and_decrypt(
    encrypted_data: bytes,
    reveal_round: Optional[int] = None,
    no_errors: bool = True,
    return_str: bool = False,
) -> bytes:
    """
    Waits for reveal round and decrypts data using TimeLock Decryption.

    Arguments:
        encrypted_data: Encrypted data to be decrypted.
        reveal_round: Reveal round to wait for. If None, will be parsed from encrypted data.
        no_errors: If True, no errors will be raised during decryption.
        return_str: convert decrypted data to string if `True`. Default is `False`.

    Raises:
        struct.error: If failed to parse reveal round from encrypted data.
        TypeError: If reveal_round is None or wrong type.
        IndexError: If provided encrypted_data does not contain reveal round.

    Returns:
        bytes: Decrypted data.

    Usage:
        import bittensor as bt
        encrypted, reveal_round = bt.timelock.encrypt("Cortex is power", 3)
    """
    if reveal_round is None:
        try:
            reveal_round = struct.unpack(
                "<Q", encrypted_data.split(TLE_ENCRYPTED_DATA_SUFFIX)[-1]
            )[0]
        except (struct.error, TypeError, IndexError):
            raise ValueError("Failed to parse reveal round from encrypted data.")

    while get_latest_round() <= reveal_round:
        # sleep Drand QuickNet period time (3 sec)
        time.sleep(3)

    return decrypt(encrypted_data, no_errors, return_str)


__all__ = [
    "decrypt",
    "encrypt",
    "get_latest_round",
    "wait_reveal_and_decrypt",
]
