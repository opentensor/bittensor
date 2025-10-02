import struct
import time

import pytest

from bittensor.extras import timelock


def test_encrypt_returns_valid_tuple():
    """Test that encrypt() returns a (bytes, int) tuple."""
    encrypted, reveal_round = timelock.encrypt("Bittensor", n_blocks=1)
    assert isinstance(encrypted, bytes)
    assert isinstance(reveal_round, int)
    assert reveal_round > 0


def test_encrypt_with_fast_block_time():
    """Test encrypt() with fast-blocks mode (block_time = 0.25s)."""
    encrypted, reveal_round = timelock.encrypt("Fast mode", 5, block_time=0.25)
    assert isinstance(encrypted, bytes)
    assert isinstance(reveal_round, int)


def test_decrypt_returns_bytes_or_none():
    """Test that decrypt() returns bytes after reveal round, or None before."""
    data = b"Decode me"
    encrypted, reveal_round = timelock.encrypt(data, 1)

    current_round = timelock.get_latest_round()
    if current_round < reveal_round:
        decrypted = timelock.decrypt(encrypted)
        assert decrypted is None
    else:
        decrypted = timelock.decrypt(encrypted)
        assert decrypted == data


def test_decrypt_raises_if_no_errors_false_and_invalid_data():
    """Test that decrypt() raises an error on invalid data when no_errors=False."""
    with pytest.raises(Exception):
        timelock.decrypt(b"corrupt data", no_errors=False)


def test_decrypt_with_return_str():
    """Test decrypt() with return_str=True returns a string."""
    plaintext = "Stringified!"
    encrypted, _ = timelock.encrypt(plaintext, 1, block_time=0.25)
    result = timelock.decrypt(encrypted, no_errors=True, return_str=True)
    if result is not None:
        assert isinstance(result, str)


def test_get_latest_round_is_monotonic():
    """Test that get_latest_round() is monotonic over time."""
    r1 = timelock.get_latest_round()
    time.sleep(3)
    r2 = timelock.get_latest_round()
    assert r2 >= r1


def test_wait_reveal_and_decrypt_auto_round():
    """Test wait_reveal_and_decrypt() without explicit reveal_round."""
    msg = "Reveal and decrypt test"
    encrypted, _ = timelock.encrypt(msg, 1)
    result = timelock.wait_reveal_and_decrypt(encrypted, return_str=True)
    assert result == msg


def test_wait_reveal_and_decrypt_manual_round():
    """Test wait_reveal_and_decrypt() with explicit reveal_round."""
    msg = "Manual round decryption"
    encrypted, reveal_round = timelock.encrypt(msg, 1)
    result = timelock.wait_reveal_and_decrypt(encrypted, reveal_round, return_str=True)
    assert result == msg


def test_unpack_reveal_round_struct():
    """Test that reveal_round can be extracted from encrypted data."""
    encrypted, reveal_round = timelock.encrypt("parse test", 1)
    parsed = struct.unpack(
        "<Q", encrypted.split(timelock.TLE_ENCRYPTED_DATA_SUFFIX)[-1]
    )[0]
    assert parsed == reveal_round
