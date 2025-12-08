"""
Unit tests for bittensor.core.timelock module.

This module provides comprehensive unit tests for the TimeLock Encryption (TLE) functionality,
which encrypts data such that it can only be decrypted after a specific amount of time
(expressed in Drand rounds).

Test Coverage:
- encrypt() function: input validation, string/bytes handling, parameter passing
- decrypt() function: error handling, return types, no_errors flag
- wait_reveal_and_decrypt() function: round parsing, waiting logic, error handling
- TLE_ENCRYPTED_DATA_SUFFIX constant validation
- Edge cases and error conditions

Contribution by Gittensor, learn more at https://gittensor.io/
"""

import struct
from unittest.mock import patch

import pytest

from bittensor.core import timelock
from bittensor.core.timelock import (
    encrypt,
    decrypt,
    wait_reveal_and_decrypt,
    TLE_ENCRYPTED_DATA_SUFFIX,
)


class TestTLEConstants:
    """Tests for module-level constants."""

    def test_tle_encrypted_data_suffix_is_bytes(self):
        """Test that TLE_ENCRYPTED_DATA_SUFFIX is bytes type."""
        assert isinstance(TLE_ENCRYPTED_DATA_SUFFIX, bytes)

    def test_tle_encrypted_data_suffix_value(self):
        """Test that TLE_ENCRYPTED_DATA_SUFFIX has expected value."""
        assert TLE_ENCRYPTED_DATA_SUFFIX == b"AES_GCM_"

    def test_module_exports(self):
        """Test that __all__ exports expected functions."""
        assert "encrypt" in timelock.__all__
        assert "decrypt" in timelock.__all__
        assert "wait_reveal_and_decrypt" in timelock.__all__
        assert "get_latest_round" in timelock.__all__


class TestEncrypt:
    """Tests for the encrypt() function."""

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_string_data(self, mock_btr_encrypt):
        """Test that encrypt() converts string to bytes before encryption."""
        mock_btr_encrypt.return_value = (b"encrypted_data", 12345)

        result = encrypt("test string", n_blocks=5)

        # Verify string was encoded to bytes
        mock_btr_encrypt.assert_called_once_with(b"test string", 5, 12.0)
        assert result == (b"encrypted_data", 12345)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_bytes_data(self, mock_btr_encrypt):
        """Test that encrypt() passes bytes data directly."""
        mock_btr_encrypt.return_value = (b"encrypted_data", 12345)

        result = encrypt(b"test bytes", n_blocks=5)

        mock_btr_encrypt.assert_called_once_with(b"test bytes", 5, 12.0)
        assert result == (b"encrypted_data", 12345)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_default_block_time(self, mock_btr_encrypt):
        """Test that encrypt() uses default block_time of 12.0 seconds."""
        mock_btr_encrypt.return_value = (b"encrypted", 100)

        encrypt(b"data", n_blocks=10)

        mock_btr_encrypt.assert_called_once_with(b"data", 10, 12.0)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_custom_block_time(self, mock_btr_encrypt):
        """Test that encrypt() passes custom block_time correctly."""
        mock_btr_encrypt.return_value = (b"encrypted", 100)

        encrypt(b"data", n_blocks=10, block_time=0.25)

        mock_btr_encrypt.assert_called_once_with(b"data", 10, 0.25)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_fast_block_time(self, mock_btr_encrypt):
        """Test encrypt() with fast-blocks mode (block_time=0.25)."""
        mock_btr_encrypt.return_value = (b"fast_encrypted", 500)

        result = encrypt("fast mode", n_blocks=15, block_time=0.25)

        mock_btr_encrypt.assert_called_once_with(b"fast mode", 15, 0.25)
        assert result == (b"fast_encrypted", 500)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_integer_block_time(self, mock_btr_encrypt):
        """Test that encrypt() accepts integer block_time."""
        mock_btr_encrypt.return_value = (b"encrypted", 100)

        encrypt(b"data", n_blocks=5, block_time=6)

        mock_btr_encrypt.assert_called_once_with(b"data", 5, 6)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_returns_tuple(self, mock_btr_encrypt):
        """Test that encrypt() returns a tuple of (bytes, int)."""
        mock_btr_encrypt.return_value = (b"encrypted_data", 99999)

        result = encrypt(b"test", n_blocks=1)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bytes)
        assert isinstance(result[1], int)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_empty_string(self, mock_btr_encrypt):
        """Test encrypt() with empty string input."""
        mock_btr_encrypt.return_value = (b"encrypted_empty", 100)

        result = encrypt("", n_blocks=1)

        mock_btr_encrypt.assert_called_once_with(b"", 1, 12.0)
        assert result == (b"encrypted_empty", 100)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_empty_bytes(self, mock_btr_encrypt):
        """Test encrypt() with empty bytes input."""
        mock_btr_encrypt.return_value = (b"encrypted_empty", 100)

        encrypt(b"", n_blocks=1)

        mock_btr_encrypt.assert_called_once_with(b"", 1, 12.0)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_unicode_string(self, mock_btr_encrypt):
        """Test encrypt() with unicode string input."""
        mock_btr_encrypt.return_value = (b"encrypted_unicode", 100)
        unicode_str = "Hello 世界 🌍"

        encrypt(unicode_str, n_blocks=1)

        mock_btr_encrypt.assert_called_once_with(unicode_str.encode(), 1, 12.0)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_large_n_blocks(self, mock_btr_encrypt):
        """Test encrypt() with large n_blocks value."""
        mock_btr_encrypt.return_value = (b"encrypted", 1000000)

        encrypt(b"data", n_blocks=100000)

        mock_btr_encrypt.assert_called_once_with(b"data", 100000, 12.0)


class TestDecrypt:
    """Tests for the decrypt() function."""

    @patch("bittensor.core.timelock._btr_decrypt")
    def test_decrypt_returns_bytes(self, mock_btr_decrypt):
        """Test that decrypt() returns bytes by default."""
        mock_btr_decrypt.return_value = b"decrypted_data"

        result = decrypt(b"encrypted_data")

        mock_btr_decrypt.assert_called_once_with(b"encrypted_data", True)
        assert result == b"decrypted_data"
        assert isinstance(result, bytes)

    @patch("bittensor.core.timelock._btr_decrypt")
    def test_decrypt_returns_none_before_reveal(self, mock_btr_decrypt):
        """Test that decrypt() returns None when reveal round not reached."""
        mock_btr_decrypt.return_value = None

        result = decrypt(b"encrypted_data")

        assert result is None

    @patch("bittensor.core.timelock._btr_decrypt")
    def test_decrypt_with_no_errors_true(self, mock_btr_decrypt):
        """Test decrypt() with no_errors=True (default)."""
        mock_btr_decrypt.return_value = b"data"

        decrypt(b"encrypted", no_errors=True)

        mock_btr_decrypt.assert_called_once_with(b"encrypted", True)

    @patch("bittensor.core.timelock._btr_decrypt")
    def test_decrypt_with_no_errors_false(self, mock_btr_decrypt):
        """Test decrypt() with no_errors=False."""
        mock_btr_decrypt.return_value = b"data"

        decrypt(b"encrypted", no_errors=False)

        mock_btr_decrypt.assert_called_once_with(b"encrypted", False)

    @patch("bittensor.core.timelock._btr_decrypt")
    def test_decrypt_with_return_str_true(self, mock_btr_decrypt):
        """Test decrypt() with return_str=True returns string."""
        mock_btr_decrypt.return_value = b"decrypted_string"

        result = decrypt(b"encrypted", return_str=True)

        assert result == "decrypted_string"
        assert isinstance(result, str)

    @patch("bittensor.core.timelock._btr_decrypt")
    def test_decrypt_with_return_str_false(self, mock_btr_decrypt):
        """Test decrypt() with return_str=False returns bytes."""
        mock_btr_decrypt.return_value = b"decrypted_bytes"

        result = decrypt(b"encrypted", return_str=False)

        assert result == b"decrypted_bytes"
        assert isinstance(result, bytes)

    @patch("bittensor.core.timelock._btr_decrypt")
    def test_decrypt_return_str_with_none_result(self, mock_btr_decrypt):
        """Test decrypt() with return_str=True when result is None."""
        mock_btr_decrypt.return_value = None

        result = decrypt(b"encrypted", return_str=True)

        assert result is None

    @patch("bittensor.core.timelock._btr_decrypt")
    def test_decrypt_with_unicode_result(self, mock_btr_decrypt):
        """Test decrypt() with unicode bytes result and return_str=True."""
        mock_btr_decrypt.return_value = "Hello 世界".encode()

        result = decrypt(b"encrypted", return_str=True)

        assert result == "Hello 世界"


class TestWaitRevealAndDecrypt:
    """Tests for the wait_reveal_and_decrypt() function."""

    @patch("bittensor.core.timelock.decrypt")
    @patch("bittensor.core.timelock.get_latest_round")
    def test_wait_reveal_with_explicit_round(self, mock_get_round, mock_decrypt):
        """Test wait_reveal_and_decrypt() with explicit reveal_round."""
        mock_get_round.return_value = 100  # Already past reveal round
        mock_decrypt.return_value = b"decrypted"

        result = wait_reveal_and_decrypt(b"encrypted", reveal_round=50)

        mock_decrypt.assert_called_once_with(b"encrypted", True, False)
        assert result == b"decrypted"

    @patch("bittensor.core.timelock.decrypt")
    @patch("bittensor.core.timelock.get_latest_round")
    def test_wait_reveal_parses_round_from_data(self, mock_get_round, mock_decrypt):
        """Test wait_reveal_and_decrypt() parses reveal_round from encrypted data."""
        # Create mock encrypted data with embedded round
        reveal_round = 12345
        encrypted_data = (
            b"some_encrypted_data"
            + TLE_ENCRYPTED_DATA_SUFFIX
            + struct.pack("<Q", reveal_round)
        )

        mock_get_round.return_value = reveal_round + 1  # Past reveal round
        mock_decrypt.return_value = b"decrypted"

        result = wait_reveal_and_decrypt(encrypted_data)

        mock_decrypt.assert_called_once()
        assert result == b"decrypted"

    @patch("bittensor.core.timelock.decrypt")
    @patch("bittensor.core.timelock.get_latest_round")
    @patch("bittensor.core.timelock.time.sleep")
    def test_wait_reveal_waits_for_round(
        self, mock_sleep, mock_get_round, mock_decrypt
    ):
        """Test wait_reveal_and_decrypt() waits until reveal round is reached."""
        # Simulate waiting: first call returns 50, second returns 100 (past reveal)
        mock_get_round.side_effect = [50, 50, 100]
        mock_decrypt.return_value = b"decrypted"

        result = wait_reveal_and_decrypt(b"encrypted", reveal_round=75)

        # Should have slept twice (while round <= 75)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(3)  # Drand QuickNet period
        assert result == b"decrypted"

    @patch("bittensor.core.timelock.decrypt")
    @patch("bittensor.core.timelock.get_latest_round")
    def test_wait_reveal_with_return_str(self, mock_get_round, mock_decrypt):
        """Test wait_reveal_and_decrypt() with return_str=True."""
        mock_get_round.return_value = 100
        mock_decrypt.return_value = "decrypted_string"

        result = wait_reveal_and_decrypt(b"encrypted", reveal_round=50, return_str=True)

        mock_decrypt.assert_called_once_with(b"encrypted", True, True)
        assert result == "decrypted_string"

    @patch("bittensor.core.timelock.decrypt")
    @patch("bittensor.core.timelock.get_latest_round")
    def test_wait_reveal_with_no_errors_false(self, mock_get_round, mock_decrypt):
        """Test wait_reveal_and_decrypt() with no_errors=False."""
        mock_get_round.return_value = 100
        mock_decrypt.return_value = b"decrypted"

        wait_reveal_and_decrypt(b"encrypted", reveal_round=50, no_errors=False)

        mock_decrypt.assert_called_once_with(b"encrypted", False, False)

    def test_wait_reveal_raises_on_invalid_data_format(self):
        """Test wait_reveal_and_decrypt() raises ValueError on invalid data format."""
        invalid_data = b"invalid_data_without_suffix"

        with pytest.raises(ValueError, match="Failed to parse reveal round"):
            wait_reveal_and_decrypt(invalid_data)

    def test_wait_reveal_raises_on_malformed_round(self):
        """Test wait_reveal_and_decrypt() raises ValueError on malformed round data."""
        # Data with suffix but invalid round bytes (too short)
        malformed_data = TLE_ENCRYPTED_DATA_SUFFIX + b"short"

        with pytest.raises(ValueError, match="Failed to parse reveal round"):
            wait_reveal_and_decrypt(malformed_data)

    def test_wait_reveal_raises_on_empty_data(self):
        """Test wait_reveal_and_decrypt() raises ValueError on empty data."""
        with pytest.raises(ValueError, match="Failed to parse reveal round"):
            wait_reveal_and_decrypt(b"")

    @patch("bittensor.core.timelock.decrypt")
    @patch("bittensor.core.timelock.get_latest_round")
    def test_wait_reveal_immediate_if_past_round(self, mock_get_round, mock_decrypt):
        """Test wait_reveal_and_decrypt() returns immediately if already past reveal round."""
        mock_get_round.return_value = 1000  # Way past reveal round
        mock_decrypt.return_value = b"decrypted"

        result = wait_reveal_and_decrypt(b"encrypted", reveal_round=100)

        # Should only call get_latest_round once (no waiting)
        assert mock_get_round.call_count == 1
        assert result == b"decrypted"


class TestRoundParsing:
    """Tests for reveal round parsing from encrypted data."""

    def test_parse_round_from_valid_encrypted_data(self):
        """Test parsing reveal round from properly formatted encrypted data."""
        reveal_round = 987654321
        encrypted_data = (
            b"prefix_data" + TLE_ENCRYPTED_DATA_SUFFIX + struct.pack("<Q", reveal_round)
        )

        parsed_round = struct.unpack(
            "<Q", encrypted_data.split(TLE_ENCRYPTED_DATA_SUFFIX)[-1]
        )[0]

        assert parsed_round == reveal_round

    def test_parse_round_little_endian(self):
        """Test that round is parsed as little-endian unsigned long long."""
        reveal_round = 0x0102030405060708
        packed = struct.pack("<Q", reveal_round)

        # Little-endian: least significant byte first
        assert packed == b"\x08\x07\x06\x05\x04\x03\x02\x01"

    def test_parse_round_max_value(self):
        """Test parsing maximum possible round value (2^64 - 1)."""
        max_round = 2**64 - 1
        encrypted_data = TLE_ENCRYPTED_DATA_SUFFIX + struct.pack("<Q", max_round)

        parsed_round = struct.unpack(
            "<Q", encrypted_data.split(TLE_ENCRYPTED_DATA_SUFFIX)[-1]
        )[0]

        assert parsed_round == max_round

    def test_parse_round_zero(self):
        """Test parsing round value of zero."""
        encrypted_data = TLE_ENCRYPTED_DATA_SUFFIX + struct.pack("<Q", 0)

        parsed_round = struct.unpack(
            "<Q", encrypted_data.split(TLE_ENCRYPTED_DATA_SUFFIX)[-1]
        )[0]

        assert parsed_round == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_zero_blocks(self, mock_btr_encrypt):
        """Test encrypt() with n_blocks=0."""
        mock_btr_encrypt.return_value = (b"encrypted", 100)

        encrypt(b"data", n_blocks=0)

        mock_btr_encrypt.assert_called_once_with(b"data", 0, 12.0)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_negative_blocks(self, mock_btr_encrypt):
        """Test encrypt() with negative n_blocks (passed to underlying function)."""
        mock_btr_encrypt.return_value = (b"encrypted", 100)

        encrypt(b"data", n_blocks=-1)

        mock_btr_encrypt.assert_called_once_with(b"data", -1, 12.0)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_float_block_time(self, mock_btr_encrypt):
        """Test encrypt() with float block_time."""
        mock_btr_encrypt.return_value = (b"encrypted", 100)

        encrypt(b"data", n_blocks=5, block_time=0.5)

        mock_btr_encrypt.assert_called_once_with(b"data", 5, 0.5)

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_with_very_small_block_time(self, mock_btr_encrypt):
        """Test encrypt() with very small block_time."""
        mock_btr_encrypt.return_value = (b"encrypted", 100)

        encrypt(b"data", n_blocks=5, block_time=0.001)

        mock_btr_encrypt.assert_called_once_with(b"data", 5, 0.001)

    @patch("bittensor.core.timelock._btr_decrypt")
    def test_decrypt_with_large_data(self, mock_btr_decrypt):
        """Test decrypt() with large encrypted data."""
        large_data = b"x" * 1000000  # 1MB
        mock_btr_decrypt.return_value = large_data

        result = decrypt(large_data)

        assert result == large_data

    @patch("bittensor.core.timelock.decrypt")
    @patch("bittensor.core.timelock.get_latest_round")
    def test_wait_reveal_with_exact_round_match(self, mock_get_round, mock_decrypt):
        """Test wait_reveal_and_decrypt() when current round equals reveal round."""
        # When current_round == reveal_round, should still wait (condition is <=)
        mock_get_round.side_effect = [100, 101]  # First equal, then past
        mock_decrypt.return_value = b"decrypted"

        with patch("bittensor.core.timelock.time.sleep"):
            result = wait_reveal_and_decrypt(b"encrypted", reveal_round=100)

        assert result == b"decrypted"

    @patch("bittensor.core.timelock._btr_encrypt")
    def test_encrypt_preserves_binary_data(self, mock_btr_encrypt):
        """Test that encrypt() correctly handles binary data with null bytes."""
        binary_data = b"\x00\x01\x02\xff\xfe\xfd"
        mock_btr_encrypt.return_value = (b"encrypted", 100)

        encrypt(binary_data, n_blocks=1)

        mock_btr_encrypt.assert_called_once_with(binary_data, 1, 12.0)


class TestModuleImports:
    """Tests for module imports and availability."""

    def test_encrypt_is_importable(self):
        """Test that encrypt function is importable from module."""
        from bittensor.core.timelock import encrypt

        assert callable(encrypt)

    def test_decrypt_is_importable(self):
        """Test that decrypt function is importable from module."""
        from bittensor.core.timelock import decrypt

        assert callable(decrypt)

    def test_wait_reveal_and_decrypt_is_importable(self):
        """Test that wait_reveal_and_decrypt function is importable from module."""
        from bittensor.core.timelock import wait_reveal_and_decrypt

        assert callable(wait_reveal_and_decrypt)

    def test_get_latest_round_is_importable(self):
        """Test that get_latest_round is re-exported from bittensor_drand."""
        from bittensor.core.timelock import get_latest_round

        assert callable(get_latest_round)

    def test_tle_suffix_is_importable(self):
        """Test that TLE_ENCRYPTED_DATA_SUFFIX constant is importable."""
        from bittensor.core.timelock import TLE_ENCRYPTED_DATA_SUFFIX

        assert TLE_ENCRYPTED_DATA_SUFFIX == b"AES_GCM_"
