from bittensor.core.errors import (
    ChainError,
    HotKeyAccountNotExists,
    map_shield_error,
)


def test_from_error():
    error = {
        "type": "Module",
        "name": "HotKeyAccountNotExists",
        "docs": ["The hotkey does not exists"],
    }

    exception = ChainError.from_error(error)

    assert isinstance(exception, HotKeyAccountNotExists)
    assert exception.args[0] == "The hotkey does not exists"


def test_from_error_unsupported_exception():
    error = {
        "type": "Module",
        "name": "UnknownException",
        "docs": ["Unknown"],
    }

    exception = ChainError.from_error(error)

    assert isinstance(exception, ChainError)
    assert exception.args[0] == error


def test_from_error_new_exception():
    error = {
        "type": "Module",
        "name": "NewException",
        "docs": ["New"],
    }

    exception = ChainError.from_error(error)

    assert isinstance(exception, ChainError)

    class NewException(ChainError):
        pass

    exception = ChainError.from_error(error)

    assert isinstance(exception, NewException)


class TestMapShieldError:
    def test_custom_error_23_maps_to_parsing_failure(self):
        msg = "Subtensor returned `SubstrateRequestException(Verification Error)` error. This means: `Custom error: 23 | Please consult docs`."
        result = map_shield_error(msg)
        assert result == "Failed to parse shielded transaction: the ciphertext has an invalid format."

    def test_custom_error_24_maps_to_invalid_key_hash(self):
        msg = "Subtensor returned `SubstrateRequestException(Verification Error)` error. This means: `Custom error: 24 | Please consult docs`."
        result = map_shield_error(msg)
        assert "key_hash" in result
        assert "does not match any known key" in result

    def test_generic_invalid_status_maps_to_catchall(self):
        msg = "Subtensor returned: Subscription abc123 invalid: {'jsonrpc': '2.0', 'params': {'result': 'invalid'}}"
        result = map_shield_error(msg)
        assert "MEV Shield extrinsic rejected as invalid" in result

    def test_unrelated_error_returned_unchanged(self):
        msg = "Something completely unrelated went wrong"
        result = map_shield_error(msg)
        assert result == msg
