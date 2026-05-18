from async_substrate_interface.errors import SubstrateRequestException

from bittensor.core.errors import (
    ChainError,
    DelegateTakeTooHigh,
    DelegateTakeTooLow,
    HotKeyAccountNotExists,
    NonAssociatedColdKey,
    SubnetNotExists,
    TxRateLimitExceeded,
    chain_error_from_substrate_exception,
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
        assert (
            result
            == "Failed to parse shielded transaction: the ciphertext has an invalid format."
        )

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


def _validity_error(code: int) -> SubstrateRequestException:
    payload = {
        "jsonrpc": "2.0",
        "id": "Xc18",
        "error": {
            "code": 1010,
            "message": "Invalid Transaction",
            "data": f"Custom error: {code}",
        },
    }
    return SubstrateRequestException(str(payload))


class TestChainErrorFromSubstrateException:
    def test_maps_hotkey_account_not_exists(self):
        exc = chain_error_from_substrate_exception(_validity_error(4))
        assert isinstance(exc, HotKeyAccountNotExists)

    def test_maps_non_associated_coldkey(self):
        exc = chain_error_from_substrate_exception(_validity_error(25))
        assert isinstance(exc, NonAssociatedColdKey)

    def test_maps_rate_limit_exceeded(self):
        exc = chain_error_from_substrate_exception(_validity_error(6))
        assert isinstance(exc, TxRateLimitExceeded)

    def test_maps_subnet_not_exists(self):
        exc = chain_error_from_substrate_exception(_validity_error(3))
        assert isinstance(exc, SubnetNotExists)

    def test_maps_delegate_take_too_low(self):
        exc = chain_error_from_substrate_exception(_validity_error(26))
        assert isinstance(exc, DelegateTakeTooLow)

    def test_maps_delegate_take_too_high(self):
        exc = chain_error_from_substrate_exception(_validity_error(27))
        assert isinstance(exc, DelegateTakeTooHigh)

    def test_unmapped_code_returns_none(self):
        assert chain_error_from_substrate_exception(_validity_error(255)) is None

    def test_non_validity_error_returns_none(self):
        err = SubstrateRequestException("not a validity error")
        assert chain_error_from_substrate_exception(err) is None

    def test_already_typed_returns_none(self):
        err = HotKeyAccountNotExists("already typed")
        assert chain_error_from_substrate_exception(err) is None
