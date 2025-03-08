from bittensor.core.errors import (
    ChainError,
    HotKeyAccountNotExists,
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
