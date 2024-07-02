"""Tests for bittensor/extrinsics/__ini__ module."""

from bittensor.utils import format_error_message


def test_format_error_message_with_right_error_message():
    # Prep
    fake_error_message = {
        "type": "SomeType",
        "name": "SomeErrorName",
        "docs": ["Some error description."],
    }

    # Call
    result = format_error_message(fake_error_message)

    # Assertions

    assert "SomeType" in result
    assert "SomeErrorName" in result
    assert "Some error description." in result


def test_format_error_message_with_empty_error_message():
    # Prep
    fake_error_message = {}

    # Call
    result = format_error_message(fake_error_message)

    # Assertions

    assert "UnknownType" in result
    assert "UnknownError" in result
    assert "Unknown Description" in result


def test_format_error_message_with_wrong_type_error_message():
    # Prep
    fake_error_message = None

    # Call
    result = format_error_message(fake_error_message)

    # Assertions

    assert "UnknownType" in result
    assert "UnknownError" in result
    assert "Unknown Description" in result
