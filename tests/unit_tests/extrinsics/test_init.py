"""Tests for bittensor/extrinsics/__ini__ module."""

from bittensor.utils import format_error_message


def test_format_error_message_with_right_error_message():
    """Verify that error message from extrinsic response parses correctly."""
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
    """Verify that empty error message from extrinsic response parses correctly."""
    # Prep
    fake_error_message = {}

    # Call
    result = format_error_message(fake_error_message)

    # Assertions

    assert "UnknownType" in result
    assert "UnknownError" in result
    assert "Unknown Description" in result


def test_format_error_message_with_wrong_type_error_message():
    """Verify that error message from extrinsic response with wrong type parses correctly."""
    # Prep
    fake_error_message = None

    # Call
    result = format_error_message(fake_error_message)

    # Assertions

    assert "UnknownType" in result
    assert "UnknownError" in result
    assert "Unknown Description" in result


def test_format_error_message_with_custom_error_message_with_index(mocker):
    """Tests error formatter if subtensor error is custom error with index."""
    # Preps
    fake_custom_error = {
        "code": 1010,
        "message": "SomeErrorName",
        "data": "Custom error: 1",
    }
    fake_subtensor_error = {
        "docs": ["Some description"],
        "fields": [],
        "index": 1,
        "name": "SomeErrorName",
    }

    fake_substrate = mocker.MagicMock()
    fake_substrate.metadata.get_metadata_pallet().errors.__getitem__().value.get = (
        mocker.Mock(
            side_effect=[fake_custom_error["message"], fake_subtensor_error["docs"]]
        )
    )

    mocker.patch(
        "substrateinterface.base.SubstrateInterface", return_value=fake_substrate
    )

    # Call
    result = format_error_message(fake_custom_error, fake_substrate)

    # Assertions
    assert (
        result
        == f"Subtensor returned `SubstrateRequestException({fake_subtensor_error['name']})` error. This means: `Some description`."
    )


def test_format_error_message_with_custom_error_message_without_index(mocker):
    """Tests error formatter if subtensor error is custom error without index."""
    # Preps
    fake_custom_error = {
        "code": 1010,
        "message": "SomeErrorType",
        "data": "Custom error description",
    }
    fake_substrate = mocker.MagicMock()
    fake_substrate.metadata.get_metadata_pallet().errors.__getitem__().value.get.return_value = fake_custom_error[
        "message"
    ]
    mocker.patch(
        "substrateinterface.base.SubstrateInterface", return_value=fake_substrate
    )

    # Call
    result = format_error_message(fake_custom_error, fake_substrate)

    # Assertions
    assert (
        result
        == f"Subtensor returned `SubstrateRequestException({fake_custom_error['message']})` error. This means: `{fake_custom_error['data']}`."
    )
