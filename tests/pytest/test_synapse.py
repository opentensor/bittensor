import pytest
from pydantic import ValidationError
from bittensor.synapse import TerminalInfo, Synapse


# Test cases for Synapse
def test_name_field():
    """Tests for the 'name' field validation."""
    # Valid case
    model = Synapse(name="Forward")
    assert model.name == "Forward"

    # Invalid case
    with pytest.raises(ValidationError):
        Synapse(name=123)  # Non-string value


def test_timeout_field():
    """Tests for the 'timeout' field validation."""
    # Valid case
    model = Synapse(timeout=12.0)
    assert model.timeout == 12.0

    # Invalid case
    with pytest.raises(ValidationError):
        Synapse(timeout="not a float")


def test_total_size_field():
    """Tests for the 'total_size' field validation."""
    # Valid case
    model = Synapse(total_size=1000)
    assert model.total_size == 1000

    # Invalid case
    with pytest.raises(ValidationError):
        Synapse(total_size="not an int")


def test_header_size_field():
    """Tests for the 'header_size' field validation."""
    # Valid case
    model = Synapse(header_size=500)
    assert model.header_size == 500

    # Invalid case
    with pytest.raises(ValidationError):
        Synapse(header_size="not an int")


def test_dendrite_field():
    """Tests for the 'dendrite' field validation."""
    dendrite_info = TerminalInfo()  # Assuming valid TerminalInfo instance
    model = Synapse(dendrite=dendrite_info)
    assert model.dendrite == dendrite_info


def test_axon_field():
    """Tests for the 'axon' field validation."""
    axon_info = TerminalInfo()  # Assuming valid TerminalInfo instance
    model = Synapse(axon=axon_info)
    assert model.axon == axon_info


def test_computed_body_hash_field():
    """Tests for the 'computed_body_hash' field validation."""
    # Valid case
    model = Synapse(computed_body_hash="0x123")
    assert model.computed_body_hash == "0x123"

    # Invalid case
    with pytest.raises(ValidationError):
        Synapse(computed_body_hash=123)  # Non-string value


def test_required_hash_fields_field():
    """Tests for the 'required_hash_fields' field validation."""
    # Valid case
    model = Synapse(required_hash_fields=["roles", "messages"])
    assert model.required_hash_fields == ["roles", "messages"]

    # Invalid case
    with pytest.raises(ValidationError):
        Synapse(required_hash_fields="not a list")