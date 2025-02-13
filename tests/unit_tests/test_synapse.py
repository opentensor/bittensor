import base64
import json
from typing import Optional, ClassVar

import pytest

from bittensor.core.synapse import Synapse


def test_parse_headers_to_inputs():
    class Test(Synapse):
        key1: list[int]

    # Define a mock headers dictionary to use for testing
    headers = {
        "bt_header_axon_nonce": "111",
        "bt_header_dendrite_ip": "12.1.1.2",
        "bt_header_input_obj_key1": base64.b64encode(
            json.dumps([1, 2, 3, 4]).encode("utf-8")
        ).decode("utf-8"),
        "timeout": "12",
        "name": "Test",
        "header_size": "111",
        "total_size": "111",
        "computed_body_hash": "0xabcdef",
    }
    print(headers)

    # Run the function to test
    inputs_dict = Test.parse_headers_to_inputs(headers)
    print(inputs_dict)
    # Check the resulting dictionary
    assert inputs_dict == {
        "axon": {"nonce": "111"},
        "dendrite": {"ip": "12.1.1.2"},
        "key1": [1, 2, 3, 4],
        "timeout": "12",
        "name": "Test",
        "header_size": "111",
        "total_size": "111",
        "computed_body_hash": "0xabcdef",
    }


def test_from_headers():
    class Test(Synapse):
        key1: list[int]

    # Define a mock headers dictionary to use for testing
    headers = {
        "bt_header_axon_nonce": "111",
        "bt_header_dendrite_ip": "12.1.1.2",
        "bt_header_input_obj_key1": base64.b64encode(
            json.dumps([1, 2, 3, 4]).encode("utf-8")
        ).decode("utf-8"),
        "timeout": "12",
        "name": "Test",
        "header_size": "111",
        "total_size": "111",
        "computed_body_hash": "0xabcdef",
    }

    # Run the function to test
    synapse = Test.from_headers(headers)

    # Check that the resulting object is an instance of YourClass
    assert isinstance(synapse, Test)

    # Check the properties of the resulting object
    # Replace with actual checks based on the structure of your class
    assert synapse.axon.nonce == 111
    assert synapse.dendrite.ip == "12.1.1.2"
    assert synapse.key1 == [1, 2, 3, 4]
    assert synapse.timeout == 12
    assert synapse.name == "Test"
    assert synapse.header_size == 111
    assert synapse.total_size == 111


def test_synapse_create():
    # Create an instance of Synapse
    synapse = Synapse()

    # Ensure the instance created is of type Synapse
    assert isinstance(synapse, Synapse)

    # Check default properties of a newly created Synapse
    assert synapse.name == "Synapse"
    assert synapse.timeout == 12.0
    assert synapse.header_size == 0
    assert synapse.total_size == 0

    # Convert the Synapse instance to a headers dictionary
    headers = synapse.to_headers()

    # Ensure the headers is a dictionary and contains the expected keys
    assert isinstance(headers, dict)
    assert "timeout" in headers
    assert "name" in headers
    assert "header_size" in headers
    assert "total_size" in headers

    # Ensure the 'name' and 'timeout' values match the Synapse's properties
    assert headers["name"] == "Synapse"
    assert headers["timeout"] == "12.0"

    # Create a new Synapse from the headers and check its 'timeout' property
    next_synapse = synapse.from_headers(synapse.to_headers())
    assert next_synapse.timeout == 12.0


def test_custom_synapse():
    # Define a custom Synapse subclass
    class Test(Synapse):
        a: int  # Carried through because required.
        b: int = None  # Not carried through headers
        c: Optional[int]  # Required, carried through headers, cannot be None
        d: Optional[list[int]]  # Required, carried though headers, cannot be None
        e: list[int]  # Carried through headers
        f: Optional[int] = (
            None  # Not Required, Not carried through headers, can be None
        )
        g: Optional[list[int]] = (
            None  # Not Required, Not carried though headers, can be None
        )

    # Create an instance of the custom Synapse subclass
    synapse = Test(
        a=1,
        c=3,
        d=[1, 2, 3, 4],
        e=[1, 2, 3, 4],
    )

    # Ensure the instance created is of type Test and has the expected properties
    assert isinstance(synapse, Test)
    assert synapse.name == "Test"
    assert synapse.a == 1
    assert synapse.b is None
    assert synapse.c == 3
    assert synapse.d == [1, 2, 3, 4]
    assert synapse.e == [1, 2, 3, 4]
    assert synapse.f is None
    assert synapse.g is None

    # Convert the Test instance to a headers dictionary
    headers = synapse.to_headers()

    # Ensure the headers contains 'a' but not 'b'
    assert "bt_header_input_obj_a" in headers
    assert "bt_header_input_obj_b" not in headers

    # Create a new Test from the headers and check its properties
    next_synapse = synapse.from_headers(synapse.to_headers())
    assert next_synapse.a == 0  # Default value is 0
    assert next_synapse.b is None
    assert next_synapse.c == 0  # Default is 0
    assert next_synapse.d == []  # Default is []
    assert next_synapse.e == []  # Empty list is default for list types
    assert next_synapse.f is None
    assert next_synapse.g is None


def test_body_hash_override():
    # Create a Synapse instance
    synapse_instance = Synapse()

    # Try to set the body_hash property and expect an AttributeError
    with pytest.raises(
        AttributeError,
        match="body_hash property is read-only and cannot be overridden.",
    ):
        synapse_instance.body_hash = []


def test_default_instance_fields_dict_consistency():
    synapse_instance = Synapse()
    assert synapse_instance.model_dump() == {
        "name": "Synapse",
        "timeout": 12.0,
        "total_size": 0,
        "header_size": 0,
        "dendrite": {
            "status_code": None,
            "status_message": None,
            "process_time": None,
            "ip": None,
            "port": None,
            "version": None,
            "nonce": None,
            "uuid": None,
            "hotkey": None,
            "signature": None,
        },
        "axon": {
            "status_code": None,
            "status_message": None,
            "process_time": None,
            "ip": None,
            "port": None,
            "version": None,
            "nonce": None,
            "uuid": None,
            "hotkey": None,
            "signature": None,
        },
        "computed_body_hash": "",
    }


class LegacyHashedSynapse(Synapse):
    """Legacy Synapse subclass that serialized `required_hash_fields`."""

    a: int
    b: int
    c: Optional[int] = None
    d: Optional[list[str]] = None
    required_hash_fields: Optional[list[str]] = ["b", "a", "d"]


class HashedSynapse(Synapse):
    a: int
    b: int
    c: Optional[int] = None
    d: Optional[list[str]] = None
    required_hash_fields: ClassVar[tuple[str, ...]] = ("a", "b", "d")


@pytest.mark.parametrize("synapse_cls", [LegacyHashedSynapse, HashedSynapse])
def test_synapse_body_hash(synapse_cls):
    synapse_instance = synapse_cls(a=1, b=2, d=["foobar"])
    assert (
        synapse_instance.body_hash
        == "ae06397d08f30f75c91395c59f05c62ac3b62b88250eb78b109213258e6ced0c"
    )

    # Extra non-hashed values should not influence the body hash
    synapse_instance_slightly_different = synapse_cls(d=["foobar"], c=3, a=1, b=2)
    assert synapse_instance.body_hash == synapse_instance_slightly_different.body_hash

    # Even if someone tries to override the required_hash_fields, it should still be the same
    synapse_instance_try_override_hash_fields = synapse_cls(
        a=1, b=2, d=["foobar"], required_hash_fields=["a"]
    )
    assert (
        synapse_instance.body_hash
        == synapse_instance_try_override_hash_fields.body_hash
    )

    # Different hashed values should result in different body hashes
    synapse_different = synapse_cls(a=1, b=2)
    assert synapse_instance.body_hash != synapse_different.body_hash
