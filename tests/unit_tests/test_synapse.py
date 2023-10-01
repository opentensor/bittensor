# The MIT License (MIT)
# Copyright © 2022 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import json
import torch
import base64
import typing
import pytest
import bittensor


def test_parse_headers_to_inputs():
    class Test(bittensor.Synapse):
        key1: typing.List[int]
        key2: bittensor.Tensor

    # Define a mock headers dictionary to use for testing
    headers = {
        "bt_header_axon_nonce": "111",
        "bt_header_dendrite_ip": "12.1.1.2",
        "bt_header_input_obj_key1": base64.b64encode(
            json.dumps([1, 2, 3, 4]).encode("utf-8")
        ).decode("utf-8"),
        "bt_header_tensor_key2": "[3]-torch.float32",
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
        "key2": bittensor.Tensor(dtype="torch.float32", shape=[3]),
        "timeout": "12",
        "name": "Test",
        "header_size": "111",
        "total_size": "111",
        "computed_body_hash": "0xabcdef",
    }


def test_from_headers():
    class Test(bittensor.Synapse):
        key1: typing.List[int]
        key2: bittensor.Tensor

    # Define a mock headers dictionary to use for testing
    headers = {
        "bt_header_axon_nonce": "111",
        "bt_header_dendrite_ip": "12.1.1.2",
        "bt_header_input_obj_key1": base64.b64encode(
            json.dumps([1, 2, 3, 4]).encode("utf-8")
        ).decode("utf-8"),
        "bt_header_tensor_key2": "[3]-torch.float32",
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
    assert synapse.key2.shape == [3]
    assert synapse.key2.dtype == "torch.float32"
    assert synapse.timeout == 12
    assert synapse.name == "Test"
    assert synapse.header_size == 111
    assert synapse.total_size == 111


def test_synapse_create():
    # Create an instance of Synapse
    synapse = bittensor.Synapse()

    # Ensure the instance created is of type Synapse
    assert isinstance(synapse, bittensor.Synapse)

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
    class Test(bittensor.Synapse):
        a: int  # Carried through because required.
        b: int = None  # Not carried through headers
        c: typing.Optional[int]  # Not carried through headers
        d: typing.Optional[typing.List[int]]  # Not carried through headers
        e: typing.List[int]  # Carried through headers
        f: bittensor.Tensor  # Carried through headers, but not buffer.

    # Create an instance of the custom Synapse subclass
    synapse = Test(
        a=1,
        c=3,
        d=[1, 2, 3, 4],
        e=[1, 2, 3, 4],
        f=bittensor.Tensor.serialize(torch.randn(10)),
    )

    # Ensure the instance created is of type Test and has the expected properties
    assert isinstance(synapse, Test)
    assert synapse.name == "Test"
    assert synapse.a == 1
    assert synapse.b == None
    assert synapse.c == 3
    assert synapse.d == [1, 2, 3, 4]
    assert synapse.e == [1, 2, 3, 4]
    assert synapse.f.shape == [10]

    # Convert the Test instance to a headers dictionary
    headers = synapse.to_headers()

    # Ensure the headers contains 'a' but not 'b'
    assert "bt_header_input_obj_a" in headers
    assert "bt_header_input_obj_b" not in headers

    # Create a new Test from the headers and check its properties
    next_synapse = synapse.from_headers(synapse.to_headers())
    assert next_synapse.a == 0  # Default value is 0
    assert next_synapse.b == None
    assert next_synapse.c == None
    assert next_synapse.d == None
    assert next_synapse.e == []  # Empty list is default for list types
    assert next_synapse.f.shape == [10]  # Shape is passed through
    assert next_synapse.f.dtype == "torch.float32"  # Type is passed through
    assert next_synapse.f.buffer == None  # Buffer is not passed through


def test_list_tensors():
    class Test(bittensor.Synapse):
        a: typing.List[bittensor.Tensor]

    synapse = Test(
        a=[bittensor.Tensor.serialize(torch.randn(10))],
    )
    headers = synapse.to_headers()
    assert "bt_header_list_tensor_a" in headers
    assert headers["bt_header_list_tensor_a"] == "['[10]-torch.float32']"
    next_synapse = synapse.from_headers(synapse.to_headers())
    assert next_synapse.a[0].dtype == "torch.float32"
    assert next_synapse.a[0].shape == [10]

    class Test(bittensor.Synapse):
        a: typing.List[bittensor.Tensor]

    synapse = Test(
        a=[
            bittensor.Tensor.serialize(torch.randn(10)),
            bittensor.Tensor.serialize(torch.randn(11)),
            bittensor.Tensor.serialize(torch.randn(12)),
        ],
    )
    headers = synapse.to_headers()
    assert "bt_header_list_tensor_a" in headers
    assert (
        headers["bt_header_list_tensor_a"]
        == "['[10]-torch.float32', '[11]-torch.float32', '[12]-torch.float32']"
    )
    next_synapse = synapse.from_headers(synapse.to_headers())
    assert next_synapse.a[0].dtype == "torch.float32"
    assert next_synapse.a[0].shape == [10]
    assert next_synapse.a[1].dtype == "torch.float32"
    assert next_synapse.a[1].shape == [11]
    assert next_synapse.a[2].dtype == "torch.float32"
    assert next_synapse.a[2].shape == [12]


def test_dict_tensors():
    class Test(bittensor.Synapse):
        a: typing.Dict[str, bittensor.Tensor]

    synapse = Test(
        a={
            "cat": bittensor.tensor(torch.randn(10)),
            "dog": bittensor.tensor(torch.randn(11)),
        },
    )
    headers = synapse.to_headers()
    assert "bt_header_dict_tensor_a" in headers
    assert (
        headers["bt_header_dict_tensor_a"]
        == "['cat-[10]-torch.float32', 'dog-[11]-torch.float32']"
    )
    next_synapse = synapse.from_headers(synapse.to_headers())
    assert next_synapse.a["cat"].dtype == "torch.float32"
    assert next_synapse.a["cat"].shape == [10]
    assert next_synapse.a["dog"].dtype == "torch.float32"
    assert next_synapse.a["dog"].shape == [11]


def test_body_hash_override():
    # Create a Synapse instance
    synapse_instance = bittensor.Synapse()

    # Try to set the body_hash property and expect an AttributeError
    with pytest.raises(
        AttributeError,
        match="body_hash property is read-only and cannot be overridden.",
    ):
        synapse_instance.body_hash = []


def test_required_fields_override():
    # Create a Synapse instance
    synapse_instance = bittensor.Synapse()

    # Try to set the required_hash_fields property and expect a TypeError
    with pytest.raises(
        TypeError,
        match='"required_hash_fields" has allow_mutation set to False and cannot be assigned',
    ):
        synapse_instance.required_hash_fields = []
