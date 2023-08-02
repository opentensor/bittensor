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
import pytest
import torch
import bittensor
import numpy


# This is a fixture that creates an example tensor for testing
@pytest.fixture
def example_tensor():
    # Create a tensor from a list using PyTorch
    data = torch.tensor([1, 2, 3, 4])

    # Serialize the tensor into a Tensor instance and return it
    return bittensor.tensor(data)


def test_deserialize(example_tensor):
    # Deserialize the tensor from the Tensor instance
    tensor = example_tensor.deserialize()

    # Check that the result is a PyTorch tensor with the correct values
    assert isinstance(tensor, torch.Tensor)
    assert tensor.tolist() == [1, 2, 3, 4]


def test_serialize(example_tensor):
    # Check that the serialized tensor is an instance of Tensor
    assert isinstance(example_tensor, bittensor.Tensor)

    # Check that the Tensor instance has the correct buffer, dtype, and shape
    assert example_tensor.buffer == example_tensor.buffer
    assert example_tensor.dtype == example_tensor.dtype
    assert example_tensor.shape == example_tensor.shape

    assert isinstance(example_tensor.tolist(), list)

    # Check that the Tensor instance has the correct buffer, dtype, and shape
    assert example_tensor.buffer == example_tensor.buffer
    assert example_tensor.dtype == example_tensor.dtype
    assert example_tensor.shape == example_tensor.shape

    assert isinstance(example_tensor.numpy(), numpy.ndarray)

    # Check that the Tensor instance has the correct buffer, dtype, and shape
    assert example_tensor.buffer == example_tensor.buffer
    assert example_tensor.dtype == example_tensor.dtype
    assert example_tensor.shape == example_tensor.shape

    assert isinstance(example_tensor.tensor(), torch.Tensor)

    # Check that the Tensor instance has the correct buffer, dtype, and shape
    assert example_tensor.buffer == example_tensor.buffer
    assert example_tensor.dtype == example_tensor.dtype
    assert example_tensor.shape == example_tensor.shape


def test_buffer_field():
    # Create a Tensor instance with a specified buffer, dtype, and shape
    tensor = bittensor.Tensor(
        buffer="0x321e13edqwds231231231232131", dtype="torch.float32", shape=[3, 3]
    )

    # Check that the buffer field matches the provided value
    assert tensor.buffer == "0x321e13edqwds231231231232131"


def test_dtype_field():
    # Create a Tensor instance with a specified buffer, dtype, and shape
    tensor = bittensor.Tensor(
        buffer="0x321e13edqwds231231231232131", dtype="torch.float32", shape=[3, 3]
    )

    # Check that the dtype field matches the provided value
    assert tensor.dtype == "torch.float32"


def test_shape_field():
    # Create a Tensor instance with a specified buffer, dtype, and shape
    tensor = bittensor.Tensor(
        buffer="0x321e13edqwds231231231232131", dtype="torch.float32", shape=[3, 3]
    )

    # Check that the shape field matches the provided value
    assert tensor.shape == [3, 3]


def test_serialize_all_types():
    bittensor.tensor(torch.tensor([1], dtype=torch.float16))
    bittensor.tensor(torch.tensor([1], dtype=torch.float32))
    bittensor.tensor(torch.tensor([1], dtype=torch.float64))
    bittensor.tensor(torch.tensor([1], dtype=torch.uint8))
    bittensor.tensor(torch.tensor([1], dtype=torch.int32))
    bittensor.tensor(torch.tensor([1], dtype=torch.int64))
    bittensor.tensor(torch.tensor([1], dtype=torch.bool))


def test_serialize_all_types_equality():
    torchtensor = torch.randn([100], dtype=torch.float16)
    assert torch.all(bittensor.tensor(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randn([100], dtype=torch.float32)
    assert torch.all(bittensor.tensor(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randn([100], dtype=torch.float64)
    assert torch.all(bittensor.tensor(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randint(255, 256, (1000,), dtype=torch.uint8)
    assert torch.all(bittensor.tensor(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randint(
        2_147_483_646, 2_147_483_647, (1000,), dtype=torch.int32
    )
    assert torch.all(bittensor.tensor(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randint(
        9_223_372_036_854_775_806, 9_223_372_036_854_775_807, (1000,), dtype=torch.int64
    )
    assert torch.all(bittensor.tensor(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randn([100], dtype=torch.float32) < 0.5
    assert torch.all(bittensor.tensor(torchtensor).tensor() == torchtensor)
