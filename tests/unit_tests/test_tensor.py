# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import numpy
import numpy as np
import pytest
import torch

from bittensor.core.tensor import Tensor


# This is a fixture that creates an example tensor for testing
@pytest.fixture
def example_tensor():
    # Create a tensor from a list using PyTorch
    data = np.array([1, 2, 3, 4])

    # Serialize the tensor into a Tensor instance and return it
    return Tensor.serialize(data)


@pytest.fixture
def example_tensor_torch(force_legacy_torch_compatible_api):
    # Create a tensor from a list using PyTorch
    data = torch.tensor([1, 2, 3, 4])

    # Serialize the tensor into a Tensor instance and return it
    return Tensor.serialize(data)


def test_deserialize(example_tensor):
    # Deserialize the tensor from the Tensor instance
    tensor = example_tensor.deserialize()

    # Check that the result is a np.array with the correct values
    assert isinstance(tensor, np.ndarray)
    assert tensor.tolist() == [1, 2, 3, 4]


def test_deserialize_torch(example_tensor_torch, force_legacy_torch_compatible_api):
    tensor = example_tensor_torch.deserialize()
    # Check that the result is a PyTorch tensor with the correct values
    assert isinstance(tensor, torch.Tensor)
    assert tensor.tolist() == [1, 2, 3, 4]


def test_serialize(example_tensor):
    # Check that the serialized tensor is an instance of Tensor
    assert isinstance(example_tensor, Tensor)

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

    assert isinstance(example_tensor.tensor(), np.ndarray)

    # Check that the Tensor instance has the correct buffer, dtype, and shape
    assert example_tensor.buffer == example_tensor.buffer
    assert example_tensor.dtype == example_tensor.dtype
    assert example_tensor.shape == example_tensor.shape


def test_serialize_torch(example_tensor_torch, force_legacy_torch_compatible_api):
    # Check that the serialized tensor is an instance of Tensor
    assert isinstance(example_tensor_torch, Tensor)

    # Check that the Tensor instance has the correct buffer, dtype, and shape
    assert example_tensor_torch.buffer == example_tensor_torch.buffer
    assert example_tensor_torch.dtype == example_tensor_torch.dtype
    assert example_tensor_torch.shape == example_tensor_torch.shape

    assert isinstance(example_tensor_torch.tolist(), list)

    # Check that the Tensor instance has the correct buffer, dtype, and shape
    assert example_tensor_torch.buffer == example_tensor_torch.buffer
    assert example_tensor_torch.dtype == example_tensor_torch.dtype
    assert example_tensor_torch.shape == example_tensor_torch.shape

    assert isinstance(example_tensor_torch.numpy(), numpy.ndarray)

    # Check that the Tensor instance has the correct buffer, dtype, and shape
    assert example_tensor_torch.buffer == example_tensor_torch.buffer
    assert example_tensor_torch.dtype == example_tensor_torch.dtype
    assert example_tensor_torch.shape == example_tensor_torch.shape

    assert isinstance(example_tensor_torch.tensor(), torch.Tensor)

    # Check that the Tensor instance has the correct buffer, dtype, and shape
    assert example_tensor_torch.buffer == example_tensor_torch.buffer
    assert example_tensor_torch.dtype == example_tensor_torch.dtype
    assert example_tensor_torch.shape == example_tensor_torch.shape


def test_buffer_field():
    # Create a Tensor instance with a specified buffer, dtype, and shape
    tensor = Tensor(
        buffer="0x321e13edqwds231231231232131", dtype="float32", shape=[3, 3]
    )

    # Check that the buffer field matches the provided value
    assert tensor.buffer == "0x321e13edqwds231231231232131"


def test_buffer_field_torch(force_legacy_torch_compatible_api):
    # Create a Tensor instance with a specified buffer, dtype, and shape
    tensor = Tensor(
        buffer="0x321e13edqwds231231231232131", dtype="torch.float32", shape=[3, 3]
    )

    # Check that the buffer field matches the provided value
    assert tensor.buffer == "0x321e13edqwds231231231232131"


def test_dtype_field():
    # Create a Tensor instance with a specified buffer, dtype, and shape
    tensor = Tensor(
        buffer="0x321e13edqwds231231231232131", dtype="float32", shape=[3, 3]
    )

    # Check that the dtype field matches the provided value
    assert tensor.dtype == "float32"


def test_dtype_field_torch(force_legacy_torch_compatible_api):
    tensor = Tensor(
        buffer="0x321e13edqwds231231231232131", dtype="torch.float32", shape=[3, 3]
    )
    assert tensor.dtype == "torch.float32"


def test_shape_field():
    # Create a Tensor instance with a specified buffer, dtype, and shape
    tensor = Tensor(
        buffer="0x321e13edqwds231231231232131", dtype="float32", shape=[3, 3]
    )

    # Check that the shape field matches the provided value
    assert tensor.shape == [3, 3]


def test_shape_field_torch(force_legacy_torch_compatible_api):
    tensor = Tensor(
        buffer="0x321e13edqwds231231231232131", dtype="torch.float32", shape=[3, 3]
    )
    assert tensor.shape == [3, 3]


def test_serialize_all_types():
    Tensor.serialize(np.array([1], dtype=np.float16))
    Tensor.serialize(np.array([1], dtype=np.float32))
    Tensor.serialize(np.array([1], dtype=np.float64))
    Tensor.serialize(np.array([1], dtype=np.uint8))
    Tensor.serialize(np.array([1], dtype=np.int32))
    Tensor.serialize(np.array([1], dtype=np.int64))
    Tensor.serialize(np.array([1], dtype=bool))


def test_serialize_all_types_torch(force_legacy_torch_compatible_api):
    Tensor.serialize(torch.tensor([1], dtype=torch.float16))
    Tensor.serialize(torch.tensor([1], dtype=torch.float32))
    Tensor.serialize(torch.tensor([1], dtype=torch.float64))
    Tensor.serialize(torch.tensor([1], dtype=torch.uint8))
    Tensor.serialize(torch.tensor([1], dtype=torch.int32))
    Tensor.serialize(torch.tensor([1], dtype=torch.int64))
    Tensor.serialize(torch.tensor([1], dtype=torch.bool))


def test_serialize_all_types_equality():
    rng = np.random.default_rng()

    tensor = rng.standard_normal((100,), dtype=np.float32)
    assert np.all(Tensor.serialize(tensor).tensor() == tensor)

    tensor = rng.standard_normal((100,), dtype=np.float64)
    assert np.all(Tensor.serialize(tensor).tensor() == tensor)

    tensor = np.random.randint(255, 256, (1000,), dtype=np.uint8)
    assert np.all(Tensor.serialize(tensor).tensor() == tensor)

    tensor = np.random.randint(2_147_483_646, 2_147_483_647, (1000,), dtype=np.int32)
    assert np.all(Tensor.serialize(tensor).tensor() == tensor)

    tensor = np.random.randint(
        9_223_372_036_854_775_806, 9_223_372_036_854_775_807, (1000,), dtype=np.int64
    )
    assert np.all(Tensor.serialize(tensor).tensor() == tensor)

    tensor = rng.standard_normal((100,), dtype=np.float32) < 0.5
    assert np.all(Tensor.serialize(tensor).tensor() == tensor)


def test_serialize_all_types_equality_torch(force_legacy_torch_compatible_api):
    torchtensor = torch.randn([100], dtype=torch.float16)
    assert torch.all(Tensor.serialize(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randn([100], dtype=torch.float32)
    assert torch.all(Tensor.serialize(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randn([100], dtype=torch.float64)
    assert torch.all(Tensor.serialize(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randint(255, 256, (1000,), dtype=torch.uint8)
    assert torch.all(Tensor.serialize(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randint(
        2_147_483_646, 2_147_483_647, (1000,), dtype=torch.int32
    )
    assert torch.all(Tensor.serialize(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randint(
        9_223_372_036_854_775_806, 9_223_372_036_854_775_807, (1000,), dtype=torch.int64
    )
    assert torch.all(Tensor.serialize(torchtensor).tensor() == torchtensor)

    torchtensor = torch.randn([100], dtype=torch.float32) < 0.5
    assert torch.all(Tensor.serialize(torchtensor).tensor() == torchtensor)
