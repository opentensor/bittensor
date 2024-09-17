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

import numpy as np

from bittensor.core.tensor import Tensor


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


def test_serialize():
    """Test the deserialization of the Tensor instance."""
    # Preps
    tensor_ = np.ndarray([3, 3], dtype=np.float32)

    # Call
    tensor = Tensor.serialize(tensor_)

    # Asserts
    assert (
        tensor.buffer
        == "hcQCbmTDxAR0eXBlozxmNMQEa2luZMQAxAVzaGFwZZIDA8QEZGF0YcQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    )
    assert tensor.dtype == "float32"
    assert tensor.shape == [3, 3]


def test_deserialize():
    """Test the deserialization of the Tensor instance."""
    # Preps
    tensor = Tensor(
        buffer="hcQCbmTDxAR0eXBlozxmNMQEa2luZMQAxAVzaGFwZZIDA8QEZGF0YcQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        dtype="float32",
        shape=[3, 3],
    )

    # Call
    result = tensor.deserialize()

    # Asserts
    assert np.array_equal(result, np.zeros((3, 3)))
