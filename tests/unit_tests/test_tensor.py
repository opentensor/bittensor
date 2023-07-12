# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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
import bittensor as bt

@pytest.fixture
def example_tensor():
    # Create an example tensor for testing
    data = torch.tensor([1, 2, 3, 4])
    return bt.Tensor.serialize(data)

def test_deserialize(example_tensor):
    # Test deserialization of tensor
    tensor = example_tensor.deserialize()
    assert isinstance(tensor, torch.Tensor)
    assert tensor.tolist() == [1, 2, 3, 4]

def test_serialize(example_tensor):
    # Test serialization of tensor
    assert isinstance(example_tensor, bt.Tensor)
    assert example_tensor.buffer == example_tensor.buffer
    assert example_tensor.dtype == example_tensor.dtype
    assert example_tensor.shape == example_tensor.shape

def test_buffer_field():
    # Test buffer field
    tensor = bt.Tensor(buffer='0x321e13edqwds231231231232131', dtype='torch.float32', shape=[3, 3])
    assert tensor.buffer == '0x321e13edqwds231231231232131'

def test_dtype_field():
    # Test dtype field
    tensor = bt.Tensor(buffer='0x321e13edqwds231231231232131', dtype='torch.float32', shape=[3, 3])
    assert tensor.dtype == 'torch.float32'

def test_shape_field():
    # Test shape field
    tensor = bt.Tensor(buffer='0x321e13edqwds231231231232131', dtype='torch.float32', shape=[3, 3])
    assert tensor.shape == [3, 3]