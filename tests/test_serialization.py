from bittensor.serializer import PyTorchSerializer
import numpy

import torch


def test_serialize():
    for _ in range(10):
        tensor_a = torch.rand([12, 23])
        content = PyTorchSerializer.serialize(tensor_a)
        tensor_b = PyTorchSerializer.deserialize(content)
        torch.all(torch.eq(tensor_a, tensor_b))
