import opentensor
import numpy

import torch

def test_serialize():
    for _ in range(10):
        tensor_a = torch.rand([12,23])
        content = opentensor.Serializer.serialize(tensor_a)
        tensor_b = opentensor.Serializer.deserialize(content)
        torch.all(torch.eq(tensor_a, tensor_b))
