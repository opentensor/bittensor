from typing import List

import os
import numpy as np
import torch

from opentensor import opentensor_pb2


def torch_to_bytes(key):
    key = key.cpu().detach().numpy()
    key = key.tobytes()
    return key


def bytes_to_torch(key):
    torchkey = torch.Tensor(np.frombuffer(key, dtype=np.float32))
    return torchkey


def new_key(dim):
    new_key = torch.rand(dim, dtype=torch.float32, requires_grad=False)
    return new_key


class Keys():
    def __init__(self, key_dim):
        self._key_dim = key_dim
        self._key_for_axon = {}
        self._axon_for_key = {}

    def addAxon(self, axon):
        key = new_key(self._key_dim)
        self._key_for_axon[axon.identity] = key
        self._axon_for_key[torch_to_bytes(key)] = axon

    def toKeys(self, axons: List[opentensor_pb2.Axon]):
        torch_keys = []
        for axon in axons:
            if axon.identity not in self._key_for_axon:
                self.addAxon(axon)
            torch_keys.append(self._key_for_axon[axon.identity])
        return torch.cat(torch_keys, dim=0).view(-1, self._key_dim)

    def toAxons(self, keys):
        axons = []
        for k in keys:
            kb = torch_to_bytes(k)
            assert (kb in self._axon_for_key)
            axons.append(self._axon_for_key[kb])
        return axons
