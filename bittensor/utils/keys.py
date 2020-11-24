from typing import List

import os
import numpy as np
import torch

from bittensor import bittensor_pb2


def torch_to_bytes(key):
    key = key.encode()
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
        self._key_for_neuron = {}
        self._neuron_for_key = {}

    def addNeuron(self, neuron):
        key = new_key(self._key_dim)
        self._key_for_neuron[neuron.public_key] = key
        self._neuron_for_key[torch_to_bytes(neuron.public_key)] = neuron

    def toKeys(self, neurons: List[bittensor_pb2.Neuron]):
        torch_keys = []
        for neuron in neurons:
            if neuron.public_key not in self._key_for_neuron:
                self.addNeuron(neuron)
            torch_keys.append(self._key_for_neuron[neuron.public_key])

        return torch.cat(torch_keys, dim=0).view(-1, self._key_dim)

    def toNeurons(self, keys):
        neurons = []
        for k in keys:
            kb = torch_to_bytes(k)
            assert (kb in self._neuron_for_key)
            neurons.append(self._neuron_for_key[kb])
        return neurons
