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
        self._key_for_synapse = {}
        self._synapse_for_key = {}

    def addSynapse(self, synapse):
        key = new_key(self._key_dim)
        self._key_for_synapse[synapse.synapse_key] = key
        self._synapse_for_key[torch_to_bytes(key)] = synapse

    def toKeys(self, synapses: List[opentensor_pb2.Synapse]):
        torch_keys = []
        for synapse in synapses:
            if synapse.synapse_key not in self._key_for_synapse:
                self.addSynapse(synapse)
            torch_keys.append(self._key_for_synapse[synapse.synapse_key])
        return torch.cat(torch_keys, dim=0).view(-1, self._key_dim)

    def toSynapses(self, keys):
        synapses = []
        for k in keys:
            kb = torch_to_bytes(k)
            assert (kb in self._synapse_for_key)
            synapses.append(self._synapse_for_key[kb])
        return synapses
