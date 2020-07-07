from opentensor import opentensor_pb2_grpc as opentensor_grpc
from opentensor import opentensor_pb2
import opentensor

from typing import List
import torch


def torch_to_bytes(key):
    key = key.cpu().detach().numpy()
    key = key.tobytes()
    return key


def bytes_to_torch(key):
    torchkey = torch.Tensor(np.frombuffer(key, dtype=np.float32))
    return torchkey


def new_node_id():
    return os.urandom(12)


def new_key(dim):
    new_key = torch.rand(dim, dtype=torch.float32, requires_grad=False)
    return new_key


class Keys():
    def __init__(self, key_dim):
        self._key_dim = key_dim
        self._key_for_node = {}
        self._node_for_key = {}

    def addNode(self, node):
        key = new_key(self._key_dim)
        self._key_for_node[node.identity] = key
        self._node_for_key[torch_to_bytes(key)] = node

    def toKeys(self, nodes: List[opentensor_pb2.Node]):
        torch_keys = []
        for node in nodes:
            if node.identity not in self._key_for_node:
                self.addNode(node)
            torch_keys.append(self._key_for_node[node.identity])
        return torch.cat(torch_keys, dim=0).view(-1, self._key_dim)

    def toNodes(self, keys):
        nodes = []
        for k in keys:
            kb = torch_to_bytes(k)
            assert (kb in self._node_for_key)
            nodes.append(self._node_for_key[kb])
        return nodes
