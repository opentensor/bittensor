from concurrent import futures

from typing import List

import grpc
import random
import threading
import torch
from torch import nn

from opentensor import opentensor_pb2_grpc as opentensor_grpc
from opentensor import opentensor_pb2
import opentensor


class Metagraph(nn.Module):
    def __init__(self, identity: opentensor.Identity, start=True):
        super().__init__()

        # Network identity key object.
        self.identity = identity

        # Inward connection handler.
        # Axon: deals with inward connections and messag queuing.
        self._axon = opentensor.Axon(self)
        if start:
            # Starts the axon grpc server
            self._axon.start()

        # Dendrite: outward connection handler.
        self._dendrite = opentensor.Dendrite(self)

        # Storage: maintains a cached and scored metagraph object.
        self._storage = opentensor.Storage(100000)

        # Gossip: makes periodic calls to the network to find new
        # service infor
        self._gossip = opentensor.Gossip(100)
        if start:
            self._gossip.start()

        # Local callable nodes.
        self._local_nodes = {}

    def start(self):
        """ Begins opentensor backend processes """
        self._axon.start()
        self._gossip.start()

    def stop(self):
        """ Ends opentensor backend processes """
        self._axon.stop()
        self._gossip.start()

    def nodes(self) -> List[opentensor_pb2.Node]:
        """ Returns a list of metagraph nodes to the caller """
        # TODO(const) should accept a query
        return self._storage.get()

    def Fwd(self, source_id, target_id, tensor):
        """ Query the node target_id with tensor """
        assert target_id in self._local_nodes
        node = self._local_nodes[target_id]
        tensor = node.fwd(source_id, tensor)
        return tensor

    def Bwd(self, request, context):
        """ Runs the backward request over the targeted node """
        pass

    def forward(self, x: List[torch.Tensor], nodes: List[opentensor_pb2.Node]):
        """ Runs a forward request through the passed nodes """
        return self._dendrite.forward(x, nodes)

    def getweights(self, nodes: List[opentensor_pb2.Node]):
        """ Returns the weights as a torch tensor for passed nodes """
        return torch.Tensor(self._storage.getweights(nodes))

    def setweights(self, nodes: List[opentensor_pb2.Node],
                   weights: torch.Tensor):
        """ Sets weights for nodes in local storage """
        self._storage.setweights(weights.cpu().detach().numpy().tolist())

    def subscribe(self, node: opentensor.Node):
        """ Subscribes a node to the graph """
        node_identity = opentensor.Identity().public_key()
        node_proto = opentensor_pb2.Node(version=1.0,
                                         public_key=self.identity.public_key(),
                                         identity=node_identity,
                                         address=self._axon.address,
                                         port=self._axon.port,
                                         indef=node.indef(),
                                         outdef=node.outdef(),
                                         definition=node.definition())
        self._storage.addlocal(node_proto)
        self._local_nodes[node_identity] = node

    def recv_gossip(self, subgraph):
        """ Recvs a gossip subgraph """
        self._storage.addsubgraph(subgraph)

    def send_gossip(self, node, subgraph):
        """ Sends a gossip subgraph and sinks results"""
        subgraph = self._dendrite.send_gossip(subgraph)
        self._storage.addsubgraph(node, subgraph)
