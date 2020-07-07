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
        self._axon_address = 'localhost'
        self._axon_port = str(random.randint(8000, 30000))
        self._axon = opentensor.Axon(self)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        opentensor_grpc.add_OpentensorServicer_to_server(
            self._axon, self._server)
        self._server.add_insecure_port('[::]:' + self._axon_port)

        # Dendrite: outward connection handler.
        self._dendrite = opentensor.Dendrite(self)

        # Internal state.
        self._nodes = {}
        self._local_nodes = {}
        self._weights = {self.identity.public_key(): {}}
        self._local_node_protos = {}

        # Build grpc server.
        self._thread = None

        if start:
            self.start()

    def __del__(self):
        self.stop()

    def start(self):
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._server.stop(0)

    def _serve(self):
        print('serving metagraph ...')
        self._server.start()

    def nodes(self) -> List[opentensor_pb2.Node]:
        return list(self._nodes.values())

    def Fwd(self, source_id, target_id, tensor):
        assert target_id in self._local_nodes
        node = self._local_nodes[target_id]
        tensor = node.fwd(source_id, tensor)
        return tensor

    def Bwd(self, request, context):
        pass

    def forward(self, x: List[torch.Tensor], nodes: List[opentensor_pb2.Node]):
        return self._dendrite.forward(x, nodes)

    def getweights(self, nodes: List[opentensor_pb2.Node]):
        weights = []
        for n in nodes:
            if n.identity not in self._weights[self.identity.public_key()]:
                weights.append(0.0)
            else:
                weights.append(self._weights[self.identity.public_key()][
                    n.identity].value)
        return torch.Tensor(weights)

    def setweights(self, nodes: List[opentensor_pb2.Node],
                   weights: torch.Tensor):
        weights = weights.cpu().detach().numpy().tolist()
        for n, w in list(zip(nodes, weights)):
            w_proto = opentensor_pb2.Weight(source=self.identity.public_key(),
                                            target=n.identity,
                                            value=w)
            self._weights[self.identity.public_key()][n.identity] = w_proto

    def subscribe(self, node: opentensor.Node):
        node_identity = opentensor.Identity().public_key()
        assert node_identity not in self._nodes
        node_proto = opentensor_pb2.Node(version=1.0,
                                         public_key=self.identity.public_key(),
                                         identity=node_identity,
                                         address=self._axon_address,
                                         port=self._axon_port,
                                         indef=node.indef(),
                                         outdef=node.outdef(),
                                         definition=node.definition())
        self._nodes[node_identity] = node_proto
        self._local_node_protos[node_identity] = node_proto
        self._local_nodes[node_identity] = node
        print(str(node_proto))
        self.refresh()

    def recv_gossip(self, metagraph):
        self._storage.recv_gossip(metagraph)

    def gossip(self):
        # Send graph.
        neuron = opentensor_pb2.Neuron(
            public_key=self.identity.public_key(),
            nodes=self._local_node_protos.values(),
            weights=self._weights[self.identity.public_key()].values(),
        )
        metagraph = opentensor_pb2.Metagraph(neurons=[neuron])
        node = random.choice(self._nodes.values())
        metagraphs = self._dendrite.gossip(node, metagraph)
        self._storage.update(metagraphs)
