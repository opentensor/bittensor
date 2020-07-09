from concurrent import futures

from typing import List

import grpc
import random
import threading
import torch
from torch import nn

from opentensor import opentensor_pb2_grpc as opentensor_grpc
from opentensor import opentensor_pb2
from opentensor import Dendrite
from opentensor import Axon
from opentensor import AxonTerminal
from opentensor import Metagraph
from opentensor import Identity
import opentensor


class Neuron(nn.Module):
    """ Opentensor Neuron """
    def __init__(self, identity: opentensor.Identity = None):
        super().__init__()

        if not identity:
            self._identity = opentensor.Identity()
        else:
            self._identity = identity

        # Create a port map
        self._metagraph_address, self._metagraph_port = opentensor.Nat.create_port_map(
        )
        self._axon_address, self._axon_port = opentensor.Nat.create_port_map()

        # Inward connection handler.
        # AxonTerminal: deals with inward connections and makes connections
        # to Axon types
        self._axon = opentensor.AxonTerminal(identity, axon_port)

        # Dendrite: outward connection handler.
        self._dendrite = opentensor.Dendrite(identity)
        # TODO (const) connection handling.

        # Metagraph: maintains a cache of axons on the network.
        self._metagraph = opentensor.Metagraph(max_size=100000,
                                               port=metagraph_port)

    def __del__(self):
        self.stop()

    def start(self):
        """ Begins opentensor backend processes """
        self._axon.start()
        self._metagraph.start()

    def stop(self):
        """ Ends opentensor backend processes """
        self._axon.stop()
        self._metagraph.start()
        opentensor.Nat.delete_port_map(self._axon_port)
        opentensor.Nat.delete_port_map(self._metagraph_port)

    def axons(self) -> List[opentensor_pb2.Axons]:
        """ Returns a list of metagraph nodes to the caller """
        # TODO(const) should accept a query
        return self._metagraph.get()

    def forward(self, x: List[torch.Tensor], axons: List[opentensor_pb2.Axon]):
        """ Runs a forward request through the passed nodes """
        return self._dendrite.forward(x, axons)

    def getweights(self, axons: List[opentensor_pb2.Axon]):
        """ Returns the weights as a torch tensor for passed nodes """
        return torch.Tensor(self._metagraph.getweights(axons))

    def setweights(self, axons: List[opentensor_pb2.Axon],
                   weights: torch.Tensor):
        """ Sets weights for nodes in local storage """
        weights = weights.cpu().detach().numpy().tolist()
        self._metagraph.setweights(axons, weights)

    def subscribe(self, axon: opentensor.Axon):
        """ Subscribes an axon to the graph """
        axon_identity = opentensor.Identity().public_key()
        axon_proto = opentensor_pb2.Axon(version=1.0,
                                         public_key=self.identity.public_key(),
                                         identity=axon_identity,
                                         address=self._axon_terminal.address,
                                         port=self._axon_terminal.port,
                                         indef=axon.indef(),
                                         outdef=axon.outdef(),
                                         definition=axon.definition())
        self._metagraph.add(axon_proto)
        self._axon_terminal.add(axon)
