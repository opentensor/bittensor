from concurrent import futures
from typing import List

import os
import sys
import random
import threading
import grpc
import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import protos.
from opentensor import opentensor_pb2
from opentensor import opentensor_pb2_grpc as opentensor_grpc

# Import objects.
from opentensor.nat import Nat
from opentensor.serializer import Serializer
from opentensor.keys import Keys
from opentensor.identity import Identity
from opentensor.axon import Axon
from opentensor.axon import AxonTerminal
from opentensor.dendrite import Dendrite
from opentensor.metagraph import Metagraph
from opentensor.gate import Gate
from opentensor.dispatcher import Dispatcher


class Neuron(nn.Module):
    """ Opentensor Neuron """
    def __init__(self, remote_ip):
        super().__init__()

        self._identity = Identity()

        # Create a port map
        self._remote_ip = remote_ip
        self._m_port = random.randint(10000, 600000)
        self._a_port = random.randint(10000, 60000)

        # Inward connection handler.
        # AxonTerminal: deals with inward connections and makes connections
        # to Axon types
        self._axon_terminal = AxonTerminal(self._identity, self._a_port)

        # Dendrite: outward connection handler.
        self._dendrite = Dendrite(self._identity)
        # TODO (const) connection handling.

        # Metagraph: maintains a cache of axons on the network.
        self._metagraph = Metagraph(self._identity,
                                    max_size=100000,
                                    port=self._m_port)

    def __del__(self):
        self.stop()

    def start(self):
        """ Begins opentensor backend processes """
        self._axon_terminal.start()
        self._metagraph.start()

    def stop(self):
        """ Ends opentensor backend processes """
        self._axon_terminal.stop()
        self._metagraph.start()

    def axons(self) -> List[opentensor_pb2.Axon]:
        """ Returns a list of metagraph nodes to the caller """
        # TODO(const) should accept a query
        return self._metagraph.get(1000)

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

    def subscribe(self, axon: Axon):
        """ Subscribes an axon to the graph """
        axon_identity = Identity().public_key()
        axon_proto = opentensor_pb2.Axon(
            version=1.0,
            neuron_key=self._identity.public_key(),
            identity=axon_identity,
            address=self._remote_ip,
            port=str(self._a_port),
            m_port=str(self._m_port),
            indef=axon.indef(),
            outdef=axon.outdef())
        self._metagraph.subscribe(axon_proto)
        self._axon_terminal.subscribe(axon_proto, axon)
