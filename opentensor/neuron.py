from loguru import logger
from typing import List

import torch
import torch.nn as nn 

# Import protos.
from opentensor import opentensor_pb2
import opentensor

class Neuron(nn.Module):
    """ Opentensor Neuron """
    def __init__(self, config: opentensor.Config):
        super().__init__()
        self._config = config
        # Inward connection handler.
        # Axon: deals with inward connections
        self._axon = opentensor.Axon(config)

        # Dendrite: outward connection handler.
        self._dendrite = opentensor.Dendrite(config)
        # TODO (const) connection handling.

        # Metagraph: maintains a cache of synapses on the network.
        self._metagraph = opentensor.Metagraph(config)

    def __del__(self):
        self.stop()

    def start(self):
        """ Begins opentensor backend processes """
        self._axon.start()
        self._metagraph.start()

    def stop(self):
        """ Ends opentensor backend processes """
        self._axon.stop()
        self._metagraph.stop()

    def synapses(self) -> List[opentensor_pb2.Synapse]:
        """ Returns a list of metagraph nodes to the caller """
        # TODO(const) should accept a query
        return self._metagraph.get(1000)

    def forward(self, x: List[torch.Tensor], synapses: List[opentensor_pb2.Synapse]):
        """ Runs a forward request through the passed nodes """
        return self._dendrite.forward(x, synapses)

    def getweights(self, synapses: List[opentensor_pb2.Synapse]):
        """ Returns the weights as a torch tensor for passed nodes """
        return torch.Tensor(self._metagraph.getweights(synapses))

    def setweights(self, synapses: List[opentensor_pb2.Synapse],
                   weights: torch.Tensor):
        """ Sets weights for nodes in local storage """
        weights = weights.cpu().detach().numpy().tolist()
        self._metagraph.setweights(synapses, weights)

    def subscribe(self, module: torch.nn.Module):
        """ Subscribes a synapse to the graph """
        synapse = hivemind.server.ExpertBackend(module)
        synapse_identity = opentensor.Identity().public_key()
        synapse_proto = opentensor_pb2.Synapse(
            version=1.0,
            neuron_key = self._config.identity.public_key(),
            identity = synapse_identity,
            address = self._config.remote_ip,
            port = self._config.axon_port,
            m_port = self._config.metagraph_port,
            indef = synapse.indef(),
            outdef = synapse.outdef())
        self._metagraph.subscribe(synapse_proto)
        self._axon.subscribe(synapse_proto, synapse)

    @property
    def identity(self):
        return self._config.identity

    @property
    def metagraph(self):
        return self._metagraph

    @property
    def axon(self):
        return self._axon

    @property
    def dendrite(self):
        return self._dendrite
