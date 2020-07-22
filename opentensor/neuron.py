from loguru import logger
from typing import List

import torch
import torch.nn as nn 

# Import protos.
from opentensor import opentensor_pb2
import opentensor

class Neuron(nn.Module):
    def __init__(self, config: opentensor.Config):
        """[summary]

        Args:
            config (opentensor.Config): [description]
        """
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
        """[summary]
        """
        self.stop()

    def start(self):
        """[summary]
        """
        self._axon.start()
        self._metagraph.start()

    def stop(self):
        """[summary]
        """
        self._axon.stop()
        self._metagraph.stop()

    def synapses(self) -> List[opentensor_pb2.Synapse]:
        """[summary]

        Returns:
            List[opentensor_pb2.Synapse]: [description]
        """
        # TODO(const) should accept a query
        return self._metagraph.get(1000)

    def forward(self, x: List[torch.Tensor], synapses: List[opentensor_pb2.Synapse]):
        """[summary]

        Args:
            x (List[torch.Tensor]): [description]
            synapses (List[opentensor_pb2.Synapse]): [description]

        Returns:
            [type]: [description]
        """
        return self._dendrite.forward(x, synapses)

    def getweights(self, synapses: List[opentensor_pb2.Synapse]):
        """[summary]

        Args:
            synapses (List[opentensor_pb2.Synapse]): [description]

        Returns:
            [type]: [description]
        """
        return torch.Tensor(self._metagraph.getweights(synapses))

    def setweights(self, synapses: List[opentensor_pb2.Synapse],
                   weights: torch.Tensor):
        """[summary]

        Args:
            synapses (List[opentensor_pb2.Synapse]): [description]
            weights (torch.Tensor): [description]
        """
        weights = weights.cpu().detach().numpy().tolist()
        self._metagraph.setweights(synapses, weights)

    def subscribe(self, module: opentensor.Synapse):
        """[summary]

        Args:
            module (opentensor.Synapse): [description]
        """
        synapse_identity = opentensor.Identity().public_key()
        synapse_proto = opentensor_pb2.Synapse(
            version = opentensor.PROTOCOL_VERSION,
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
