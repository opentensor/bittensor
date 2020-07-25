from loguru import logger
from typing import List

import torch
import torch.nn as nn 

# Import protos.
from opentensor import opentensor_pb2
import opentensor

class Neuron(nn.Module):
    """ Auto-grad friendly Torch NN module which maintains a connection to a network of other neuons connected across the web. 
    """
    def __init__(self, config: opentensor.Config):
        """Initializes a new background Neuron object.

        Args:
            config (opentensor.Config): Config object containing information relevant to create a new neuron object.
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

    def synapses(self) -> List[opentensor_pb2.Synapse]:
        """Returns an unordered list of Synapse objects.

        Returns:
            List[opentensor_pb2.Synapse]: List of opentensor_pb2.Synapses from the metagraph.
        """
        # TODO(const) should accept a query
        return self._metagraph.get(1000)

    def forward(self, x: List[torch.Tensor], synapses: List[opentensor_pb2.Synapse]):
        """Makes a series of forward requests to synapses passing corresponding inputs.

        Args:
            x (List[torch.Tensor]): List of torch.Tensor inputs for corresponding Synapses.
            synapses (List[opentensor_pb2.Synapse]): List of Synapse objects.

        Returns:
            [type]: List of torch.Tensor responses from Synapse service definitions.
        """
        return self._dendrite.forward(x, synapses)

    def getweights(self, synapses: List[opentensor_pb2.Synapse]):
        """Get the weights for list of Synapse endpoints.

        Args:
            synapses (List[opentensor_pb2.Synapse]): Synapses to get weights for.

        Returns:
            [type]: Weights set for each synapse.
        """
        return torch.Tensor(self._metagraph.getweights(synapses))

    def setweights(self, synapses: List[opentensor_pb2.Synapse],
                   weights: torch.Tensor):
        """Sets the weights for these synapses given equal length list of weights.

        Args:
            synapses (List[opentensor_pb2.Synapse]): Synapses to set weights.
            weights (torch.Tensor): Weights to set.
        """
        weights = weights.cpu().detach().numpy().tolist()
        self._metagraph.setweights(synapses, weights)

    def subscribe(self, module: opentensor.Synapse):
        """Subscribes a synapse class object to the metagraph.

        Args:
            module (opentensor.Synapse): opentensor.Synapse class object to subscribe.
        """
        # Create a new opentensor_pb2.Synapse proto.
        synapse_identity = opentensor.Identity().public_key()
        synapse_proto = opentensor_pb2.Synapse(
            version = opentensor.PROTOCOL_VERSION,
            neuron_key = self._config.identity.public_key(),
            identity = synapse_identity,
            address = self._config.remote_ip,
            port = self._config.axon_port,
            indef = synapse.indef(),
            outdef = synapse.outdef())
        self._metagraph.subscribe(synapse_proto)
        self._axon.subscribe(synapse_proto, synapse)

    def __del__(self):
        """Stops background threads and destroys object.
        """
        self.stop()

    def start(self):
        """Starts background threads.
        """
        self._axon.start()
        self._metagraph.start()

    def stop(self):
        """Stops background threads.
        """
        self._axon.stop()
        self._metagraph.stop()

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
