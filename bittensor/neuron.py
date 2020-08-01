from loguru import logger
from typing import List

import torch
import torch.nn as nn 

# Import protos.
from bittensor import bittensor_pb2
import bittensor
import bittensor

class Neuron(nn.Module):
    """ Auto-grad friendly Torch NN module which maintains a connection to a network of other neuons connected across the web. 
    """
    def __init__(self, config: bittensor.Config):
        """Initializes a new background Neuron object.

        Args:
            config (bittensor.Config): Config object containing information relevant to create a new neuron object.
        """
        super().__init__()
        self._config = config
        # Inward connection handler.
        # Axon: deals with inward connections
        self._axon = bittensor.Axon(config)

        # Dendrite: outward connection handler.
        self._dendrite = bittensor.Dendrite(config)
        # TODO (const) connection handling.

        # Metagraph: maintains a cache of synapses on the network.
        self._metagraph = bittensor.Metagraph(config)

    def synapses(self) -> List[bittensor_pb2.Synapse]:
        """Returns an unordered list of Synapse objects.

        Returns:
            List[bittensor_pb2.Synapse]: List of bittensor_pb2.Synapses from the metagraph.
        """
        # TODO(const) should accept a query
        return self._metagraph.get_synapses(1000)

    def forward(self, x: List[torch.Tensor], synapses: List[bittensor_pb2.Synapse]):
        """Makes a series of forward requests to synapses passing corresponding inputs.

        Args:
            x (List[torch.Tensor]): List of torch.Tensor inputs for corresponding Synapses.
            synapses (List[bittensor_pb2.Synapse]): List of Synapse objects.

        Returns:
            [type]: List of torch.Tensor responses from Synapse service definitions.
        """
        return self._dendrite.forward(synapses, x)

    def getweights(self, synapses: List[bittensor_pb2.Synapse]):
        """Get the weights for list of Synapse endpoints.

        Args:
            synapses (List[bittensor_pb2.Synapse]): Synapses to get weights for.

        Returns:
            [type]: Weights set for each synapse.
        """
        return torch.Tensor(self._metagraph.getweights(synapses))

    def setweights(self, synapses: List[bittensor_pb2.Synapse],
                   weights: torch.Tensor):
        """Sets the weights for these synapses given equal length list of weights.

        Args:
            synapses (List[bittensor_pb2.Synapse]): Synapses to set weights.
            weights (torch.Tensor): Weights to set.
        """
        weights = weights.cpu().detach().numpy().tolist()
        self._metagraph.setweights(synapses, weights)

    def subscribe(self, synapse: bittensor.Synapse):
        """Subscribes a synapse class object to the metagraph.

        Args:
            module (bittensor.Synapse): bittensor.Synapse class object to subscribe.
        """
        # Create a new bittensor_pb2.Synapse proto.
        synapse_proto = bittensor_pb2.Synapse(
            version = bittensor.__version__,
            neuron_key = self._config.neuron_key,
            synapse_key = synapse.synapse_key(),
            address = self._config.remote_ip,
            port = self._config.axon_port,
            indef = synapse.indef(),
            outdef = synapse.outdef())
        logger.info("subscribe synapse: {}", synapse_proto)
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
    def neuron_key(self):
        return self._config.neuron_key

    @property
    def metagraph(self):
        return self._metagraph

    @property
    def axon(self):
        return self._axon

    @property
    def dendrite(self):
        return self._dendrite
