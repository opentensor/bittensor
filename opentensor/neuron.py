from loguru import logger
from typing import List

import random
import requests
import torch
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter

# Import protos.
from opentensor import opentensor_pb2
import opentensor

class Neuron(nn.Module):
    """ Opentensor Neuron """
    def __init__(self,
                 identity: opentensor.Identity = None,
                 bootstrap: str = None,
                 writer: SummaryWriter = None):
        super().__init__()
        
        if writer == None:
            self._writer = SummaryWriter()
        else:
            self._writer = writer

        if identity == None:
            self._identity = opentensor.Identity()
        else:
            self._identity = identity

        if bootstrap == None:
            bootstrap = "165.227.216.95:8080"

        # Create a port map
        self._remote_ip = requests.get('https://api.ipify.org').text
        self._m_port = random.randint(10000, 60000)
        self._a_port = random.randint(10000, 60000)
        logger.info('Serving metagraph on: {}',
                    self._remote_ip + ":" + str(self._m_port))
        logger.info("Serving synapse terminal on {}",
                    self._remote_ip + ":" + str(self._a_port))

        # Inward connection handler.
        # Axon: deals with inward connections
        self._axon = opentensor.Axon(self._identity, self._a_port,
                                           self._writer)

        # Dendrite: outward connection handler.
        self._dendrite = opentensor.Dendrite(self._identity, self._remote_ip)
        # TODO (const) connection handling.

        # Metagraph: maintains a cache of synapses on the network.
        self._metagraph = opentensor.Metagraph(self._identity,
                                    max_size=100000,
                                    port=self._m_port,
                                    remote_ip=self._remote_ip,
                                    bootstrap=bootstrap)

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

    def subscribe(self, synapse: opentensor.Synapse):
        """ Subscribes an synapse to the graph """
        synapse_identity = opentensor.Identity().public_key()
        synapse_proto = opentensor_pb2.Synapse(
            version=1.0,
            neuron_key=self._identity.public_key(),
            identity=synapse_identity,
            address=self._remote_ip,
            port=str(self._a_port),
            m_port=str(self._m_port),
            indef=synapse.indef(),
            outdef=synapse.outdef())
        self._metagraph.subscribe(synapse_proto)
        self._axon.subscribe(synapse_proto, synapse)

    @property
    def identity(self):
        return self._identity

    @property
    def metagraph(self):
        return self._metagraph

    @property
    def axon(self):
        return self._axon

    @property
    def dendrite(self):
        return self._dendrite
