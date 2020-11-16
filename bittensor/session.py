import struct
import socket
import bittensor

from loguru import logger
from substrateinterface import Keypair
from torch.utils.tensorboard import SummaryWriter

class BTSession:
    def __init__(self, config: bittensor.Config, keypair: Keypair):
        self.config = config 
        self.__keypair = keypair
        self.metagraph = bittensor.Metagraph(self.config)
        self.axon = bittensor.Axon(self.config)
        self.dendrite = bittensor.Dendrite(self.config)
        self.tbwriter = SummaryWriter(log_dir=self.config.logdir)

    def serve(self, synapse: bittensor.Synapse):
        # Serve the synapse object on the grpc endpoint.
        self.axon.serve(synapse)

    def start(self):
        # Stop background grpc threads for serving synapse objects.
        self.axon.start()

    def stop(self):
        # Stop background grpc threads for serving synapse objects.
        self.axon.stop()

    def synapses (self):
       return self.metagraph.synapses()

    def subscribe (self):
       self.metagraph.subscribe()

    def unsubscribe (self):
        self.metagraph.unsubscribe()
