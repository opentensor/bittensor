import struct
import socket
import bittensor
import subprocess

from bittensor.synapse import Synapse
from bittensor.dendrite import Dendrite
from bittensor.axon import Axon
from bittensor.metagraph import Metagraph
from bittensor.pysubtensor import WSClient

from loguru import logger
from substrateinterface import Keypair
from torch.utils.tensorboard import SummaryWriter

class SubtensorProcess:
    def __init__(self, config):
        self._config = config
        self._process = process
    
    def start(self):
        args = ['./subtensor/target/release/node-subtensor', '--dev']
        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def stop(self):
        self._process.kill()

class BTSession:
    def __init__(self, config, keypair: Keypair):
        self.config = config 
        self.__keypair = keypair
        self.metagraph = Metagraph(self.config, self.__keypair)
        self.axon = Axon(self.config, self.__keypair)
        self.dendrite = Dendrite(self.config, self.__keypair)
        self.tbwriter = SummaryWriter(log_dir=self.config.session_settings.logdir)
        self.subtensor_process = SubtensorProcess(self.config)

    def serve(self, synapse: Synapse):
        # Serve the synapse object on the grpc endpoint.
        self.axon.serve(synapse)

    def start(self):
        # Stop background grpc threads for serving synapse objects.
        self.axon.start()
        self.subtensor_process.start()

    def stop(self):
        # Stop background grpc threads for serving synapse objects.
        self.axon.stop()
        self.subtensor_process.stop()

    def synapses (self):
       return self.metagraph.synapses()

    def subscribe (self):
       self.metagraph.subscribe()

    def unsubscribe (self):
        self.metagraph.unsubscribe()
