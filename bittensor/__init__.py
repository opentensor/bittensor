from loguru import logger
import os
import sys
from substrateinterface import Keypair
import time
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer

import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bittensor.config import Config
from bittensor.config import ConfigService
from bittensor.config import SynapseConfig
from bittensor.crypto import Crypto
from bittensor.serializer import PyTorchSerializer
from bittensor.synapse import Synapse, SynapseOutput
from bittensor.axon import Axon
from bittensor.dendrite import Dendrite
from bittensor.metagraph import Metagraph
from bittensor.subtensor import Subtensor
from bittensor.utils.keys import Keys
from bittensor.utils.gate import Gate
from bittensor.utils.dispatcher import Dispatcher
from bittensor.utils.router import Router
import bittensor.utils.batch_transforms

__version__ = '0.0.0'
__network_dim__ = 512
__tokenizer__ = GPT2Tokenizer.from_pretrained("gpt2")
__tokenizer__.pad_token = '[PAD]'
__tokenizer__.mask_token = -100
__vocab_size__ = 204483

# Default logger
logger_config = {
    "handlers": [{
        "sink":
            sys.stdout,
        "format":
            "<level>{level: <8}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    }]
}
logger.configure(**logger_config)

# Global bittensor neuron objects.
__config = None
metagraph = None
dendrite = None
subtensor = None
axon = None
tbwriter = None
keypair = None

def init(argparser: argparse.ArgumentParser):
    global __config

    config_service = ConfigService()
    __config = config_service.create("config.ini", argparser)
    if not __config:
        logger.error("Invalid configuration. Aborting")
        quit(-1)

    __config.log()

    global keypair
    # TOOD(const): mnemonic loading.
    keypair = Keypair.create_from_uri('//' + __config.substrate_uri)

    # Build and start the metagraph background object.
    # The metagraph is responsible for connecting to the blockchain
    # and finding the other neurons on the network.
    global metagraph
    metagraph = bittensor.Metagraph(__config, keypair)

    # Build and start the Axon server.
    # The axon server serves synapse objects (models)
    # allowing other neurons to make queries through a dendrite.
    global axon
    axon = bittensor.Axon(__config)

    # Build the dendrite and router.
    # The dendrite is a torch nn.Module object which makes calls to synapses across the network
    # The router is responsible for learning which synapses to call.
    global dendrite
    dendrite = bittensor.Dendrite(__config, keypair)

    global subtensor
    subtensor = bittensor.Subtensor(__config)

    # Build bittensor tbwriter for tensorboard.
    # Logs are stored in datapath/neuronkey/logs/
    global tbwriter
    tbwriter = SummaryWriter(log_dir=__config.logdir)

def height():
    if subtensor == None:
        logger.error("INIT: bittensor is not initialized. Call bittensor.init() before calling bittensor.height().")
        quit(-1)
    return subtensor.height()

def balance():
    if subtensor == None or keypair == None:
        logger.error("INIT: bittensor is not initialized. Call bittensor.init() before calling bittensor.balance().")
        quit(-1)
    return subtensor.get_balance(keypair)

def log_output(step, output):
    bittensor.tbwriter.add_scalar('remote target loss', output.remote_target_loss.item(), step)
    bittensor.tbwriter.add_scalar('local target loss', output.local_target_loss.item(), step)
    bittensor.tbwriter.add_scalar('distilation loss', output.distillation_loss.item(), step)
    bittensor.tbwriter.add_scalar('loss', output.loss.item(), step)

def serve(synapse: Synapse):
    if metagraph == None or axon == None:
        logger.error("INIT: bittensor is not initialized. Call bittensor.init() before calling bittensor.serve().")
        quit(-1)

    # Subscribe the synapse object to the network.
    metagraph.subscribe(synapse)

    # Serve the synapse object on the grpc endpoint.
    axon.serve(synapse)

def start():
    if metagraph == None or axon == None:
        logger.error("INIT: bittensor is not initialized. Call bittensor.init() before stopping or starting backend.")
        quit(-1)

    # Start background threads for gossiping peers.
    metagraph.start()

    # Stop background grpc threads for serving synapse objects.
    axon.start()


def stop():
    if metagraph == None or axon == None:
        logger.error("INIT: bittensor is not initialized. Call bittensor.init() before stopping or starting backend.")
        quit(-1)

    # Start background threads for gossiping peers.
    metagraph.stop()

    # Stop background grpc threads for serving synapse objects.
    axon.stop()

def get_config():
    if not __config:
        logger.error("INIT: bittensor is not initialized. Call bittensor.init() before getting the config")
        quit(-1)

    return __config

