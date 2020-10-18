from concurrent import futures
import grpc
from loguru import logger
import os
import random
import requests
import sys
import threading
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bittensor.config import Config
from bittensor.crypto import Crypto
from bittensor.serializer import PyTorchSerializer
from bittensor.synapse import Synapse
from bittensor.axon import Axon
from bittensor.dendrite import Dendrite
from bittensor.metagraph import Metagraph
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
config = {
    "handlers": [{
        "sink":
            sys.stdout,
        "format":
            "<level>{level: <8}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    }]
}
logger.configure(**config)

# Global bittensor neuron objects.
metagraph = None
dendrite = None
axon = None


def init(config: bittensor.Config):
    # Build and start the metagraph background object.
    # The metagraph is responsible for connecting to the blockchain
    # and finding the other neurons on the network.
    global metagraph
    metagraph = bittensor.Metagraph(config)

    # Build and start the Axon server.
    # The axon server serves synapse objects (models)
    # allowing other neurons to make queries through a dendrite.
    global axon
    axon = bittensor.Axon(config)

    # Build the dendrite and router.
    # The dendrite is a torch nn.Module object which makes calls to synapses across the network
    # The router is responsible for learning which synapses to call.
    global dendrite
    dendrite = bittensor.Dendrite(config)


def serve(synapse: Synapse):
    # Subscribe the synapse object to the network.
    metagraph.subscribe(synapse)

    # Serve the synapse object on the grpc endpoint.
    axon.serve(synapse)


def start():
    # Start background threads for gossiping peers.
    metagraph.start()

    # Stop background grpc threads for serving synapse objects.
    axon.start()


def stop():
    # Start background threads for gossiping peers.
    metagraph.stop()

    # Stop background grpc threads for serving synapse objects.
    axon.stop()
