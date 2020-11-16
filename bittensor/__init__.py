from loguru import logger
import os
import sys
import socket
import struct
from torch import nn
from substrateinterface import SubstrateInterface, Keypair
from transformers import GPT2Tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bittensor.config import Config
from bittensor.config import ConfigService
from bittensor.config import SynapseConfig
from bittensor.crypto import Crypto
from bittensor.serializer import PyTorchSerializer
from bittensor.synapse import Synapse, SynapseOutput
from bittensor.session import BTSession
from bittensor.axon import Axon
from bittensor.dendrite import Dendrite
from bittensor.metagraph import Metagraph
from bittensor.utils.keys import Keys
from bittensor.utils.gate import Gate
from bittensor.utils.dispatcher import Dispatcher
from bittensor.utils.router import Router
from bittensor.neuron import Neuron
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

# Initialize the global bittensor session object.
session = None
def init(config: bittensor.Config, keypair: Keypair):
    global session
    session = bittensor.BTSession(config, keypair)
    return session

