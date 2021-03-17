# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import argparse
from munch import Munch
from loguru import logger

import sys
import torch
from loguru import logger

# Bittensor code and protocol version.
__version__ = '1.0.5'

# Tensor dimension.
# NOTE (const): if/when this increases peers must be responsible for trimming or expanding output to this size.
__network_dim__ = 512 # All network responses have shape = [ __batch_size__, __sequence_dim__, __network_dim__ ]

# Substrate chain block time (seconds).
__blocktime__ = 6

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

# Load components.
import bittensor.bittensor_pb2 as proto
import bittensor.bittensor_pb2_grpc as grpc
from bittensor.axon import Axon as Axon
from bittensor.config import Config as Config
from bittensor.executor import Executor as Executor
from bittensor.dendrite import Dendrite as Dendrite
from bittensor.metagraph import Metagraph as Metagraph
from bittensor.metagraph import ChainState as ChainState
from bittensor.metagraph import TorchChainState as TorchChainState
from bittensor.neuron import Neuron as Neuron
from bittensor.nucleus import Nucleus as Nucleus
from bittensor.receptor import Receptor as Receptor
from bittensor.subtensor import Subtensor as Subtensor
from bittensor.wallet import Wallet as Wallet
import bittensor.substrate

import multiprocessing.managers
from multiprocessing.managers import BaseManager

# Create shared memory Manager
BaseManager.register('Subtensor', bittensor.Subtensor)
BaseManager.register('Metagraph', bittensor.Metagraph)
BaseManager.register('_Dendrite', bittensor.dendrite._Dendrite)
BaseManager.register('Axon', bittensor.Axon)
manager = BaseManager()
manager.start()

# Create instance components

# An encapsulation of bittensor objects (metagraph, subtensor, axon, dendrite)
neuron = None

# Config object used by bittensor objects.
config = None

# Wallet used to initialize bittensor.
wallet = None

# Holds/updates the chain state as a torch object.
metagraph = None

# Maintain RPC connections to other node in the network. 
dendrite = None

# Recieves and queues messages for processing.
axon = None

# An interface to the chain endpoint.
subtensor = None

def help():
    r""" Prints bittensor config arguments to stdout
    """
    parser = argparse.ArgumentParser(); 
    bittensor.Neuron.add_args(parser)
    parser.print_help()

def default_config() -> Munch:
    parser = argparse.ArgumentParser(); 
    Neuron.add_args(parser) 
    config = bittensor.Config.to_config(parser); 
    return config

def add_args(parser: argparse.ArgumentParser):
    bittensor.Wallet.add_args( parser )
    bittensor.Subtensor.add_args( parser )
    bittensor.Metagraph.add_args( parser )
    bittensor.Axon.add_args(parser)
    bittensor.Dendrite.add_args( parser )
    try:
        parser.add_argument('--neuron.modality', default=0, type=int, 
                            help='''Neuron network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''')
        parser.add_argument('--neuron.multiprocessing', default=False, type=bool, 
                            help='''Are bittensor components process safe objects or run from a single thread.''')
        parser.add_argument('--neuron.debug', default=False, type=bool, 
                            help='''True if forward and backward calls print response messages to the screen''')
    except:
        pass

def check_config(config: Munch):
    bittensor.Axon.check_config( config )
    bittensor.Subtensor.check_config( config )
    bittensor.Metagraph.check_config( config )
    bittensor.Dendrite.check_config( config )
    assert config.neuron.modality == bittensor.proto.Modality.TEXT, 'Only TEXT modalities are allowed at this time.'


def init( with_config: Munch = None, with_wallet: 'bittensor.Wallet' = None, **kwargs ):
    r""" Creates bittensor background objects.
    """
    global neuron
    global subtensor
    global metagraph
    global dendrite
    global axon
    global wallet
    global config
    neuron = Neuron(config = with_config, wallet = with_wallet, **kwargs)
    axon = neuron.axon
    metagraph = neuron.metagraph
    dendrite = neuron.dendrite
    subtensor = neuron.subtensor
    wallet = neuron.wallet
    config = neuron.config

# Tokenizer
# NOTE (const): tokenizers are guaranteed to improve and expand as time progresses. We version the tokenizer here.
# neurons must be aware that versions will increase and be ready to convert between tokenizers.
# TODO (const): Add functionality to allow tokenizer conversion. i.e. for input token conversion.
__vocab_size__ = (50278 + 100)  # Must match the __tokenizer__() vocab size.
def __tokenizer__(  version = __version__ ):
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=False)
    tokenizer.padding_side = "left"
    tokenizer.add_prefix_space = False
    tokenizer.add_special_tokens({'bos_token': "[BOS]"}) # A special token representing the beginning of a sentence.
    tokenizer.add_special_tokens({'eos_token': "[EOS]"}) # A special token representing the end of a sentence.
    tokenizer.add_special_tokens({'unk_token': "[UNK]"}) # A special token representing an out-of-vocabulary token.
    tokenizer.add_special_tokens({'sep_token': "[SEP]"}) # A special token separating two different sentences in the same input (used by BERT for instance)
    tokenizer.add_special_tokens({'pad_token': "[PAD]"}) # A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.
    tokenizer.add_special_tokens({'cls_token': "[CLS]"}) # A special token representing the class of the input (used by BERT for instance).
    tokenizer.add_special_tokens({'mask_token': "[MASK]"}) # A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT).
    additional_special_tokens = [
        "<s>NOTUSED",  # Used by BARThez
        "</s>NOTUSED", # Used by BARThez
        "<eop>", # Used by MarianMT
        "<eod>", # Used by MarianMT
        "<formula>", # Used by Transformer XL
        "<mask_1>" # Used by Pegasus
        "<special0>", # Used by XLM
        "<special1>", # Used by XLM
        "<special2>", # Used by XLM
        "<special3>", # Used by XLM
        "<special4>", # Used by XLM
        "<special5>", # Used by XLM
        "<special6>", # Used by XLM
        "<special7>", # Used by XLM
        "<special8>", # Used by XLM
        "<special9>", # Used by XLM
    ]
    tokenizer.additional_special_tokens = additional_special_tokens
    global __vocab_size__
    __vocab_size__ = len(tokenizer) + len(additional_special_tokens) + 100 # Plus 100 for eventual token size increase.

    return tokenizer

# Hardcoded entry point nodes. 
__akira_entrypoints__ = [
    '104.248.52.148:9944',
    '142.93.194.110:9944',
    '162.243.175.73:9944',
    '165.227.92.237:9944',
    '167.172.141.223:9944',
    '174.138.32.166:9944',
    '206.189.194.236:9944',
    '68.183.130.145:9944',
    '68.183.140.221:9944',
    '68.183.140.251:9944'
]
__kusanagi_entrypoints__ = [
    '142.93.203.149:9944',
    '157.230.11.1:9944',
    '157.230.11.116:9944',
    '157.230.11.31:9944',
    '157.230.11.36:9944',
    '157.230.11.53:9944',
    '157.230.3.108:9944',
    '159.65.236.189:9944',
    '165.227.81.42:9944',
    '206.189.207.173:9944'
]
__boltzmann_entrypoints__ = [
    'feynman.boltzmann.bittensor.com:9944',
    '157.230.223.68:9944'
]
__local_entrypoints__ = [
    '127.0.0.1:9944'
]
