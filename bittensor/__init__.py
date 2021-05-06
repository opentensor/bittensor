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

import sys
import random
from loguru import logger

# Nest Asyncio for colab support.
import nest_asyncio
nest_asyncio.apply()

import bittensor.bittensor_pb2 as proto
import bittensor.bittensor_pb2_grpc as grpc

# Bittensor code and protocol version.
__version__ = '1.0.3'

# Tensor dimension.
# NOTE (const): if/when this increases peers must be responsible for trimming or expanding output to this size.
__network_dim__ = 512 # All network responses have shape = [ __batch_size__, __sequence_dim__, __network_dim__ ]

# Substrate chain block time (seconds).
__blocktime__ = 6

# Load components.
import bittensor.axon
import bittensor.config 
import bittensor.neuron
import bittensor.miner
import bittensor.executor
import bittensor.cli
import bittensor.dendrite
import bittensor.metagraph
import bittensor.logging
import bittensor.receptor
import bittensor.subtensor
import bittensor.synapse
import bittensor.wallet

# ---- LOGGING ----
__debug_on__ = False 
bittensor.logging.init_logger()

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
