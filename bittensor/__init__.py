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
import bittensor.tokenizer
import bittensor.config 
import bittensor.logging
import bittensor.receptor
import bittensor.router
import bittensor.subtensor
import bittensor.nucleus

from bittensor.axons import axon
from bittensor.dendrites import dendrite
from bittensor.clis import cli
from bittensor.executors import executor
from bittensor.wallets import wallet
from bittensor.metagraphs import metagraph

# ---- LOGGING ----
__debug_on__ = False
bittensor.logging.init_logger()

# ---- Tokenizer ----
__tokenizer__ = bittensor.tokenizer.get_tokenizer_for_version( __version__ )
__vocab_size__ = len(__tokenizer__) + len(__tokenizer__.additional_special_tokens) + 100 # Plus 100 for eventual token size increase.

# Hardcoded entry point nodes. 
__akira_entrypoints__ = [
    "fermi.akira.bittensor.com:9944",
    "copernicus.akira.bittensor.com:9944",
    "buys.akira.bittensor.com:9944",
    "nobel.akira.bittensor.com:9944",
    "mendeleev.akira.bittensor.com:9944",
    "rontgen.akira.bittensor.com:9944",
    "feynman.akira.bittensor.com:9944",
    "bunsen.akira.bittensor.com:9944",
    "berkeley.akira.bittensor.com:9944",
    "huygens.akira.bittensor.com:9944"
]
__kusanagi_entrypoints__ = [
    "fermi.kusanagi.bittensor.com:9944",
    "copernicus.kusanagi.bittensor.com:9944",
    "buys.kusanagi.bittensor.com:9944",
    "nobel.kusanagi.bittensor.com:9944",
    "mendeleev.kusanagi.bittensor.com:9944",
    "rontgen.kusanagi.bittensor.com:9944",
    "feynman.kusanagi.bittensor.com:9944",
    "bunsen.kusanagi.bittensor.com:9944",
    "berkeley.kusanagi.bittensor.com:9944",
    "huygens.kusanagi.bittensor.com:9944"
]
__boltzmann_entrypoints__ = [
    'feynman.boltzmann.bittensor.com:9944',
]
__local_entrypoints__ = [
    '127.0.0.1:9944'
]