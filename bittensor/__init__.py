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

from rich.console import Console
from rich.traceback import install
from prometheus_client import Info

import nest_asyncio
nest_asyncio.apply()

# Bittensor code and protocol version.
__version__ = '4.0.1'
version_split = __version__.split(".")
__version_as_int__ = (100 * int(version_split[0])) + (10 * int(version_split[1])) + (1 * int(version_split[2]))
__new_signature_version__ = 360

# Turn off rich console locals trace.
from rich.traceback import install
install(show_locals=False)

# Rich console.
__console__ = Console()
__use_console__ = True

# Remove overdue locals in debug training.
install(show_locals=False)

def turn_console_off():
    from io import StringIO
    __use_console__ = False
    __console__ = Console(file=StringIO(), stderr=False)


# Vocabulary dimension.
#__vocab_size__ = len( tokenizer ) + len( tokenizer.additional_special_tokens) + 100 # Plus 100 for eventual token size increase.
__vocab_size__ = 50258

# Tensor dimension.
# NOTE (const): if/when this increases peers must be responsible for trimming or expanding output to this size.
__network_dim__ = 1024 # All network responses have shape = [ __batch_size__, __sequence_dim__, __network_dim__ ]

# Substrate chain block time (seconds).
__blocktime__ = 12

# Pip address for versioning
__pipaddress__ = 'https://pypi.org/pypi/bittensor/json'

# Raw github url for delegates registry file
__delegates_details_url__: str = "https://raw.githubusercontent.com/opentensor/bittensor-delegates/main/public/delegates.json"

# Substrate ss58_format
__ss58_format__ = 42

# Wallet ss58 address length
__ss58_address_length__ = 48

__networks__ = [ 'local', 'nakamoto', 'finney']

__datasets__ = ['ArXiv', 'BookCorpus2', 'Books3', 'DMMathematics', 'EnronEmails', 'EuroParl', 'Gutenberg_PG', 'HackerNews', 'NIHExPorter', 'OpenSubtitles', 'PhilPapers', 'UbuntuIRC', 'YoutubeSubtitles']

__nakamoto_entrypoint__ = "ws://AtreusLB-2c6154f73e6429a9.elb.us-east-2.amazonaws.com:9944"

__nobunaga_entrypoint__ = "wss://stagingnode.opentensor.ai:443"

__finney_entrypoint__ = "wss://entrypoint-finney.opentensor.ai:443"

# Needs to use wss://
__bellagene_entrypoint__ = "wss://parachain.opentensor.ai:443"

__local_entrypoint__ = "ws://127.0.0.1:9944"

__tao_symbol__: str = chr(0x03C4)

__rao_symbol__: str = chr(0x03C1)

# Block Explorers map network to explorer url
## Must all be polkadotjs explorer urls
__network_explorer_map__ = {
    'local': "https://explorer.finney.opentensor.ai/#/explorer",
    'nakamoto': "https://explorer.nakamoto.opentensor.ai/#/explorer",
    'endpoint': "https://explorer.finney.opentensor.ai/#/explorer",
    'finney': "https://explorer.finney.opentensor.ai/#/explorer"
}

# Avoid collisions with other processes
from .utils.test_utils import get_random_unused_port
mock_subtensor_port = get_random_unused_port()
__mock_entrypoint__ = f"localhost:{mock_subtensor_port}"

__mock_chain_db__ = './tmp/mock_chain_db'

# --- Type Registry ---
__type_registry__ = {
    'types': {
        'Balance': 'u64', # Need to override default u128
    },
}

# --- Prometheus ---
__prometheus_version__ = "0.1.0"
prometheus_version__split = __prometheus_version__.split(".")
__prometheus_version__as_int__ = (100 * int(prometheus_version__split[0])) + (10 * int(prometheus_version__split[1])) + (1 * int(prometheus_version__split[2]))
try:
    bt_promo_info = Info("bittensor_info", "Information about the installed bittensor package.")
    bt_promo_info.info ( 
        {
            '__version__': str(__version__),
            '__version_as_int__': str(__version_as_int__),
            '__vocab_size__': str(__vocab_size__),
            '__network_dim__': str(__network_dim__),
            '__blocktime__': str(__blocktime__),
            '__prometheus_version__': str(__prometheus_version__),
            '__prometheus_version__as_int__': str(__prometheus_version__as_int__),
        } 
    )
except ValueError: 
    # This can silently fail if we import bittensor twice in the same process.
    # We simply pass over this error. 
    pass

# ---- Config ----
from bittensor._config import config as config

# ---- LOGGING ----
from bittensor._logging import logging as logging

# ---- Protos ----
import bittensor._proto.bittensor_pb2 as proto
import bittensor._proto.bittensor_pb2_grpc as grpc

# ---- Neurons ----
import bittensor._neuron as neurons

# ---- Utils ----
from bittensor.utils import unbiased_topk as unbiased_topk
from bittensor._cli.commands import utils as cli_utils

# ---- Factories -----
from bittensor.utils.balance import Balance as Balance
from bittensor._cli import cli as cli
from bittensor._axon import axon as axon
from bittensor._wallet import wallet as wallet
from bittensor._keyfile import keyfile as keyfile
from bittensor._receptor import receptor as receptor
from bittensor._endpoint import endpoint as endpoint
from bittensor._dendrite import dendrite as dendrite
from bittensor._metagraph import metagraph as metagraph
from bittensor._prometheus import prometheus as prometheus
from bittensor._subtensor import subtensor as subtensor
from bittensor._tokenizer import tokenizer as tokenizer
from bittensor._serializer import serializer as serializer
from bittensor._synapse import synapse  as synapse 
from bittensor._dataset import dataset as dataset
from bittensor._receptor import receptor_pool as receptor_pool
from bittensor._wandb import wandb as wandb
from bittensor._threadpool import prioritythreadpool as prioritythreadpool

# ---- Classes -----
from bittensor._cli.cli_impl import CLI as CLI
from bittensor._axon.axon_impl import Axon as Axon
from bittensor._subtensor.chain_data import AxonInfo as AxonInfo
from bittensor._config.config_impl import Config as Config
from bittensor._subtensor.chain_data import DelegateInfo as DelegateInfo
from bittensor._wallet.wallet_impl import Wallet as Wallet
from bittensor._keyfile.keyfile_impl import Keyfile as Keyfile
from bittensor._receptor.receptor_impl import Receptor as Receptor
from bittensor._endpoint.endpoint_impl import Endpoint as Endpoint
from bittensor._dendrite.dendrite_impl import Dendrite as Dendrite
from bittensor._metagraph.metagraph_impl import Metagraph as Metagraph
from bittensor._subtensor.chain_data import NeuronInfo as NeuronInfo
from bittensor._subtensor.chain_data import NeuronInfoLite as NeuronInfoLite
from bittensor._subtensor.chain_data import PrometheusInfo as PrometheusInfo
from bittensor._subtensor.subtensor_impl import Subtensor as Subtensor
from bittensor._serializer.serializer_impl import Serializer as Serializer
from bittensor._subtensor.chain_data import SubnetInfo as SubnetInfo
from bittensor._dataset.dataset_impl import Dataset as Dataset
from bittensor._receptor.receptor_pool_impl import ReceptorPool as ReceptorPool
from bittensor._threadpool.priority_thread_pool_impl import PriorityThreadPoolExecutor as PriorityThreadPoolExecutor
from bittensor._ipfs.ipfs_impl import Ipfs as Ipfs
from bittensor._synapse.synapse_impl import Synapse as Synapse
from bittensor._synapse.text_causallm_impl import TextCausalLM as TextCausalLM
from bittensor._synapse.text_causallmnext_impl import TextCausalLMNext as TextCausalLMNext
from bittensor._synapse.text_lasthiddenstate_impl import TextLastHiddenState as TextLastHiddenState
from bittensor._synapse.text_seq2seq_impl import TextSeq2Seq as TextSeq2Seq

# ---- Errors and Exceptions -----
from bittensor._keyfile.keyfile_impl import KeyFileError as KeyFileError

# DEFAULTS
defaults = Config()
defaults.netuid = 1
subtensor.add_defaults( defaults )
dendrite.add_defaults( defaults )
axon.add_defaults( defaults )
prometheus.add_defaults( defaults )
wallet.add_defaults( defaults )
dataset.add_defaults( defaults )
wandb.add_defaults( defaults )
logging.add_defaults( defaults )

from substrateinterface import Keypair as Keypair 
