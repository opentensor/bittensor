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

import torch
from typing import Union, List, Dict
from rich.console import Console
from rich.traceback import install
from prometheus_client import Info
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any, Tuple

import nest_asyncio
nest_asyncio.apply()

# Bittensor code and protocol version.
__version__ = '5.0.0'
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

__networks__ = [ 'local', 'finney']

__datasets__ = ['ArXiv', 'BookCorpus2', 'Books3', 'DMMathematics', 'EnronEmails', 'EuroParl', 'Gutenberg_PG', 'HackerNews', 'NIHExPorter', 'OpenSubtitles', 'PhilPapers', 'UbuntuIRC', 'YoutubeSubtitles']

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
# Duplicate import for ease of use.
from bittensor._logging import logging as logging
from bittensor._logging import logging as logger

# ---- Protos ----
import bittensor._proto.bittensor_pb2 as proto
import bittensor._proto.bittensor_pb2_grpc as grpc

# ---- Utils ----
from bittensor.utils import unbiased_topk as unbiased_topk
from bittensor.utils.tokenizer_utils import topk_token_phrases
from bittensor.utils.tokenizer_utils import compact_topk_token_phrases
from bittensor.utils.tokenizer_utils import unravel_topk_token_phrases
from bittensor.utils.tokenizer_utils import prep_tokenizer
from bittensor._cli.commands import utils as cli_utils

# ---- Factories -----
from bittensor.utils.balance import Balance as Balance
from bittensor._cli import cli as cli
from bittensor._axon import axon as axon
from bittensor._axon import axon_info as axon_info
from bittensor._wallet import wallet as wallet
from bittensor._keyfile import keyfile as keyfile
from bittensor._metagraph import metagraph as metagraph
from bittensor._prometheus import prometheus as prometheus
from bittensor._subtensor import subtensor as subtensor
from bittensor._tokenizer import tokenizer as tokenizer
from bittensor._serializer import serializer as serializer
from bittensor._dataset import dataset as dataset
from bittensor._threadpool import prioritythreadpool as prioritythreadpool
from bittensor._blacklist import blacklist  as blacklist
from bittensor._priority import priority as priority

# ---- Classes -----
from bittensor._cli.cli_impl import CLI as CLI
from bittensor._config.config_impl import Config as Config
from bittensor._subtensor.chain_data import DelegateInfo as DelegateInfo
from bittensor._wallet.wallet_impl import Wallet as Wallet
from bittensor._keyfile.keyfile_impl import Keyfile as Keyfile
from bittensor._subtensor.chain_data import NeuronInfo as NeuronInfo
from bittensor._subtensor.chain_data import NeuronInfoLite as NeuronInfoLite
from bittensor._subtensor.chain_data import PrometheusInfo as PrometheusInfo
from bittensor._subtensor.subtensor_impl import Subtensor as Subtensor
from bittensor._serializer.serializer_impl import Serializer as Serializer
from bittensor._subtensor.chain_data import SubnetInfo as SubnetInfo
from bittensor._dataset.dataset_impl import Dataset as Dataset
from bittensor._threadpool.priority_thread_pool_impl import PriorityThreadPoolExecutor as PriorityThreadPoolExecutor
from bittensor._ipfs.ipfs_impl import Ipfs as Ipfs

# ---- Errors and Exceptions -----
from bittensor._keyfile.keyfile_impl import KeyFileError as KeyFileError

from bittensor._proto.bittensor_pb2 import ForwardTextPromptingRequest
from bittensor._proto.bittensor_pb2 import ForwardTextPromptingResponse
from bittensor._proto.bittensor_pb2 import MultiForwardTextPromptingRequest
from bittensor._proto.bittensor_pb2 import MultiForwardTextPromptingResponse
from bittensor._proto.bittensor_pb2 import BackwardTextPromptingRequest
from bittensor._proto.bittensor_pb2 import BackwardTextPromptingResponse

# ---- Synapses -----
from bittensor._synapse.synapse import Synapse
from bittensor._synapse.synapse import SynapseCall
from bittensor._synapse.text_prompting.synapse import TextPromptingSynapse

# ---- Dendrites -----
from bittensor._dendrite.dendrite import Dendrite
from bittensor._dendrite.dendrite import DendriteCall
from bittensor._dendrite.text_prompting.dendrite import TextPromptingDendrite as text_prompting
from bittensor._dendrite.text_prompting.dendrite_pool import TextPromptingDendritePool as text_prompting_pool

# ---- Base Miners -----
from bittensor._neuron.base_miner_neuron import BaseMinerNeuron as base_miner_neuron
from bittensor._neuron.base_validator import BaseValidator as base_validator
from bittensor._neuron.base_prompting_miner import BasePromptingMiner
from bittensor._neuron.base_huggingface_miner import HuggingFaceMiner

# ---- Errors and Exceptions -----
from bittensor._keyfile.keyfile_impl import KeyFileError as KeyFileError

# ---- Errors and Exceptions -----
from bittensor._keyfile.keyfile_impl import KeyFileError as KeyFileError

# DEFAULTS
defaults = Config()
defaults.netuid = 1
subtensor.add_defaults( defaults )
axon.add_defaults( defaults )
prioritythreadpool.add_defaults( defaults )
prometheus.add_defaults( defaults )
wallet.add_defaults( defaults )
dataset.add_defaults( defaults )
logging.add_defaults( defaults )

from substrateinterface import Keypair as Keypair

# Logging helpers.
def trace():
    logging.set_trace(True)

def debug():
    logging.set_debug(True)

default_prompt = '''
You are Chattensor.
Chattensor is a research project by Opentensor Cortex.
Chattensor is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Chattensor is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
'''

default_prompting_validator_key = '5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3'

__context_prompting_llm = None
def prompt(
        content: Union[ str, List[str], List[Dict[ str ,str ]]],
        wallet_name: str = "default",
        hotkey: str = default_prompting_validator_key,
        subtensor_: Optional['Subtensor'] = None,
        axon_: Optional['axon_info'] = None,
        return_all: bool = False,
    ) -> str:
    global __context_prompting_llm
    if __context_prompting_llm == None:
        __context_prompting_llm = prompting(
            wallet_name = wallet_name,
            hotkey = hotkey,
            subtensor_ = subtensor_,
            axon_ = axon_,
        )
    return __context_prompting_llm( content = content, return_all = return_all )

class prompting ( torch.nn.Module ):
    _axon: 'axon_info'
    _dendrite: 'Dendrite'
    _subtensor: 'Subtensor'
    _hotkey: str
    _keypair: 'Keypair'

    def __init__(
        self,
        wallet_name: str = "default",
        hotkey: str = default_prompting_validator_key,
        subtensor_: Optional['Subtensor'] = None,
        axon_: Optional['axon_info'] = None,
        use_coldkey: bool = False
    ):
        super(prompting, self).__init__()
        self._hotkey = hotkey
        self._subtensor = subtensor() if subtensor_ is None else subtensor_
        if use_coldkey:
            self._keypair = wallet( name = wallet_name ).create_if_non_existent().coldkey
        else:
            self._keypair = wallet( name = wallet_name ).create_if_non_existent().hotkey

        if axon_ is not None:
            self._axon = axon_
        else:
            self._metagraph = metagraph( 1 )
            self._axon = self._metagraph.axons[ self._metagraph.hotkeys.index( self._hotkey ) ]
        self._dendrite = text_prompting(
            keypair = self._keypair,
            axon = self._axon
        )

    @staticmethod
    def format_content( content: Union[ str, List[str], List[Dict[ str ,str ]]] ) -> Tuple[ List[str], List[str ]]:
        if isinstance( content, str ):
            return ['system', 'user'], [ default_prompt, content ]
        elif isinstance( content, list ):
            if isinstance( content[0], str ):
                return ['user' for _ in content ], content
            elif isinstance( content[0], dict ):
                return [ dictitem[ list(dictitem.keys())[0] ] for dictitem in content ], [ dictitem[ list(dictitem.keys())[1] ] for dictitem in content ]
            else:
                raise ValueError('content has invalid type {}'.format( type( content )))
        else:
            raise ValueError('content has invalid type {}'.format( type( content )))

    def forward(
            self,
            content: Union[ str, List[str], List[Dict[ str ,str ]]],
            timeout: float = 24,
            return_call: bool = False,
            return_all: bool = False,
        ) -> Union[str, List[str]]:
        roles, messages = self.format_content( content )
        if not return_all:
            return self._dendrite.forward(
                roles = roles,
                messages = messages,
                timeout = timeout
            ).completion
        else:
            return self._dendrite.multi_forward(
                roles = roles,
                messages = messages,
                timeout = timeout
            ).multi_completions


    async def async_forward(
            self,
            content: Union[ str, List[str], List[Dict[ str ,str ]]],
            timeout: float = 24,
            return_all: bool = False,
        ) -> Union[str, List[str]]:
        roles, messages = self.format_content( content )
        if not return_all:
            return await self._dendrite.async_forward(
                    roles = roles,
                    messages = messages,
                    timeout = timeout
                ).completion
        else:
            return self._dendrite.async_multi_forward(
                roles = roles,
                messages = messages,
                timeout = timeout
            ).multi_completions

class BittensorLLM(LLM):
    """Wrapper around Bittensor Prompting Subnetwork.
This Python file implements the BittensorLLM class, a wrapper around the Bittensor Prompting Subnetwork for easy integration into language models. The class provides a query method to receive responses from the subnetwork for a given user message and an implementation of the _call method to return the best response. The class can be initialized with various parameters such as the wallet name and chain endpoint.

    Example:
        .. code-block:: python

            from bittensor import BittensorLLM
            btllm = BittensorLLM(wallet_name="default")
    """

    wallet_name: str = 'default'
    hotkey: str = default_prompting_validator_key
    llm: prompting = None
    def __init__(self, subtensor_: Optional['Subtensor'] = None, axon_: Optional['axon_info'] = None, **data):
        super().__init__(**data)
        self.llm = prompting(wallet_name=self.wallet_name, hotkey=self.hotkey, subtensor_=subtensor_, axon_=axon_ )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"wallet_name": self.wallet_name, "hotkey_name": self.hotkey}

    @property
    def _llm_type(self) -> str:
        return "BittensorLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the LLM with the given prompt and stop tokens."""
        return self.llm(prompt)


