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

# Bittensor code and protocol version.
__version__ = '5.2.0'
version_split = __version__.split(".")
__version_as_int__ = (100 * int(version_split[0])) + (10 * int(version_split[1])) + (1 * int(version_split[2]))
__new_signature_version__ = 360

# Rich console.
__console__ = Console()
__use_console__ = True

# Remove overdue locals in debug training.
install(show_locals=False)

def turn_console_off():
    from io import StringIO
    __use_console__ = False
    __console__ = Console(file=StringIO(), stderr=False)

# Logging helpers.
def trace( on:bool = True ):
    logging.set_trace( on )

def debug( on:bool = True ):
    logging.set_debug( on )

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

__finney_entrypoint__ = "wss://entrypoint-finney.opentensor.ai:443"

__finney_test_entrypoint__ = "wss://test.finney.opentensor.ai:443/"

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


# --- Type Registry ---
__type_registry__ = {
    'types': {
        'Balance': 'u64', # Need to override default u128
    },
}

from substrateinterface import Keypair as Keypair
from .config import config as config
from .keyfile import keyfile as keyfile
from .wallet import wallet as wallet
from .utils import *
from .utils.balance import Balance as Balance
from .chain_data import *
from .errors import *
from .subtensor import subtensor as subtensor
from .cli import cli as cli
from .logging import logging as logging
from .metagraph import metagraph as metagraph
from .threadpool import PriorityThreadPoolExecutor

from .protocol import * 
from .tensor import *
from .axon import axon as axon
from .dendrite import dendrite as dendrite