# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# Bittensor code and protocol version.

__version__ = "7.3.0"

_version_split = __version__.split(".")
__version_info__ = tuple(int(part) for part in _version_split)
_version_int_base = 1000
assert max(__version_info__) < _version_int_base

__version_as_int__: int = sum(e * (_version_int_base**i) for i, e in enumerate(reversed(__version_info__)))
assert __version_as_int__ < 2 ** 31  # fits in int32

import os
import warnings

from bittensor_wallet.errors import KeyFileError  # noqa: F401
from bittensor_wallet.keyfile import (  # noqa: F401
    serialized_keypair_to_keyfile_data,
    deserialize_keypair_from_keyfile_data,
    validate_password,
    ask_password_to_encrypt,
    keyfile_data_is_encrypted_nacl,
    keyfile_data_is_encrypted_ansible,
    keyfile_data_is_encrypted_legacy,
    keyfile_data_is_encrypted,
    keyfile_data_encryption_method,
    legacy_encrypt_keyfile_data,
    encrypt_keyfile_data,
    get_coldkey_password_from_environment,
    decrypt_keyfile_data,
    Keyfile,
)
from bittensor_wallet.wallet import display_mnemonic_msg, Wallet  # noqa: F401
from rich.console import Console
from rich.traceback import install
from substrateinterface import Keypair  # noqa: F401

from .btcli.cli import cli as cli, COMMANDS as ALL_COMMANDS
from .core import settings
from .core.axon import Axon
from .core.chain_data import (
    AxonInfo,
    NeuronInfo,
    NeuronInfoLite,
    PrometheusInfo,
    DelegateInfo,
    StakeInfo,
    SubnetInfo,
    SubnetHyperparameters,
    IPInfo,
    ProposalCallData,
    ProposalVoteData,
)
from .core.config import (  # noqa: F401
    InvalidConfigFile,
    DefaultConfig,
    Config,
    T,
)
from .core.dendrite import dendrite as dendrite
from .core.errors import (
    BlacklistedException,
    ChainConnectionError,
    ChainError,
    ChainQueryError,
    ChainTransactionError,
    IdentityError,
    InternalServerError,
    InvalidRequestNameError,
    MetadataError,
    NominationError,
    NotDelegateError,
    NotRegisteredError,
    NotVerifiedException,
    PostProcessException,
    PriorityException,
    RegistrationError,
    RunException,
    StakeError,
    SynapseDendriteNoneException,
    SynapseParsingError,
    TransferError,
    UnknownSynapseError,
    UnstakeError,
)
from .core.metagraph import metagraph as metagraph
from .core.settings import blocktime
from .core.stream import StreamingSynapse
from .core.subnets import SubnetsAPI as SubnetsAPI
from .core.subtensor import Subtensor
from .core.synapse import TerminalInfo, Synapse
from .core.tensor import tensor, Tensor
from .core.threadpool import PriorityThreadPoolExecutor as PriorityThreadPoolExecutor
from .mock.subtensor_mock import MockSubtensor as MockSubtensor
from .utils import (
    ss58_to_vec_u8,
    unbiased_topk,
    version_checking,
    strtobool,
    strtobool_with_default,
    get_explorer_root_url_by_network_from_map,
    get_explorer_root_url_by_network_from_map,
    get_explorer_url_for_network,
    ss58_address_to_bytes,
    U16_NORMALIZED_FLOAT,
    U64_NORMALIZED_FLOAT,
    u8_key_to_ss58,
    hash,
    wallet_utils,
)
from .utils.balance import Balance as Balance
from .utils.btlogging import logging


configs = [
    Axon.config(),
    Subtensor.config(),
    PriorityThreadPoolExecutor.config(),
    Wallet.config(),
    logging.get_config(),
]
defaults = Config.merge_all(configs)


def __getattr__(name):
    if name == "version_split":
        warnings.warn(
            "version_split is deprecated and will be removed in future versions. Use __version__ instead.",
            DeprecationWarning,
        )
        return _version_split
    raise AttributeError(f"module {__name__} has no attribute {name}")


# Rich console.
__console__ = Console()
__use_console__ = True

# Remove overdue locals in debug training.
install(show_locals=False)


def turn_console_off():
    global __use_console__
    global __console__
    from io import StringIO

    __use_console__ = False
    __console__ = Console(file=StringIO(), stderr=False)


def turn_console_on():
    global __use_console__
    global __console__
    __use_console__ = True
    __console__ = Console()


# Logging helpers.
def trace(on: bool = True):
    logging.set_trace(on)


def debug(on: bool = True):
    logging.set_debug(on)


turn_console_off()


def __apply_nest_asyncio():
    """
    Apply nest_asyncio if the environment variable NEST_ASYNCIO is set to "1" or not set.
    If not set, warn the user that the default will change in the future.
    """
    nest_asyncio_env = os.getenv("NEST_ASYNCIO")

    if nest_asyncio_env == "1" or nest_asyncio_env is None:
        if nest_asyncio_env is None:
            warnings.warn(
                """NEST_ASYNCIO implicitly set to '1'. In the future, the default value will be '0'.
                If you use `nest_asyncio`, make sure to add it explicitly to your project dependencies,
                as it will be removed from `bittensor` package dependencies in the future.
                To silence this warning, explicitly set the environment variable, e.g. `export NEST_ASYNCIO=0`.""",
                DeprecationWarning,
            )
        # Install and apply nest asyncio to allow the async functions to run in a .ipynb
        import nest_asyncio
        nest_asyncio.apply()


__apply_nest_asyncio()

# Backwards compatibility with previous bittensor versions.
axon = Axon
config = Config
keyfile = Keyfile
wallet = Wallet
subtensor = Subtensor

__blocktime__ = blocktime
__network_explorer_map__ = settings.network_explorer_map
__pipaddress__ = settings.pipaddress
__ss58_format__ = settings.ss58_format
__type_registry__ = settings.type_registry
__ss58_address_length__ = settings.ss58_address_length

__networks__ = settings.networks

__finney_entrypoint__ = settings.finney_entrypoint
__finney_test_entrypoint__ = settings.finney_test_entrypoint
__archive_entrypoint__ = settings.archive_entrypoint
__local_entrypoint__ = settings.local_entrypoint

__tao_symbol__ = settings.tao_symbol
__rao_symbol__ = settings.rao_symbol
