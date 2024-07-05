# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022-2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

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
import os
import warnings
from typing import Optional


if (NEST_ASYNCIO_ENV := os.getenv("NEST_ASYNCIO")) in ("1", None):
    if NEST_ASYNCIO_ENV is None:
        warnings.warn(
            "NEST_ASYNCIO implicitly set to '1'. In the future, the default value will be '0'."
            "If you use `nest_asyncio` make sure to add it explicitly to your project dependencies,"
            "as it will be removed from `bittensor` package dependencies in the future."
            "To silence this warning, explicitly set the environment variable, e.g. `export NEST_ASYNCIO=0`.",
            DeprecationWarning,
        )
    # Install and apply nest asyncio to allow the async functions
    # to run in a .ipynb
    import nest_asyncio

    nest_asyncio.apply()


import asyncio

# from . import v2 as bittensor
from .v2 import (
    turn_console_off,
    turn_console_on,
    trace,
    debug,
    BlacklistedException,
    ChainConnectionError,
    ChainError,
    ChainQueryError,
    ChainTransactionError,
    IdentityError,
    InternalServerError,
    InvalidRequestNameError,
    KeyFileError,
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
    Keypair,
    InvalidConfigFile,
    DefaultConfig,
    T,
    config,
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
    Mockkeyfile,
    keyfile,
    display_mnemonic_msg,
    wallet,
    ss58_to_vec_u8,
    unbiased_topk,
    version_checking,
    strtobool,
    strtobool_with_default,
    get_explorer_root_url_by_network_from_map,
    get_explorer_url_for_network,
    ss58_address_to_bytes,
    u16_normalized_float,
    u64_normalized_float,
    u8_key_to_ss58,
    get_hash,
    wallet_utils,
    Balance,
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
    logging,
    btlogging,
    subtensor_module,
    ALL_COMMANDS,
    Metagraph,
    PriorityThreadPoolExecutor,
    TerminalInfo,
    Synapse,
    StreamingSynapse,
    tensor,
    Tensor,
    axon,
    dendrite,
    MockKeyfile,
    MockSubtensor,
    MockWallet,
    SubnetsAPI,
    configs,
    defaults,
    Console,
)
from .core.cli import Cli as cli

turn_console_off = turn_console_off
turn_console_on = turn_console_on
trace = trace
debug = debug
BlacklistedException = BlacklistedException
ChainConnectionError = ChainConnectionError
ChainError = ChainError
ChainQueryError = ChainQueryError
ChainTransactionError = ChainTransactionError
IdentityError = IdentityError
InternalServerError = InternalServerError
InvalidRequestNameError = InvalidRequestNameError
KeyFileError = KeyFileError
MetadataError = MetadataError
NominationError = NominationError
NotDelegateError = NotDelegateError
NotRegisteredError = NotRegisteredError
NotVerifiedException = NotVerifiedException
PostProcessException = PostProcessException
PriorityException = PriorityException
RegistrationError = RegistrationError
RunException = RunException
StakeError = StakeError
SynapseDendriteNoneException = SynapseDendriteNoneException
SynapseParsingError = SynapseParsingError
TransferError = TransferError
UnknownSynapseError = UnknownSynapseError
UnstakeError = UnstakeError
Keypair = Keypair
InvalidConfigFile = InvalidConfigFile
DefaultConfig = DefaultConfig
T = T
config = config
serialized_keypair_to_keyfile_data = serialized_keypair_to_keyfile_data
deserialize_keypair_from_keyfile_data = deserialize_keypair_from_keyfile_data
validate_password = validate_password
ask_password_to_encrypt = ask_password_to_encrypt
keyfile_data_is_encrypted_nacl = keyfile_data_is_encrypted_nacl
keyfile_data_is_encrypted_ansible = keyfile_data_is_encrypted_ansible
keyfile_data_is_encrypted_legacy = keyfile_data_is_encrypted_legacy
keyfile_data_is_encrypted = keyfile_data_is_encrypted
keyfile_data_encryption_method = keyfile_data_encryption_method
legacy_encrypt_keyfile_data = legacy_encrypt_keyfile_data
encrypt_keyfile_data = encrypt_keyfile_data
get_coldkey_password_from_environment = get_coldkey_password_from_environment
decrypt_keyfile_data = decrypt_keyfile_data
Mockkeyfile = Mockkeyfile
keyfile = keyfile
display_mnemonic_msg = display_mnemonic_msg
wallet = wallet
ss58_to_vec_u8 = ss58_to_vec_u8
unbiased_topk = unbiased_topk
version_checking = version_checking
strtobool = strtobool
strtobool_with_default = strtobool_with_default
get_explorer_root_url_by_network_from_map = get_explorer_root_url_by_network_from_map
get_explorer_url_for_network = get_explorer_url_for_network
ss58_address_to_bytes = ss58_address_to_bytes
u16_normalized_float = u16_normalized_float
u64_normalized_float = u64_normalized_float
u8_key_to_ss58 = u8_key_to_ss58
get_hash = get_hash
wallet_utils = wallet_utils
Balance = Balance
AxonInfo = AxonInfo
NeuronInfo = NeuronInfo
NeuronInfoLite = NeuronInfoLite
PrometheusInfo = PrometheusInfo
DelegateInfo = DelegateInfo
StakeInfo = StakeInfo
SubnetInfo = SubnetInfo
SubnetHyperparameters = SubnetHyperparameters
IPInfo = IPInfo
ProposalCallData = ProposalCallData
ProposalVoteData = ProposalVoteData
logging = logging
btlogging = btlogging
subtensor_module = subtensor_module
cli = cli
ALL_COMMANDS = ALL_COMMANDS
Metagraph = Metagraph
PriorityThreadPoolExecutor = PriorityThreadPoolExecutor
TerminalInfo = TerminalInfo
Synapse = Synapse
StreamingSynapse = StreamingSynapse
tensor = tensor
Tensor = Tensor
axon = axon
dendrite = dendrite
MockKeyfile = MockKeyfile
MockSubtensor = MockSubtensor
MockWallet = MockWallet
SubnetsAPI = SubnetsAPI
configs = configs
defaults = defaults


# Bittensor code and protocol version.
__version__ = "7.2.0"

_version_split = __version__.split(".")
__version_info__ = tuple(int(part) for part in _version_split)
_version_int_base = 1000
assert max(__version_info__) < _version_int_base

__version_as_int__: int = sum(
    e * (_version_int_base**i) for i, e in enumerate(reversed(__version_info__))
)
assert __version_as_int__ < 2**31  # fits in int32
__new_signature_version__ = 360

# Rich console.
__console__ = Console()
__use_console__ = True

# Substrate chain block time (seconds).
__blocktime__ = 12

# Pip address for versioning
__pipaddress__ = "https://pypi.org/pypi/bittensor/json"

# Raw GitHub url for delegates registry file
__delegates_details_url__: str = "https://raw.githubusercontent.com/opentensor/bittensor-delegates/main/public/delegates.json"

# Substrate ss58_format
__ss58_format__ = 42

# Wallet ss58 address length
__ss58_address_length__ = 48

__networks__ = ["local", "finney", "test", "archive"]

__finney_entrypoint__ = "wss://entrypoint-finney.opentensor.ai:443"

__finney_test_entrypoint__ = "wss://test.finney.opentensor.ai:443/"

__archive_entrypoint__ = "wss://archive.chain.opentensor.ai:443/"

# Needs to use wss://
__bellagene_entrypoint__ = "wss://parachain.opentensor.ai:443"

if (
    BT_SUBTENSOR_CHAIN_ENDPOINT := os.getenv("BT_SUBTENSOR_CHAIN_ENDPOINT")
) is not None:
    __local_entrypoint__ = BT_SUBTENSOR_CHAIN_ENDPOINT
else:
    __local_entrypoint__ = "ws://127.0.0.1:9944"

__tao_symbol__: str = chr(0x03C4)

__rao_symbol__: str = chr(0x03C1)

# Block Explorers map network to explorer url
# Must all be polkadotjs explorer urls
__network_explorer_map__ = {
    "opentensor": {
        "local": "https://polkadot.js.org/apps/?rpc=wss%3A%2F%2Fentrypoint-finney.opentensor.ai%3A443#/explorer",
        "endpoint": "https://polkadot.js.org/apps/?rpc=wss%3A%2F%2Fentrypoint-finney.opentensor.ai%3A443#/explorer",
        "finney": "https://polkadot.js.org/apps/?rpc=wss%3A%2F%2Fentrypoint-finney.opentensor.ai%3A443#/explorer",
    },
    "taostats": {
        "local": "https://x.taostats.io",
        "endpoint": "https://x.taostats.io",
        "finney": "https://x.taostats.io",
    },
}

# --- Type Registry ---
__type_registry__ = {
    "types": {
        "Balance": "u64",  # Need to override default u128
    },
    "runtime_api": {
        "NeuronInfoRuntimeApi": {
            "methods": {
                "get_neuron_lite": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                        {
                            "name": "uid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                },
                "get_neurons_lite": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                },
            }
        },
        "StakeInfoRuntimeApi": {
            "methods": {
                "get_stake_info_for_coldkey": {
                    "params": [
                        {
                            "name": "coldkey_account_vec",
                            "type": "Vec<u8>",
                        },
                    ],
                    "type": "Vec<u8>",
                },
                "get_stake_info_for_coldkeys": {
                    "params": [
                        {
                            "name": "coldkey_account_vecs",
                            "type": "Vec<Vec<u8>>",
                        },
                    ],
                    "type": "Vec<u8>",
                },
            },
        },
        "ValidatorIPRuntimeApi": {
            "methods": {
                "get_associated_validator_ip_info_for_subnet": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                },
            },
        },
        "SubnetInfoRuntimeApi": {
            "methods": {
                "get_subnet_hyperparams": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                }
            }
        },
        "SubnetRegistrationRuntimeApi": {
            "methods": {"get_network_registration_cost": {"params": [], "type": "u64"}}
        },
    },
}


def metagraph(
    netuid: int,
    network: str = "finney",
    lite: bool = True,
    sync: bool = True,
    subtensor: Optional["Subtensor"] = None,
):
    return asyncio.run(Metagraph(netuid, network, lite, sync, subtensor))


class Subtensor:
    def __init__(self, *args, **kwargs):
        self._async_instance = subtensor_module.Subtensor(*args, **kwargs)

    def __getattr__(self, item):
        attr = getattr(self._async_instance, item)
        if asyncio.iscoroutinefunction(attr):

            def sync_wrapper(*args, **kwargs):
                return asyncio.run(attr(*args, **kwargs))

            return sync_wrapper
        return attr


subtensor = Subtensor
