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

"""
The Bittensor Compatibility Module is designed to ensure seamless integration and functionality with legacy versions of
the Bittensor framework, specifically up to and including version 7.3.0. This module addresses changes and deprecated
features in recent versions, allowing users to maintain compatibility with older systems and projects.
"""
import sys
import importlib

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
from substrateinterface import Keypair  # noqa: F401

from bittensor.core import settings
from bittensor.core.axon import Axon
from bittensor.core.chain_data import (  # noqa: F401
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
from bittensor.core.config import (  # noqa: F401
    InvalidConfigFile,
    DefaultConfig,
    Config,
    T,
)
from bittensor.core.dendrite import Dendrite  # noqa: F401
from bittensor.core.errors import (  # noqa: F401
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
from bittensor.core.metagraph import Metagraph
from bittensor.core.settings import blocktime
from bittensor.core.stream import StreamingSynapse   # noqa: F401
from bittensor.core.subtensor import Subtensor
from bittensor.core.synapse import TerminalInfo, Synapse   # noqa: F401
from bittensor.core.tensor import tensor, Tensor   # noqa: F401
from bittensor.core.threadpool import PriorityThreadPoolExecutor as PriorityThreadPoolExecutor   # noqa: F401
from bittensor.mock.subtensor_mock import MockSubtensor as MockSubtensor   # noqa: F401
from bittensor.utils import (  # noqa: F401
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
    get_hash
)
from bittensor.utils.balance import Balance as Balance   # noqa: F401
from bittensor.utils.subnets import SubnetsAPI  # noqa: F401


# Backwards compatibility with previous bittensor versions.
axon = Axon
config = Config
keyfile = Keyfile
metagraph = Metagraph
wallet = Wallet
subtensor = Subtensor
synapse = Synapse

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

# Makes the `bittensor.api.extrinsics` subpackage available as `bittensor.extrinsics` for backwards compatibility.
extrinsics = importlib.import_module('bittensor.api.extrinsics')
sys.modules['bittensor.extrinsics'] = extrinsics
