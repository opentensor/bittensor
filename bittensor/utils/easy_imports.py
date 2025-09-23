"""
The Bittensor Compatibility Module is designed to ensure seamless integration and functionality with legacy versions of
the Bittensor framework, specifically up to and including version 7.3.0. This module addresses changes and deprecated
features in recent versions, allowing users to maintain compatibility with older systems and projects.
"""

import importlib
import sys

from bittensor_wallet import Keypair
from bittensor_wallet.errors import KeyFileError
from bittensor_wallet.keyfile import (
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
from bittensor_wallet.wallet import Wallet

from bittensor.core import settings, timelock
from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.core.axon import Axon
from bittensor.core.chain_data import (
    AxonInfo,
    ChainIdentity,
    DelegateInfo,
    DelegateInfoLite,
    DynamicInfo,
    IPInfo,
    MetagraphInfo,
    MetagraphInfoEmissions,
    MetagraphInfoParams,
    MetagraphInfoPool,
    NeuronInfo,
    NeuronInfoLite,
    PrometheusInfo,
    ProposalCallData,
    ProposalVoteData,
    ScheduledColdkeySwapInfo,
    SelectiveMetagraphIndex,
    StakeInfo,
    SubnetHyperparameters,
    SubnetIdentity,
    SubnetInfo,
    SubnetState,
    WeightCommitInfo,
)
from bittensor.core.config import Config
from bittensor.core.dendrite import Dendrite
from bittensor.core.errors import (
    BlacklistedException,
    ChainConnectionError,
    ChainError,
    ChainQueryError,
    ChainTransactionError,
    DelegateTakeTooHigh,
    DelegateTakeTooLow,
    DelegateTxRateLimitExceeded,
    DuplicateChild,
    HotKeyAccountNotExists,
    IdentityError,
    InternalServerError,
    InvalidChild,
    InvalidRequestNameError,
    MetadataError,
    NominationError,
    NonAssociatedColdKey,
    NotDelegateError,
    NotEnoughStakeToSetChildkeys,
    NotRegisteredError,
    NotVerifiedException,
    PostProcessException,
    PriorityException,
    ProportionOverflow,
    RegistrationError,
    RegistrationNotPermittedOnRootSubnet,
    RunException,
    StakeError,
    SubnetNotExists,
    SynapseDendriteNoneException,
    SynapseParsingError,
    TooManyChildren,
    TransferError,
    TxRateLimitExceeded,
    UnknownSynapseError,
    UnstakeError,
)
from bittensor.core.metagraph import Metagraph
from bittensor.core.settings import BLOCKTIME
from bittensor.core.stream import StreamingSynapse
from bittensor.core.subtensor import Subtensor
from bittensor.core.subtensor_api import SubtensorApi
from bittensor.core.synapse import TerminalInfo, Synapse
from bittensor.core.tensor import Tensor
from bittensor.core.threadpool import PriorityThreadPoolExecutor
from bittensor.utils import (
    ss58_to_vec_u8,
    version_checking,
    strtobool,
    get_explorer_url_for_network,
    ss58_address_to_bytes,
    u16_normalized_float,
    u64_normalized_float,
    get_hash,
)
from bittensor.utils.balance import Balance
from bittensor.utils.balance import tao, rao
from bittensor.utils.btlogging import logging
from bittensor.utils.mock.subtensor_mock import MockSubtensor
from bittensor.utils.subnets import SubnetsAPI


# Backwards compatibility with previous bittensor versions.
async_subtensor = AsyncSubtensor
axon = Axon
config = Config
dendrite = Dendrite
keyfile = Keyfile
metagraph = Metagraph
wallet = Wallet
subtensor = Subtensor
synapse = Synapse

# Makes the `bittensor.utils.mock` subpackage available as `bittensor.mock` for backwards compatibility.
mock_subpackage = importlib.import_module("bittensor.utils.mock")
sys.modules["bittensor.mock"] = mock_subpackage

# Makes the `bittensor.core.extrinsics` subpackage available as `bittensor.extrinsics` for backwards compatibility.
extrinsics_subpackage = importlib.import_module("bittensor.core.extrinsics")
sys.modules["bittensor.extrinsics"] = extrinsics_subpackage


# Logging helpers.
def trace(on: bool = True):
    """
    Enables or disables trace logging.

    Parameters:
        on: If True, enables trace logging. If False, disables trace logging.
    """
    logging.set_trace(on)


def debug(on: bool = True):
    """
    Enables or disables debug logging.

    Parameters:
        on: If True, enables debug logging. If False, disables debug logging.
    """
    logging.set_debug(on)


def warning(on: bool = True):
    """
    Enables or disables warning logging.

    Parameters:
        on: If True, enables warning logging. If False, disables warning logging and sets default (WARNING) level.
    """
    logging.set_warning(on)


def info(on: bool = True):
    """
    Enables or disables info logging.

    Parameters:
        on: If True, enables info logging. If False, disables info logging and sets default (WARNING) level.
    """
    logging.set_info(on)


__all__ = [
    "Keypair",
    "KeyFileError",
    "serialized_keypair_to_keyfile_data",
    "deserialize_keypair_from_keyfile_data",
    "validate_password",
    "ask_password_to_encrypt",
    "keyfile_data_is_encrypted_nacl",
    "keyfile_data_is_encrypted_ansible",
    "keyfile_data_is_encrypted_legacy",
    "keyfile_data_is_encrypted",
    "keyfile_data_encryption_method",
    "legacy_encrypt_keyfile_data",
    "encrypt_keyfile_data",
    "get_coldkey_password_from_environment",
    "decrypt_keyfile_data",
    "Keyfile",
    "Wallet",
    "settings",
    "timelock",
    "AsyncSubtensor",
    "Axon",
    "AxonInfo",
    "ChainIdentity",
    "DelegateInfo",
    "DelegateInfoLite",
    "DynamicInfo",
    "IPInfo",
    "MetagraphInfo",
    "MetagraphInfoEmissions",
    "MetagraphInfoParams",
    "MetagraphInfoPool",
    "NeuronInfo",
    "NeuronInfoLite",
    "PrometheusInfo",
    "ProposalCallData",
    "ProposalVoteData",
    "ScheduledColdkeySwapInfo",
    "SelectiveMetagraphIndex",
    "StakeInfo",
    "SubnetHyperparameters",
    "SubnetIdentity",
    "SubnetInfo",
    "SubnetState",
    "WeightCommitInfo",
    "Config",
    "Dendrite",
    "BlacklistedException",
    "ChainConnectionError",
    "ChainError",
    "ChainQueryError",
    "ChainTransactionError",
    "DelegateTakeTooHigh",
    "DelegateTakeTooLow",
    "DelegateTxRateLimitExceeded",
    "DuplicateChild",
    "HotKeyAccountNotExists",
    "IdentityError",
    "InternalServerError",
    "InvalidChild",
    "InvalidRequestNameError",
    "MetadataError",
    "NominationError",
    "NonAssociatedColdKey",
    "NotDelegateError",
    "NotEnoughStakeToSetChildkeys",
    "NotRegisteredError",
    "NotVerifiedException",
    "PostProcessException",
    "PriorityException",
    "ProportionOverflow",
    "RegistrationError",
    "RegistrationNotPermittedOnRootSubnet",
    "RunException",
    "StakeError",
    "SubnetNotExists",
    "SynapseDendriteNoneException",
    "SynapseParsingError",
    "TooManyChildren",
    "TransferError",
    "TxRateLimitExceeded",
    "UnknownSynapseError",
    "UnstakeError",
    "Metagraph",
    "BLOCKTIME",
    "StreamingSynapse",
    "Subtensor",
    "SubtensorApi",
    "TerminalInfo",
    "Synapse",
    "Tensor",
    "PriorityThreadPoolExecutor",
    "ss58_to_vec_u8",
    "version_checking",
    "strtobool",
    "get_explorer_url_for_network",
    "ss58_address_to_bytes",
    "u16_normalized_float",
    "u64_normalized_float",
    "get_hash",
    "Balance",
    "tao",
    "rao",
    "logging",
    "MockSubtensor",
    "SubnetsAPI",
    "async_subtensor",
    "axon",
    "config",
    "dendrite",
    "keyfile",
    "metagraph",
    "wallet",
    "subtensor",
    "synapse",
    "trace",
    "debug",
    "warning",
    "info",
    "mock_subpackage",
    "extrinsics_subpackage",
]
