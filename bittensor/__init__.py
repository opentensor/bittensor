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

from rich.console import Console
from rich.traceback import install

# Install and apply nest asyncio to allow the async functions
# to run in a .ipynb
import nest_asyncio

nest_asyncio.apply()

# Bittensor code and protocol version.
__version__ = "7.0.1"

version_split = __version__.split(".")
__version_as_int__: int = (
    (100 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
__new_signature_version__ = 360

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


turn_console_off()


# Logging helpers.
def trace(on: bool = True):
    logging.set_trace(on)


def debug(on: bool = True):
    logging.set_debug(on)


# Substrate chain block time (seconds).
__blocktime__ = 12

# Pip address for versioning
__pipaddress__ = "https://pypi.org/pypi/bittensor/json"

# Raw github url for delegates registry file
__delegates_details_url__: str = "https://raw.githubusercontent.com/opentensor/bittensor-delegates/main/public/delegates.json"

# Substrate ss58_format
__ss58_format__ = 42

# Wallet ss58 address length
__ss58_address_length__ = 48

__networks__ = ["dtao", "local", "finney", "test", "archive", "dev"]

__dtao_entrypoint__ = "wss://dtao-demo.chain.opentensor.ai:443"

__finney_entrypoint__ = "wss://entrypoint-finney.opentensor.ai:443"

__finney_test_entrypoint__ = "wss://test.finney.opentensor.ai:443/"

__archive_entrypoint__ = "wss://archive.chain.opentensor.ai:443/"

__dev_entrypoint__ = "wss://dev.chain.opentensor.ai:443 "

# Needs to use wss://
__bellagene_entrypoint__ = "wss://parachain.opentensor.ai:443"

__local_entrypoint__ = "ws://127.0.0.1:9946"

__tao_symbol__: str = chr(0x03C4)

__rao_symbol__: str = chr(0x03C1)

# Block Explorers map network to explorer url
## Must all be polkadotjs explorer urls
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
        "StakeInfoRuntimeApi": {
            "methods": {
                "get_stake_info_for_coldkey": {
                    "params": [{"name": "coldkey_account_vec", "type": "Vec<u8>"}],
                    "type": "Vec<u8>",
                },
                "get_stake_info_for_coldkeys": {
                    "params": [
                        {"name": "coldkey_account_vecs", "type": "Vec<Vec<u8>>"}
                    ],
                    "type": "Vec<u8>",
                },
                "get_subnet_stake_info_for_coldkeys": {
                    "params": [
                        {"name": "coldkey_account_vecs", "type": "Vec<Vec<u8>>"},
                        {"name": "netuid", "type": "u16"},
                    ],
                    "type": "Vec<u8>",
                },
                "get_subnet_stake_info_for_coldkey": {
                    "params": [
                        {"name": "coldkey_account_vec", "type": "Vec<u8>"},
                        {"name": "netuid", "type": "u16"},
                    ],
                    "type": "Vec<u8>",
                },
                "get_total_subnet_stake": {
                    "params": [{"name": "netuid", "type": "u16"}],
                    "type": "Vec<u8>",
                },
            }
        },
    },
}

from .errors import (
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
)

from substrateinterface import Keypair as Keypair
from .config import InvalidConfigFile, DefaultConfig, config, T
from .keyfile import (
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
    keyfile,
    Mockkeyfile,
)
from .wallet import display_mnemonic_msg, wallet

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
from .chain_data import (
    AxonInfo,
    DynamicPoolInfoV2,
    NeuronInfo,
    NeuronInfoLite,
    PrometheusInfo,
    DelegateInfo,
    StakeInfo,
    SubnetInfoV2,
    SubnetHyperparameters,
    IPInfo,
    ProposalCallData,
    ProposalVoteData,
)

from . import subtensor as subtensor_module
from .subtensor import subtensor as subtensor
from .cli import cli as cli, COMMANDS as ALL_COMMANDS
from .btlogging import logging
from .metagraph import metagraph as metagraph
from .threadpool import PriorityThreadPoolExecutor as PriorityThreadPoolExecutor

from .synapse import TerminalInfo, Synapse
from .stream import StreamingSynapse
from .tensor import tensor, Tensor
from .axon import axon as axon
from .dendrite import dendrite as dendrite

from .mock.keyfile_mock import MockKeyfile as MockKeyfile
from .mock.subtensor_mock import MockSubtensor as MockSubtensor
from .mock.wallet_mock import MockWallet as MockWallet

from .subnets import SubnetsAPI as SubnetsAPI

configs = [
    axon.config(),
    subtensor.config(),
    PriorityThreadPoolExecutor.config(),
    wallet.config(),
    logging.get_config(),
]
defaults = config.merge_all(configs)

units = [
    "\u03C4",  # τ (tau, 0)
    "\u03B1",  # α (alpha, 1)
    "\u03B2",  # β (beta, 2)
    "\u03B3",  # γ (gamma, 3)
    "\u03B4",  # δ (delta, 4)
    "\u03B5",  # ε (epsilon, 5)
    "\u03B6",  # ζ (zeta, 6)
    "\u03B7",  # η (eta, 7)
    "\u03B8",  # θ (theta, 8)
    "\u03B9",  # ι (iota, 9)
    "\u03BA",  # κ (kappa, 10)
    "\u03BB",  # λ (lambda, 11)
    "\u03BC",  # μ (mu, 12)
    "\u03BD",  # ν (nu, 13)
    "\u03BE",  # ξ (xi, 14)
    "\u03BF",  # ο (omicron, 15)
    "\u03C0",  # π (pi, 16)
    "\u03C1",  # ρ (rho, 17)
    "\u03C3",  # σ (sigma, 18)
    "\u03C4",  # τ (tau, 19)
    "\u03C5",  # υ (upsilon, 20)
    "\u03C6",  # φ (phi, 21)
    "\u03C7",  # χ (chi, 22)
    "\u03C8",  # ψ (psi, 23)
    "\u03C9",  # ω (omega, 24)
    # Hebrew letters
    "\u05D0",  # א (aleph, 25)
    "\u05D1",  # ב (bet, 26)
    "\u05D2",  # ג (gimel, 27)
    "\u05D3",  # ד (dalet, 28)
    "\u05D4",  # ה (he, 29)
    "\u05D5",  # ו (vav, 30)
    "\u05D6",  # ז (zayin, 31)
    "\u05D7",  # ח (het, 32)
    "\u05D8",  # ט (tet, 33)
    "\u05D9",  # י (yod, 34)
    "\u05DA",  # ך (final kaf, 35)
    "\u05DB",  # כ (kaf, 36)
    "\u05DC",  # ל (lamed, 37)
    "\u05DD",  # ם (final mem, 38)
    "\u05DE",  # מ (mem, 39)
    "\u05DF",  # ן (final nun, 40)
    "\u05E0",  # נ (nun, 41)
    "\u05E1",  # ס (samekh, 42)
    "\u05E2",  # ע (ayin, 43)
    "\u05E3",  # ף (final pe, 44)
    "\u05E4",  # פ (pe, 45)
    "\u05E5",  # ץ (final tsadi, 46)
    "\u05E6",  # צ (tsadi, 47)
    "\u05E7",  # ק (qof, 48)
    "\u05E8",  # ר (resh, 49)
    "\u05E9",  # ש (shin, 50)
    "\u05EA",  # ת (tav, 51)
    # Arabic letters
    "\u0627",  # ا (alef, 52)
    "\u0628",  # ب (ba, 53)
    "\u062A",  # ت (ta, 54)
    "\u062B",  # ث (tha, 55)
    "\u062C",  # ج (jeem, 56)
    "\u062D",  # ح (ha, 57)
    "\u062E",  # خ (kha, 58)
    "\u062F",  # د (dal, 59)
    "\u0630",  # ذ (dhal, 60)
    "\u0631",  # ر (ra, 61)
    "\u0632",  # ز (zay, 62)
    "\u0633",  # س (seen, 63)
    "\u0634",  # ش (sheen, 64)
    "\u0635",  # ص (sad, 65)
    "\u0636",  # ض (dad, 66)
    "\u0637",  # ط (ta, 67)
    "\u0638",  # ظ (dha, 68)
    "\u0639",  # ع (ain, 69)
    "\u063A",  # غ (ghain, 70)
    "\u0641",  # ف (fa, 71)
    "\u0642",  # ق (qaf, 72)
    "\u0643",  # ك (kaf, 73)
    "\u0644",  # ل (lam, 74)
    "\u0645",  # م (meem, 75)
    "\u0646",  # ن (noon, 76)
    "\u0647",  # ه (ha, 77)
    "\u0648",  # و (waw, 78)
    "\u0649",  # ى (alef maksura, 79)
    "\u064A",  # ي (ya, 80)
    # Runic Alphabet
    "\u16A0",  # ᚠ (Fehu, wealth, 81)
    "\u16A2",  # ᚢ (Uruz, strength, 82)
    "\u16A6",  # ᚦ (Thurisaz, giant, 83)
    "\u16A8",  # ᚨ (Ansuz, god, 84)
    "\u16B1",  # ᚱ (Raidho, ride, 85)
    "\u16B3",  # ᚲ (Kaunan, ulcer, 86)
    "\u16C7",  # ᛇ (Eihwaz, yew, 87)
    "\u16C9",  # ᛉ (Algiz, protection, 88)
    "\u16D2",  # ᛒ (Berkanan, birch, 89)
    # Ogham Alphabet
    "\u1680",  #   (Space, 90)
    "\u1681",  # ᚁ (Beith, birch, 91)
    "\u1682",  # ᚂ (Luis, rowan, 92)
    "\u1683",  # ᚃ (Fearn, alder, 93)
    "\u1684",  # ᚄ (Sail, willow, 94)
    "\u1685",  # ᚅ (Nion, ash, 95)
    "\u169B",  # ᚛ (Forfeda, 96)
    # Georgian Alphabet (Mkhedruli)
    "\u10D0",  # ა (Ani, 97)
    "\u10D1",  # ბ (Bani, 98)
    "\u10D2",  # გ (Gani, 99)
    "\u10D3",  # დ (Doni, 100)
    "\u10D4",  # ე (Eni, 101)
    "\u10D5",  # ვ (Vini, 102)
    # Armenian Alphabet
    "\u0531",  # Ա (Ayp, 103)
    "\u0532",  # Բ (Ben, 104)
    "\u0533",  # Գ (Gim, 105)
    "\u0534",  # Դ (Da, 106)
    "\u0535",  # Ե (Ech, 107)
    "\u0536",  # Զ (Za, 108)
    "\u055E",  # ՞ (Question mark, 109)
    # Cyrillic Alphabet
    "\u0400",  # Ѐ (Ie with grave, 110)
    "\u0401",  # Ё (Io, 111)
    "\u0402",  # Ђ (Dje, 112)
    "\u0403",  # Ѓ (Gje, 113)
    "\u0404",  # Є (Ukrainian Ie, 114)
    "\u0405",  # Ѕ (Dze, 115)
    # Coptic Alphabet
    "\u2C80",  # Ⲁ (Alfa, 116)
    "\u2C81",  # ⲁ (Small Alfa, 117)
    "\u2C82",  # Ⲃ (Vida, 118)
    "\u2C83",  # ⲃ (Small Vida, 119)
    "\u2C84",  # Ⲅ (Gamma, 120)
    "\u2C85",  # ⲅ (Small Gamma, 121)
    # Brahmi Script
    "\u11000",  # 𑀀 (A, 122)
    "\u11001",  # 𑀁 (Aa, 123)
    "\u11002",  # 𑀂 (I, 124)
    "\u11003",  # 𑀃 (Ii, 125)
    "\u11005",  # 𑀅 (U, 126)
    # Tifinagh Alphabet
    "\u2D30",  # ⴰ (Ya, 127)
    "\u2D31",  # ⴱ (Yab, 128)
]
