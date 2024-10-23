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

__version__ = "8.2.0"

import os
import re
import warnings
from pathlib import Path

from munch import munchify
from rich.console import Console
from rich.traceback import install

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

bt_console = __console__


HOME_DIR = Path.home()
USER_BITTENSOR_DIR = HOME_DIR / ".bittensor"
WALLETS_DIR = USER_BITTENSOR_DIR / "wallets"
MINERS_DIR = USER_BITTENSOR_DIR / "miners"

# Bittensor networks name
NETWORKS = ["local", "finney", "test", "archive"]

DEFAULT_ENDPOINT = "wss://entrypoint-finney.opentensor.ai:443"
DEFAULT_NETWORK = NETWORKS[1]

# Create dirs if they don't exist
WALLETS_DIR.mkdir(parents=True, exist_ok=True)
MINERS_DIR.mkdir(parents=True, exist_ok=True)


# Bittensor endpoints (Needs to use wss://)
FINNEY_ENTRYPOINT = "wss://entrypoint-finney.opentensor.ai:443"
FINNEY_TEST_ENTRYPOINT = "wss://test.finney.opentensor.ai:443/"
ARCHIVE_ENTRYPOINT = "wss://archive.chain.opentensor.ai:443/"
RAO_ENDPOINT = "wss://rao.chain.opentensor.ai:443/"
LOCAL_ENTRYPOINT = os.getenv("BT_SUBTENSOR_CHAIN_ENDPOINT") or "ws://127.0.0.1:9946"

# Currency Symbols Bittensor
TAO_SYMBOL: str = chr(0x03C4)
RAO_SYMBOL: str = chr(0x03C1)

# Pip address for versioning
PIPADDRESS = "https://pypi.org/pypi/bittensor/json"

# Substrate chain block time (seconds).
BLOCKTIME = 12

# Substrate ss58_format
SS58_FORMAT = 42

# Wallet ss58 address length
SS58_ADDRESS_LENGTH = 48

# Raw GitHub url for delegates registry file
DELEGATES_DETAILS_URL = "https://raw.githubusercontent.com/opentensor/bittensor-delegates/main/public/delegates.json"

# Block Explorers map network to explorer url
# Must all be polkadotjs explorer urls
NETWORK_EXPLORER_MAP = {
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
TYPE_REGISTRY: dict[str, dict] = {
    "types": {
        "Balance": "u64",  # Need to override default u128
    },
    "runtime_api": {
        "DelegateInfoRuntimeApi": {
            "methods": {
                "get_delegated": {
                    "params": [
                        {
                            "name": "coldkey",
                            "type": "Vec<u8>",
                        },
                    ],
                    "type": "Vec<u8>",
                },
                "get_delegates": {
                    "params": [],
                    "type": "Vec<u8>",
                },
            }
        },
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
                "get_neuron": {
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
                "get_neurons": {
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
                },
                "get_subnet_info": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                },
                "get_subnets_info": {
                    "params": [],
                    "type": "Vec<u8>",
                },
            }
        },
        "SubnetRegistrationRuntimeApi": {
            "methods": {"get_network_registration_cost": {"params": [], "type": "u64"}}
        },
        "ColdkeySwapRuntimeApi": {
            "methods": {
                "get_scheduled_coldkey_swap": {
                    "params": [
                        {
                            "name": "coldkey_account_vec",
                            "type": "Vec<u8>",
                        },
                    ],
                    "type": "Vec<u8>",
                },
                "get_remaining_arbitration_period": {
                    "params": [
                        {
                            "name": "coldkey_account_vec",
                            "type": "Vec<u8>",
                        },
                    ],
                    "type": "Vec<u8>",
                },
                "get_coldkey_swap_destinations": {
                    "params": [
                        {
                            "name": "coldkey_account_vec",
                            "type": "Vec<u8>",
                        },
                    ],
                    "type": "Vec<u8>",
                },
            }
        },
    },
}


_BT_AXON_PORT = os.getenv("BT_AXON_PORT")
_BT_AXON_MAX_WORKERS = os.getenv("BT_AXON_MAX_WORKERS")
_BT_PRIORITY_MAX_WORKERS = os.getenv("BT_PRIORITY_MAX_WORKERS")
_BT_PRIORITY_MAXSIZE = os.getenv("BT_PRIORITY_MAXSIZE")

DEFAULTS = munchify(
    {
        "axon": {
            "port": int(_BT_AXON_PORT) if _BT_AXON_PORT else 8091,
            "ip": os.getenv("BT_AXON_IP") or "[::]",
            "external_port": os.getenv("BT_AXON_EXTERNAL_PORT") or None,
            "external_ip": os.getenv("BT_AXON_EXTERNAL_IP") or None,
            "max_workers": int(_BT_AXON_MAX_WORKERS) if _BT_AXON_MAX_WORKERS else 10,
        },
        "logging": {
            "debug": os.getenv("BT_LOGGING_DEBUG") or False,
            "trace": os.getenv("BT_LOGGING_TRACE") or False,
            "record_log": os.getenv("BT_LOGGING_RECORD_LOG") or False,
            "logging_dir": os.getenv("BT_LOGGING_LOGGING_DIR") or str(MINERS_DIR),
        },
        "priority": {
            "max_workers": int(_BT_PRIORITY_MAX_WORKERS)
            if _BT_PRIORITY_MAX_WORKERS
            else 5,
            "maxsize": int(_BT_PRIORITY_MAXSIZE) if _BT_PRIORITY_MAXSIZE else 10,
        },
        "subtensor": {
            "chain_endpoint": DEFAULT_ENDPOINT,
            "network": DEFAULT_NETWORK,
            "_mock": False,
        },
        "wallet": {
            "name": "default",
            "hotkey": "default",
            "path": str(WALLETS_DIR),
        },
    }
)


# Parsing version without any literals.
__version__ = re.match(r"^\d+\.\d+\.\d+", __version__).group(0)

version_split = __version__.split(".")
_version_info = tuple(int(part) for part in version_split)
_version_int_base = 1000
assert max(_version_info) < _version_int_base

version_as_int: int = sum(
    e * (_version_int_base**i) for i, e in enumerate(reversed(_version_info))
)
assert version_as_int < 2**31  # fits in int32


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

units = [
    "\u03c4",  # τ (tau, 0)
    "\u03b1",  # α (alpha, 1)
    "\u03b2",  # β (beta, 2)
    "\u03b3",  # γ (gamma, 3)
    "\u03b4",  # δ (delta, 4)
    "\u03b5",  # ε (epsilon, 5)
    "\u03b6",  # ζ (zeta, 6)
    "\u03b7",  # η (eta, 7)
    "\u03b8",  # θ (theta, 8)
    "\u03b9",  # ι (iota, 9)
    "\u03ba",  # κ (kappa, 10)
    "\u03bb",  # λ (lambda, 11)
    "\u03bc",  # μ (mu, 12)
    "\u03bd",  # ν (nu, 13)
    "\u03be",  # ξ (xi, 14)
    "\u03bf",  # ο (omicron, 15)
    "\u03c0",  # π (pi, 16)
    "\u03c1",  # ρ (rho, 17)
    "\u03c3",  # σ (sigma, 18)
    "t",  # t (tau, 19)
    "\u03c5",  # υ (upsilon, 20)
    "\u03c6",  # φ (phi, 21)
    "\u03c7",  # χ (chi, 22)
    "\u03c8",  # ψ (psi, 23)
    "\u03c9",  # ω (omega, 24)
    # Hebrew letters
    "\u05d0",  # א (aleph, 25)
    "\u05d1",  # ב (bet, 26)
    "\u05d2",  # ג (gimel, 27)
    "\u05d3",  # ד (dalet, 28)
    "\u05d4",  # ה (he, 29)
    "\u05d5",  # ו (vav, 30)
    "\u05d6",  # ז (zayin, 31)
    "\u05d7",  # ח (het, 32)
    "\u05d8",  # ט (tet, 33)
    "\u05d9",  # י (yod, 34)
    "\u05da",  # ך (final kaf, 35)
    "\u05db",  # כ (kaf, 36)
    "\u05dc",  # ל (lamed, 37)
    "\u05dd",  # ם (final mem, 38)
    "\u05de",  # מ (mem, 39)
    "\u05df",  # ן (final nun, 40)
    "\u05e0",  # נ (nun, 41)
    "\u05e1",  # ס (samekh, 42)
    "\u05e2",  # ע (ayin, 43)
    "\u05e3",  # ף (final pe, 44)
    "\u05e4",  # פ (pe, 45)
    "\u05e5",  # ץ (final tsadi, 46)
    "\u05e6",  # צ (tsadi, 47)
    "\u05e7",  # ק (qof, 48)
    "\u05e8",  # ר (resh, 49)
    "\u05e9",  # ש (shin, 50)
    "\u05ea",  # ת (tav, 51)
    # Georgian Alphabet (Mkhedruli)
    "\u10d0",  # ა (Ani, 97)
    "\u10d1",  # ბ (Bani, 98)
    "\u10d2",  # გ (Gani, 99)
    "\u10d3",  # დ (Doni, 100)
    "\u10d4",  # ე (Eni, 101)
    "\u10d5",  # ვ (Vini, 102)
    # Armenian Alphabet
    "\u0531",  # Ա (Ayp, 103)
    "\u0532",  # Բ (Ben, 104)
    "\u0533",  # Գ (Gim, 105)
    "\u0534",  # Դ (Da, 106)
    "\u0535",  # Ե (Ech, 107)
    "\u0536",  # Զ (Za, 108)
    # "\u055e",  # ՞ (Question mark, 109)
    # Runic Alphabet
    "\u16a0",  # ᚠ (Fehu, wealth, 81)
    "\u16a2",  # ᚢ (Uruz, strength, 82)
    "\u16a6",  # ᚦ (Thurisaz, giant, 83)
    "\u16a8",  # ᚨ (Ansuz, god, 84)
    "\u16b1",  # ᚱ (Raidho, ride, 85)
    "\u16b3",  # ᚲ (Kaunan, ulcer, 86)
    "\u16c7",  # ᛇ (Eihwaz, yew, 87)
    "\u16c9",  # ᛉ (Algiz, protection, 88)
    "\u16d2",  # ᛒ (Berkanan, birch, 89)
    # Cyrillic Alphabet
    "\u0400",  # Ѐ (Ie with grave, 110)
    "\u0401",  # Ё (Io, 111)
    "\u0402",  # Ђ (Dje, 112)
    "\u0403",  # Ѓ (Gje, 113)
    "\u0404",  # Є (Ukrainian Ie, 114)
    "\u0405",  # Ѕ (Dze, 115)
    # Coptic Alphabet
    "\u2c80",  # Ⲁ (Alfa, 116)
    "\u2c81",  # ⲁ (Small Alfa, 117)
    "\u2c82",  # Ⲃ (Vida, 118)
    "\u2c83",  # ⲃ (Small Vida, 119)
    "\u2c84",  # Ⲅ (Gamma, 120)
    "\u2c85",  # ⲅ (Small Gamma, 121)
    # Arabic letters
    "\u0627",  # ا (alef, 52)
    "\u0628",  # ب (ba, 53)
    "\u062a",  # ت (ta, 54)
    "\u062b",  # ث (tha, 55)
    "\u062c",  # ج (jeem, 56)
    "\u062d",  # ح (ha, 57)
    "\u062e",  # خ (kha, 58)
    "\u062f",  # د (dal, 59)
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
    "\u063a",  # غ (ghain, 70)
    "\u0641",  # ف (fa, 71)
    "\u0642",  # ق (qaf, 72)
    "\u0643",  # ك (kaf, 73)
    "\u0644",  # ل (lam, 74)
    "\u0645",  # م (meem, 75)
    "\u0646",  # ن (noon, 76)
    "\u0647",  # ه (ha, 77)
    "\u0648",  # و (waw, 78)
    "\u0649",  # ى (alef maksura, 79)
    "\u064a",  # ي (ya, 80)
    # Ogham Alphabet
    "\u1680",  #   (Space, 90)
    "\u1681",  # ᚁ (Beith, birch, 91)
    "\u1682",  # ᚂ (Luis, rowan, 92)
    "\u1683",  # ᚃ (Fearn, alder, 93)
    "\u1684",  # ᚄ (Sail, willow, 94)
    "\u1685",  # ᚅ (Nion, ash, 95)
    "\u169b",  # ᚛ (Forfeda, 96)
    # Brahmi Script
    "\u11000",  # 𑀀 (A, 122)
    "\u11001",  # 𑀁 (Aa, 123)
    "\u11002",  # 𑀂 (I, 124)
    "\u11003",  # 𑀃 (Ii, 125)
    "\u11005",  # 𑀅 (U, 126)
    # Tifinagh Alphabet
    "\u2d30",  # ⴰ (Ya, 127)
    "\u2d31",  # ⴱ (Yab, 128)
]
