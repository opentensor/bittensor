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
