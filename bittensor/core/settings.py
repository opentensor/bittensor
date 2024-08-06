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

__version__ = "7.3.0"

import os
import re
from pathlib import Path
from rich.console import Console
from rich.traceback import install

from munch import munchify


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

DEFAULT_ENDPOINT = "wss://entrypoint-finney.opentensor.ai:443"
DEFAULT_NETWORK = "finney"

# Create dirs if they don't exist
WALLETS_DIR.mkdir(parents=True, exist_ok=True)
MINERS_DIR.mkdir(parents=True, exist_ok=True)

# Bittensor networks name
networks = ["local", "finney", "test", "archive"]

# Bittensor endpoints (Needs to use wss://)
finney_entrypoint = "wss://entrypoint-finney.opentensor.ai:443"
finney_test_entrypoint = "wss://test.finney.opentensor.ai:443/"
archive_entrypoint = "wss://archive.chain.opentensor.ai:443/"
local_entrypoint = os.getenv("BT_SUBTENSOR_CHAIN_ENDPOINT") or "ws://127.0.0.1:9944"

# Currency Symbols Bittensor
tao_symbol: str = chr(0x03C4)
rao_symbol: str = chr(0x03C1)

# Pip address for versioning
pipaddress = "https://pypi.org/pypi/bittensor/json"

# Substrate chain block time (seconds).
blocktime = 12

# Substrate ss58_format
ss58_format = 42

# Wallet ss58 address length
ss58_address_length = 48

# Raw GitHub url for delegates registry file
delegates_details_url = "https://raw.githubusercontent.com/opentensor/bittensor-delegates/main/public/delegates.json"

# Block Explorers map network to explorer url
# Must all be polkadotjs explorer urls
network_explorer_map = {
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
type_registry: dict = {
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

defaults = Munch = munchify(
    {
        "axon": {
            "port": os.getenv("BT_AXON_PORT") or 8091,
            "ip": os.getenv("BT_AXON_IP") or "[::]",
            "external_port": os.getenv("BT_AXON_EXTERNAL_PORT") or None,
            "external_ip": os.getenv("BT_AXON_EXTERNAL_IP") or None,
            "max_workers": os.getenv("BT_AXON_MAX_WORKERS") or 10,
        },
        "logging": {
            "debug": os.getenv("BT_LOGGING_DEBUG") or False,
            "trace": os.getenv("BT_LOGGING_TRACE") or False,
            "record_log": os.getenv("BT_LOGGING_RECORD_LOG") or False,
            "logging_dir": os.getenv("BT_LOGGING_LOGGING_DIR") or str(MINERS_DIR),
        },
        "priority": {
            "max_workers": os.getenv("BT_PRIORITY_MAX_WORKERS") or 5,
            "maxsize": os.getenv("BT_PRIORITY_MAXSIZE") or 10,
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
