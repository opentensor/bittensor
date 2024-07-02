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
import asyncio

import bittensor
from bittensor import *

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
    return asyncio.run(bittensor.metagraph(netuid, network, lite, sync, subtensor))


class Subtensor:
    def __init__(self, *args, **kwargs):
        self._async_instance = bittensor.Subtensor(*args, **kwargs)

    def __getattr__(self, item):
        attr = getattr(self._async_instance, item)
        if asyncio.iscoroutinefunction(attr):
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(attr(*args, **kwargs))

            return sync_wrapper
        return attr


subtensor = Subtensor
