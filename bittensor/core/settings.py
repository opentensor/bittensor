import importlib.metadata
import os
import re
from pathlib import Path

from munch import munchify

ROOT_TAO_STAKE_WEIGHT = 0.18

READ_ONLY = os.getenv("READ_ONLY") == "1"

HOME_DIR = Path.home()
USER_BITTENSOR_DIR = HOME_DIR / ".bittensor"
WALLETS_DIR = USER_BITTENSOR_DIR / "wallets"
MINERS_DIR = USER_BITTENSOR_DIR / "miners"

TAO_APP_BLOCK_EXPLORER = "https://www.tao.app/block/"

__version__ = importlib.metadata.version("bittensor")


if not READ_ONLY:
    # Create dirs if they don't exist
    WALLETS_DIR.mkdir(parents=True, exist_ok=True)
    MINERS_DIR.mkdir(parents=True, exist_ok=True)

# Bittensor networks name
NETWORKS = ["finney", "test", "archive", "local", "latent-lite"]

# Bittensor endpoints (Needs to use wss://)
FINNEY_ENTRYPOINT = "wss://entrypoint-finney.opentensor.ai:443"
FINNEY_TEST_ENTRYPOINT = "wss://test.finney.opentensor.ai:443"
ARCHIVE_ENTRYPOINT = "wss://archive.chain.opentensor.ai:443"
LOCAL_ENTRYPOINT = os.getenv("BT_SUBTENSOR_CHAIN_ENDPOINT") or "ws://127.0.0.1:9944"
LATENT_LITE_ENTRYPOINT = "wss://lite.sub.latent.to:443"

NETWORK_MAP = {
    NETWORKS[0]: FINNEY_ENTRYPOINT,
    NETWORKS[1]: FINNEY_TEST_ENTRYPOINT,
    NETWORKS[2]: ARCHIVE_ENTRYPOINT,
    NETWORKS[3]: LOCAL_ENTRYPOINT,
    NETWORKS[4]: LATENT_LITE_ENTRYPOINT,
}

REVERSE_NETWORK_MAP = {
    FINNEY_ENTRYPOINT: NETWORKS[0],
    FINNEY_TEST_ENTRYPOINT: NETWORKS[1],
    ARCHIVE_ENTRYPOINT: NETWORKS[2],
    LOCAL_ENTRYPOINT: NETWORKS[3],
    LATENT_LITE_ENTRYPOINT: NETWORKS[4],
}

DEFAULT_NETWORK = NETWORKS[0]
DEFAULT_ENDPOINT = NETWORK_MAP[DEFAULT_NETWORK]

# Currency Symbols Bittensor
TAO_SYMBOL: str = chr(0x03C4)
RAO_SYMBOL: str = chr(0x03C1)

# Pip address for versioning
PIPADDRESS = "https://pypi.org/pypi/bittensor/json"

# Substrate chain block time (seconds).
BLOCKTIME = 12

# Wallet ss58 address length
SS58_ADDRESS_LENGTH = 48

# Default period for extrinsics Era
# details https://paritytech.github.io/polkadot-sdk/master/src/sp_runtime/generic/era.rs.html#65-72
DEFAULT_PERIOD = 128

# Default MEV Shield protection setting for extrinsics.
# When enabled, transactions are encrypted to protect against Miner Extractable Value (MEV) attacks.
DEFAULT_MEV_PROTECTION = os.getenv("BT_MEV_PROTECTION", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Public_key size for ML-KEM-768 (must be exactly 1184 bytes)
MLKEM768_PUBLIC_KEY_SIZE = 1184

# Block Explorers map network to explorer url
# Must all be polkadotjs explorer urls
NETWORK_EXPLORER_MAP = {
    "opentensor": {
        "local": "https://polkadot.js.org/apps/?rpc=wss%3A%2F%2Fentrypoint-finney.opentensor.ai%3A443#/explorer",
        "endpoint": "https://polkadot.js.org/apps/?rpc=wss%3A%2F%2Fentrypoint-finney.opentensor.ai%3A443#/explorer",
        "finney": "https://polkadot.js.org/apps/?rpc=wss%3A%2F%2Fentrypoint-finney.opentensor.ai%3A443#/explorer",
    },
    "taostats": {
        "local": "https://taostats.io",
        "endpoint": "https://taostats.io",
        "finney": "https://taostats.io",
    },
}

# --- Type Registry ---
TYPE_REGISTRY: dict[str, dict] = {
    "types": {
        "Balance": "u64",  # Need to override default u128
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
            "debug": bool(os.getenv("BT_LOGGING_DEBUG")) or False,
            "trace": bool(os.getenv("BT_LOGGING_TRACE")) or False,
            "info": bool(os.getenv("BT_LOGGING_INFO")) or False,
            "record_log": bool(os.getenv("BT_LOGGING_RECORD_LOG")) or False,
            "logging_dir": None
            if READ_ONLY
            else os.getenv("BT_LOGGING_LOGGING_DIR") or str(MINERS_DIR),
            "enable_third_party_loggers": os.getenv(
                "BT_LOGGING_ENABLE_THIRD_PARTY_LOGGERS"
            )
            or False,
        },
        "priority": {
            "max_workers": int(_BT_PRIORITY_MAX_WORKERS)
            if _BT_PRIORITY_MAX_WORKERS
            else 5,
            "maxsize": int(_BT_PRIORITY_MAXSIZE) if _BT_PRIORITY_MAXSIZE else 10,
        },
        "subtensor": {
            "chain_endpoint": os.getenv("BT_SUBTENSOR_CHAIN_ENDPOINT")
            or DEFAULT_ENDPOINT,
            "network": os.getenv("BT_SUBTENSOR_NETWORK") or DEFAULT_NETWORK,
            "_mock": False,
        },
        "wallet": {
            "name": os.getenv("BT_WALLET_NAME") or "default",
            "hotkey": os.getenv("BT_WALLET_HOTKEY") or "default",
            "path": os.getenv("BT_WALLET_PATH") or str(WALLETS_DIR),
        },
        "config": False,
        "strict": False,
        "no_version_checking": False,
    }
)


# Parsing version without any literals.
__version__ = re.match(r"^\d+\.\d+\.\d+", __version__).group(0)

__version_split = __version__.split(".")
_version_info = tuple(int(part) for part in __version_split)
_version_int_base = 1000
assert max(_version_info) < _version_int_base

version_as_int: int = sum(
    e * (_version_int_base**i) for i, e in enumerate(reversed(_version_info))
)
assert version_as_int < 2**31  # fits in int32
