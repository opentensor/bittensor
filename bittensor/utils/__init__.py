import ast
import decimal
import hashlib
from collections import namedtuple
from typing import Any, Literal, Union, Optional, TYPE_CHECKING
from urllib.parse import urlparse

import scalecodec
from async_substrate_interface.utils import (
    hex_to_bytes,
)
from bittensor_wallet import Keypair
from bittensor_wallet.errors import KeyFileError, PasswordError
from scalecodec import ss58_decode, is_valid_ss58_address as _is_valid_ss58_address

from bittensor.core import settings
from bittensor.core.settings import SS58_FORMAT
from bittensor.utils.btlogging import logging
from .registration import torch, use_torch
from .version import version_checking, check_version, VersionCheckError

if TYPE_CHECKING:
    from bittensor_wallet import Wallet


# redundant aliases
logging = logging
torch = torch
use_torch = use_torch
version_checking = version_checking
check_version = check_version
VersionCheckError = VersionCheckError
ss58_decode = ss58_decode
hex_to_bytes = hex_to_bytes


RAOPERTAO = 1e9
U16_MAX = 65535
U64_MAX = 18446744073709551615

UnlockStatus = namedtuple("UnlockStatus", ["success", "message"])


class Certificate(str):
    def __new__(cls, data: Union[str, dict]):
        if isinstance(data, dict):
            tuple_ascii = data["public_key"][0]
            string = chr(data["algorithm"]) + "".join(chr(i) for i in tuple_ascii)
        else:
            string = data
        return str.__new__(cls, string)


def decode_hex_identity_dict(info_dictionary: dict[str, Any]) -> dict[str, Any]:
    """Decodes a dictionary of hexadecimal identities."""
    decoded_info = {}
    for k, v in info_dictionary.items():
        if isinstance(v, dict):
            item = next(iter(v.values()))
        else:
            item = v

        if isinstance(item, tuple):
            try:
                decoded_info[k] = bytes(item).decode()
            except UnicodeDecodeError:
                print(f"Could not decode: {k}: {item}")
        else:
            decoded_info[k] = item
    return decoded_info


def ss58_to_vec_u8(ss58_address: str) -> list[int]:
    ss58_bytes: bytes = ss58_address_to_bytes(ss58_address)
    encoded_address: list[int] = [int(byte) for byte in ss58_bytes]
    return encoded_address


def strtobool(val: str) -> Union[bool, Literal["==SUPRESS=="]]:
    """
    Converts a string to a boolean value.

    truth-y values are 'y', 'yes', 't', 'true', 'on', and '1';
    false-y values are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def _get_explorer_root_url_by_network_from_map(
    network: str, network_map: dict[str, dict[str, str]]
) -> Optional[dict[str, str]]:
    """
    Returns the explorer root url for the given network name from the given network map.

    Args:
        network(str): The network to get the explorer url for.
        network_map(dict[str, str]): The network map to get the explorer url from.

    Returns:
        The explorer url for the given network.
        Or None if the network is not in the network map.
    """
    explorer_urls: Optional[dict[str, str]] = {}
    for entity_nm, entity_network_map in network_map.items():
        if network in entity_network_map:
            explorer_urls[entity_nm] = entity_network_map[network]

    return explorer_urls


def get_explorer_url_for_network(
    network: str, block_hash: str, network_map: dict[str, dict[str, str]]
) -> Optional[dict[str, str]]:
    """
    Returns the explorer url for the given block hash and network.

    Args:
        network(str): The network to get the explorer url for.
        block_hash(str): The block hash to get the explorer url for.
        network_map(dict[str, dict[str, str]]): The network maps to get the explorer urls from.

    Returns:
        The explorer url for the given block hash and network.
        Or None if the network is not known.
    """

    explorer_urls: Optional[dict[str, str]] = {}
    # Will be None if the network is not known. i.e. not in network_map
    explorer_root_urls: Optional[dict[str, str]] = (
        _get_explorer_root_url_by_network_from_map(network, network_map)
    )

    if explorer_root_urls != {}:
        # We are on a known network.
        explorer_opentensor_url = (
            f"{explorer_root_urls.get('opentensor')}/query/{block_hash}"
        )
        explorer_taostats_url = (
            f"{explorer_root_urls.get('taostats')}/extrinsic/{block_hash}"
        )
        explorer_urls["opentensor"] = explorer_opentensor_url
        explorer_urls["taostats"] = explorer_taostats_url

    return explorer_urls


def ss58_address_to_bytes(ss58_address: str) -> bytes:
    """Converts a ss58 address to a bytes object."""
    account_id_hex: str = scalecodec.ss58_decode(ss58_address, SS58_FORMAT)
    return bytes.fromhex(account_id_hex)


def u16_normalized_float(x: int) -> float:
    return float(x) / float(U16_MAX)


def u64_normalized_float(x: int) -> float:
    return float(x) / float(U64_MAX)


def float_to_u64(value: float) -> int:
    """Converts a float to a u64 int"""

    value = decimal.Decimal(str(value))

    if not (0 <= value <= 1):
        raise ValueError("Input value must be between 0 and 1")

    return int(value * U64_MAX)


def get_hash(content, encoding="utf-8"):
    sha3 = hashlib.sha3_256()

    # Update the hash object with the concatenated string
    sha3.update(content.encode(encoding))

    # Produce the hash
    return sha3.hexdigest()


def format_error_message(error_message: Union[dict, Exception]) -> str:
    """
    Formats an error message from the Subtensor error information for use in extrinsics.

    Args:
        error_message: A dictionary containing the error information from Subtensor, or a SubstrateRequestException
                       containing dictionary literal args.

    Returns:
        str: A formatted error message string.
    """
    err_name = "UnknownError"
    err_type = "UnknownType"
    err_description = "Unknown Description"

    if isinstance(error_message, Exception):
        # generally gotten through SubstrateRequestException args
        new_error_message = None
        for arg in error_message.args:
            try:
                d = ast.literal_eval(arg)
                if isinstance(d, dict):
                    if "error" in d:
                        new_error_message = d["error"]
                        break
                    elif all(x in d for x in ["code", "message", "data"]):
                        new_error_message = d
                        break
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                pass
        if new_error_message is None:
            return_val = " ".join(error_message.args)
            return f"Subtensor returned: {return_val}"
        else:
            error_message = new_error_message

    if isinstance(error_message, dict):
        # subtensor error structure
        if (
            error_message.get("code")
            and error_message.get("message")
            and error_message.get("data")
        ):
            err_name = "SubstrateRequestException"
            err_type = error_message.get("message", "")
            err_data = error_message.get("data", "")

            # subtensor custom error marker
            if err_data.startswith("Custom error:"):
                err_description = f"{err_data} | Please consult https://docs.bittensor.com/subtensor-nodes/subtensor-error-messages"
            else:
                err_description = err_data

        elif (
            error_message.get("type")
            and error_message.get("name")
            and error_message.get("docs")
        ):
            err_type = error_message.get("type", err_type)
            err_name = error_message.get("name", err_name)
            err_docs = error_message.get("docs", [err_description])
            err_description = err_docs[0] if err_docs else err_description

    return f"Subtensor returned `{err_name}({err_type})` error. This means: `{err_description}`."


def is_valid_ss58_address(address: str) -> bool:
    """
    Checks if the given address is a valid ss58 address.

    Args:
        address(str): The address to check.

    Returns:
        True if the address is a valid ss58 address for Bittensor, False otherwise.
    """
    try:
        return _is_valid_ss58_address(
            address, valid_ss58_format=SS58_FORMAT
        ) or _is_valid_ss58_address(
            address, valid_ss58_format=42
        )  # Default substrate ss58 format (legacy)
    except IndexError:
        return False


def _is_valid_ed25519_pubkey(public_key: Union[str, bytes]) -> bool:
    """
    Checks if the given public_key is a valid ed25519 key.

    Args:
        public_key(Union[str, bytes]): The public_key to check.

    Returns:
        True if the public_key is a valid ed25519 key, False otherwise.

    """
    try:
        if isinstance(public_key, str):
            if len(public_key) != 64 and len(public_key) != 66:
                raise ValueError("a public_key should be 64 or 66 characters")
        elif isinstance(public_key, bytes):
            if len(public_key) != 32:
                raise ValueError("a public_key should be 32 bytes")
        else:
            raise ValueError("public_key must be a string or bytes")

        keypair = Keypair(public_key=public_key)

        ss58_addr = keypair.ss58_address
        return ss58_addr is not None

    except (ValueError, IndexError):
        return False


def is_valid_bittensor_address_or_public_key(address: Union[str, bytes]) -> bool:
    """
    Checks if the given address is a valid destination address.

    Args:
        address(Union[str, bytes]): The address to check.

    Returns:
        True if the address is a valid destination address, False otherwise.
    """
    if isinstance(address, str):
        # Check if ed25519
        if address.startswith("0x"):
            return _is_valid_ed25519_pubkey(address)
        else:
            # Assume ss58 address
            return is_valid_ss58_address(address)
    elif isinstance(address, bytes):
        # Check if ed25519
        return _is_valid_ed25519_pubkey(address)
    else:
        # Invalid address type
        return False


def validate_chain_endpoint(endpoint_url: str) -> tuple[bool, str]:
    """Validates if the provided endpoint URL is a valid WebSocket URL."""
    parsed = urlparse(endpoint_url)
    if parsed.scheme not in ("ws", "wss"):
        return False, (
            f"Invalid URL or network name provided: ({endpoint_url}).\n"
            "Allowed network names are finney, test, local. "
            "Valid chain endpoints should use the scheme `ws` or `wss`.\n"
        )
    if not parsed.netloc:
        return False, "Invalid URL passed as the endpoint"
    return True, ""


def unlock_key(
    wallet: "Wallet",
    unlock_type="coldkey",
    raise_error=False,
) -> "UnlockStatus":
    """
    Attempts to decrypt a wallet's coldkey or hotkey

    Args:
        wallet: a Wallet object
        unlock_type: the key type, 'coldkey' or 'hotkey'
        raise_error: if False, will return (False, error msg), if True will raise the otherwise-caught exception.

    Returns:
        UnlockStatus for success status of unlock, with error message if unsuccessful

    Raises:
        bittensor_wallet.errors.PasswordError: incorrect password
        bittensor_wallet.errors.KeyFileError: keyfile is corrupt, non-writable, or non-readable, or non-existent
    """
    if unlock_type == "coldkey":
        unlocker = "unlock_coldkey"
    elif unlock_type == "hotkey":
        unlocker = "unlock_hotkey"
    else:
        raise ValueError(
            f"Invalid unlock type provided: {unlock_type}. Must be 'coldkey' or 'hotkey'."
        )
    try:
        getattr(wallet, unlocker)()
        return UnlockStatus(True, "")
    except PasswordError:
        if raise_error:
            raise

        err_msg = f"The password used to decrypt your {unlock_type.capitalize()} keyfile is invalid."
        return UnlockStatus(False, err_msg)
    except KeyFileError:
        if raise_error:
            raise

        err_msg = f"{unlock_type.capitalize()} keyfile is corrupt, non-writable, or non-readable, or non-existent."
        return UnlockStatus(False, err_msg)


def determine_chain_endpoint_and_network(
    network: str,
) -> tuple[Optional[str], Optional[str]]:
    """Determines the chain endpoint and network from the passed network or chain_endpoint.

    Arguments:
        network (str): The network flag. The choices are: ``finney`` (main network), ``archive`` (archive network
            +300 blocks), ``local`` (local running network), ``test`` (test network).

    Returns:
        tuple[Optional[str], Optional[str]]: The network and chain endpoint flag. If passed, overrides the
            ``network`` argument.
    """

    if network is None:
        return None, None
    if network in settings.NETWORKS:
        return network, settings.NETWORK_MAP[network]

    substrings_map = {
        "entrypoint-finney.opentensor.ai": ("finney", settings.FINNEY_ENTRYPOINT),
        "test.finney.opentensor.ai": ("test", settings.FINNEY_TEST_ENTRYPOINT),
        "archive.chain.opentensor.ai": ("archive", settings.ARCHIVE_ENTRYPOINT),
        "subvortex": ("subvortex", settings.SUBVORTEX_ENTRYPOINT),
        "127.0.0.1": ("local", network),
        "localhost": ("local", network),
    }

    for substring, result in substrings_map.items():
        if substring in network and validate_chain_endpoint(network):
            return result

    return "unknown", network
