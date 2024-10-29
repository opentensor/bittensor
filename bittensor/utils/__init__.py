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

from urllib.parse import urlparse
import ast
import hashlib
from typing import Any, Literal, Union, Optional, TYPE_CHECKING

import scalecodec
from bittensor_wallet import Keypair
from substrateinterface.utils import ss58

from bittensor.core.settings import SS58_FORMAT, bt_err_console
from bittensor.utils.btlogging import logging
from .registration import torch, use_torch
from .version import version_checking, check_version, VersionCheckError

if TYPE_CHECKING:
    from bittensor.utils.async_substrate_interface import AsyncSubstrateInterface
    from substrateinterface import SubstrateInterface

RAOPERTAO = 1e9
U16_MAX = 65535
U64_MAX = 18446744073709551615


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


def get_hash(content, encoding="utf-8"):
    sha3 = hashlib.sha3_256()

    # Update the hash object with the concatenated string
    sha3.update(content.encode(encoding))

    # Produce the hash
    return sha3.hexdigest()


def format_error_message(error_message: Union[dict, Exception], substrate: Union["AsyncSubstrateInterface", "SubstrateInterface"]) -> str:
    """
    Formats an error message from the Subtensor error information for use in extrinsics.

    Args:
        error_message: A dictionary containing the error information from Subtensor, or a SubstrateRequestException
                       containing dictionary literal args.
        substrate: The initialised SubstrateInterface object to use.

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
            except ValueError:
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
            if err_data.startswith("Custom error:") and substrate:
                if substrate.metadata:
                    try:
                        pallet = substrate.metadata.get_metadata_pallet(
                            "SubtensorModule"
                        )
                        error_index = int(err_data.split("Custom error:")[-1])

                        error_dict = pallet.errors[error_index].value
                        err_type = error_dict.get("message", err_type)
                        err_docs = error_dict.get("docs", [])
                        err_description = err_docs[0] if err_docs else err_description
                    except (AttributeError, IndexError):
                        bt_err_console.print(
                            "Substrate pallets data unavailable. This is usually caused by an uninitialized substrate."
                        )
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


# Subnet 24 uses this function
def is_valid_ss58_address(address: str) -> bool:
    """
    Checks if the given address is a valid ss58 address.

    Args:
        address(str): The address to check.

    Returns:
        True if the address is a valid ss58 address for Bittensor, False otherwise.
    """
    try:
        return ss58.is_valid_ss58_address(
            address, valid_ss58_format=SS58_FORMAT
        ) or ss58.is_valid_ss58_address(
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


def decode_hex_identity_dict(info_dictionary) -> dict[str, Any]:
    """
    Decodes hex-encoded strings in a dictionary.

    This function traverses the given dictionary, identifies hex-encoded strings, and decodes them into readable strings. It handles nested dictionaries and lists within the dictionary.

    Args:
        info_dictionary (dict): The dictionary containing hex-encoded strings to decode.

    Returns:
        dict: The dictionary with decoded strings.

    Examples:
        input_dict = {
        ...     "name": {"value": "0x6a6f686e"},
        ...     "additional": [
        ...         [{"data": "0x64617461"}]
        ...     ]
        ... }
        decode_hex_identity_dict(input_dict)
        {'name': 'john', 'additional': [('data', 'data')]}
    """

    def get_decoded(data: str) -> str:
        """Decodes a hex-encoded string."""
        try:
            return bytes.fromhex(data[2:]).decode()
        except UnicodeDecodeError:
            print(f"Could not decode: {key}: {item}")

    for key, value in info_dictionary.items():
        if isinstance(value, dict):
            item = list(value.values())[0]
            if isinstance(item, str) and item.startswith("0x"):
                try:
                    info_dictionary[key] = get_decoded(item)
                except UnicodeDecodeError:
                    print(f"Could not decode: {key}: {item}")
            else:
                info_dictionary[key] = item
        if key == "additional":
            additional = []
            for item in value:
                additional.append(
                    tuple(
                        get_decoded(data=next(iter(sub_item.values())))
                        for sub_item in item
                    )
                )
            info_dictionary[key] = additional

    return info_dictionary


def validate_chain_endpoint(endpoint_url: str) -> tuple[bool, str]:
    """Validates if the provided endpoint URL is a valid WebSocket URL."""
    parsed = urlparse(endpoint_url)
    if parsed.scheme not in ("ws", "wss"):
        return False, (
            f"Invalid URL or network name provided: [bright_cyan]({endpoint_url})[/bright_cyan].\n"
            "Allowed network names are [bright_cyan]finney, test, local[/bright_cyan]. "
            "Valid chain endpoints should use the scheme [bright_cyan]`ws` or `wss`[/bright_cyan].\n"
        )
    if not parsed.netloc:
        return False, "Invalid URL passed as the endpoint"
    return True, ""
