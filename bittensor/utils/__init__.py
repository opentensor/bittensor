# The MIT License (MIT)
# Copyright © 2022 Opentensor Foundation
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

from typing import Callable, Union, List, Optional, Dict, Literal

import bittensor
import hashlib
import requests
import torch
import scalecodec

from .wallet_utils import *  # noqa F401

RAOPERTAO = 1e9
U16_MAX = 65535
U64_MAX = 18446744073709551615


def ss58_to_vec_u8(ss58_address: str) -> List[int]:
    ss58_bytes: bytes = bittensor.utils.ss58_address_to_bytes(ss58_address)
    encoded_address: List[int] = [int(byte) for byte in ss58_bytes]
    return encoded_address


def unbiased_topk(values, k, dim=0, sorted=True, largest=True):
    r"""Selects topk as in torch.topk but does not bias lower indices when values are equal.
    Args:
        values: (torch.Tensor)
            Values to index into.
        k: (int):
            Number to take.

    Return:
        topk: (torch.Tensor):
            topk k values.
        indices: (torch.LongTensor)
            indices of the topk values.
    """
    permutation = torch.randperm(values.shape[dim])
    permuted_values = values[permutation]
    topk, indices = torch.topk(
        permuted_values, k, dim=dim, sorted=sorted, largest=largest
    )
    return topk, permutation[indices]


def version_checking(timeout: int = 15):
    try:
        bittensor.logging.debug(
            f"Checking latest Bittensor version at: {bittensor.__pipaddress__}"
        )
        response = requests.get(bittensor.__pipaddress__, timeout=timeout)
        latest_version = response.json()["info"]["version"]
        version_split = latest_version.split(".")
        latest_version_as_int = (
            (100 * int(version_split[0]))
            + (10 * int(version_split[1]))
            + (1 * int(version_split[2]))
        )

        if latest_version_as_int > bittensor.__version_as_int__:
            print(
                "\u001b[33mBittensor Version: Current {}/Latest {}\nPlease update to the latest version at your earliest convenience. "
                "Run the following command to upgrade:\n\n\u001b[0mpython -m pip install --upgrade bittensor".format(
                    bittensor.__version__, latest_version
                )
            )

    except requests.exceptions.Timeout:
        bittensor.logging.error("Version check failed due to timeout")
    except requests.exceptions.RequestException as e:
        bittensor.logging.error(f"Version check failed due to request failure: {e}")


def strtobool_with_default(
    default: bool,
) -> Callable[[str], Union[bool, Literal["==SUPRESS=="]]]:
    """
    Creates a strtobool function with a default value.

    Args:
        default(bool): The default value to return if the string is empty.

    Returns:
        The strtobool function with the default value.
    """
    return lambda x: strtobool(x) if x != "" else default


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


def get_explorer_root_url_by_network_from_map(
    network: str, network_map: Dict[str, Dict[str, str]]
) -> Optional[Dict[str, str]]:
    r"""
    Returns the explorer root url for the given network name from the given network map.

    Args:
        network(str): The network to get the explorer url for.
        network_map(Dict[str, str]): The network map to get the explorer url from.

    Returns:
        The explorer url for the given network.
        Or None if the network is not in the network map.
    """
    explorer_urls: Optional[Dict[str, str]] = {}
    for entity_nm, entity_network_map in network_map.items():
        if network in entity_network_map:
            explorer_urls[entity_nm] = entity_network_map[network]

    return explorer_urls


def get_explorer_url_for_network(
    network: str, block_hash: str, network_map: Dict[str, str]
) -> Optional[List[str]]:
    r"""
    Returns the explorer url for the given block hash and network.

    Args:
        network(str): The network to get the explorer url for.
        block_hash(str): The block hash to get the explorer url for.
        network_map(Dict[str, Dict[str, str]]): The network maps to get the explorer urls from.

    Returns:
        The explorer url for the given block hash and network.
        Or None if the network is not known.
    """

    explorer_urls: Optional[Dict[str, str]] = {}
    # Will be None if the network is not known. i.e. not in network_map
    explorer_root_urls: Optional[
        Dict[str, str]
    ] = get_explorer_root_url_by_network_from_map(network, network_map)

    if explorer_root_urls != {}:
        # We are on a known network.
        explorer_opentensor_url = "{root_url}/query/{block_hash}".format(
            root_url=explorer_root_urls.get("opentensor"), block_hash=block_hash
        )
        explorer_taostats_url = "{root_url}/extrinsic/{block_hash}".format(
            root_url=explorer_root_urls.get("taostats"), block_hash=block_hash
        )
        explorer_urls["opentensor"] = explorer_opentensor_url
        explorer_urls["taostats"] = explorer_taostats_url

    return explorer_urls


def ss58_address_to_bytes(ss58_address: str) -> bytes:
    """Converts a ss58 address to a bytes object."""
    account_id_hex: str = scalecodec.ss58_decode(
        ss58_address, bittensor.__ss58_format__
    )
    return bytes.fromhex(account_id_hex)


def U16_NORMALIZED_FLOAT(x: int) -> float:
    return float(x) / float(U16_MAX)


def U64_NORMALIZED_FLOAT(x: int) -> float:
    return float(x) / float(U64_MAX)


def u8_key_to_ss58(u8_key: List[int]) -> str:
    r"""
    Converts a u8-encoded account key to an ss58 address.

    Args:
        u8_key (List[int]): The u8-encoded account key.
    """
    # First byte is length, then 32 bytes of key.
    return scalecodec.ss58_encode(bytes(u8_key).hex(), bittensor.__ss58_format__)


def hash(content, encoding="utf-8"):
    sha3 = hashlib.sha3_256()

    # Update the hash object with the concatenated string
    sha3.update(content.encode(encoding))

    # Produce the hash
    return sha3.hexdigest()
