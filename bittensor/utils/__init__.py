# The MIT License (MIT)
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc
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

import asyncio
import hashlib
from typing import Callable, List, Dict, Literal, Tuple

import numpy as np
import scalecodec

import bittensor
from .registration import torch, use_torch
from .version import version_checking, check_version, VersionCheckError
from .wallet_utils import *  # noqa F401

RAOPERTAO = 1e9
U16_MAX = 65535
U64_MAX = 18446744073709551615


def ss58_to_vec_u8(ss58_address: str) -> List[int]:
    """
    Converts an SS58 address to a list of integers (vector of u8).

    Args:
        ss58_address (str): The SS58 address to be converted.

    Returns:
        List[int]: A list of integers representing the byte values of the SS58 address.
    """
    ss58_bytes: bytes = ss58_address_to_bytes(ss58_address)
    encoded_address: List[int] = [int(byte) for byte in ss58_bytes]
    return encoded_address


def _unbiased_topk(
    values: Union[np.ndarray, "torch.Tensor"],
    k: int,
    dim=0,
    sorted_=True,
    largest=True,
    axis=0,
    return_type: str = "numpy",
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple["torch.Tensor", "torch.LongTensor"]]:
    """Selects topk as in torch.topk but does not bias lower indices when values are equal.

    Args:
        values: (np.ndarray) if using numpy, (torch.Tensor) if using torch: Values to index into.
        k: (int): Number to take.
        dim: (int): Dimension to index into (used by Torch)
        sorted_: (bool): Whether to sort indices.
        largest: (bool): Whether to take the largest value.
        axis: (int): Axis along which to index into (used by Numpy)
        return_type: (str): Whether or use torch or numpy approach

    Return:
        topk: (np.ndarray) if using numpy, (torch.Tensor) if using torch: topk k values.
        indices: (np.ndarray) if using numpy, (torch.LongTensor) if using torch: indices of the topk values.
    """
    if return_type == "torch":
        permutation = torch.randperm(values.shape[dim])
        permuted_values = values[permutation]
        topk, indices = torch.topk(
            permuted_values, k, dim=dim, sorted=sorted_, largest=largest
        )
        return topk, permutation[indices]
    else:
        if dim != 0 and axis == 0:
            # Ensures a seamless transition for calls made to this function that specified args by keyword
            axis = dim

        permutation = np.random.permutation(values.shape[axis])
        permuted_values = np.take(values, permutation, axis=axis)
        indices = np.argpartition(permuted_values, -k, axis=axis)[-k:]
        if not sorted_:
            indices = np.sort(indices, axis=axis)
        if not largest:
            indices = indices[::-1]
        topk = np.take(permuted_values, indices, axis=axis)
        return topk, permutation[indices]


def unbiased_topk(
    values: Union[np.ndarray, "torch.Tensor"],
    k: int,
    dim: int = 0,
    sorted_: bool = True,
    largest: bool = True,
    axis: int = 0,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple["torch.Tensor", "torch.LongTensor"]]:
    """Selects topk as in torch.topk but does not bias lower indices when values are equal.

    Args:
        values: (np.ndarray) if using numpy, (torch.Tensor) if using torch: Values to index into.
        k: (int): Number to take.
        dim: (int): Dimension to index into (used by Torch)
        sorted_: (bool):  Whether to sort indices.
        largest: (bool): Whether to take the largest value.
        axis: (int): Axis along which to index into (used by Numpy)

    Return:
        topk: (np.ndarray) if using numpy, (torch.Tensor) if using torch: topk k values.
        indices: (np.ndarray) if using numpy, (torch.LongTensor) if using torch: indices of the topk values.
    """
    if use_torch():
        return _unbiased_topk(
            values, k, dim, sorted_, largest, axis, return_type="torch"
        )
    else:
        return _unbiased_topk(
            values, k, dim, sorted_, largest, axis, return_type="numpy"
        )


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
    """
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
    network: str, block_hash: str, network_map: Dict[str, Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """
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
    explorer_root_urls: Optional[Dict[str, str]] = (
        get_explorer_root_url_by_network_from_map(network, network_map)
    )

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


def u16_normalized_float(x: int) -> float:
    return float(x) / float(U16_MAX)


def u64_normalized_float(x: int) -> float:
    return float(x) / float(U64_MAX)


def u8_key_to_ss58(u8_key: List[int]) -> str:
    """
    Converts a u8-encoded account key to an ss58 address.

    Args:
        u8_key (List[int]): The u8-encoded account key.
    """
    # First byte is length, then 32 bytes of key.
    return scalecodec.ss58_encode(bytes(u8_key).hex(), bittensor.__ss58_format__)


def get_hash(content, encoding="utf-8"):
    """
    Generates a SHA3-256 hash for the given content.

    Args:
        content (str): The content to be hashed.
        encoding (str, optional): The encoding used to convert the content to bytes. Default is ``utf-8``.

    Returns:
        str: The hexadecimal representation of the SHA3-256 hash of the content.
    """
    sha3 = hashlib.sha3_256()

    # Update the hash object with the concatenated string
    sha3.update(content.encode(encoding))

    # Produce the hash
    return sha3.hexdigest()


def format_error_message(error_message: dict) -> str:
    """
    Formats an error message from the Subtensor error information to using in extrinsics.

    Args:
        error_message (dict): A dictionary containing the error information from Subtensor.

    Returns:
        str: A formatted error message string.
    """
    err_type = "UnknownType"
    err_name = "UnknownError"
    err_description = "Unknown Description"

    if isinstance(error_message, dict):
        err_type = error_message.get("type", err_type)
        err_name = error_message.get("name", err_name)
        err_docs = error_message.get("docs", [])
        err_description = err_docs[0] if len(err_docs) > 0 else err_description
    return f"Subtensor returned `{err_name} ({err_type})` error. This means: `{err_description}`"


def get_async_result(cor_func, *args, **kwargs):
    """
    Executes an asynchronous function and returns its result.

    This function creates a new coroutine for the given asynchronous function to avoid the `cannot reuse already awaited coroutine` error, and runs it using `asyncio.run`.

    Args:
        cor_func (coroutine): The asynchronous function to be executed.
        *args: Positional arguments to pass to the asynchronous function.
        **kwargs: Keyword arguments to pass to the asynchronous function.

    Returns:
        The result of the asynchronous function.

        Examples:
        >>> async def sample_coroutine(x, y):
        ...     # some async staff here
        ...     return x + y
        ...
        >>> result = get_async_result(sample_coroutine, 2, 3)
        >>> print(result)
        5
        >>> async def fetch_data(uri):
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get(uri) as response:
        ...             return await response.text()
        ...
        >>> url = "https://www.example.com"
        >>> html_content = get_async_result(fetch_data, url)
        >>> print(html_content)

        >>> async def main():
        ...     await asyncio.sleep(2)
        ...     return "Finished"
        ...
        >>> result = get_async_result(main)
        >>> print(result)
        "Finished"
    """

    # To avoid the `cannot reuse already awaited coroutine` error, I need to create a new coroutine every time.
    def create_coroutine(cor_func_, *args_, **kwargs_):
        return cor_func_(*args_, **kwargs_)

    return asyncio.run(create_coroutine(cor_func, *args, **kwargs))
