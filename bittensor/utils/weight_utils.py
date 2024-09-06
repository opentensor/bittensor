"""
Conversion for weight between chain representation and np.array or torch.Tensor
"""

# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

import hashlib
import logging
from typing import Tuple, List, Union

import numpy as np
from numpy.typing import NDArray
from scalecodec import ScaleBytes, U16, Vec
from substrateinterface import Keypair

import bittensor
from bittensor.utils.registration import torch, use_torch, legacy_torch_api_compat

U32_MAX = 4294967295
U16_MAX = 65535


@legacy_torch_api_compat
def normalize_max_weight(
    x: Union[NDArray[np.float32], "torch.FloatTensor"], limit: float = 0.1
) -> Union[NDArray[np.float32], "torch.FloatTensor"]:
    r"""Normalizes the tensor x so that sum(x) = 1 and the max value is not greater than the limit.
    Args:
        x (:obj:`np.float32`):
            Tensor to be max_value normalized.
        limit: float:
            Max value after normalization.
    Returns:
        y (:obj:`np.float32`):
            Normalized x tensor.
    """
    epsilon = 1e-7  # For numerical stability after normalization

    weights = x.copy()
    values = np.sort(weights)

    if x.sum() == 0 or x.shape[0] * limit <= 1:
        return np.ones_like(x) / x.shape[0]
    else:
        estimation = values / values.sum()

        if estimation.max() <= limit:
            return weights / weights.sum()

        # Find the cumulative sum and sorted tensor
        cumsum = np.cumsum(estimation, 0)

        # Determine the index of cutoff
        estimation_sum = np.array(
            [(len(values) - i - 1) * estimation[i] for i in range(len(values))]
        )
        n_values = (estimation / (estimation_sum + cumsum + epsilon) < limit).sum()

        # Determine the cutoff based on the index
        cutoff_scale = (limit * cumsum[n_values - 1] - epsilon) / (
            1 - (limit * (len(estimation) - n_values))
        )
        cutoff = cutoff_scale * values.sum()

        # Applying the cutoff
        weights[weights > cutoff] = cutoff

        y = weights / weights.sum()

        return y


def convert_weight_uids_and_vals_to_tensor(
    n: int, uids: List[int], weights: List[int]
) -> Union[NDArray[np.float32], "torch.FloatTensor"]:
    r"""Converts weights and uids from chain representation into a np.array (inverse operation from convert_weights_and_uids_for_emit)
    Args:
        n: int:
            number of neurons on network.
        uids (:obj:`List[int],`):
            Tensor of uids as destinations for passed weights.
        weights (:obj:`List[int],`):
            Tensor of weights.
    Returns:
        row_weights ( np.float32 or torch.FloatTensor ):
            Converted row weights.
    """
    row_weights = (
        torch.zeros([n], dtype=torch.float32)
        if use_torch()
        else np.zeros([n], dtype=np.float32)
    )
    for uid_j, wij in list(zip(uids, weights)):
        row_weights[uid_j] = float(
            wij
        )  # assumes max-upscaled values (w_max = U16_MAX).
    row_sum = row_weights.sum()
    if row_sum > 0:
        row_weights /= row_sum  # normalize
    return row_weights


def convert_root_weight_uids_and_vals_to_tensor(
    n: int, uids: List[int], weights: List[int], subnets: List[int]
) -> Union[NDArray[np.float32], "torch.FloatTensor"]:
    r"""Converts root weights and uids from chain representation into a np.array or torch FloatTensor (inverse operation from convert_weights_and_uids_for_emit)
    Args:
        n: int:
            number of neurons on network.
        uids (:obj:`List[int],`):
            Tensor of uids as destinations for passed weights.
        weights (:obj:`List[int],`):
            Tensor of weights.
        subnets (:obj:`List[int],`):
            list of subnets on the network
    Returns:
        row_weights ( np.float32 ):
            Converted row weights.
    """

    row_weights = (
        torch.zeros([n], dtype=torch.float32)
        if use_torch()
        else np.zeros([n], dtype=np.float32)
    )
    for uid_j, wij in list(zip(uids, weights)):
        if uid_j in subnets:
            index_s = subnets.index(uid_j)
            row_weights[index_s] = float(
                wij
            )  # assumes max-upscaled values (w_max = U16_MAX).
        else:
            logging.warning(
                f"Incorrect Subnet uid {uid_j} in Subnets {subnets}. The subnet is unavailable at the moment."
            )
            continue
    row_sum = row_weights.sum()
    if row_sum > 0:
        row_weights /= row_sum  # normalize
    return row_weights


def convert_bond_uids_and_vals_to_tensor(
    n: int, uids: List[int], bonds: List[int]
) -> Union[NDArray[np.int64], "torch.LongTensor"]:
    r"""Converts bond and uids from chain representation into a np.array.
    Args:
        n: int:
            number of neurons on network.
        uids (:obj:`List[int],`):
            Tensor of uids as destinations for passed bonds.
        bonds (:obj:`List[int],`):
            Tensor of bonds.
    Returns:
        row_bonds ( np.float32 ):
            Converted row bonds.
    """
    row_bonds = (
        torch.zeros([n], dtype=torch.int64)
        if use_torch()
        else np.zeros([n], dtype=np.int64)
    )
    for uid_j, bij in list(zip(uids, bonds)):
        row_bonds[uid_j] = int(bij)
    return row_bonds


def convert_weights_and_uids_for_emit(
    uids: Union[NDArray[np.int64], "torch.LongTensor"],
    weights: Union[NDArray[np.float32], "torch.FloatTensor"],
) -> Tuple[List[int], List[int]]:
    r"""Converts weights into integer u32 representation that sum to MAX_INT_WEIGHT.
    Args:
        uids (:obj:`np.int64,`):
            Tensor of uids as destinations for passed weights.
        weights (:obj:`np.float32,`):
            Tensor of weights.
    Returns:
        weight_uids (List[int]):
            Uids as a list.
        weight_vals (List[int]):
            Weights as a list.
    """
    # Checks.
    weights = weights.tolist()
    uids = uids.tolist()
    if min(weights) < 0:
        raise ValueError(
            "Passed weight is negative cannot exist on chain {}".format(weights)
        )
    if min(uids) < 0:
        raise ValueError("Passed uid is negative cannot exist on chain {}".format(uids))
    if len(uids) != len(weights):
        raise ValueError(
            "Passed weights and uids must have the same length, got {} and {}".format(
                len(uids), len(weights)
            )
        )
    if sum(weights) == 0:
        return [], []  # Nothing to set on chain.
    else:
        max_weight = float(max(weights))
        weights = [
            float(value) / max_weight for value in weights
        ]  # max-upscale values (max_weight = 1).

    weight_vals = []
    weight_uids = []
    for i, (weight_i, uid_i) in enumerate(list(zip(weights, uids))):
        uint16_val = round(
            float(weight_i) * int(U16_MAX)
        )  # convert to int representation.

        # Filter zeros
        if uint16_val != 0:  # Filter zeros
            weight_vals.append(uint16_val)
            weight_uids.append(uid_i)

    return weight_uids, weight_vals


def process_weights_for_netuid(
    uids: Union[NDArray[np.int64], "torch.Tensor"],
    weights: Union[NDArray[np.float32], "torch.Tensor"],
    netuid: int,
    subtensor: "bittensor.subtensor",
    metagraph: "bittensor.metagraph" = None,
    exclude_quantile: int = 0,
) -> Union[
    Tuple["torch.Tensor", "torch.FloatTensor"],
    Tuple[NDArray[np.int64], NDArray[np.float32]],
]:
    bittensor.logging.debug("process_weights_for_netuid()")
    bittensor.logging.debug("weights", *weights)
    bittensor.logging.debug("netuid", netuid)
    bittensor.logging.debug("subtensor", subtensor)
    bittensor.logging.debug("metagraph", metagraph)

    # Get latest metagraph from chain if metagraph is None.
    if metagraph is None:
        metagraph = subtensor.metagraph(netuid)

    # Cast weights to floats.
    if use_torch():
        if not isinstance(weights, torch.FloatTensor):
            weights = weights.type(torch.float32)
    else:
        if not isinstance(weights, np.float32):
            weights = weights.astype(np.float32)

    # Network configuration parameters from an subtensor.
    # These parameters determine the range of acceptable weights for each neuron.
    quantile = exclude_quantile / U16_MAX
    min_allowed_weights = subtensor.min_allowed_weights(netuid=netuid)
    max_weight_limit = subtensor.max_weight_limit(netuid=netuid)
    bittensor.logging.debug("quantile", quantile)
    bittensor.logging.debug("min_allowed_weights", min_allowed_weights)
    bittensor.logging.debug("max_weight_limit", max_weight_limit)

    # Find all non zero weights.
    non_zero_weight_idx = (
        torch.argwhere(weights > 0).squeeze(dim=1)
        if use_torch()
        else np.argwhere(weights > 0).squeeze(axis=1)
    )
    non_zero_weight_uids = uids[non_zero_weight_idx]
    non_zero_weights = weights[non_zero_weight_idx]
    nzw_size = non_zero_weights.numel() if use_torch() else non_zero_weights.size
    if nzw_size == 0 or metagraph.n < min_allowed_weights:
        bittensor.logging.warning("No non-zero weights returning all ones.")
        final_weights = (
            torch.ones((metagraph.n)).to(metagraph.n) / metagraph.n
            if use_torch()
            else np.ones((metagraph.n), dtype=np.int64) / metagraph.n
        )
        bittensor.logging.debug("final_weights", *final_weights)
        final_weights_count = (
            torch.tensor(list(range(len(final_weights))))
            if use_torch()
            else np.arange(len(final_weights))
        )
        return (
            (final_weights_count, final_weights)
            if use_torch()
            else (final_weights_count, final_weights)
        )

    elif nzw_size < min_allowed_weights:
        bittensor.logging.warning(
            "No non-zero weights less then min allowed weight, returning all ones."
        )
        # ( const ): Should this be np.zeros( ( metagraph.n ) ) to reset everyone to build up weight?
        weights = (
            torch.ones((metagraph.n)).to(metagraph.n) * 1e-5
            if use_torch()
            else np.ones((metagraph.n), dtype=np.int64) * 1e-5
        )  # creating minimum even non-zero weights
        weights[non_zero_weight_idx] += non_zero_weights
        bittensor.logging.debug("final_weights", *weights)
        normalized_weights = bittensor.utils.weight_utils.normalize_max_weight(
            x=weights, limit=max_weight_limit
        )
        nw_arange = (
            torch.tensor(list(range(len(normalized_weights))))
            if use_torch()
            else np.arange(len(normalized_weights))
        )
        return nw_arange, normalized_weights

    bittensor.logging.debug("non_zero_weights", *non_zero_weights)

    # Compute the exclude quantile and find the weights in the lowest quantile
    max_exclude = max(0, len(non_zero_weights) - min_allowed_weights) / len(
        non_zero_weights
    )
    exclude_quantile = min([quantile, max_exclude])
    lowest_quantile = (
        non_zero_weights.quantile(exclude_quantile)
        if use_torch()
        else np.quantile(non_zero_weights, exclude_quantile)
    )
    bittensor.logging.debug("max_exclude", max_exclude)
    bittensor.logging.debug("exclude_quantile", exclude_quantile)
    bittensor.logging.debug("lowest_quantile", lowest_quantile)

    # Exclude all weights below the allowed quantile.
    non_zero_weight_uids = non_zero_weight_uids[lowest_quantile <= non_zero_weights]
    non_zero_weights = non_zero_weights[lowest_quantile <= non_zero_weights]
    bittensor.logging.debug("non_zero_weight_uids", *non_zero_weight_uids)
    bittensor.logging.debug("non_zero_weights", *non_zero_weights)

    # Normalize weights and return.
    normalized_weights = bittensor.utils.weight_utils.normalize_max_weight(
        x=non_zero_weights, limit=max_weight_limit
    )
    bittensor.logging.debug("final_weights", *normalized_weights)

    return non_zero_weight_uids, normalized_weights


def generate_weight_hash(
    address: str,
    netuid: int,
    uids: List[int],
    values: List[int],
    version_key: int,
    salt: List[int],
) -> str:
    """
    Generate a valid commit hash from the provided weights.

    Args:
        address (str): The account identifier. Wallet ss58_address.
        netuid (int): The network unique identifier.
        uids (List[int]): The list of UIDs.
        salt (List[int]): The salt to add to hash.
        values (List[int]): The list of weight values.
        version_key (int): The version key.

    Returns:
        str: The generated commit hash.
    """
    # Encode data using SCALE codec
    wallet_address = ScaleBytes(Keypair(ss58_address=address).public_key)
    netuid = ScaleBytes(netuid.to_bytes(2, "little"))

    vec_uids = Vec(data=None, sub_type="U16")
    vec_uids.value = [U16(ScaleBytes(uid.to_bytes(2, "little"))) for uid in uids]
    uids = ScaleBytes(vec_uids.encode().data)

    vec_values = Vec(data=None, sub_type="U16")
    vec_values.value = [
        U16(ScaleBytes(value.to_bytes(2, "little"))) for value in values
    ]
    values = ScaleBytes(vec_values.encode().data)

    version_key = ScaleBytes(version_key.to_bytes(8, "little"))

    vec_salt = Vec(data=None, sub_type="U16")
    vec_salt.value = [U16(ScaleBytes(salts.to_bytes(2, "little"))) for salts in salt]
    salt = ScaleBytes(vec_salt.encode().data)

    data = wallet_address + netuid + uids + values + salt + version_key

    # Generate Blake2b hash of the data tuple
    blake2b_hash = hashlib.blake2b(data.data, digest_size=32)

    # Convert the hash to hex string and add "0x" prefix
    commit_hash = "0x" + blake2b_hash.hexdigest()

    return commit_hash
