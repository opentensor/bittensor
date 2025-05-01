"""Module with helper functions for extrinsics."""

from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

from bittensor.utils.balance import Balance
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.chain_data import StakeInfo
    from bittensor.utils.registration import torch


def get_old_stakes(
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    netuids: list[int],
    all_stakes: list["StakeInfo"],
) -> list["Balance"]:
    """
    Retrieve the previous staking balances for a wallet's hotkeys across given netuids.

    This function searches through the provided staking data to find the stake amounts for the specified hotkeys and
    netuids associated with the wallet's coldkey. If no match is found for a particular hotkey and netuid combination,
    a default balance of zero is returned.

    Args:
        wallet: The wallet containing the coldkey to compare with stake data.
        hotkey_ss58s: List of hotkey SS58 addresses for which stakes are retrieved.
        netuids: List of network unique identifiers (netuids) corresponding to the hotkeys.
        all_stakes: A collection of all staking information to search through.

    Returns:
        list[Balance]: A list of Balances, each representing the stake for a given hotkey and netuid.
    """
    stake_lookup = {
        (stake.hotkey_ss58, stake.coldkey_ss58, stake.netuid): stake.stake
        for stake in all_stakes
    }
    return [
        stake_lookup.get(
            (hotkey_ss58, wallet.coldkeypub.ss58_address, netuid),
            Balance.from_tao(0),  # Default to 0 balance if no match found
        )
        for hotkey_ss58, netuid in zip(hotkey_ss58s, netuids)
    ]


def convert_and_normalize_weights_and_uids(
        uids: Union[NDArray[np.int64], "torch.LongTensor", list],
        weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
) -> tuple[list[int], list[int]]:
    """Converts weights and uids to numpy arrays if they are not already.

    Arguments:
        uids (Union[NDArray[np.int64], torch.LongTensor, list]): The ``uint64`` uids of destination neurons.
        weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The weights to set. These must be ``float`` s
            and correspond to the passed ``uid`` s.

    Returns:
        weight_uids, weight_vals: Bytes converted weights and uids
    """
    if isinstance(uids, list):
        uids = np.array(uids, dtype=np.int64)
    if isinstance(weights, list):
        weights = np.array(weights, dtype=np.float32)

    # Reformat and normalize.
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids, weights
    )
    return weight_uids, weight_vals
