"""Module with helper functions for extrinsics."""

from typing import TYPE_CHECKING

from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from scalecodec import GenericCall
    from bittensor_wallet import Wallet, Keypair
    from bittensor.core.chain_data import StakeInfo
    from bittensor.core.subtensor import Subtensor


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


def get_unstaking_fee(
    subtensor: "Subtensor", netuid: int, call: "GenericCall", keypair: "Keypair"
):
    """
    Get unstaking fee for a given extrinsic call and keypair for a given SN's netuid.

    Arguments:
        subtensor: The Subtensor instance.
        netuid: The SN's netuid.
        call: The extrinsic call.
        keypair: The keypair associated with the extrinsic.

    Returns:
        Balance object representing the unstaking fee in RAO.
    """
    payment_info = subtensor.substrate.get_payment_info(call=call, keypair=keypair)
    return Balance.from_rao(amount=payment_info["partial_fee"]).set_unit(netuid=netuid)
