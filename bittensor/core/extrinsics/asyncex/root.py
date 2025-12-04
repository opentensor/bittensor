import asyncio
from typing import Optional, TYPE_CHECKING, Literal

from bittensor.core.chain_data import RootClaimType
from bittensor.core.extrinsics.asyncex.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.pallets import SubtensorModule
from bittensor.core.settings import DEFAULT_MEV_PROTECTION
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor.core.types import UIDs


async def _get_limits(subtensor: "AsyncSubtensor") -> tuple[int, float]:
    """
    Retrieves the minimum allowed weights and maximum weight limit for the given subnet.

    These values are fetched asynchronously using `asyncio.gather` to run both requests concurrently.

    Parameters:
        subtensor: The AsyncSubtensor instance.

    Returns:
        tuple[int, float]:
            - `min_allowed_weights`: The minimum allowed weights.
            - `max_weight_limit`: The maximum weight limit, normalized to a float value.
    """
    # Get weight restrictions.
    maw, mwl = await asyncio.gather(
        subtensor.get_hyperparameter("MinAllowedWeights", netuid=0),
        subtensor.get_hyperparameter("MaxWeightsLimit", netuid=0),
    )
    min_allowed_weights = int(maw)
    max_weight_limit = u16_normalized_float(int(mwl))
    return min_allowed_weights, max_weight_limit


async def root_register_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Registers the neuron to the root network.

    Parameters:
        subtensor: Subtensor instance to interact with the blockchain.
        wallet: Bittensor Wallet instance.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type="both"
            )
        ).success:
            return unlocked

        netuid = 0
        logging.debug(
            f"Registering on netuid [blue]{netuid}[/blue] on network: [blue]{subtensor.network}[/blue]."
        )

        block_hash = await subtensor.get_block_hash()
        recycle_call, balance = await asyncio.gather(
            subtensor.get_hyperparameter(
                param_name="Burn",
                netuid=netuid,
                block_hash=block_hash,
            ),
            subtensor.get_balance(
                wallet.coldkeypub.ss58_address,
                block_hash=block_hash,
            ),
        )

        current_recycle = Balance.from_rao(int(recycle_call))

        if balance < current_recycle:
            return ExtrinsicResponse(
                False,
                f"Insufficient balance {balance} to register neuron. Current recycle is {current_recycle} TAO.",
            ).with_log()

        logging.debug(
            f"Checking if hotkey ([blue]{wallet.hotkey.ss58_address}[/blue]) is registered on root."
        )
        is_registered = await subtensor.is_hotkey_registered(
            netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
        )
        if is_registered:
            return ExtrinsicResponse(message="Already registered on root network.")

        call = await SubtensorModule(subtensor).root_register(
            hotkey=wallet.hotkey.ss58_address
        )

        if mev_protection:
            response = await submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            response = await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )

        if not response.success:
            logging.error(f"[red]{response.message}[/red]")
            await asyncio.sleep(0.5)
            return response

        # Successful registration, final check for neuron and pubkey
        uid = await subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=wallet.hotkey.ss58_address, netuid=netuid
        )
        if uid is not None:
            response.data = {
                "hotkey_ss58": wallet.hotkey.ss58_address,
                "netuid": netuid,
                "uid": uid,
            }
            logging.debug(
                f"Hotkey {wallet.hotkey.ss58_address} registered in subnet {netuid} with UID: {uid}."
            )
            return response

        # neuron not found, try again
        return ExtrinsicResponse(False, "Unknown error. Neuron not found.").with_log()

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


async def set_root_claim_type_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    new_root_claim_type: "Literal['Swap', 'Keep'] | RootClaimType | dict",
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """Sets the root claim type for the coldkey in provided wallet.

    Parameters:
        subtensor: Subtensor instance to interact with the blockchain.
        wallet: Bittensor Wallet instance.
        new_root_claim_type: The new root claim type to set. Can be:
            - String: "Swap" or "Keep"
            - RootClaimType: RootClaimType.Swap, RootClaimType.Keep
            - Dict: {"KeepSubnets": {"subnets": [1, 2, 3]}}
            - Callable: RootClaimType.KeepSubnets([1, 2, 3])
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        normalized_type = RootClaimType.normalize(new_root_claim_type)

        call = await SubtensorModule(subtensor).set_root_claim_type(
            new_root_claim_type=normalized_type
        )

        if mev_protection:
            return await submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            return await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


async def claim_root_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuids: "UIDs",
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """Claims the root emissions for a coldkey.

    Parameters:
        subtensor: Subtensor instance to interact with the blockchain.
        wallet: Bittensor Wallet instance.
        netuids: The netuids to claim root emissions for.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        call = await SubtensorModule(subtensor).claim_root(subnets=netuids)

        if mev_protection:
            return await submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            return await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
