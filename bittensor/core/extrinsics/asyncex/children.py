from typing import TYPE_CHECKING, Optional

from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import float_to_u64, unlock_key, get_function_name

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor


async def set_children_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    hotkey: str,
    netuid: int,
    children: list[tuple[float, str]],
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Allows a coldkey to set children-keys.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: bittensor wallet instance.
        hotkey: The ``SS58`` address of the neuron's hotkey.
        netuid: The netuid value.
        children: A list of children with their proportions.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Waits for the transaction to be included in a block.
        wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Raises:
        DuplicateChild: There are duplicates in the list of children.
        InvalidChild: Child is the hotkey.
        NonAssociatedColdKey: The coldkey does not own the hotkey or the child is the same as the hotkey.
        NotEnoughStakeToSetChildkeys: Parent key doesn't have minimum own stake.
        ProportionOverflow: The sum of the proportions does exceed uint64.
        RegistrationNotPermittedOnRootSubnet: Attempting to register a child on the root network.
        SubNetworkDoesNotExist: Attempting to register to a non-existent network.
        TooManyChildren: Too many children in request.
        TxRateLimitExceeded: Hotkey hit the rate limit.
        bittensor_wallet.errors.KeyFileError: Failed to decode keyfile data.
        bittensor_wallet.errors.PasswordError: Decryption failed or wrong password for decryption provided.
    """
    unlock = unlock_key(wallet, raise_error=raise_error)

    if not unlock.success:
        return ExtrinsicResponse(
            False, unlock.message, extrinsic_function=get_function_name()
        )

    async with subtensor.substrate as substrate:
        call = await substrate.compose_call(
            call_module="SubtensorModule",
            call_function="set_children",
            call_params={
                "children": [
                    (
                        float_to_u64(proportion),
                        child_hotkey,
                    )
                    for proportion, child_hotkey in children
                ],
                "hotkey": hotkey,
                "netuid": netuid,
            },
        )

        response = await subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            calling_function=get_function_name(),
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return response

        if response.success:
            response.message = f"Success with {response.extrinsic_function} response."
            return response

        return response


async def root_set_pending_childkey_cooldown_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    cooldown: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Allows a root coldkey to set children-keys.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked).
        cooldown: The cooldown period in blocks.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Waits for the transaction to be included in a block.
        wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    if not (unlock := unlock_key(wallet)).success:
        return ExtrinsicResponse(
            False, unlock.message, extrinsic_function=get_function_name()
        )

    async with subtensor.substrate as substrate:
        call = await substrate.compose_call(
            call_module="SubtensorModule",
            call_function="set_pending_childkey_cooldown",
            call_params={"cooldown": cooldown},
        )

        sudo_call = await substrate.compose_call(
            call_module="Sudo",
            call_function="sudo",
            call_params={"call": call},
        )

        response = await subtensor.sign_and_send_extrinsic(
            call=sudo_call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            calling_function=get_function_name(),
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return response

        if response.success:
            response.message = f"Success with {response.extrinsic_function} response."
            return response

        return response
