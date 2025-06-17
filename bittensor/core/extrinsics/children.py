from typing import TYPE_CHECKING, Optional
from bittensor.utils import float_to_u64, unlock_key

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def set_children_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey: str,
    netuid: int,
    children: list[tuple[float, str]],
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    raise_error: bool = False,
    period: Optional[int] = None,
):
    """
    Allows a coldkey to set children-keys.

    Arguments:
        subtensor: bittensor subtensor.
        wallet: bittensor wallet instance.
        hotkey: The ``SS58`` address of the neuron's hotkey.
        netuid: The netuid value.
        children: A list of children with their proportions.
        wait_for_inclusion: Waits for the transaction to be included in a block.
        wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.

    Returns:
        tuple[bool, str]: A tuple where the first element is a boolean indicating success or failure of the operation,
            and the second element is a message providing additional information.

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
        return False, unlock.message

    call = subtensor.substrate.compose_call(
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

    success, message = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        raise_error=raise_error,
        period=period,
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return True, message

    if success:
        return True, "Success with `set_children_extrinsic` response."

    return True, message


def root_set_pending_childkey_cooldown_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    cooldown: int,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
) -> tuple[bool, str]:
    """
    Allows a coldkey to set children-keys.
    """
    unlock = unlock_key(wallet)

    if not unlock.success:
        return False, unlock.message

    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="set_pending_childkey_cooldown",
        call_params={"cooldown": cooldown},
    )

    sudo_call = subtensor.substrate.compose_call(
        call_module="Sudo",
        call_function="sudo",
        call_params={"call": call},
    )

    success, message = subtensor.sign_and_send_extrinsic(
        call=sudo_call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return True, message

    if success:
        return (
            True,
            "Success with `root_set_pending_childkey_cooldown_extrinsic` response.",
        )

    return True, message
