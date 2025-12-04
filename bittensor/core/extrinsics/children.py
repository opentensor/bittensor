from typing import TYPE_CHECKING, Optional

from bittensor.core.extrinsics.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.pallets import SubtensorModule, Sudo
from bittensor.core.settings import DEFAULT_MEV_PROTECTION
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import float_to_u64

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def set_children_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: str,
    netuid: int,
    children: list[tuple[float, str]],
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> "ExtrinsicResponse":
    """
    Allows a coldkey to set children-keys.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: bittensor wallet instance.
        hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
        netuid: The netuid value.
        children: A list of children with their proportions.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Waits for the transaction to be included in a block.
        wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Raises:
        DuplicateChild: There are duplicates in the list of children.
        InvalidChild: Child is the hotkey.
        NonAssociatedColdKey: The coldkey does not own the hotkey or the child is the same as the hotkey.
        NotEnoughStakeToSetChildkeys: Parent key doesn't have minimum own stake.
        ProportionOverflow: The sum of the proportions does exceed uint64.
        RegistrationNotPermittedOnRootSubnet: Attempting to register a child on the root network.
        SubnetNotExists: Attempting to register to a non-existent network.
        TooManyChildren: Too many children in request.
        TxRateLimitExceeded: Hotkey hit the rate limit.
        bittensor_wallet.errors.KeyFileError: Failed to decode keyfile data.
        bittensor_wallet.errors.PasswordError: Decryption failed or wrong password for decryption provided.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        call = SubtensorModule(subtensor).set_children(
            netuid=netuid,
            hotkey=hotkey_ss58,
            children=[
                (float_to_u64(proportion), child_hotkey)
                for proportion, child_hotkey in children
            ],
        )

        if mev_protection:
            response = submit_encrypted_extrinsic(
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
            response = subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def root_set_pending_childkey_cooldown_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    cooldown: int,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Allows a root coldkey to set children-keys.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked).
        cooldown: The cooldown period in blocks.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Waits for the transaction to be included in a block.
        wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        call = SubtensorModule(subtensor).set_pending_childkey_cooldown(
            cooldown=cooldown
        )

        sudo_call = Sudo(subtensor).sudo(call=call)

        if mev_protection:
            response = submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=sudo_call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            response = subtensor.sign_and_send_extrinsic(
                call=sudo_call,
                wallet=wallet,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
