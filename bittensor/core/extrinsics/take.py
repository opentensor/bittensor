from typing import TYPE_CHECKING, Optional, Literal

from bittensor_wallet.bittensor_wallet import Wallet

from bittensor.core.types import ExtrinsicResponse

if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor


def set_take_extrinsic(
    subtensor: "Subtensor",
    wallet: Wallet,
    hotkey_ss58: str,
    take: int,
    action: Literal["increase_take", "decrease_take"],
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Sets the delegate 'take' percentage for a neuron identified by its hotkey.

    Parameters:
        subtensor: The Subtensor instance.
        wallet: The wallet to sign the extrinsic.
        hotkey_ss58: SS58 address of the hotkey to set take for.
        take: The percentage of rewards that the delegate claims from nominators.
        action: The call function to use to set the take. Can be either "increase_take" or "decrease_take".
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function=action,
            call_params={
                "hotkey": hotkey_ss58,
                "take": take,
            },
        )
        return subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
