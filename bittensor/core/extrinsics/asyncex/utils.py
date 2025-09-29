from typing import TYPE_CHECKING, Optional

from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from scalecodec import GenericCall
    from bittensor_wallet import Keypair
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor_wallet import Wallet


async def get_extrinsic_fee(
    subtensor: "AsyncSubtensor",
    call: "GenericCall",
    keypair: "Keypair",
    netuid: Optional[int] = None,
):
    """
    Get extrinsic fee for a given extrinsic call and keypair for a given SN's netuid.

    Parameters:
        subtensor: The Subtensor instance.
        netuid: The SN's netuid.
        call: The extrinsic call.
        keypair: The keypair associated with the extrinsic.

    Returns:
        Balance object representing the extrinsic fee in RAO.
    """
    payment_info = await subtensor.substrate.get_payment_info(
        call=call, keypair=keypair
    )
    return Balance.from_rao(amount=payment_info["partial_fee"]).set_unit(
        netuid=netuid or 0
    )


async def sudo_call_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    call_function: str,
    call_params: dict,
    call_module: str = "AdminUtils",
    sign_with: str = "coldkey",
    use_nonce: bool = False,
    nonce_key: str = "hotkey",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Execute a sudo call extrinsic.

    Parameters:
        subtensor: AsyncSubtensor instance.
        wallet: The wallet instance.
        call_function: The call function to execute.
        call_params: The call parameters.
        call_module: The call module.
        sign_with: The keypair to sign the extrinsic with.
        use_nonce: Whether to use a nonce.
        nonce_key: The key to use for the nonce.
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
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type=sign_with
            )
        ).success:
            return unlocked

        sudo_call = await subtensor.substrate.compose_call(
            call_module="Sudo",
            call_function="sudo",
            call_params={
                "call": await subtensor.substrate.compose_call(
                    call_module=call_module,
                    call_function=call_function,
                    call_params=call_params,
                )
            },
        )
        return await subtensor.sign_and_send_extrinsic(
            call=sudo_call,
            wallet=wallet,
            sign_with=sign_with,
            use_nonce=use_nonce,
            nonce_key=nonce_key,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
