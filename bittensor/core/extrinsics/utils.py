"""Module with helper functions for extrinsics."""

from typing import TYPE_CHECKING, Optional

from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
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

    Parameters:
        wallet: The wallet containing the coldkey to compare with stake data.
        hotkey_ss58s: List of hotkey SS58 addresses for which stakes are retrieved.
        netuids: List of network unique identifiers (netuids) corresponding to the hotkeys.
        all_stakes: A collection of all staking information to search through.

    Returns:
        A list of Balances, each representing the stake for a given hotkey and netuid.
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


def sudo_call_extrinsic(
    subtensor: "Subtensor",
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
    root_call: bool = False,
) -> ExtrinsicResponse:
    """Execute a sudo call extrinsic.

    Parameters:
        subtensor: The Subtensor instance.
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
        root_call: False, if the subnet owner makes a call.

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

        call = subtensor.compose_call(
            call_module=call_module,
            call_function=call_function,
            call_params=call_params,
        )
        if not root_call:
            call = subtensor.compose_call(
                call_module="Sudo",
                call_function="sudo",
                call_params={"call": call},
            )

        return subtensor.sign_and_send_extrinsic(
            call=call,
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
