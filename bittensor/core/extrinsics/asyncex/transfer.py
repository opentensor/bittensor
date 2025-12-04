import asyncio
from typing import TYPE_CHECKING, Optional

from bittensor.core.extrinsics.asyncex.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.pallets import Balances
from bittensor.core.extrinsics.utils import get_transfer_fn_params
from bittensor.core.settings import (
    DEFAULT_MEV_PROTECTION,
    NETWORK_EXPLORER_MAP,
    DEFAULT_NETWORK,
)
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import (
    get_explorer_url_for_network,
    is_valid_bittensor_address_or_public_key,
)
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor_wallet import Wallet


async def transfer_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    destination_ss58: str,
    amount: Optional[Balance],
    keep_alive: bool = True,
    transfer_all: bool = False,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """Transfers funds from this wallet to the destination public key address.

    Parameters:
        subtensor: The Subtensor instance.
        wallet: The wallet to sign the extrinsic.
        destination_ss58: Destination public key address (ss58_address or ed25519) of recipient.
        amount: Amount to stake as Bittensor balance. `None` if transferring all.
        transfer_all: Whether to transfer all funds from this wallet to the destination address.
        keep_alive: If set, keeps the account alive by keeping the balance above the existential deposit.
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

        if amount is None and not transfer_all:
            return ExtrinsicResponse(
                False, "If not transferring all, `amount` must be specified."
            ).with_log()

        # Validate destination address.
        if not is_valid_bittensor_address_or_public_key(destination_ss58):
            return ExtrinsicResponse(
                False, f"Invalid destination SS58 address: {destination_ss58}"
            ).with_log()

        # check existential deposit and fee
        logging.debug("Fetching existential and fee.")
        block_hash = await subtensor.substrate.get_chain_head()
        old_balance, existential_deposit = await asyncio.gather(
            subtensor.get_balance(
                wallet.coldkeypub.ss58_address, block_hash=block_hash
            ),
            subtensor.get_existential_deposit(block_hash=block_hash),
        )

        fee = await subtensor.get_transfer_fee(
            wallet=wallet,
            destination_ss58=destination_ss58,
            amount=amount,
            keep_alive=keep_alive,
        )

        if not keep_alive:
            # Check if the transfer should keep_alive the account
            existential_deposit = Balance(0)

        # Check if we have enough balance.
        if transfer_all:
            if (old_balance - fee) < existential_deposit:
                return ExtrinsicResponse(
                    False, "Not enough balance to transfer all stake."
                ).with_log()

        elif old_balance < (amount + fee + existential_deposit):
            return ExtrinsicResponse(
                False,
                f"Not enough balance for transfer {amount} to {destination_ss58}. "
                f"Account balance is {old_balance}. Transfers fee is {fee}.",
            ).with_log()

        call_function, call_params = get_transfer_fn_params(
            amount, destination_ss58, keep_alive
        )

        call = await getattr(Balances(subtensor), call_function)(**call_params)

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
        response.transaction_tao_fee = fee

        if response.success:
            block_hash = await subtensor.get_block_hash()

            if subtensor.network == DEFAULT_NETWORK:
                logging.debug("Fetching explorer URLs")
                explorer_urls = get_explorer_url_for_network(
                    subtensor.network, block_hash, NETWORK_EXPLORER_MAP
                )
                if explorer_urls:
                    logging.debug(
                        f"[green]Opentensor Explorer Link: {explorer_urls.get('opentensor')}[/green]"
                    )
                    logging.debug(
                        f"[green]Taostats Explorer Link: {explorer_urls.get('taostats')}[/green]"
                    )

            new_balance = await subtensor.get_balance(wallet.coldkeypub.ss58_address)
            logging.debug(
                f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
            )
            response.data = {
                "balance_before": old_balance,
                "balance_after": new_balance,
            }
            return response

        logging.error(f"[red]{response.message}[/red]")
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
