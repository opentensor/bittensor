import asyncio
from typing import TYPE_CHECKING, Optional

from bittensor.core.extrinsics.params import get_transfer_fn_params
from bittensor.core.settings import NETWORK_EXPLORER_MAP, DEFAULT_NETWORK
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
    destination: str,
    amount: Optional[Balance],
    keep_alive: bool = True,
    transfer_all: bool = False,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Transfers funds from this wallet to the destination public key address.

    Parameters:
        subtensor: The Subtensor instance.
        wallet: The wallet to sign the extrinsic.
        destination: Destination public key address (ss58_address or ed25519) of recipient.
        amount: Amount to stake as Bittensor balance. `None` if transferring all.
        transfer_all: Whether to transfer all funds from this wallet to the destination address.
        keep_alive: If set, keeps the account alive by keeping the balance above the existential deposit.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        bool: True if the subnet registration was successful, False otherwise.
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
        if not is_valid_bittensor_address_or_public_key(destination):
            return ExtrinsicResponse(
                False, f"Invalid destination SS58 address: {destination}"
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
            wallet=wallet, dest=destination, amount=amount, keep_alive=keep_alive
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
                f"Not enough balance for transfer {amount} to {destination}. "
                f"Account balance is {old_balance}. Transfers fee is {fee}.",
            ).with_log()

        call_function, call_params = get_transfer_fn_params(
            amount, destination, keep_alive
        )

        call = await subtensor.compose_call(
            call_module="Balances",
            call_function=call_function,
            call_params=call_params,
        )

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
