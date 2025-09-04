import asyncio
from typing import TYPE_CHECKING, Optional

from bittensor.core.settings import NETWORK_EXPLORER_MAP
from bittensor.utils import (
    get_explorer_url_for_network,
    get_transfer_fn_params,
    is_valid_bittensor_address_or_public_key,
    unlock_key,
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
) -> bool:
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
    if amount is None and not transfer_all:
        logging.error("If not transferring all, `amount` must be specified.")
        return False

    # Validate destination address.
    if not is_valid_bittensor_address_or_public_key(destination):
        logging.error(
            f":cross_mark: [red]Invalid destination SS58 address[/red]: {destination}"
        )
        return False

    logging.info(f"Initiating transfer on network: {subtensor.network}")
    # Unlock wallet coldkey.
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    # Check balance.
    logging.info(
        f":satellite: [magenta]Checking balance and fees on chain [/magenta] [blue]{subtensor.network}[/blue]"
    )
    # check existential deposit and fee
    logging.debug("Fetching existential and fee")
    block_hash = await subtensor.substrate.get_chain_head()
    account_balance, existential_deposit = await asyncio.gather(
        subtensor.get_balance(wallet.coldkeypub.ss58_address, block_hash=block_hash),
        subtensor.get_existential_deposit(block_hash=block_hash),
    )

    fee = await subtensor.get_transfer_fee(
        wallet=wallet, dest=destination, value=amount, keep_alive=keep_alive
    )

    if not keep_alive:
        # Check if the transfer should keep_alive the account
        existential_deposit = Balance(0)

    # Check if we have enough balance.
    if transfer_all is True:
        if (account_balance - fee) < existential_deposit:
            logging.error("Not enough balance to transfer")
            return False
    elif account_balance < (amount + fee + existential_deposit):
        logging.error(":cross_mark: [red]Not enough balance[/red]")
        logging.error(f"\t\tBalance:\t[blue]{account_balance}[/blue]")
        logging.error(f"\t\tAmount:\t[blue]{amount}[/blue]")
        logging.error(f"\t\tFor fee:\t[blue]{fee}[/blue]")
        return False

    logging.info(":satellite: [magenta]Transferring...</magenta")

    call_function, call_params = get_transfer_fn_params(amount, destination, keep_alive)

    call = await subtensor.substrate.compose_call(
        call_module="Balances",
        call_function=call_function,
        call_params=call_params,
    )

    success, message = await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
    )

    if success:
        block_hash = await subtensor.get_block_hash()
        logging.success(":white_heavy_check_mark: [green]Finalized[/green]")
        logging.info(f"[green]Block Hash:[/green] [blue]{block_hash}[/blue]")

        if subtensor.network == "finney":
            logging.debug("Fetching explorer URLs")
            explorer_urls = get_explorer_url_for_network(
                subtensor.network, block_hash, NETWORK_EXPLORER_MAP
            )
            if explorer_urls:
                logging.info(
                    f"[green]Opentensor Explorer Link: {explorer_urls.get('opentensor')}[/green]"
                )
                logging.info(
                    f"[green]Taostats Explorer Link: {explorer_urls.get('taostats')}[/green]"
                )

        logging.info(":satellite: [magenta]Checking Balance...[magenta]")
        new_balance = await subtensor.get_balance(wallet.coldkeypub.ss58_address)
        logging.info(
            f"Balance: [blue]{account_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        return True

    logging.error(f":cross_mark: [red]Failed[/red]: {message}")
    return False
