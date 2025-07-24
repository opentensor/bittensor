import asyncio
from typing import TYPE_CHECKING, Optional

from bittensor.core.settings import NETWORK_EXPLORER_MAP
from bittensor.utils import (
    get_explorer_url_for_network,
    is_valid_bittensor_address_or_public_key,
    unlock_key,
)
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor_wallet import Wallet


async def _do_transfer(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    destination: str,
    amount: Optional[Balance],
    keep_alive: bool = True,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
) -> tuple[bool, str, str]:
    """
    Makes transfer from wallet to destination public key address.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): initialized AsyncSubtensor object used for transfer
        wallet (bittensor_wallet.Wallet): Bittensor wallet object to make transfer from.
        destination (str): Destination public key address (ss58_address or ed25519) of recipient.
        amount (bittensor.utils.balance.Balance): Amount to stake as Bittensor balance.
        keep_alive (bool): If `True`, will keep the existential deposit in the account.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
            `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted.
            If the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        success, block hash, formatted error message
    """
    call_params = {"dest": destination}
    if amount is None:
        call_function = "transfer_all"
        if keep_alive:
            call_params["keep_alive"] = True
        else:
            call_params["keep_alive"] = False
    else:
        call_params["amount"] = amount.rao
        if keep_alive:
            call_function = "transfer_keep_alive"
        else:
            call_function = "transfer_allow_death"

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
    )

    # We only wait here if we expect finalization.
    if not wait_for_finalization and not wait_for_inclusion:
        return True, "", message

    # Otherwise continue with finalization.
    if success:
        block_hash_ = await subtensor.get_block_hash()
        return True, block_hash_, "Success with response."

    return False, "", message


async def transfer_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    dest: str,
    amount: Optional[Balance],
    transfer_all: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    keep_alive: bool = True,
    period: Optional[int] = None,
) -> bool:
    """Transfers funds from this wallet to the destination public key address.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): initialized AsyncSubtensor object used for transfer
        wallet (bittensor_wallet.Wallet): Bittensor wallet object to make transfer from.
        dest (str): Destination public key address (ss58_address or ed25519) of recipient.
        amount (Optional[bittensor.utils.balance.Balance]): Amount to stake as Bittensor balance. `None` if
            transferring all.
        transfer_all (bool): Whether to transfer all funds from this wallet to the destination address.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
            `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        keep_alive (bool): If set, keeps the account alive by keeping the balance above the existential deposit.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted.
            If the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        success (bool): Flag is `True` if extrinsic was finalized or included in the block. If we did not wait for
            finalization / inclusion, the response is `True`, regardless of its inclusion.
    """
    destination = dest

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
        wallet=wallet, dest=destination, value=amount
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
    success, block_hash, err_msg = await _do_transfer(
        subtensor=subtensor,
        wallet=wallet,
        destination=destination,
        amount=amount,
        wait_for_finalization=wait_for_finalization,
        wait_for_inclusion=wait_for_inclusion,
        period=period,
    )

    if success:
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

    logging.error(f":cross_mark: [red]Failed[/red]: {err_msg}")
    return False
