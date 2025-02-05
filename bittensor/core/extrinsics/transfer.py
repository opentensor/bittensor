from typing import TYPE_CHECKING

from bittensor.core.settings import NETWORK_EXPLORER_MAP
from bittensor.utils.balance import Balance
from bittensor.utils import (
    is_valid_bittensor_address_or_public_key,
    unlock_key,
    get_explorer_url_for_network,
    format_error_message,
)
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def _do_transfer(
    subtensor: "Subtensor",
    wallet: "Wallet",
    destination: str,
    amount: Balance,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> tuple[bool, str, str]:
    """
    Makes transfer from wallet to destination public key address.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): the Subtensor object used for transfer
        wallet (bittensor_wallet.Wallet): Bittensor wallet object to make transfer from.
        destination (str): Destination public key address (ss58_address or ed25519) of recipient.
        amount (bittensor.utils.balance.Balance): Amount to stake as Bittensor balance.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):  If set, waits for the extrinsic to be finalized on the chain before returning
            `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success, block hash, formatted error message
    """
    call = subtensor.substrate.compose_call(
        call_module="Balances",
        call_function="transfer_allow_death",
        call_params={"dest": destination, "value": amount.rao},
    )
    extrinsic = subtensor.substrate.create_signed_extrinsic(
        call=call, keypair=wallet.coldkey
    )
    response = subtensor.substrate.submit_extrinsic(
        extrinsic=extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    # We only wait here if we expect finalization.
    if not wait_for_finalization and not wait_for_inclusion:
        return True, "", "Success, extrinsic submitted without waiting."

    # Otherwise continue with finalization.
    if response.is_success:
        block_hash_ = response.block_hash
        return True, block_hash_, "Success with response."

    return False, "", format_error_message(response.error_message)


def transfer_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    dest: str,
    amount: Balance,
    transfer_all: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    keep_alive: bool = True,
) -> bool:
    """Transfers funds from this wallet to the destination public key address.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): the Subtensor object used for transfer
        wallet (bittensor_wallet.Wallet): Bittensor wallet object to make transfer from.
        dest (str): Destination public key address (ss58_address or ed25519) of recipient.
        amount (bittensor.utils.balance.Balance): Amount to stake as Bittensor balance.
        transfer_all (bool): Whether to transfer all funds from this wallet to the destination address.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):  If set, waits for the extrinsic to be finalized on the chain before returning
            `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        keep_alive (bool): If set, keeps the account alive by keeping the balance above the existential deposit.

    Returns:
        success (bool): Flag is `True` if extrinsic was finalized or included in the block. If we did not wait for
            finalization / inclusion, the response is `True`, regardless of its inclusion.
    """
    destination = dest
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
    block = subtensor.get_current_block()
    account_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address, block=block)
    if not keep_alive:
        # Check if the transfer should keep_alive the account
        existential_deposit = Balance(0)
    else:
        existential_deposit = subtensor.get_existential_deposit(block=block)

    fee = subtensor.get_transfer_fee(wallet=wallet, dest=destination, value=amount)

    # Check if we have enough balance.
    if transfer_all is True:
        amount = account_balance - fee - existential_deposit
        if amount < Balance(0):
            logging.error("Not enough balance to transfer")
            return False

    if account_balance < (amount + fee + existential_deposit):
        logging.error(":cross_mark: [red]Not enough balance[/red]")
        logging.error(f"\t\tBalance:\t[blue]{account_balance}[/blue]")
        logging.error(f"\t\tAmount:\t[blue]{amount}[/blue]")
        logging.error(f"\t\tFor fee:\t[blue]{fee}[/blue]")
        return False

    logging.info(":satellite: [magenta]Transferring...</magenta")
    success, block_hash, err_msg = _do_transfer(
        subtensor=subtensor,
        wallet=wallet,
        destination=destination,
        amount=amount,
        wait_for_finalization=wait_for_finalization,
        wait_for_inclusion=wait_for_inclusion,
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
        new_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        logging.info(
            f"Balance: [blue]{account_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        return True
    else:
        logging.error(f":cross_mark: [red]Failed[/red]: {err_msg}")
        return False
