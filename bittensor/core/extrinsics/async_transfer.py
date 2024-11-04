import asyncio
from typing import TYPE_CHECKING

from bittensor_wallet import Wallet
from bittensor_wallet.errors import KeyFileError
from substrateinterface.exceptions import SubstrateRequestException

from bittensor.core.settings import NETWORK_EXPLORER_MAP
from bittensor.utils import (
    format_error_message,
    get_explorer_url_for_network,
    is_valid_bittensor_address_or_public_key,
)
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor


async def transfer_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    destination: str,
    amount: "Balance",
    transfer_all: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    keep_alive: bool = True,
) -> bool:
    """Transfers funds from this wallet to the destination public key address.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): initialized AsyncSubtensor object used for transfer
        wallet (bittensor_wallet.Wallet): Bittensor wallet object to make transfer from.
        destination (str): Destination public key address (ss58_address or ed25519) of recipient.
        amount (bittensor.utils.balance.Balance): Amount to stake as Bittensor balance.
        transfer_all (bool): Whether to transfer all funds from this wallet to the destination address.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):  If set, waits for the extrinsic to be finalized on the chain before returning `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        keep_alive (bool): If set, keeps the account alive by keeping the balance above the existential deposit.
    
    Returns:
        success (bool): Flag is `True` if extrinsic was finalized or included in the block. If we did not wait for finalization / inclusion, the response is `True`, regardless of its inclusion.
    """

    async def get_transfer_fee() -> Balance:
        """
        Calculates the transaction fee for transferring tokens from a wallet to a specified destination address.
        This function simulates the transfer to estimate the associated cost, taking into account the current
        network conditions and transaction complexity.
        """
        call = await subtensor.substrate.compose_call(
            call_module="Balances",
            call_function="transfer_allow_death",
            call_params={"dest": destination, "value": amount.rao},
        )

        try:
            payment_info = await subtensor.substrate.get_payment_info(
                call=call, keypair=wallet.coldkeypub
            )
        except SubstrateRequestException as e:
            payment_info = {"partialFee": int(2e7)}  # assume  0.02 Tao
            logging.error(f":cross_mark: <red>Failed to get payment info</red>:")
            logging.error(f"\t\t{format_error_message(e, subtensor.substrate)}")
            logging.error(
                f"\t\tDefaulting to default transfer fee: {payment_info['partialFee']}"
            )

        return Balance.from_rao(payment_info["partialFee"])

    async def do_transfer() -> tuple[bool, str, str]:
        """
        Makes transfer from wallet to destination public key address.

        Returns:
            success, block hash, formatted error message
        """
        call = await subtensor.substrate.compose_call(
            call_module="Balances",
            call_function="transfer_allow_death",
            call_params={"dest": destination, "value": amount.rao},
        )
        extrinsic = await subtensor.substrate.create_signed_extrinsic(
            call=call, keypair=wallet.coldkey
        )
        response = await subtensor.substrate.submit_extrinsic(
            extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        # We only wait here if we expect finalization.
        if not wait_for_finalization and not wait_for_inclusion:
            return True, "", ""

        # Otherwise continue with finalization.
        await response.process_events()
        if await response.is_success:
            block_hash_ = response.block_hash
            return True, block_hash_, ""
        else:
            return (
                False,
                "",
                format_error_message(
                    await response.error_message, substrate=subtensor.substrate
                ),
            )

    # Validate destination address.
    if not is_valid_bittensor_address_or_public_key(destination):
        logging.error(
            f":cross_mark: <red>Invalid destination SS58 address</red>:[bold white]\n  {destination}[/bold white]"
        )
        return False
    logging.info(f"Initiating transfer on network: {subtensor.network}")
    # Unlock wallet coldkey.
    try:
        wallet.unlock_coldkey()
    except KeyFileError:
        logging.error("Error decrypting coldkey (possibly incorrect password)")
        return False

    # Check balance.
    logging.info(
        f":satellite: <magenta>Checking balance and fees on chain </magenta> <blue>{subtensor.network}</blue>"
    )
    # check existential deposit and fee
    logging.debug("Fetching existential and fee")
    block_hash = await subtensor.substrate.get_chain_head()
    account_balance_, existential_deposit = await asyncio.gather(
        subtensor.get_balance(wallet.coldkeypub.ss58_address, block_hash=block_hash),
        subtensor.get_existential_deposit(block_hash=block_hash),
    )
    account_balance = account_balance_[wallet.coldkeypub.ss58_address]
    fee = await get_transfer_fee()

    if not keep_alive:
        # Check if the transfer should keep_alive the account
        existential_deposit = Balance(0)

    # Check if we have enough balance.
    if transfer_all is True:
        amount = account_balance - fee - existential_deposit
        if amount < Balance(0):
            logging.error("Not enough balance to transfer")
            return False

    if account_balance < (amount + fee + existential_deposit):
        logging.error(":cross_mark: <red>Not enough balance</red>")
        logging.error(f"\t\tBalance:\t<blue>{account_balance}</blue>")
        logging.error(f"\t\tAmount:\t<blue>{amount}</blue>")
        logging.error(f"\t\tFor fee:\t<blue>{fee}</blue>")
        return False

    logging.info(":satellite: <magenta>Transferring...</magenta")
    success, block_hash, err_msg = await do_transfer()

    if success:
        logging.success(":white_heavy_check_mark: [green]Finalized</green>")
        logging.info(f"[green]Block Hash:</green> <blue>{block_hash}</blue>")

        if subtensor.network == "finney":
            logging.debug("Fetching explorer URLs")
            explorer_urls = get_explorer_url_for_network(
                subtensor.network, block_hash, NETWORK_EXPLORER_MAP
            )
            if explorer_urls != {} and explorer_urls:
                logging.info(
                    f"[green]Opentensor Explorer Link: {explorer_urls.get('opentensor')}</green>"
                )
                logging.info(
                    f"[green]Taostats Explorer Link: {explorer_urls.get('taostats')}</green>"
                )
    else:
        logging.error(f":cross_mark: <red>Failed</red>: {err_msg}")

    if success:
        logging.info(":satellite: <magenta>Checking Balance...<magenta>")
        new_balance = await subtensor.get_balance(
            wallet.coldkeypub.ss58_address, reuse_block=False
        )
        logging.info(
            f"Balance: [blue]{account_balance}</blue> :arrow_right: [green]{new_balance[wallet.coldkeypub.ss58_address]}</green>"
        )
        return True

    return False
