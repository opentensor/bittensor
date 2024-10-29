import asyncio
from typing import TYPE_CHECKING
from bittensor_wallet import Wallet
from bittensor_wallet.errors import KeyFileError
from rich.prompt import Confirm
from substrateinterface.exceptions import SubstrateRequestException

from bittensor.core.settings import NETWORK_EXPLORER_MAP, bt_console as console, bt_err_console as err_console, print_verbose, print_error
from bittensor.utils.balance import Balance

from bittensor.utils import (
    format_error_message,
    get_explorer_url_for_network,
    is_valid_bittensor_address_or_public_key,
)

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor


async def transfer_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    destination: str,
    amount: Balance,
    transfer_all: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    keep_alive: bool = True,
    prompt: bool = False,
) -> bool:
    """Transfers funds from this wallet to the destination public key address.

    :param subtensor: initialized AsyncSubtensor object used for transfer
    :param wallet: Bittensor wallet object to make transfer from.
    :param destination: Destination public key address (ss58_address or ed25519) of recipient.
    :param amount: Amount to stake as Bittensor balance.
    :param transfer_all: Whether to transfer all funds from this wallet to the destination address.
    :param wait_for_inclusion: If set, waits for the extrinsic to enter a block before returning `True`,
                               or returns `False` if the extrinsic fails to enter the block within the timeout.
    :param wait_for_finalization:  If set, waits for the extrinsic to be finalized on the chain before returning
                                   `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
    :param keep_alive: If set, keeps the account alive by keeping the balance above the existential deposit.
    :param prompt: If `True`, the call waits for confirmation from the user before proceeding.
    :return: success: Flag is `True` if extrinsic was finalized or included in the block. If we did not wait for
                      finalization / inclusion, the response is `True`, regardless of its inclusion.
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
            err_console.print(
                f":cross_mark: [red]Failed to get payment info[/red]:[bold white]\n"
                f"  {format_error_message(e, subtensor.substrate)}[/bold white]\n"
                f"  Defaulting to default transfer fee: {payment_info['partialFee']}"
            )

        return Balance.from_rao(payment_info["partialFee"])

    async def do_transfer() -> tuple[bool, str, str]:
        """
        Makes transfer from wallet to destination public key address.
        :return: success, block hash, formatted error message
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
            return False, "", format_error_message(await response.error_message)

    # Validate destination address.
    if not is_valid_bittensor_address_or_public_key(destination):
        err_console.print(
            f":cross_mark: [red]Invalid destination SS58 address[/red]:[bold white]\n  {destination}[/bold white]"
        )
        return False
    console.print(f"[dark_orange]Initiating transfer on network: {subtensor.network}")
    # Unlock wallet coldkey.
    try:
        wallet.unlock_coldkey()
    except KeyFileError:
        err_console.print("Error decrypting coldkey (possibly incorrect password)")
        return False

    # Check balance.
    with console.status(
        f":satellite: Checking balance and fees on chain [white]{subtensor.network}[/white]",
        spinner="aesthetic",
    ) as status:
        # check existential deposit and fee
        print_verbose("Fetching existential and fee", status)
        block_hash = await subtensor.substrate.get_chain_head()
        account_balance_, existential_deposit = await asyncio.gather(
            subtensor.get_balance(
                wallet.coldkeypub.ss58_address, block_hash=block_hash
            ),
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
            print_error("Not enough balance to transfer")
            return False

    if account_balance < (amount + fee + existential_deposit):
        err_console.print(
            ":cross_mark: [bold red]Not enough balance[/bold red]:\n\n"
            f"  balance: [bright_cyan]{account_balance}[/bright_cyan]\n"
            f"  amount: [bright_cyan]{amount}[/bright_cyan]\n"
            f"  for fee: [bright_cyan]{fee}[/bright_cyan]"
        )
        return False

    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            "Do you want to transfer:[bold white]\n"
            f"  amount: [bright_cyan]{amount}[/bright_cyan]\n"
            f"  from: [light_goldenrod2]{wallet.name}[/light_goldenrod2] : [bright_magenta]{wallet.coldkey.ss58_address}\n[/bright_magenta]"
            f"  to: [bright_magenta]{destination}[/bright_magenta]\n  for fee: [bright_cyan]{fee}[/bright_cyan]"
        ):
            return False

    with console.status(":satellite: Transferring...", spinner="earth") as status:
        success, block_hash, err_msg = await do_transfer()

        if success:
            console.print(":white_heavy_check_mark: [green]Finalized[/green]")
            console.print(f"[green]Block Hash: {block_hash}[/green]")

            if subtensor.network == "finney":
                print_verbose("Fetching explorer URLs", status)
                explorer_urls = get_explorer_url_for_network(
                    subtensor.network, block_hash, NETWORK_EXPLORER_MAP
                )
                if explorer_urls != {} and explorer_urls:
                    console.print(
                        f"[green]Opentensor Explorer Link: {explorer_urls.get('opentensor')}[/green]"
                    )
                    console.print(
                        f"[green]Taostats Explorer Link: {explorer_urls.get('taostats')}[/green]"
                    )
        else:
            console.print(f":cross_mark: [red]Failed[/red]: {err_msg}")

    if success:
        with console.status(":satellite: Checking Balance...", spinner="aesthetic"):
            new_balance = await subtensor.get_balance(
                wallet.coldkeypub.ss58_address, reuse_block=False
            )
            console.print(
                f"Balance:\n"
                f"  [blue]{account_balance}[/blue] :arrow_right: [green]{new_balance[wallet.coldkey.ss58_address]}[/green]"
            )
            return True

    return False
