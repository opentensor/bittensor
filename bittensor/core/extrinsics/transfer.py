# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Optional, Union, TYPE_CHECKING

from retry import retry

from bittensor.core.extrinsics.utils import submit_extrinsic
from bittensor.core.settings import NETWORK_EXPLORER_MAP
from bittensor.utils import (
    get_explorer_url_for_network,
    format_error_message,
    is_valid_bittensor_address_or_public_key,
)
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.networking import ensure_connected

# For annotation purposes
if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor_wallet import Wallet


# Chain call for `transfer_extrinsic`
@ensure_connected
def do_transfer(
    self: "Subtensor",
    wallet: "Wallet",
    dest: str,
    transfer_balance: "Balance",
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> tuple[bool, Optional[str], Optional[dict]]:
    """Sends a transfer extrinsic to the chain.

    Args:
        self (subtensor.core.subtensor.Subtensor): The Subtensor instance object.
        wallet (bittensor_wallet.Wallet): Wallet object.
        dest (str): Destination public key address.
        transfer_balance (bittensor.utils.balance.Balance): Amount to transfer.
        wait_for_inclusion (bool): If ``true``, waits for inclusion.
        wait_for_finalization (bool): If ``true``, waits for finalization.

    Returns:
        success (bool): ``True`` if transfer was successful.
        block_hash (str): Block hash of the transfer. On success and if wait_for_ finalization/inclusion is ``True``.
        error (dict): Error message from subtensor if transfer failed.
    """

    @retry(delay=1, tries=3, backoff=2, max_delay=4)
    def make_substrate_call_with_retry():
        call = self.substrate.compose_call(
            call_module="Balances",
            call_function="transfer_allow_death",
            call_params={"dest": dest, "value": transfer_balance.rao},
        )
        extrinsic = self.substrate.create_signed_extrinsic(
            call=call, keypair=wallet.coldkey
        )
        response = submit_extrinsic(
            substrate=self.substrate,
            extrinsic=extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        # We only wait here if we expect finalization.
        if not wait_for_finalization and not wait_for_inclusion:
            return True, None, None

        # Otherwise continue with finalization.
        response.process_events()
        if response.is_success:
            block_hash = response.block_hash
            return True, block_hash, None
        else:
            return False, None, response.error_message

    return make_substrate_call_with_retry()


# Community uses this extrinsic directly and via `subtensor.transfer`
def transfer_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    dest: str,
    amount: Union["Balance", float],
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    keep_alive: bool = True,
) -> bool:
    """Transfers funds from this wallet to the destination public key address.

    Args:
        subtensor (subtensor.core.subtensor.Subtensor): The Subtensor instance object.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object to make transfer from.
        dest (str, ss58_address or ed25519): Destination public key address of receiver.
        amount (Union[Balance, int]): Amount to stake as Bittensor balance, or ``float`` interpreted as Tao.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        keep_alive (bool): If set, keeps the account alive by keeping the balance above the existential deposit.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    # Validate destination address.
    if not is_valid_bittensor_address_or_public_key(dest):
        logging.error(f"<red>Invalid destination address: {dest}</red>")
        return False

    if isinstance(dest, bytes):
        # Convert bytes to hex string.
        dest = "0x" + dest.hex()

    # Unlock wallet coldkey.
    wallet.unlock_coldkey()

    # Convert to bittensor.Balance
    if not isinstance(amount, Balance):
        transfer_balance = Balance.from_tao(amount)
    else:
        transfer_balance = amount

    # Check balance.
    logging.info(":satellite: <magenta>Checking Balance...</magenta>")
    account_balance = subtensor.get_balance(wallet.coldkey.ss58_address)
    # check existential deposit.
    existential_deposit = subtensor.get_existential_deposit()

    logging.info(":satellite: <magenta>Transferring...</magenta>")
    fee = subtensor.get_transfer_fee(
        wallet=wallet, dest=dest, value=transfer_balance.rao
    )

    if not keep_alive:
        # Check if the transfer should keep_alive the account
        existential_deposit = Balance(0)

    # Check if we have enough balance.
    if account_balance < (transfer_balance + fee + existential_deposit):
        logging.error(":cross_mark: <red>Not enough balance</red>:")
        logging.info(f"\t\tBalance: \t<blue>{account_balance}</blue>")
        logging.info(f"\t\tAmount: \t<blue>{transfer_balance}</blue>")
        logging.info(f"\t\tFor fee: \t<blue>{fee}</blue>")
        return False

    logging.info(":satellite: <magenta>Transferring...</magenta>")
    logging.info(f"\tAmount: <blue>{transfer_balance}</blue>")
    logging.info(f"\tfrom: <blue>{wallet.name}:{wallet.coldkey.ss58_address}</blue>")
    logging.info(f"\tTo: <blue>{dest}</blue>")
    logging.info(f"\tFor fee: <blue>{fee}</blue>")

    success, block_hash, error_message = do_transfer(
        self=subtensor,
        wallet=wallet,
        dest=dest,
        transfer_balance=transfer_balance,
        wait_for_finalization=wait_for_finalization,
        wait_for_inclusion=wait_for_inclusion,
    )

    if success:
        logging.success(":white_heavy_check_mark: <green>Finalized</green>")
        logging.info(f"<green>Block Hash:</green> <blue>{block_hash}</blue>")

        explorer_urls = get_explorer_url_for_network(
            subtensor.network, block_hash, NETWORK_EXPLORER_MAP
        )
        if explorer_urls != {} and explorer_urls:
            logging.info(
                f"<green>Opentensor Explorer Link: {explorer_urls.get('opentensor')}</green>"
            )
            logging.info(
                f"<green>Taostats Explorer Link: {explorer_urls.get('taostats')}</green>"
            )
    else:
        logging.error(
            f":cross_mark: <red>Failed</red>: {format_error_message(error_message, substrate=subtensor.substrate)}"
        )

    if success:
        logging.info(":satellite: <magenta>Checking Balance...</magenta>")
        new_balance = subtensor.get_balance(wallet.coldkey.ss58_address)
        logging.success(
            f"Balance: <blue>{account_balance}</blue> :arrow_right: <green>{new_balance}</green>"
        )
        return True

    return False
