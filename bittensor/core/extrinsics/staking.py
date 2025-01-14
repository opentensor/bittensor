from time import sleep
from typing import Union, Optional, TYPE_CHECKING

from bittensor.core.errors import StakeError, NotRegisteredError
from bittensor.utils import format_error_message, unlock_key
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.networking import ensure_connected

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


@ensure_connected
def _do_stake(
    self: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: str,
    netuid: int,
    amount: "Balance",
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """Sends a stake extrinsic to the chain.

    Args:
        self (subtensor): Subtensor instance.
        wallet (bittensor_wallet.Wallet): Wallet object that can sign the extrinsic.
        hotkey_ss58 (str): Hotkey ``ss58`` address to stake to.
        amount (bittensor.utils.balance.Balance): Amount to stake.
        wait_for_inclusion (bool): If ``true``, waits for inclusion before returning.
        wait_for_finalization (bool): If ``true``, waits for finalization before returning.

    Returns:
        success (bool): ``True`` if the extrinsic was successful.

    Raises:
        bittensor.core.errors.StakeError: If the extrinsic failed.
    """

    call = self.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="add_stake",
        call_params={
            "hotkey": hotkey_ss58,
            "amount_staked": amount.rao,
            "netuid": netuid,
        },
    )
    extrinsic = self.substrate.create_signed_extrinsic(
        call=call, keypair=wallet.coldkey
    )
    response = self.substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    # We only wait here if we expect finalization.
    if not wait_for_finalization and not wait_for_inclusion:
        return True

    response.process_events()
    if response.is_success:
        return True
    else:
        raise StakeError(format_error_message(response.error_message))


def _check_threshold_amount(
    subtensor: "Subtensor", stake_balance: Balance
) -> tuple[bool, Balance]:
    """
    Checks if the new stake balance will be above the minimum required stake threshold.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance.
        stake_balance (Balance): the balance to check for threshold limits.

    Returns:
        success, threshold (bool, Balance): ``true`` if the staking balance is above the threshold, or ``false`` if the staking balance is below the threshold. The threshold balance required to stake.
    """
    min_req_stake: Balance = subtensor.get_minimum_required_stake()
    if min_req_stake > stake_balance:
        return False, min_req_stake
    else:
        return True, min_req_stake


def __do_add_stake_single(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: str,
    amount: "Balance",
    netuid: int,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """
    Executes a stake call to the chain using the wallet and the amount specified.

    Args:
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        hotkey_ss58 (str): Hotkey to stake to.
        amount (bittensor.utils.balance.Balance): Amount to stake as Bittensor balance object.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.

    Raises:
        bittensor.core.errors.StakeError: If the extrinsic fails to be finalized or included in the block.
        bittensor.core.errors.NotDelegateError: If the hotkey is not a delegate.
        bittensor.core.errors.NotRegisteredError: If the hotkey is not registered in any subnets.

    """
    # Decrypt keys,
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    success = _do_stake(
        self=subtensor,
        wallet=wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    return success


def add_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: Optional[str] = None,
    netuid: Optional[int] = None,
    amount: Optional[Union[Balance, float, int]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """Adds the specified amount of stake to passed hotkey ``uid``.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance.
        wallet (Wallet): Bittensor wallet object.
        hotkey_ss58 (Optional[str]): The ``ss58`` address of the hotkey account to stake to defaults to the wallet's hotkey.
        amount (Union[Balance, float]): Amount to stake as Bittensor balance, or ``float`` interpreted as Tao.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.

    Raises:
        bittensor.core.errors.NotRegisteredError: If the wallet is not registered on the chain.
        bittensor.core.errors.NotDelegateError: If the hotkey is not a delegate on the chain.
    """
    # Decrypt keys,
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    # Default to wallet's own hotkey if the value is not passed.
    if hotkey_ss58 is None:
        hotkey_ss58 = wallet.hotkey.ss58_address

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

    # Get current stake
    old_stake = subtensor.get_stake_for_coldkey_and_hotkey(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
    )
    if old_stake is not None:
        old_stake = old_stake.stake
    else:
        old_stake = Balance.from_tao(0)

    # Grab the existential deposit.
    existential_deposit = subtensor.get_existential_deposit()

    # Convert to bittensor.Balance
    if amount is None:
        # Stake it all.
        staking_balance = Balance.from_tao(old_balance.tao)
    elif not isinstance(amount, Balance):
        staking_balance = Balance.from_tao(amount)
    else:
        staking_balance = amount

    # Leave existential balance to keep key alive.
    if staking_balance > old_balance - existential_deposit:
        # If we are staking all, we need to leave at least the existential deposit.
        staking_balance = old_balance - existential_deposit
    else:
        staking_balance = staking_balance

    # Check enough to stake.
    if staking_balance > old_balance:
        logging.error(":cross_mark: [red]Not enough stake:[/red]")
        logging.error(f"\t\tbalance:{old_balance}")
        logging.error(f"\t\tamount: {staking_balance}")
        logging.error(f"\t\twallet: {wallet.name}")
        return False

    try:
        logging.info(
            f":satellite: [magenta]Staking to:[/magenta] [blue]netuid: {netuid}, amount: {staking_balance} on {subtensor.network}[/blue] [magenta]...[/magenta]"
        )
        staking_response: bool = __do_add_stake_single(
            subtensor=subtensor,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            amount=staking_balance,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if staking_response is True:  # If we successfully staked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            logging.info(
                f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
            )
            new_balance = subtensor.get_balance(address=wallet.coldkeypub.ss58_address)

            # Get new stake
            new_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
            )
            if new_stake is not None:
                new_stake = new_stake.stake
            else:
                new_stake = Balance.from_tao(0)

            logging.info(
                f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
            )
            logging.info(
                f"Stake:[blue]{old_stake}[/blue] :arrow_right: [green]{new_stake}[/green]"
            )
            return True
        else:
            logging.error(":cross_mark: [red]Failed[/red]: Error unknown.")
            return False

    except NotRegisteredError:
        logging.error(
            ":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(
                wallet.hotkey_str
            )
        )
        return False
    except StakeError as e:
        logging.error(f":cross_mark: [red]Stake Error: {e}[/red]")
        return False


def add_stake_multiple_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    netuids: list[int],
    amounts: Optional[list[Union[Balance, float, int]]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """Adds stake to each ``hotkey_ss58`` in the list, using each amount, from a common coldkey.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object for the coldkey.
        hotkey_ss58s (List[str]): List of hotkeys to stake to.
        amounts (List[Union[Balance, float]]): List of amounts to stake. If ``None``, stake all to the first hotkey.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or included in the block. Flag is ``true`` if any wallet was staked. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if not isinstance(hotkey_ss58s, list) or not all(
        isinstance(hotkey_ss58, str) for hotkey_ss58 in hotkey_ss58s
    ):
        raise TypeError("hotkey_ss58s must be a list of str")

    if len(hotkey_ss58s) == 0:
        return True

    if amounts is not None and len(amounts) != len(hotkey_ss58s):
        raise ValueError("amounts must be a list of the same length as hotkey_ss58s")

    if netuids is not None and len(netuids) != len(hotkey_ss58s):
        raise ValueError("netuids must be a list of the same length as hotkey_ss58s")

    if amounts is not None and not all(
        isinstance(amount, (Balance, float, int)) for amount in amounts
    ):
        raise TypeError(
            "amounts must be a [list of bittensor.Balance or float] or None"
        )

    if amounts is None:
        amounts = [None] * len(hotkey_ss58s)
    else:
        # Convert to Balance
        amounts = [
            Balance.from_tao(amount) if isinstance(amount, (float, int)) else amount
            for amount in amounts
        ]

        if sum(amount.tao for amount in amounts) == 0:
            # Staking 0 tao
            return True

    # Decrypt keys,
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    old_balance = inital_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

    # Get the old stakes.
    old_stakes = []
    all_stakes = subtensor.get_stake_for_coldkey(
        coldkey_ss58=wallet.coldkeypub.ss58_address
    )
    for hotkey_ss58, netuid in zip(hotkey_ss58s, netuids):
        stake = next(
            (
                stake.stake
                for stake in all_stakes
                if stake.hotkey_ss58 == hotkey_ss58
                and stake.coldkey_ss58 == wallet.coldkeypub.ss58_address
                and stake.netuid == netuid
            ),
            Balance.from_tao(0),  # Default to 0 balance if no match found
        )
        old_stakes.append(stake)

    # Remove existential balance to keep key alive.
    # Keys must maintain a balance of at least 1000 rao to stay alive.
    total_staking_rao = sum(
        [amount.rao if amount is not None else 0 for amount in amounts]
    )
    if total_staking_rao == 0:
        # Staking all to the first wallet.
        if old_balance.rao > 1000:
            old_balance -= Balance.from_rao(1000)

    elif total_staking_rao < 1000:
        # Staking less than 1000 rao to the wallets.
        pass
    else:
        # Staking more than 1000 rao to the wallets.
        # Reduce the amount to stake to each wallet to keep the balance above 1000 rao.
        percent_reduction = 1 - (1000 / total_staking_rao)
        amounts = [
            Balance.from_tao(amount.tao * percent_reduction) for amount in amounts
        ]

    successful_stakes = 0
    for idx, (hotkey_ss58, amount, old_stake, netuid) in enumerate(
        zip(hotkey_ss58s, amounts, old_stakes, netuids)
    ):
        staking_all = False
        # Convert to bittensor.Balance
        if amount is None:
            # Stake it all.
            staking_balance = Balance.from_tao(old_balance.tao)
            staking_all = True
        else:
            # Amounts are cast to balance earlier in the function
            assert isinstance(amount, Balance)
            staking_balance = amount

        # Check enough to stake
        if staking_balance > old_balance:
            logging.error(
                f":cross_mark: [red]Not enough balance[/red]: [green]{old_balance}[/green] to stake: [blue]{staking_balance}[/blue] from wallet: [white]{wallet.name}[/white]"
            )
            continue

        try:
            staking_response: bool = __do_add_stake_single(
                subtensor=subtensor,
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
                amount=staking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # If we successfully staked.
            if staking_response:
                # We only wait here if we expect finalization.

                if idx < len(hotkey_ss58s) - 1:
                    # Wait for tx rate limit.
                    tx_rate_limit_blocks = subtensor.tx_rate_limit()
                    if tx_rate_limit_blocks > 0:
                        logging.error(
                            f":hourglass: [yellow]Waiting for tx rate limit: [white]{tx_rate_limit_blocks}[/white] blocks[/yellow]"
                        )
                        sleep(tx_rate_limit_blocks * 12)  # 12 seconds per block

                if not wait_for_finalization and not wait_for_inclusion:
                    old_balance -= staking_balance
                    successful_stakes += 1
                    if staking_all:
                        # If staked all, no need to continue
                        break

                    continue

                logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

                # Get new stake
                new_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    netuid=netuid,
                )
                if new_stake is not None:
                    new_stake = new_stake.stake
                else:
                    new_stake = Balance.from_tao(0)

                block = subtensor.get_current_block()
                new_balance = subtensor.get_balance(
                    wallet.coldkeypub.ss58_address, block=block
                )
                logging.info(
                    f"Stake ({hotkey_ss58}) on netuid {netuid}: [blue]{old_stake}[/blue] :arrow_right: [green]{new_stake}[/green]"
                )
                old_balance = new_balance
                successful_stakes += 1
                if staking_all:
                    # If staked all, no need to continue
                    break

            else:
                logging.error(":cross_mark: [red]Failed[/red]: Error unknown.")
                continue

        except NotRegisteredError:
            logging.error(
                ":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(
                    hotkey_ss58
                )
            )
            continue
        except StakeError as e:
            logging.error(":cross_mark: [red]Stake Error: {}[/red]".format(e))
            continue

    if successful_stakes != 0:
        logging.info(
            f":satellite: [magenta]Checking Balance on:[/magenta] ([blue]{subtensor.network}[/blue]) [magenta]...[/magenta]"
        )
        new_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        logging.info(
            f"Balance: [blue]{inital_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        return True

    return False
