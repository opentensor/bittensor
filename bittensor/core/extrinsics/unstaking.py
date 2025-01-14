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
def _do_unstake(
    self: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: str,
    netuid: int,
    amount: "Balance",
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """Sends an unstake extrinsic to the chain.

    Args:
        wallet (bittensor_wallet.Wallet): Wallet object that can sign the extrinsic.
        hotkey_ss58 (str): Hotkey ``ss58`` address to unstake from.
        amount (bittensor.utils.balance.Balance): Amount to unstake.
        wait_for_inclusion (bool): If ``true``, waits for inclusion before returning.
        wait_for_finalization (bool): If ``true``, waits for finalization before returning.

    Returns:
        success (bool): ``True`` if the extrinsic was successful.

    Raises:
        StakeError: If the extrinsic failed.
    """

    call = self.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="remove_stake",
        call_params={
            "hotkey": hotkey_ss58,
            "amount_unstaked": amount.rao,
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


def __do_remove_stake_single(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: str,
    netuid: int,
    amount: "Balance",
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """
    Executes an unstake call to the chain using the wallet and the amount specified.

    Args:
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        hotkey_ss58 (str): Hotkey address to unstake from.
        amount (bittensor.utils.balance.Balance): Amount to unstake as Bittensor balance object.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.

    Raises:
        bittensor.core.errors.StakeError: If the extrinsic fails to be finalized or included in the block.
        bittensor.core.errors.NotRegisteredError: If the hotkey is not registered in any subnets.

    """
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    success = _do_unstake(
        self=subtensor,
        wallet=wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    return success


def unstake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: Optional[str] = None,
    netuid: Optional[int] = None,
    amount: Optional[Union[Balance, float]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """Removes stake into the wallet coldkey from the specified hotkey ``uid``.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        hotkey_ss58 (Optional[str]): The ``ss58`` address of the hotkey to unstake from. By default, the wallet hotkey is used.
        amount (Union[Balance, float]): Amount to stake as Bittensor balance, or ``float`` interpreted as Tao.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    # Decrypt keys,
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    if hotkey_ss58 is None:
        hotkey_ss58 = wallet.hotkey.ss58_address  # Default to wallet's own hotkey.

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

    old_stake = subtensor.get_stake_for_coldkey_and_hotkey(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
    )
    if old_stake is not None:
        old_stake = old_stake.stake
    else:
        old_stake = Balance.from_tao(0)

    # Convert to bittensor.Balance
    if amount is None:
        # Unstake it all.
        unstaking_balance = old_stake
    elif not isinstance(amount, Balance):
        unstaking_balance = Balance.from_tao(amount)
    else:
        unstaking_balance = amount

    # Check enough to unstake.
    stake_on_uid = old_stake
    if unstaking_balance > stake_on_uid:
        logging.error(
            f":cross_mark: [red]Not enough stake[/red]: [green]{stake_on_uid}[/green] to unstake: [blue]{unstaking_balance}[/blue] from hotkey: [yellow]{wallet.hotkey_str}[/yellow]"
        )
        return False

    try:
        logging.info(
            f":satellite: [magenta]Unstaking from chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
        )
        staking_response: bool = __do_remove_stake_single(
            subtensor=subtensor,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            amount=unstaking_balance,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if staking_response is True:  # If we successfully unstaked.
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
                f"Stake: [blue]{old_stake}[/blue] :arrow_right: [green]{new_stake}[/green]"
            )
            return True
        else:
            logging.error(":cross_mark: [red]Failed[/red]: Unknown Error.")
            return False

    except NotRegisteredError:
        logging.error(
            f":cross_mark: [red]Hotkey: {wallet.hotkey_str} is not registered.[/red]"
        )
        return False
    except StakeError as e:
        logging.error(":cross_mark: [red]Stake Error: {}[/red]".format(e))
        return False


def unstake_multiple_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    netuids: list[int],
    amounts: Optional[list[Union[Balance, float, int]]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """Removes stake from each ``hotkey_ss58`` in the list, using each amount, to a common coldkey.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance.
        wallet (bittensor_wallet.Wallet): The wallet with the coldkey to unstake to.
        hotkey_ss58s (List[str]): List of hotkeys to unstake from.
        amounts (List[Union[Balance, float]]): List of amounts to unstake. If ``None``, unstake all.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or included in the block. Flag is ``true`` if any wallet was unstaked. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if not isinstance(hotkey_ss58s, list) or not all(
        isinstance(hotkey_ss58, str) for hotkey_ss58 in hotkey_ss58s
    ):
        raise TypeError("hotkey_ss58s must be a list of str")

    if len(hotkey_ss58s) == 0:
        return True

    if netuids is not None and len(netuids) != len(hotkey_ss58s):
        raise ValueError("netuids must be a list of the same length as hotkey_ss58s")

    if amounts is not None and len(amounts) != len(hotkey_ss58s):
        raise ValueError("amounts must be a list of the same length as hotkey_ss58s")

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

    # Unlock coldkey.
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

    _stakes = subtensor.get_stake_for_coldkey(
        coldkey_ss58=wallet.coldkeypub.ss58_address
    )
    old_stakes = []
    for hotkey_ss58, netuid in zip(hotkey_ss58s, netuids):
        stake = next(
            (
                stake.stake
                for stake in _stakes
                if stake.hotkey_ss58 == hotkey_ss58
                and stake.coldkey_ss58 == wallet.coldkeypub.ss58_address
                and stake.netuid == netuid
            ),
            Balance.from_tao(0),  # Default to 0 balance if no match found
        )
        old_stakes.append(stake)

    successful_unstakes = 0
    for idx, (hotkey_ss58, amount, old_stake, netuid) in enumerate(
        zip(hotkey_ss58s, amounts, old_stakes, netuids)
    ):
        # Covert to bittensor.Balance
        if amount is None:
            # Unstake it all.
            unstaking_balance = old_stake
        else:
            unstaking_balance = (
                amount if isinstance(amount, Balance) else Balance.from_tao(amount)
            )

        # Check enough to unstake.
        stake_on_uid = old_stake
        if unstaking_balance > stake_on_uid:
            logging.error(
                f":cross_mark: [red]Not enough stake[/red]: [green]{stake_on_uid}[/green] to unstake: [blue]{unstaking_balance}[/blue] from hotkey: [blue]{wallet.hotkey_str}[/blue]."
            )
            continue

        try:
            logging.info(
                f":satellite: [magenta]Unstaking from chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
            )
            staking_response: bool = __do_remove_stake_single(
                subtensor=subtensor,
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
                amount=unstaking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if staking_response is True:  # If we successfully unstaked.
                # We only wait here if we expect finalization.

                if idx < len(hotkey_ss58s) - 1:
                    # Wait for tx rate limit.
                    tx_rate_limit_blocks = subtensor.tx_rate_limit()
                    if tx_rate_limit_blocks > 0:
                        logging.info(
                            f":hourglass: [yellow]Waiting for tx rate limit: [white]{tx_rate_limit_blocks}[/white] blocks[/yellow]"
                        )
                        sleep(tx_rate_limit_blocks * 12)  # 12 seconds per block

                if not wait_for_finalization and not wait_for_inclusion:
                    successful_unstakes += 1
                    continue

                logging.info(":white_heavy_check_mark: [green]Finalized[/green]")

                logging.info(
                    f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]..."
                )

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
                    f"Stake ({hotkey_ss58}) on netuid {netuid}: [blue]{stake_on_uid}[/blue] :arrow_right: [green]{new_stake}[/green]"
                )
                successful_unstakes += 1
            else:
                logging.error(":cross_mark: [red]Failed: Unknown Error.[/red]")
                continue

        except NotRegisteredError:
            logging.error(
                f":cross_mark: [red]Hotkey[/red] [blue]{hotkey_ss58}[/blue] [red]is not registered.[/red]"
            )
            continue
        except StakeError as e:
            logging.error(":cross_mark: [red]Stake Error: {}[/red]".format(e))
            continue

    if successful_unstakes != 0:
        logging.info(
            f":satellite: [magenta]Checking Balance on:[/magenta] ([blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
        )
        new_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        logging.info(
            f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        return True

    return False
