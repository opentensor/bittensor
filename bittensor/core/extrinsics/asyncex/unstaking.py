import asyncio
from typing import Optional, TYPE_CHECKING

from bittensor.core.errors import StakeError, NotRegisteredError
from bittensor.utils import unlock_key
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.core.extrinsics.utils import get_old_stakes

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor


async def unstake_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    hotkey_ss58: Optional[str] = None,
    netuid: Optional[int] = None,
    amount: Optional[Balance] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """Removes stake into the wallet coldkey from the specified hotkey ``uid``.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): AsyncSubtensor instance.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        hotkey_ss58 (Optional[str]): The ``ss58`` address of the hotkey to unstake from. By default, the wallet hotkey
            is used.
        netuid (Optional[int]): The subnet uid to unstake from.
        amount (Union[Balance, float]): Amount to stake as Bittensor balance, or ``float`` interpreted as Tao.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``True``, or
            returns ``False`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
            ``True``, or returns ``False`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``True`` if extrinsic was finalized or included in the block. If we did not wait for
            finalization / inclusion, the response is ``True``.
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
    block_hash = await subtensor.substrate.get_chain_head()
    old_balance, old_stake = await asyncio.gather(
        subtensor.get_balance(wallet.coldkeypub.ss58_address, block_hash=block_hash),
        subtensor.get_stake(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            block_hash=block_hash,
        ),
    )

    # Convert to bittensor.Balance
    if amount is None:
        # Unstake it all.
        unstaking_balance = old_stake
        logging.warning(
            f"Didn't receive any unstaking amount. Unstaking all existing stake: [blue]{old_stake}[/blue] "
            f"from hotkey: [blue]{hotkey_ss58}[/blue]"
        )
    else:
        unstaking_balance = amount
        unstaking_balance.set_unit(netuid)

    # Check enough to unstake.
    stake_on_uid = old_stake
    if unstaking_balance > stake_on_uid:
        logging.error(
            f":cross_mark: [red]Not enough stake[/red]: [green]{stake_on_uid}[/green] to unstake: "
            f"[blue]{unstaking_balance}[/blue] from hotkey: [yellow]{wallet.hotkey_str}[/yellow]"
        )
        return False

    try:
        logging.info(
            f"Unstaking [blue]{unstaking_balance}[/blue] from hotkey: [magenta]{hotkey_ss58}[/magenta] on netuid: "
            f"[blue]{netuid}[/blue]"
        )

        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="remove_stake",
            call_params={
                "hotkey": hotkey_ss58,
                "amount_unstaked": unstaking_balance.rao,
                "netuid": netuid,
            },
        )
        staking_response, err_msg = await subtensor.sign_and_send_extrinsic(
            call, wallet, wait_for_inclusion, wait_for_finalization,
            nonce_key="coldkeypub", sign_with="coldkey", use_nonce=True
        )

        if staking_response is True:  # If we successfully unstaked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            logging.info(
                f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] "
                f"[magenta]...[/magenta]"
            )
            new_block_hash = await subtensor.substrate.get_chain_head()
            new_balance, new_stake = await asyncio.gather(
                subtensor.get_balance(
                    wallet.coldkeypub.ss58_address, block_hash=new_block_hash
                ),
                subtensor.get_stake(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    netuid=netuid,
                    block_hash=new_block_hash,
                ),
            )
            logging.info(
                f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
            )
            logging.info(
                f"Stake: [blue]{old_stake}[/blue] :arrow_right: [green]{new_stake}[/green]"
            )
            return True
        else:
            logging.error(f":cross_mark: [red]Failed: {err_msg}.[/red]")
            return False

    except NotRegisteredError:
        logging.error(
            f":cross_mark: [red]Hotkey: {wallet.hotkey_str} is not registered.[/red]"
        )
        return False
    except StakeError as e:
        logging.error(f":cross_mark: [red]Stake Error: {e}[/red]")
        return False


async def unstake_multiple_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    netuids: list[int],
    amounts: Optional[list[Balance]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """Removes stake from each ``hotkey_ss58`` in the list, using each amount, to a common coldkey.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance.
        wallet (bittensor_wallet.Wallet): The wallet with the coldkey to unstake to.
        hotkey_ss58s (List[str]): List of hotkeys to unstake from.
        netuids (List[int]): List of netuids to unstake from.
        amounts (List[Union[Balance, float]]): List of amounts to unstake. If ``None``, unstake all.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``True``, or
            returns ``False`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
            ``True``, or returns ``False`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``True`` if extrinsic was finalized or included in the block. Flag is ``True`` if any
            wallet was unstaked. If we did not wait for finalization / inclusion, the response is ``True``.
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
        isinstance(amount, (Balance, float)) for amount in amounts
    ):
        raise TypeError(
            "amounts must be a [list of bittensor.Balance or float] or None"
        )

    if amounts is None:
        amounts = [None] * len(hotkey_ss58s)
    else:
        # Convert to Balance
        amounts = [amount.set_unit(netuid) for amount, netuid in zip(amounts, netuids)]
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

    block_hash = await subtensor.substrate.get_chain_head()

    all_stakes, old_balance = await asyncio.gather(
        subtensor.get_stake_for_coldkey(
            coldkey_ss58=wallet.coldkeypub.ss58_address, block_hash=block_hash
        ),
        subtensor.get_balance(wallet.coldkeypub.ss58_address, block_hash=block_hash),
    )

    old_stakes: list[Balance] = get_old_stakes(
        wallet=wallet, hotkey_ss58s=hotkey_ss58s, netuids=netuids, all_stakes=all_stakes
    )

    successful_unstakes = 0
    for idx, (hotkey_ss58, amount, old_stake, netuid) in enumerate(
        zip(hotkey_ss58s, amounts, old_stakes, netuids)
    ):
        # Covert to bittensor.Balance
        if amount is None:
            # Unstake it all.
            unstaking_balance = old_stake
            logging.warning(
                f"Didn't receive any unstaking amount. Unstaking all existing stake: [blue]{old_stake}[/blue] "
                f"from hotkey: [blue]{hotkey_ss58}[/blue]"
            )
        else:
            unstaking_balance = amount

        # Check enough to unstake.
        stake_on_uid = old_stake
        if unstaking_balance > stake_on_uid:
            logging.error(
                f":cross_mark: [red]Not enough stake[/red]: [green]{stake_on_uid}[/green] to unstake: "
                f"[blue]{unstaking_balance}[/blue] from hotkey: [blue]{wallet.hotkey_str}[/blue]."
            )
            continue

        try:
            logging.info(
                f"Unstaking [blue]{unstaking_balance}[/blue] from hotkey: [magenta]{hotkey_ss58}[/magenta] on netuid: "
                f"[blue]{netuid}[/blue]"
            )
            call = await subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="remove_stake",
                call_params={
                    "hotkey": hotkey_ss58,
                    "amount_unstaked": unstaking_balance.rao,
                    "netuid": netuid,
                },
            )

            staking_response, err_msg = await subtensor.sign_and_send_extrinsic(
                call,
                wallet,
                wait_for_inclusion,
                wait_for_finalization,
                nonce_key="coldkeypub",
                sign_with="coldkey",
                use_nonce=True
            )

            if staking_response is True:  # If we successfully unstaked.
                # We only wait here if we expect finalization.

                if idx < len(hotkey_ss58s) - 1:
                    # Wait for tx rate limit.
                    tx_rate_limit_blocks = await subtensor.tx_rate_limit()
                    if tx_rate_limit_blocks > 0:
                        logging.info(
                            f":hourglass: [yellow]Waiting for tx rate limit: "
                            f"[white]{tx_rate_limit_blocks}[/white] blocks[/yellow]"
                        )
                        await asyncio.sleep(
                            tx_rate_limit_blocks * 12
                        )  # 12 seconds per block

                if not wait_for_finalization and not wait_for_inclusion:
                    successful_unstakes += 1
                    continue

                logging.info(":white_heavy_check_mark: [green]Finalized[/green]")

                logging.info(
                    f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] "
                    f"[magenta]...[/magenta]..."
                )
                block_hash = await subtensor.substrate.get_chain_head()
                new_stake = await subtensor.get_stake(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    netuid=netuid,
                    block_hash=block_hash,
                )
                logging.info(
                    f"Stake ({hotkey_ss58}): [blue]{stake_on_uid}[/blue] :arrow_right: [green]{new_stake}[/green]"
                )
                successful_unstakes += 1
            else:
                logging.error(f":cross_mark: [red]Failed: {err_msg}.[/red]")
                continue

        except NotRegisteredError:
            logging.error(
                f":cross_mark: [red]Hotkey[/red] [blue]{hotkey_ss58}[/blue] [red]is not registered.[/red]"
            )
            continue
        except StakeError as e:
            logging.error(f":cross_mark: [red]Stake Error: {e}[/red]")
            continue

    if successful_unstakes != 0:
        logging.info(
            f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] "
            f"[magenta]...[/magenta]"
        )
        block_hash = await subtensor.substrate.get_chain_head()
        new_balance = await subtensor.get_balance(
            wallet.coldkeypub.ss58_address, block_hash=block_hash
        )
        logging.info(
            f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        return True

    return False
