import asyncio
from typing import Optional, Sequence, TYPE_CHECKING, cast

from bittensor.core.errors import StakeError, NotRegisteredError
from bittensor.utils import unlock_key
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor


async def _get_threshold_amount(
    subtensor: "AsyncSubtensor", block_hash: str
) -> "Balance":
    """Fetches the minimum required stake threshold from the chain."""
    min_req_stake_ = await subtensor.substrate.query(
        module="SubtensorModule",
        storage_function="NominatorMinRequiredStake",
        block_hash=block_hash,
    )
    min_req_stake: "Balance" = Balance.from_rao(min_req_stake_)
    return min_req_stake


async def _check_threshold_amount(
    subtensor: "AsyncSubtensor",
    balance: "Balance",
    block_hash: str,
    min_req_stake: Optional["Balance"] = None,
) -> tuple[bool, "Balance"]:
    """Checks if the new stake balance will be above the minimum required stake threshold."""
    if not min_req_stake:
        min_req_stake = await _get_threshold_amount(subtensor, block_hash)

    if min_req_stake > balance:
        return False, min_req_stake
    return True, min_req_stake


async def add_stake_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    old_balance: Optional["Balance"] = None,
    hotkey_ss58: Optional[str] = None,
    amount: Optional["Balance"] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """
    Adds the specified amount of stake to passed hotkey `uid`.

    Arguments:
        subtensor: the initialized SubtensorInterface object to use
        wallet: Bittensor wallet object.
        old_balance: the balance prior to the staking
        hotkey_ss58: The `ss58` address of the hotkey account to stake to defaults to the wallet's hotkey.
        amount: Amount to stake as Bittensor balance, `None` if staking all.
        wait_for_inclusion: If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization: If set, waits for the extrinsic to be finalized on the chain before returning `True`,
            or returns `False` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success: Flag is `True` if extrinsic was finalized or included in the block. If we did not wait for
                      finalization/inclusion, the response is `True`.
    """

    # Decrypt keys,
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    # Default to wallet's own hotkey if the value is not passed.
    if hotkey_ss58 is None:
        hotkey_ss58 = wallet.hotkey.ss58_address

    # Flag to indicate if we are using the wallet's own hotkey.
    own_hotkey: bool

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    if not old_balance:
        old_balance = await subtensor.get_balance(wallet.coldkeypub.ss58_address)
    block_hash = await subtensor.substrate.get_chain_head()

    # Get hotkey owner
    hotkey_owner = await subtensor.get_hotkey_owner(
        hotkey_ss58=hotkey_ss58, block_hash=block_hash
    )
    own_hotkey = wallet.coldkeypub.ss58_address == hotkey_owner
    if not own_hotkey:
        # This is not the wallet's own hotkey, so we are delegating.
        if not await subtensor.is_hotkey_delegate(hotkey_ss58, block_hash=block_hash):
            logging.debug(f"Hotkey {hotkey_ss58} is not a delegate on the chain.")
            return False

    # Get current stake and existential deposit
    old_stake, existential_deposit = await asyncio.gather(
        subtensor.get_stake_for_coldkey_and_hotkey(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            hotkey_ss58=hotkey_ss58,
            block_hash=block_hash,
        ),
        subtensor.get_existential_deposit(block_hash=block_hash),
    )

    # Convert to bittensor.Balance
    if amount is None:
        # Stake it all.
        staking_balance = Balance.from_tao(old_balance.tao)
    else:
        staking_balance = Balance.from_tao(amount.tao)

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

    # If nominating, we need to check if the new stake balance will be above the minimum required stake threshold.
    if not own_hotkey:
        new_stake_balance = old_stake + staking_balance
        is_above_threshold, threshold = await _check_threshold_amount(
            subtensor, new_stake_balance, block_hash
        )
        if not is_above_threshold:
            logging.error(
                f":cross_mark: [red]New stake balance of {new_stake_balance} is below the minimum required "
                f"nomination stake threshold {threshold}.[/red]"
            )
            return False

    try:
        logging.info(
            f":satellite: [magenta]Staking to:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
        )
        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="add_stake",
            call_params={"hotkey": hotkey_ss58, "amount_staked": staking_balance.rao},
        )
        staking_response, err_msg = await subtensor.sign_and_send_extrinsic(
            call, wallet, wait_for_inclusion, wait_for_finalization
        )
        if staking_response is True:  # If we successfully staked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            logging.info(
                f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
            )
            new_block_hash = await subtensor.substrate.get_chain_head()
            new_balance, new_stake = await asyncio.gather(
                subtensor.get_balance(
                    wallet.coldkeypub.ss58_address, block_hash=new_block_hash
                ),
                subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    block_hash=new_block_hash,
                ),
            )

            logging.info("Balance:")
            logging.info(
                f"[blue]{old_balance}[/blue] :arrow_right: {new_balance}[/green]"
            )
            logging.info("Stake:")
            logging.info(
                f"[blue]{old_stake}[/blue] :arrow_right: [green]{new_stake}[/green]"
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


async def add_stake_multiple_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    old_balance: Optional["Balance"] = None,
    amounts: Optional[list["Balance"]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """Adds stake to each ``hotkey_ss58`` in the list, using each amount, from a common coldkey.

    Arguments:
        subtensor: The initialized SubtensorInterface object.
        wallet: Bittensor wallet object for the coldkey.
        old_balance: The balance of the wallet prior to staking.
        hotkey_ss58s: List of hotkeys to stake to.
        amounts: List of amounts to stake. If `None`, stake all to the first hotkey.
        wait_for_inclusion: If set, waits for the extrinsic to enter a block before returning `True`, or returns `False`
            if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization: If set, waits for the extrinsic to be finalized on the chain before returning `True`, or
            returns `False` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success: `True` if extrinsic was finalized or included in the block. `True` if any wallet was staked. If we did
            not wait for finalization/inclusion, the response is `True`.
    """
    if not isinstance(hotkey_ss58s, list) or not all(
        isinstance(hotkey_ss58, str) for hotkey_ss58 in hotkey_ss58s
    ):
        raise TypeError("hotkey_ss58s must be a list of str")

    if len(hotkey_ss58s) == 0:
        return True

    if amounts is not None and len(amounts) != len(hotkey_ss58s):
        raise ValueError("amounts must be a list of the same length as hotkey_ss58s")

    new_amounts: Sequence[Optional[Balance]]
    if amounts is None:
        new_amounts = [None] * len(hotkey_ss58s)
    else:
        new_amounts = [Balance.from_tao(amount) for amount in amounts]
        if sum(amount.tao for amount in new_amounts) == 0:
            # Staking 0 tao
            return True

    # Decrypt keys,
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    block_hash = await subtensor.substrate.get_chain_head()
    old_stakes = await asyncio.gather(
        *[
            subtensor.get_stake_for_coldkey_and_hotkey(
                hk, wallet.coldkeypub.ss58_address, block_hash=block_hash
            )
            for hk in hotkey_ss58s
        ]
    )

    # Remove existential balance to keep key alive.
    # Keys must maintain a balance of at least 1000 rao to stay alive.
    total_staking_rao = sum(
        [amount.rao if amount is not None else 0 for amount in new_amounts]
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
        new_amounts = [
            Balance.from_tao(amount.tao * percent_reduction)
            for amount in cast(Sequence[Balance], new_amounts)
        ]

    successful_stakes = 0
    for idx, (hotkey_ss58, amount, old_stake) in enumerate(
        zip(hotkey_ss58s, new_amounts, old_stakes)
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
            call = await subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="add_stake",
                call_params={
                    "hotkey": hotkey_ss58,
                    "amount_staked": staking_balance.rao,
                },
            )
            staking_response, err_msg = await subtensor.sign_and_send_extrinsic(
                call, wallet, wait_for_inclusion, wait_for_finalization
            )

            if staking_response is True:  # If we successfully staked.
                # We only wait here if we expect finalization.

                if idx < len(hotkey_ss58s) - 1:
                    # Wait for tx rate limit.
                    tx_query = await subtensor.substrate.query(
                        module="SubtensorModule",
                        storage_function="TxRateLimit",
                        block_hash=block_hash,
                    )
                    tx_rate_limit_blocks: int = tx_query
                    if tx_rate_limit_blocks > 0:
                        logging.error(
                            f":hourglass: [yellow]Waiting for tx rate limit: [white]{tx_rate_limit_blocks}[/white] "
                            f"blocks[/yellow]"
                        )
                        # 12 seconds per block
                        await asyncio.sleep(tx_rate_limit_blocks * 12)

                if not wait_for_finalization and not wait_for_inclusion:
                    old_balance -= staking_balance
                    successful_stakes += 1
                    if staking_all:
                        # If staked all, no need to continue
                        break

                    continue

                logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

                new_block_hash = await subtensor.substrate.get_chain_head()
                new_stake, new_balance = await asyncio.gather(
                    subtensor.get_stake_for_coldkey_and_hotkey(
                        coldkey_ss58=wallet.coldkeypub.ss58_address,
                        hotkey_ss58=hotkey_ss58,
                        block_hash=new_block_hash,
                    ),
                    subtensor.get_balance(
                        wallet.coldkeypub.ss58_address, block_hash=new_block_hash
                    ),
                )
                logging.info(
                    "Stake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        hotkey_ss58, old_stake, new_stake
                    )
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
            f":satellite: [magenta]Checking Balance on:[/magenta] ([blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
        )
        new_balance = await subtensor.get_balance(wallet.coldkeypub.ss58_address)
        logging.info(
            f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        return True

    return False
