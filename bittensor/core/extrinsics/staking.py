import time
from typing import Union, Optional, TYPE_CHECKING, Sequence

from bittensor.core.errors import StakeError, NotRegisteredError
from bittensor.utils import unlock_key
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def add_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: Optional[str] = None,
    amount: Optional[Union["Balance", float]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """
    Adds the specified amount of stake to passed hotkey `uid`.

    Arguments:
        subtensor: the Subtensor object to use
        wallet: Bittensor wallet object.
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

    def _check_threshold_amount(
        balance: "Balance",
        block_hash: str,
        min_req_stake: Optional["Balance"] = None,
    ) -> tuple[bool, "Balance"]:
        """Checks if the new stake balance will be above the minimum required stake threshold."""
        if not min_req_stake:
            min_req_stake_ = subtensor.substrate.query(
                module="SubtensorModule",
                storage_function="NominatorMinRequiredStake",
                block_hash=block_hash,
            )
            min_req_stake = Balance.from_rao(min_req_stake_)
        if min_req_stake > balance:
            return False, min_req_stake
        return True, min_req_stake

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
    old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
    block = subtensor.get_current_block()

    # Get hotkey owner
    hotkey_owner = subtensor.get_hotkey_owner(hotkey_ss58=hotkey_ss58, block=block)
    own_hotkey = wallet.coldkeypub.ss58_address == hotkey_owner
    if not own_hotkey:
        # This is not the wallet's own hotkey, so we are delegating.
        if not subtensor.is_hotkey_delegate(hotkey_ss58, block=block):
            logging.debug(f"Hotkey {hotkey_ss58} is not a delegate on the chain.")
            return False

    # Get current stake and existential deposit
    old_stake = subtensor.get_stake_for_coldkey_and_hotkey(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        hotkey_ss58=hotkey_ss58,
        block=block,
    )
    existential_deposit = subtensor.get_existential_deposit(block=block)

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
        is_above_threshold, threshold = _check_threshold_amount(
            new_stake_balance, block_hash=subtensor.get_block_hash(block)
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
        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="add_stake",
            call_params={"hotkey": hotkey_ss58, "amount_staked": staking_balance.rao},
        )
        staking_response, err_msg = subtensor.sign_and_send_extrinsic(
            call, wallet, wait_for_inclusion, wait_for_finalization
        )
        if staking_response is True:  # If we successfully staked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            logging.info(
                f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] "
                "[magenta]...[/magenta]"
            )
            new_block = subtensor.get_current_block()
            new_balance = subtensor.get_balance(
                wallet.coldkeypub.ss58_address, block=new_block
            )
            new_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=hotkey_ss58,
                block=new_block,
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

    # TODO I don't think these are used. Maybe should just catch SubstrateRequestException?
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
    amounts: Optional[list[Union["Balance", float]]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """Adds stake to each ``hotkey_ss58`` in the list, using each amount, from a common coldkey.

    Arguments:
        subtensor: The initialized SubtensorInterface object.
        wallet: Bittensor wallet object for the coldkey.
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

    def get_old_stakes(block_hash: str) -> dict[str, Balance]:
        calls = [
            (
                subtensor.substrate.create_storage_key(
                    "SubtensorModule",
                    "Stake",
                    [hotkey_ss58, wallet.coldkeypub.ss58_address],
                    block_hash=block_hash,
                )
            )
            for hotkey_ss58 in hotkey_ss58s
        ]
        batch_call = subtensor.substrate.query_multi(calls, block_hash=block_hash)
        results = {}
        for item in batch_call:
            results.update({item[0].params[0]: Balance.from_rao(item[1] or 0)})
        return results

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
    block = subtensor.get_current_block()
    old_stakes: dict[str, Balance] = get_old_stakes(subtensor.get_block_hash(block))

    # Remove existential balance to keep key alive.
    # Keys must maintain a balance of at least 1000 rao to stay alive.
    total_staking_rao = sum(
        [amount.rao if amount is not None else 0 for amount in new_amounts]
    )
    old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address, block=block)
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
            Balance.from_tao(amount.tao * percent_reduction) for amount in new_amounts
        ]

    successful_stakes = 0
    for idx, (hotkey_ss58, amount) in enumerate(zip(hotkey_ss58s, new_amounts)):
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
                f":cross_mark: [red]Not enough balance[/red]: [green]{old_balance}[/green] to stake: "
                f"[blue]{staking_balance}[/blue] from wallet: [white]{wallet.name}[/white]"
            )
            continue

        try:
            call = subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="add_stake",
                call_params={
                    "hotkey": hotkey_ss58,
                    "amount_staked": staking_balance.rao,
                },
            )
            staking_response, err_msg = subtensor.sign_and_send_extrinsic(
                call, wallet, wait_for_inclusion, wait_for_finalization
            )

            if staking_response is True:  # If we successfully staked.
                # We only wait here if we expect finalization.

                if idx < len(hotkey_ss58s) - 1:
                    # Wait for tx rate limit.
                    tx_query = subtensor.substrate.query(
                        module="SubtensorModule", storage_function="TxRateLimit"
                    )
                    tx_rate_limit_blocks: int = tx_query
                    if tx_rate_limit_blocks > 0:
                        logging.error(
                            f":hourglass: [yellow]Waiting for tx rate limit: [white]{tx_rate_limit_blocks}[/white] "
                            f"blocks[/yellow]"
                        )
                        # 12 seconds per block
                        time.sleep(tx_rate_limit_blocks * 12)

                if not wait_for_finalization and not wait_for_inclusion:
                    old_balance -= staking_balance
                    successful_stakes += 1
                    if staking_all:
                        # If staked all, no need to continue
                        break

                    continue

                logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

                new_block = subtensor.get_current_block()
                new_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    block=new_block,
                )
                new_balance = subtensor.get_balance(
                    wallet.coldkeypub.ss58_address, block=new_block
                )
                logging.info(
                    "Stake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        hotkey_ss58, old_stakes[hotkey_ss58], new_stake
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
            f":satellite: [magenta]Checking Balance on:[/magenta] ([blue]{subtensor.network}[/blue] "
            f"[magenta]...[/magenta]"
        )
        new_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        logging.info(
            f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        return True

    return False
