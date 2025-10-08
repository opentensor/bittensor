from typing import Optional, TYPE_CHECKING

from async_substrate_interface.errors import SubstrateRequestException

from bittensor.core.errors import BalanceTypeError
from bittensor.core.extrinsics.params import UnstakingParams
from bittensor.core.extrinsics.utils import get_old_stakes
from bittensor.core.types import ExtrinsicResponse, UIDs
from bittensor.utils import format_error_message
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def unstake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    hotkey_ss58: str,
    amount: Balance,
    allow_partial_stake: bool = False,
    rate_tolerance: float = 0.005,
    safe_unstaking: bool = False,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Removes stake into the wallet coldkey from the specified hotkey ``uid``.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor wallet object.
        hotkey_ss58: The ``ss58`` address of the hotkey to unstake from.
        netuid: Subnet unique id.
        amount: Amount to stake as Bittensor balance.
        allow_partial_stake: If true, allows partial unstaking if price tolerance exceeded.
        rate_tolerance: Maximum allowed price decrease percentage (0.005 = 0.5%).
        safe_unstaking: If true, enables price safety checks.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        block = subtensor.get_current_block()
        old_balance = subtensor.get_balance(
            address=wallet.coldkeypub.ss58_address, block=block
        )
        old_stake = subtensor.get_stake(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            block=block,
        )

        # unstaking_balance = amount
        amount.set_unit(netuid)

        # Check enough to unstake.
        if amount > old_stake:
            return ExtrinsicResponse(
                False,
                f"Not enough stake: {old_stake} to unstake: {amount} from hotkey: {hotkey_ss58}",
            ).with_log()

        if safe_unstaking:
            pool = subtensor.subnet(netuid=netuid)

            call_function = "remove_stake_limit"

            call_params = UnstakingParams.remove_stake_limit(
                netuid=netuid,
                hotkey_ss58=hotkey_ss58,
                amount=amount,
                allow_partial_stake=allow_partial_stake,
                rate_tolerance=rate_tolerance,
                pool=pool,
            )

            logging_message = (
                f":satellite: [magenta]Safe Unstaking from:[/magenta] "
                f"netuid: [green]{netuid}[/green], amount: [green]{amount}[/green], "
                f"tolerance percentage: [green]{rate_tolerance * 100}%[/green], "
                f"price limit: [green]{Balance.from_rao(call_params['limit_price'])}[/green], "
                f"original price: [green]{pool.price.tao}[/green], "
                f"with partial unstake: [green]{allow_partial_stake}[/green] "
                f"on [blue]{subtensor.network}[/blue]"
            )

        else:
            call_function = "remove_stake"
            call_params = UnstakingParams.remove_stake(
                netuid=netuid,
                hotkey_ss58=hotkey_ss58,
                amount=amount,
            )
            logging_message = (
                f":satellite: [magenta]Unstaking from:[/magenta] "
                f"netuid: [green]{netuid}[/green], amount: [green]{amount}[/green] "
                f"on [blue]{subtensor.network}[/blue]"
            )

        logging.debug(logging_message)
        call = subtensor.compose_call(
            call_module="SubtensorModule",
            call_function=call_function,
            call_params=call_params,
        )

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            nonce_key="coldkeypub",
            use_nonce=True,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            sim_swap = subtensor.sim_swap(
                origin_netuid=netuid,
                destination_netuid=0,
                amount=amount,
            )
            response.transaction_tao_fee = sim_swap.tao_fee
            response.transaction_alpha_fee = sim_swap.alpha_fee.set_unit(netuid)

            if not wait_for_finalization and not wait_for_inclusion:
                return response

            logging.debug("[green]Finalized[/green]")

            new_block = subtensor.get_current_block()
            new_balance = subtensor.get_balance(
                wallet.coldkeypub.ss58_address, block=new_block
            )
            new_stake = subtensor.get_stake(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
                block=new_block,
            )
            logging.debug(
                f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
            )
            logging.debug(
                f"Stake: [blue]{old_stake}[/blue] :arrow_right: [green]{new_stake}[/green]"
            )
            response.data = {
                "balance_before": old_balance,
                "balance_after": new_balance,
                "stake_before": old_stake,
                "stake_after": new_stake,
            }
            return response

        if safe_unstaking and "Custom error: 8" in response.message:
            response.message = "Price exceeded tolerance limit. Either increase price tolerance or enable partial staking."

        logging.error(f"[red]{response.message}[/red]")
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def unstake_all_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    hotkey_ss58: str,
    rate_tolerance: Optional[float] = 0.005,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Unstakes all TAO/Alpha associated with a hotkey from the specified subnets on the Bittensor network.

    Parameters:
        subtensor: Subtensor instance.
        wallet: The wallet of the stake owner.
        netuid: The unique identifier of the subnet.
        hotkey_ss58: The SS58 address of the hotkey to unstake from.
        rate_tolerance: The maximum allowed price change ratio when unstaking. For example, 0.005 = 0.5% maximum
            price decrease. If not passed (None), then unstaking goes without price limit.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        pool = subtensor.subnet(netuid=netuid) if rate_tolerance else None
        call_params = UnstakingParams.remove_stake_full_limit(
            netuid=netuid,
            hotkey_ss58=hotkey_ss58,
            rate_tolerance=rate_tolerance,
            pool=pool,
        )

        call = subtensor.compose_call(
            call_module="SubtensorModule",
            call_function="remove_stake_full_limit",
            call_params=call_params,
        )

        return subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            nonce_key="coldkeypub",
            use_nonce=True,
            period=period,
            raise_error=raise_error,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def unstake_multiple_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuids: UIDs,
    hotkey_ss58s: list[str],
    amounts: Optional[list[Balance]] = None,
    rate_tolerance: Optional[float] = 0.05,
    unstake_all: bool = False,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Removes stake from each ``hotkey_ss58`` in the list, using each amount, to a common coldkey.

    Parameters:
        subtensor: Subtensor instance.
        wallet: The wallet with the coldkey to unstake to.
        netuids: List of subnets unique IDs to unstake from.
        hotkey_ss58s: List of hotkeys to unstake from.
        amounts: List of amounts to unstake. If ``None``, unstake all.
        rate_tolerance: Maximum allowed price decrease percentage (0.005 = 0.5%).
        unstake_all: If true, unstakes all tokens.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note:
        The `data` field in the returned `ExtrinsicResponse` contains the results of each individual internal
        `unstake_extrinsic` or `unstake_all_extrinsic` call. Each entry maps a tuple key `(idx, hotkey_ss58, netuid)` to
        either:
            - the corresponding `ExtrinsicResponse` object if the unstaking attempt was executed, or
            - `None` if the unstaking was skipped due to failing validation (e.g., wrong balance, zero amount, etc.).
        In the key, `idx` is the index the unstake attempt. This allows the caller to inspect which specific operations
        were attempted and which were not.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        # or amounts or unstake_all (no both)
        if amounts and unstake_all:
            raise ValueError("Cannot specify both `amounts` and `unstake_all`.")

        if amounts is not None and not all(
            isinstance(amount, Balance) for amount in amounts
        ):
            raise BalanceTypeError("`amounts` must be a list of Balance or None.")

        if amounts is None:
            amounts = [None] * len(hotkey_ss58s)
        else:
            # Convert to Balance
            amounts = [
                amount.set_unit(netuid) for amount, netuid in zip(amounts, netuids)
            ]
            if sum(amount.tao for amount in amounts) == 0:
                # Staking 0 tao
                return ExtrinsicResponse(True, "Success")

        if not all(
            [
                isinstance(netuids, list),
                isinstance(hotkey_ss58s, list),
                isinstance(amounts, list),
            ]
        ):
            raise TypeError(
                "The `netuids`, `hotkey_ss58s` and `amounts` must be lists."
            )

        if len(hotkey_ss58s) == 0:
            return ExtrinsicResponse(True, "Success")

        if not len(netuids) == len(hotkey_ss58s) == len(amounts):
            raise ValueError(
                "The number of items in `netuids`, `hotkey_ss58s` and `amounts` must be the same."
            )

        if not all(isinstance(hotkey_ss58, str) for hotkey_ss58 in hotkey_ss58s):
            raise TypeError("`hotkey_ss58s` must be a list of str.")

        if amounts is not None and len(amounts) != len(hotkey_ss58s):
            raise ValueError(
                "`amounts` must be a list of the same length as `hotkey_ss58s`."
            )

        if netuids is not None and len(netuids) != len(hotkey_ss58s):
            raise ValueError(
                "`netuids` must be a list of the same length as `hotkey_ss58s`."
            )

        block = subtensor.get_current_block()
        old_balance = subtensor.get_balance(
            address=wallet.coldkeypub.ss58_address, block=block
        )
        all_stakes = subtensor.get_stake_info_for_coldkey(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            block=block,
        )
        old_stakes = get_old_stakes(
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            netuids=netuids,
            all_stakes=all_stakes,
        )

        successful_unstakes = 0
        data = {}
        for idx, (hotkey_ss58, amount, old_stake, netuid) in enumerate(
            zip(hotkey_ss58s, amounts, old_stakes, netuids)
        ):
            data.update({(idx, hotkey_ss58, netuid): None})

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
            if unstaking_balance > old_stake:
                logging.warning(
                    f"[red]Not enough stake[/red]: [green]{old_stake}[/green] to unstake: "
                    f"[blue]{unstaking_balance}[/blue] from hotkey: [blue]{hotkey_ss58}[/blue]."
                )
                continue

            try:
                logging.debug(
                    f"Unstaking [blue]{amount}[/blue] from hotkey [blue]{hotkey_ss58}[/blue] on netuid "
                    f"[blue]{netuid}[/blue]."
                )
                if unstake_all:
                    response = unstake_all_extrinsic(
                        subtensor=subtensor,
                        wallet=wallet,
                        hotkey_ss58=hotkey_ss58,
                        netuid=netuid,
                        rate_tolerance=rate_tolerance,
                        period=period,
                        raise_error=raise_error,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )
                else:
                    response = unstake_extrinsic(
                        subtensor=subtensor,
                        wallet=wallet,
                        netuid=netuid,
                        hotkey_ss58=hotkey_ss58,
                        amount=unstaking_balance,
                        period=period,
                        raise_error=raise_error,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )

                data.update({(idx, hotkey_ss58, netuid): response})

                if response.success:
                    if not wait_for_finalization and not wait_for_inclusion:
                        successful_unstakes += 1
                        continue

                    logging.debug("[green]Finalized[/green]")

                    new_stake = subtensor.get_stake(
                        coldkey_ss58=wallet.coldkeypub.ss58_address,
                        hotkey_ss58=hotkey_ss58,
                        netuid=netuid,
                    )
                    logging.debug(
                        f"Stake ({hotkey_ss58}) in subnet {netuid}: "
                        f"[blue]{old_stake}[/blue] :arrow_right: [green]{new_stake}[/green]."
                    )
                    successful_unstakes += 1
                    continue

                logging.warning(
                    f"Unstaking from hotkey_ss58 {hotkey_ss58} in subnet {netuid} was not successful."
                )

            except SubstrateRequestException as error:
                logging.error(
                    f"[red]Add Stake Multiple error: {format_error_message(error)}[/red]"
                )
                if raise_error:
                    raise

        if len(netuids) > successful_unstakes > 0:
            success = False
            message = "Some unstake were successful."
        elif successful_unstakes == len(netuids):
            success = True
            message = "Success"
        else:
            success = False
            message = "No one unstake were successful."

        if (
            new_balance := subtensor.get_balance(address=wallet.coldkeypub.ss58_address)
        ) != old_balance:
            logging.debug(
                f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
            )
            data.update({"balance_before": old_balance, "balance_after": new_balance})

        response = ExtrinsicResponse(success, message, data=data)
        if response.success:
            return response
        return response.with_log()

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
