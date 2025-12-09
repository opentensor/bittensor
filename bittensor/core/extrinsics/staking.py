from typing import Optional, TYPE_CHECKING, Sequence

from async_substrate_interface.errors import SubstrateRequestException

from bittensor.core.errors import BalanceTypeError
from bittensor.core.extrinsics.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.pallets import SubtensorModule
from bittensor.core.extrinsics.utils import get_old_stakes
from bittensor.core.settings import DEFAULT_MEV_PROTECTION
from bittensor.core.types import ExtrinsicResponse, UIDs
from bittensor.utils import format_error_message
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def add_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    hotkey_ss58: str,
    amount: Balance,
    safe_staking: bool = False,
    allow_partial_stake: bool = False,
    rate_tolerance: float = 0.005,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Adds a stake from the specified wallet to the neuron identified by the SS58 address of its hotkey in specified subnet.
    Staking is a fundamental process in the Bittensor network that enables neurons to participate actively and earn
    incentives.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object.
        netuid: The unique identifier of the subnet to which the neuron belongs.
        hotkey_ss58: The `ss58` address of the hotkey account to stake to default to the wallet's hotkey.
        amount: Amount to stake as Bittensor balance in TAO always.
        safe_staking: If True, enables price safety checks.
        allow_partial_stake: If True, allows partial unstaking if price tolerance exceeded.
        rate_tolerance: Maximum allowed price increase percentage (0.005 = 0.5%).
        mev_protection: If True, encrypts and submits the staking transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Raises:
        SubstrateRequestException: Raised if the extrinsic fails to be included in the block within the timeout.

    Notes:
        The `data` field in the returned `ExtrinsicResponse` contains extra information about the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        block = subtensor.get_current_block()

        # Get current stake and existential deposit
        old_stake = subtensor.get_stake(
            hotkey_ss58=hotkey_ss58,
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            netuid=netuid,
            block=block,
        )
        existential_deposit = subtensor.get_existential_deposit(block=block)

        # Leave existential balance to keep key alive.
        if old_balance <= existential_deposit:
            return ExtrinsicResponse(
                False,
                f"Balance ({old_balance}) is not enough to cover existential deposit `{existential_deposit}`.",
            ).with_log()

        # Leave existential balance to keep key alive.
        if amount > old_balance - existential_deposit:
            # If we are staking all, we need to leave at least the existential deposit.
            amount = old_balance - existential_deposit

        # Check enough to stake.
        if amount > old_balance:
            message = "Not enough stake"
            logging.debug(f":cross_mark: [red]{message}:[/red]")
            logging.debug(f"\t\tbalance:{old_balance}")
            logging.debug(f"\t\tamount: {amount}")
            logging.debug(f"\t\twallet: {wallet.name}")
            return ExtrinsicResponse(False, f"{message}.").with_log()

        if safe_staking:
            pool = subtensor.subnet(netuid=netuid)

            price_with_tolerance = (
                pool.price.tao
                if pool.netuid == 0
                else pool.price.tao * (1 + rate_tolerance)
            )

            limit_price = Balance.from_tao(price_with_tolerance).rao

            logging.debug(
                f"Safe Staking to: [blue]netuid: [green]{netuid}[/green], amount: [green]{amount}[/green], "
                f"tolerance percentage: [green]{rate_tolerance * 100}%[/green], "
                f"price limit: [green]{Balance.from_tao(limit_price)}[/green], "
                f"original price: [green]{pool.price}[/green], "
                f"with partial stake: [green]{allow_partial_stake}[/green] "
                f"on [blue]{subtensor.network}[/blue]."
            )

            call = SubtensorModule(subtensor).add_stake_limit(
                hotkey=hotkey_ss58,
                netuid=netuid,
                amount_staked=amount.rao,
                limit_price=limit_price,
                allow_partial=allow_partial_stake,
            )

        else:
            logging.debug(
                f"Staking to: [blue]netuid: [green]{netuid}[/green], amount: [green]{amount}[/green] "
                f"on [blue]{subtensor.network}[/blue]."
            )
            call = SubtensorModule(subtensor).add_stake(
                netuid=netuid, hotkey=hotkey_ss58, amount_staked=amount.rao
            )

        block_before = subtensor.block
        if mev_protection:
            response = submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            response = subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                use_nonce=True,
                nonce_key="coldkeypub",
                period=period,
                raise_error=raise_error,
            )
        if response.success:
            sim_swap = subtensor.sim_swap(
                origin_netuid=0,
                destination_netuid=netuid,
                amount=amount,
                block=block_before,
            )
            response.transaction_tao_fee = sim_swap.tao_fee
            response.transaction_alpha_fee = sim_swap.alpha_fee.set_unit(netuid)

            if not wait_for_finalization and not wait_for_inclusion:
                return response
            logging.debug("[green]Finalized.[/green]")

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

        if safe_staking and "Custom error: 8" in response.message:
            response.message = "Price exceeded tolerance limit. Either increase price tolerance or enable partial staking."

        logging.error(f"[red]{response.message}[/red]")
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def add_stake_multiple_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuids: UIDs,
    hotkey_ss58s: list[str],
    amounts: list[Balance],
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Adds stake to each ``hotkey_ss58`` in the list, using each amount, from a common coldkey on subnet with
    corresponding netuid.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object for the coldkey.
        netuids: List of netuids to stake to.
        hotkey_ss58s: List of hotkeys to stake to.
        amounts: List of corresponding TAO amounts to bet for each netuid and hotkey.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note:
        The `data` field in the returned `ExtrinsicResponse` contains the results of each individual internal
        `add_stake_extrinsic` call. Each entry maps a tuple key `(idx, hotkey_ss58, netuid)` to either:
            - the corresponding `ExtrinsicResponse` object if the staking attempt was executed, or
            - `None` if the staking was skipped due to failing validation (e.g., wrong balance, zero amount, etc.).
        In the key, `idx` is the index the stake attempt. This allows the caller to inspect which specific operations
        were attempted and which were not.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

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

        if not all(isinstance(a, Balance) for a in amounts):
            raise BalanceTypeError("Each `amount` must be an instance of Balance.")

        new_amounts: Sequence[Optional[Balance]] = [
            amount.set_unit(netuid) for amount, netuid in zip(amounts, netuids)
        ]

        if sum(amount.tao for amount in new_amounts) == 0:
            # Staking 0 tao
            return ExtrinsicResponse(True, "Success")

        block = subtensor.get_current_block()
        all_stakes = subtensor.get_stake_info_for_coldkey(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
        )
        old_stakes: list[Balance] = get_old_stakes(
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            netuids=netuids,
            all_stakes=all_stakes,
        )

        # Remove existential balance to keep key alive. Keys must maintain a balance of at least 1000 rao to stay alive.
        total_staking_rao = sum(
            [amount.rao if amount is not None else 0 for amount in new_amounts]
        )
        old_balance = initial_balance = subtensor.get_balance(
            address=wallet.coldkeypub.ss58_address, block=block
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
                for amount in new_amounts
            ]

        successful_stakes = 0
        data = {}
        for idx, (hotkey_ss58, amount, old_stake, netuid) in enumerate(
            zip(hotkey_ss58s, new_amounts, old_stakes, netuids)
        ):
            data.update({(idx, hotkey_ss58, netuid): None})

            # Check enough to stake
            if amount > old_balance:
                logging.warning(
                    f"Not enough balance: [green]{old_balance}[/green] to stake "
                    f"[blue]{amount}[/blue] from wallet: [white]{wallet.name}[/white] "
                    f"with hotkey: [blue]{hotkey_ss58}[/blue] on netuid [blue]{netuid}[/blue]."
                )
                continue

            try:
                logging.debug(
                    f"Staking [blue]{amount}[/blue] to hotkey [blue]{hotkey_ss58}[/blue] on netuid "
                    f"[blue]{netuid}[/blue]."
                )
                response = add_stake_extrinsic(
                    subtensor=subtensor,
                    wallet=wallet,
                    netuid=netuid,
                    hotkey_ss58=hotkey_ss58,
                    amount=amount,
                    mev_protection=mev_protection,
                    period=period,
                    raise_error=raise_error,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    wait_for_revealed_execution=wait_for_revealed_execution,
                )

                data.update({(idx, hotkey_ss58, netuid): response})

                if response.success:
                    if not wait_for_finalization and not wait_for_inclusion:
                        old_balance -= amount
                        successful_stakes += 1
                        continue

                    logging.debug("[green]Finalized[/green]")

                    new_block = subtensor.get_current_block()
                    new_stake = subtensor.get_stake(
                        coldkey_ss58=wallet.coldkeypub.ss58_address,
                        hotkey_ss58=hotkey_ss58,
                        netuid=netuid,
                        block=new_block,
                    )
                    new_balance = subtensor.get_balance(
                        wallet.coldkeypub.ss58_address, block=new_block
                    )
                    logging.debug(
                        f"Stake ({hotkey_ss58}) on netuid {netuid}: [blue]{old_stake}[/blue] :arrow_right: "
                        f"[green]{new_stake}[/green]"
                    )
                    logging.debug(
                        f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
                    )
                    old_balance = new_balance
                    successful_stakes += 1
                    continue

                logging.warning(
                    f"Staking amount {amount} to hotkey_ss58 {hotkey_ss58} in subnet {netuid} was not successful."
                )

            except SubstrateRequestException as error:
                logging.error(
                    f"[red]Add Stake Multiple error: {format_error_message(error)}[/red]"
                )
                if raise_error:
                    raise

        if len(netuids) > successful_stakes > 0:
            success = False
            message = "Some stake were successful."
        elif successful_stakes == len(netuids):
            success = True
            message = "Success"
        else:
            success = False
            message = "No one stake were successful."

        if (
            new_balance := subtensor.get_balance(wallet.coldkeypub.ss58_address)
        ) != old_balance:
            logging.debug(
                f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
            )
            data.update(
                {"balance_before": initial_balance, "balance_after": new_balance}
            )

        response = ExtrinsicResponse(success, message, data=data)
        if response.success:
            return response
        return response.with_log()

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def set_auto_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    hotkey_ss58: str,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """Sets the coldkey to automatically stake to the hotkey within specific subnet mechanism.

    Parameters:
        subtensor: AsyncSubtensor instance.
        wallet: Bittensor Wallet instance.
        netuid: The subnet unique identifier.
        hotkey_ss58: The SS58 address of the validator's hotkey to which the miner automatically stakes all rewards
            received from the specified subnet immediately upon receipt.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        call = SubtensorModule(subtensor).set_coldkey_auto_stake_hotkey(
            netuid=netuid, hotkey=hotkey_ss58
        )

        if mev_protection:
            response = submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            response = subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if response.success:
            logging.debug(response.message)
            return response

        logging.error(response.message)
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
