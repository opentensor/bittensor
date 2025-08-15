"""
This module provides asynchronous functionalities for registering a wallet with the subtensor network using
Proof-of-Work (PoW).

Extrinsics:
- register_extrinsic: Registers the wallet to the subnet.
- burned_register_extrinsic: Registers the wallet to chain by recycling TAO.
"""

import asyncio
from typing import Optional, Union, TYPE_CHECKING

from bittensor.utils import unlock_key
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.registration import log_no_torch_error, create_pow_async, torch

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor.utils.registration.pow import POWSolution


async def _do_burned_register(
    subtensor: "AsyncSubtensor",
    netuid: int,
    wallet: "Wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    period: Optional[int] = None,
) -> tuple[bool, str]:
    """
    Performs a burned register extrinsic call to the Subtensor chain.

    This method sends a registration transaction to the Subtensor blockchain using the burned register mechanism.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): Subtensor instance.
        netuid (int): The network unique identifier to register on.
        wallet (bittensor_wallet.Wallet): The wallet to be registered.
        wait_for_inclusion (bool): Whether to wait for the transaction to be included in a block. Default is False.
        wait_for_finalization (bool): Whether to wait for the transaction to be finalized. Default is True.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating success or failure, and an optional error
            message.
    """

    # create extrinsic call
    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="burned_register",
        call_params={
            "netuid": netuid,
            "hotkey": wallet.hotkey.ss58_address,
        },
    )
    return await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
    )


async def burned_register_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    period: Optional[int] = None,
) -> bool:
    """Registers the wallet to chain by recycling TAO.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): Subtensor instance.
        wallet (bittensor.wallet): Bittensor wallet object.
        netuid (int): The ``netuid`` of the subnet to register on.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``True``, or
            returns ``False`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
            ``True``, or returns ``False`` if the extrinsic fails to be finalized within the timeout.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        success (bool): Flag is ``True`` if extrinsic was finalized or included in the block. If we did not wait for
            finalization / inclusion, the response is ``True``.
    """

    async def get_registration_fee():
        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="burned_register",
            call_params={
                "netuid": netuid,
                "hotkey": wallet.hotkey.ss58_address,
            },
            block_hash=block_hash,
        )
        payment_info = await subtensor.substrate.get_payment_info(
            call, wallet.coldkeypub
        )
        return Balance.from_rao(payment_info["partial_fee"])

    block_hash = await subtensor.substrate.get_chain_head()
    if not await subtensor.subnet_exists(netuid, block_hash=block_hash):
        logging.error(
            f":cross_mark: [red]Failed error:[/red] subnet [blue]{netuid}[/blue] does not exist."
        )
        return False

    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    logging.info(
        f":satellite: [magenta]Checking Account on subnet[/magenta] [blue]{netuid}[/blue][magenta] ...[/magenta]"
    )

    # We could do this as_completed because we don't need old_balance and recycle
    # if neuron is null, but the complexity isn't worth it considering the small performance
    # gains we'd hypothetically receive in this situation
    neuron, old_balance, recycle_amount, registration_fee = await asyncio.gather(
        subtensor.get_neuron_for_pubkey_and_subnet(
            wallet.hotkey.ss58_address, netuid=netuid, block_hash=block_hash
        ),
        subtensor.get_balance(wallet.coldkeypub.ss58_address, block_hash=block_hash),
        subtensor.recycle(netuid=netuid, block_hash=block_hash),
        get_registration_fee(),
    )

    if not neuron.is_null:
        logging.info(":white_heavy_check_mark: [green]Already Registered[/green]")
        logging.info(f"\t\tuid: [blue]{neuron.uid}[/blue]")
        logging.info(f"\t\tnetuid: [blue]{neuron.netuid}[/blue]")
        logging.info(f"\t\thotkey: [blue]{neuron.hotkey}[/blue]")
        logging.info(f"\t\tcoldkey: [blue]{neuron.coldkey}[/blue]")
        return True

    logging.debug(":satellite: [magenta]Recycling TAO for Registration...[/magenta]")
    logging.info(
        f"Recycling {recycle_amount} to register on subnet:{netuid} for fee {registration_fee}"
    )

    success, err_msg = await _do_burned_register(
        subtensor=subtensor,
        netuid=netuid,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
    )

    if not success:
        logging.error(f":cross_mark: [red]Failed error:[/red] {err_msg}")
        await asyncio.sleep(0.5)
        return False
    # Successful registration, final check for neuron and pubkey
    else:
        logging.info(":satellite: [magenta]Checking Balance...[/magenta]")
        block_hash = await subtensor.substrate.get_chain_head()
        new_balance = await subtensor.get_balance(
            wallet.coldkeypub.ss58_address, block_hash=block_hash
        )

        logging.info(
            f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        is_registered = await subtensor.is_hotkey_registered(
            netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
        )
        if is_registered:
            logging.info(":white_heavy_check_mark: [green]Registered[/green]")
            return True
        else:
            # neuron not found, try again
            logging.error(":cross_mark: [red]Unknown error. Neuron not found.[/red]")
            return False


async def _do_pow_register(
    subtensor: "AsyncSubtensor",
    netuid: int,
    wallet: "Wallet",
    pow_result: "POWSolution",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    period: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """Sends a (POW) register extrinsic to the chain.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): The subtensor to send the extrinsic to.
        netuid (int): The subnet to register on.
        wallet (bittensor.wallet): The wallet to register.
        pow_result (POWSolution): The PoW result to register.
        wait_for_inclusion (bool): If ``True``, waits for the extrinsic to be included in a block. Default to `False`.
        wait_for_finalization (bool): If ``True``, waits for the extrinsic to be finalized. Default to `True`.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        success (bool): ``True`` if the extrinsic was included in a block.
        error (Optional[str]): ``None`` on success or not waiting for inclusion/finalization, otherwise the error
            message.
    """
    # create extrinsic call
    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="register",
        call_params={
            "netuid": netuid,
            "block_number": pow_result.block_number,
            "nonce": pow_result.nonce,
            "work": [int(byte_) for byte_ in pow_result.seal],
            "hotkey": wallet.hotkey.ss58_address,
            "coldkey": wallet.coldkeypub.ss58_address,
        },
    )
    return await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
    )


async def register_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[list[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
    period: Optional[int] = None,
) -> bool:
    """Registers the wallet to the chain.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): initialized AsyncSubtensor object to use for chain
            interactions
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        netuid (int): The ``netuid`` of the subnet to register on.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
            `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        max_allowed_attempts (int): Maximum number of attempts to register the wallet.
        output_in_place (bool): Whether the POW solving should be outputted to the console as it goes along.
        cuda (bool): If `True`, the wallet should be registered using CUDA device(s).
        dev_id: The CUDA device id to use, or a list of device ids.
        tpb: The number of threads per block (CUDA).
        num_processes: The number of processes to use to register.
        update_interval: The number of nonces to solve between updates.
        log_verbose: If `True`, the registration process will log more information.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        `True` if extrinsic was finalized or included in the block. If we did not wait for finalization/inclusion, the
            response is `True`.
    """
    block_hash = await subtensor.substrate.get_chain_head()
    logging.debug("[magenta]Checking subnet status... [/magenta]")
    if not await subtensor.subnet_exists(netuid, block_hash=block_hash):
        logging.error(
            f":cross_mark: [red]Failed error:[/red] subnet [blue]{netuid}[/blue] does not exist."
        )
        return False

    logging.info(
        f":satellite: [magenta]Checking Account on subnet[/magenta] [blue]{netuid}[/blue] [magenta]...[/magenta]"
    )
    neuron = await subtensor.get_neuron_for_pubkey_and_subnet(
        hotkey_ss58=wallet.hotkey.ss58_address, netuid=netuid, block_hash=block_hash
    )

    if not neuron.is_null:
        logging.info(":white_heavy_check_mark: [green]Already Registered[/green]")
        logging.info(f"\t\tuid: [blue]{neuron.uid}[/blue]")
        logging.info(f"\t\tnetuid: [blue]{neuron.netuid}[/blue]")
        logging.info(f"\t\thotkey: [blue]{neuron.hotkey}[/blue]")
        logging.info(f"\t\tcoldkey: [blue]{neuron.coldkey}[/blue]")
        return True

    logging.debug(
        f"Registration hotkey: <blue>{wallet.hotkey.ss58_address}</blue>, <green>Public</green> coldkey: "
        f"<blue>{wallet.coldkey.ss58_address}</blue> in the network: <blue>{subtensor.network}</blue>."
    )

    if not torch:
        log_no_torch_error()
        return False

    # Attempt rolling registration.
    attempts = 1

    while True:
        logging.info(
            f":satellite: [magenta]Registering...[/magenta] [blue]({attempts}/{max_allowed_attempts})[/blue]"
        )
        # Solve latest POW.
        if cuda:
            if not torch.cuda.is_available():
                return False

            pow_result = await create_pow_async(
                subtensor=subtensor,
                wallet=wallet,
                netuid=netuid,
                output_in_place=output_in_place,
                cuda=cuda,
                dev_id=dev_id,
                tpb=tpb,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )
        else:
            pow_result = await create_pow_async(
                subtensor=subtensor,
                wallet=wallet,
                netuid=netuid,
                output_in_place=output_in_place,
                cuda=cuda,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )

        # pow failed
        if not pow_result:
            # might be registered already on this subnet
            is_registered = await subtensor.is_hotkey_registered(
                netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
            )
            if is_registered:
                logging.error(
                    f":white_heavy_check_mark: [green]Already registered on netuid:[/green] [blue]{netuid}[/blue]"
                )
                return True

        # pow successful, proceed to submit pow to chain for registration
        else:
            logging.info(":satellite: [magenta]Submitting POW...[/magenta]")
            # check if a pow result is still valid
            while not await pow_result.is_stale_async(subtensor=subtensor):
                result: tuple[bool, Optional[str]] = await _do_pow_register(
                    subtensor=subtensor,
                    netuid=netuid,
                    wallet=wallet,
                    pow_result=pow_result,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    period=period,
                )

                success, err_msg = result
                if not success:
                    # Look error here
                    # https://github.com/opentensor/subtensor/blob/development/pallets/subtensor/src/errors.rs

                    if "HotKeyAlreadyRegisteredInSubNet" in err_msg:
                        logging.info(
                            f":white_heavy_check_mark: [green]Already Registered on subnet:[/green] "
                            f"[blue]{netuid}[/blue]."
                        )
                        return True
                    logging.error(f":cross_mark: [red]Failed[/red]: {err_msg}")
                    await asyncio.sleep(0.5)

                # Successful registration, final check for neuron and pubkey
                if success:
                    logging.info(":satellite: Checking Registration status...")
                    is_registered = await subtensor.is_hotkey_registered(
                        netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
                    )
                    if is_registered:
                        logging.success(
                            ":white_heavy_check_mark: [green]Registered[/green]"
                        )
                        return True
                    else:
                        # neuron not found, try again
                        logging.error(
                            ":cross_mark: [red]Unknown error. Neuron not found.[/red]"
                        )
                        continue
            else:
                # Exited loop because pow is no longer valid.
                logging.error("[red]POW is stale.[/red]")
                # Try again.

        if attempts < max_allowed_attempts:
            # Failed registration, retry pow
            attempts += 1
            logging.error(
                f":satellite: [magenta]Failed registration, retrying pow ...[/magenta] "
                f"[blue]({attempts}/{max_allowed_attempts})[/blue]"
            )
        else:
            # Failed to register after max attempts.
            logging.error("[red]No more attempts.[/red]")
            return False


async def register_subnet_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    period: Optional[int] = None,
) -> bool:
    """
    Registers a new subnetwork on the Bittensor blockchain asynchronously.

    Args:
        subtensor (AsyncSubtensor): The async subtensor interface to send the extrinsic.
        wallet (Wallet): The wallet to be used for subnet registration.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning true.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning true.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        bool: True if the subnet registration was successful, False otherwise.
    """
    balance = await subtensor.get_balance(wallet.coldkeypub.ss58_address)
    burn_cost = await subtensor.get_subnet_burn_cost()

    if burn_cost > balance:
        logging.error(
            f"Insufficient balance {balance} to register subnet. Current burn cost is {burn_cost} TAO"
        )
        return False

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="register_network",
        call_params={
            "hotkey": wallet.hotkey.ss58_address,
            "mechid": 1,
        },
    )

    success, message = await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return True

    if success:
        logging.success(
            ":white_heavy_check_mark: [green]Successfully registered subnet[/green]"
        )
        return True

    logging.error(f"Failed to register subnet: {message}")
    return False


async def set_subnet_identity_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    subnet_name: str,
    github_repo: str,
    subnet_contact: str,
    subnet_url: str,
    logo_url: str,
    discord: str,
    description: str,
    additional: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    period: Optional[int] = None,
) -> tuple[bool, str]:
    """
    Set the identity information for a given subnet.

    Arguments:
        subtensor (AsyncSubtensor): An instance of the Subtensor class to interact with the blockchain.
        wallet (Wallet): A wallet instance used to sign and submit the extrinsic.
        netuid (int): The unique ID for the subnet.
        subnet_name (str): The name of the subnet to assign the identity information.
        github_repo (str): URL of the GitHub repository related to the subnet.
        subnet_contact (str): Subnet's contact information, e.g., email or contact link.
        subnet_url (str): The URL of the subnet's primary web portal.
        logo_url (str): The URL of the logo's primary web portal.
        discord (str): Discord server or contact for the subnet.
        description (str): A textual description of the subnet.
        additional (str): Any additional metadata or information related to the subnet.
        wait_for_inclusion (bool): Whether to wait for the extrinsic inclusion in a block (default: False).
        wait_for_finalization (bool): Whether to wait for the extrinsic finalization in a block (default: True).
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        tuple[bool, str]: A tuple where the first element indicates success or failure (True/False), and the second
            element contains a descriptive message.
    """

    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False, unlock.message

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="set_subnet_identity",
        call_params={
            "hotkey": wallet.hotkey.ss58_address,
            "netuid": netuid,
            "subnet_name": subnet_name,
            "github_repo": github_repo,
            "subnet_contact": subnet_contact,
            "subnet_url": subnet_url,
            "logo_url": logo_url,
            "discord": discord,
            "description": description,
            "additional": additional,
        },
    )

    success, message = await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return True, message

    if success:
        logging.success(
            f":white_heavy_check_mark: [green]Identities for subnet[/green] [blue]{netuid}[/blue] [green]are set.[/green]"
        )
        return True, f"Identities for subnet {netuid} are set."

    logging.error(
        f":cross_mark: Failed to set identity for subnet [blue]{netuid}[/blue]: {message}"
    )
    return False, f"Failed to set identity for subnet {netuid}: {message}"
