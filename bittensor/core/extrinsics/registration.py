"""
This module provides sync functionalities for registering a wallet with the subtensor network using Proof-of-Work (PoW).
"""

import time
from typing import Optional, Union, TYPE_CHECKING

from bittensor.core.errors import RegistrationError
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.btlogging import logging
from bittensor.utils.registration import create_pow, log_no_torch_error, torch

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def burned_register_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Registers the wallet to chain by recycling TAO.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor wallet object.
        netuid: The ``netuid`` of the subnet to register on.
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
        if not subtensor.subnet_exists(netuid, block=block):
            return ExtrinsicResponse(False, f"Subnet {netuid} does not exist.")

        neuron = subtensor.get_neuron_for_pubkey_and_subnet(
            wallet.hotkey.ss58_address, netuid=netuid, block=block
        )

        old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address, block=block)

        if not neuron.is_null:
            message = "Already registered."
            logging.debug(f"[green]{message}[/green]")
            logging.debug(f"\t\tuid: [blue]{neuron.uid}[/blue]")
            logging.debug(f"\t\tnetuid: [blue]{neuron.netuid}[/blue]")
            logging.debug(f"\t\thotkey: [blue]{neuron.hotkey}[/blue]")
            logging.debug(f"\t\tcoldkey: [blue]{neuron.coldkey}[/blue]")
            return ExtrinsicResponse(
                message=message, data={"neuron": neuron, "old_balance": old_balance}
            )

        recycle_amount = subtensor.recycle(netuid=netuid, block=block)
        logging.debug(f"Recycling {recycle_amount} to register on subnet:{netuid}")

        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="burned_register",
            call_params={
                "netuid": netuid,
                "hotkey": wallet.hotkey.ss58_address,
            },
        )
        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        extrinsic_fee = response.extrinsic_fee
        logging.debug(
            f"The registration fee for SN #[blue]{netuid}[/blue] is [blue]{extrinsic_fee}[/blue]."
        )
        if not response.success:
            logging.error(f"[red]{response.message}[/red]")
            time.sleep(0.5)
            return response

        # Successful registration, final check for neuron and pubkey
        new_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

        logging.debug(
            f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        is_registered = subtensor.is_hotkey_registered(
            netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
        )

        response.data = {
            "neuron": neuron,
            "balance_before": old_balance,
            "balance_after": new_balance,
            "recycle_amount": recycle_amount,
        }

        if is_registered:
            logging.debug("[green]Registered.[/green]")
            return response

        # neuron not found
        message = f"Neuron with hotkey {wallet.hotkey.ss58_address} not found in subnet {netuid} after registration."
        return ExtrinsicResponse(
            success=False,
            message=message,
            extrinsic=response.extrinsic,
            error=RegistrationError(message),
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def register_subnet_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Registers a new subnetwork on the Bittensor blockchain.

    Parameters:
        subtensor: The subtensor interface to send the extrinsic.
        wallet: The wallet to be used for subnet registration.
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

        balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        burn_cost = subtensor.get_subnet_burn_cost()

        if burn_cost > balance:
            return ExtrinsicResponse(
                False,
                f"Insufficient balance {balance} to register subnet. Current burn cost is {burn_cost} TAO.",
            )

        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="register_network",
            call_params={
                "hotkey": wallet.hotkey.ss58_address,
            },
        )

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return response

        if response.success:
            logging.debug("[green]Successfully registered subnet.[/green]")
            return response

        logging.error(f"Failed to register subnet: {response.message}")
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def register_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[list[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Registers a neuron on the Bittensor subnet with provided netuid using the provided wallet.

    Parameters:
        subtensor: Subtensor object to use for chain interactions
        wallet: Bittensor wallet object.
        netuid: The ``netuid`` of the subnet to register on.
        max_allowed_attempts: Maximum number of attempts to register the wallet.
        output_in_place: Whether the POW solving should be outputted to the console as it goes along.
        cuda: If `True`, the wallet should be registered using CUDA device(s).
        dev_id: The CUDA device id to use, or a list of device ids.
        tpb: The number of threads per block (CUDA).
        num_processes: The number of processes to use to register.
        update_interval: The number of nonces to solve between updates.
        log_verbose: If `True`, the registration process will log more information.
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
        if not subtensor.subnet_exists(netuid, block=block):
            return ExtrinsicResponse(False, f"Subnet {netuid} does not exist.")

        neuron = subtensor.get_neuron_for_pubkey_and_subnet(
            hotkey_ss58=wallet.hotkey.ss58_address, netuid=netuid, block=block
        )

        if not neuron.is_null:
            message = "Already registered."
            logging.debug(f"[green]{message}[/green]")
            logging.debug(f"\t\tuid: [blue]{neuron.uid}[/blue]")
            logging.debug(f"\t\tnetuid: [blue]{neuron.netuid}[/blue]")
            logging.debug(f"\t\thotkey: [blue]{neuron.hotkey}[/blue]")
            logging.debug(f"\t\tcoldkey: [blue]{neuron.coldkey}[/blue]")
            return ExtrinsicResponse(message=message, data={"neuron": neuron})

        logging.debug(
            f"Registration hotkey: [blue]{wallet.hotkey.ss58_address}[/blue], Public coldkey: "
            f"[blue]{wallet.coldkey.ss58_address}[/blue] in the network: [blue]{subtensor.network}[/blue]."
        )

        if not torch:
            log_no_torch_error()
            return ExtrinsicResponse(False, "Torch is not installed.")

        # Attempt rolling registration.
        attempts = 1

        while True:
            # Solve latest POW.
            if cuda:
                if not torch.cuda.is_available():
                    return ExtrinsicResponse(False, "CUDA not available.")

                logging.debug(f"Creating a POW with CUDA.")
                pow_result = create_pow(
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
                logging.debug(f"Creating a POW.")
                pow_result = create_pow(
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
                is_registered = subtensor.is_hotkey_registered(
                    netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
                )
                if is_registered:
                    message = f"Already registered in subnet {netuid}."
                    logging.debug(f"[green]{message}[/green]")
                    return ExtrinsicResponse(message=message)

            # pow successful, proceed to submit pow to chain for registration
            else:
                # check if a pow result is still valid
                while not pow_result.is_stale(subtensor=subtensor):
                    # create extrinsic call
                    call = subtensor.substrate.compose_call(
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
                    response = subtensor.sign_and_send_extrinsic(
                        call=call,
                        wallet=wallet,
                        period=period,
                        raise_error=raise_error,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )

                    if not response.success:
                        # Look error here
                        # https://github.com/opentensor/subtensor/blob/development/pallets/subtensor/src/errors.rs
                        if "HotKeyAlreadyRegisteredInSubNet" in response.message:
                            logging.debug(
                                f"[green]Already registered on subnet:[/green] [blue]{netuid}[/blue]."
                            )
                            return response
                        time.sleep(0.5)

                    if response.success:
                        is_registered = subtensor.is_hotkey_registered(
                            netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
                        )
                        if is_registered:
                            logging.debug("[green]Registered.[/green]")
                            return response

                        # neuron not found, try again
                        logging.warning("[red]Unknown error. Neuron not found.[/red]")
                        continue
                else:
                    # Exited loop because pow is no longer valid.
                    logging.warning("[red]POW is stale.[/red]")
                    # Try again.

            if attempts < max_allowed_attempts:
                # Failed registration, retry pow
                attempts += 1
                logging.warning(
                    f"Failed registration, retrying pow ... [blue]({attempts}/{max_allowed_attempts})[/blue]"
                )
            else:
                # Failed to register after max attempts.
                return ExtrinsicResponse(False, "No more attempts.")

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def set_subnet_identity_extrinsic(
    subtensor: "Subtensor",
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
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Set the identity information for a given subnet.

    Parameters:
        subtensor: An instance of the Subtensor class to interact with the blockchain.
        wallet: A wallet instance used to sign and submit the extrinsic.
        netuid: The unique ID for the subnet.
        subnet_name: The name of the subnet to assign the identity information.
        github_repo: URL of the GitHub repository related to the subnet.
        subnet_contact: Subnet's contact information, e.g., email or contact link.
        subnet_url: The URL of the subnet's primary web portal.
        logo_url: The URL of the logo's primary web portal.
        discord: Discord server or contact for the subnet.
        description: A textual description of the subnet.
        additional: Any additional metadata or information related to the subnet.
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

        call = subtensor.substrate.compose_call(
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

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return response

        if response.success:
            logging.debug(
                f"[green]Identities for subnet[/green] [blue]{netuid}[/blue] [green]are set.[/green]"
            )
            return response

        logging.error(
            f"[red]Failed to set identity for subnet {netuid}: {response.message}[/red]"
        )
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
