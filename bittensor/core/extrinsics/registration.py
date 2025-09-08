"""
This module provides sync functionalities for registering a wallet with the subtensor network using Proof-of-Work (PoW).

Extrinsics:
- register_extrinsic: Registers the wallet to the subnet.
- burned_register_extrinsic: Registers the wallet to chain by recycling TAO.
"""

import time
from typing import Optional, Union, TYPE_CHECKING

from bittensor.core.extrinsics.utils import get_extrinsic_fee
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import unlock_key, get_function_name
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
    block = subtensor.get_current_block()
    if not subtensor.subnet_exists(netuid, block=block):
        logging.error(
            f":cross_mark: [red]Failed error:[/red] subnet [blue]{netuid}[/blue] does not exist."
        )
        return ExtrinsicResponse(
            False,
            f"Subnet #{netuid} does not exist",
            extrinsic_function=get_function_name(),
        )

    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return ExtrinsicResponse(
            False, unlock.message, extrinsic_function=get_function_name()
        )

    logging.info(
        f":satellite: [magenta]Checking Account on subnet[/magenta] [blue]{netuid}[/blue][magenta] ...[/magenta]"
    )
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(
        wallet.hotkey.ss58_address, netuid=netuid, block=block
    )

    old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address, block=block)

    if not neuron.is_null:
        message = "Already registered."
        logging.info(f":white_heavy_check_mark: [green]{message}[/green]")
        logging.info(f"\t\tuid: [blue]{neuron.uid}[/blue]")
        logging.info(f"\t\tnetuid: [blue]{neuron.netuid}[/blue]")
        logging.info(f"\t\thotkey: [blue]{neuron.hotkey}[/blue]")
        logging.info(f"\t\tcoldkey: [blue]{neuron.coldkey}[/blue]")
        return ExtrinsicResponse(
            message=message, extrinsic_function=get_function_name()
        )

    recycle_amount = subtensor.recycle(netuid=netuid, block=block)
    logging.debug(":satellite: [magenta]Recycling TAO for Registration...[/magenta]")
    logging.info(f"Recycling {recycle_amount} to register on subnet:{netuid}")

    # create extrinsic call
    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="burned_register",
        call_params={
            "netuid": netuid,
            "hotkey": wallet.hotkey.ss58_address,
        },
    )
    fee = get_extrinsic_fee(subtensor=subtensor, call=call, keypair=wallet.coldkeypub)
    logging.info(
        f"The registration fee for SN #[blue]{netuid}[/blue] is [blue]{fee}[/blue]."
    )
    response = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
        calling_function=get_function_name(),
    )

    if not response.success:
        logging.error(f":cross_mark: [red]Failed error:[/red] {response.message}")
        time.sleep(0.5)
        return response

    # TODO: It is worth deleting everything below and simply returning the result without additional verification. This
    #  should be the responsibility of the user. We will also reduce the number of calls to the chain.
    # Successful registration, final check for neuron and pubkey
    logging.info(":satellite: [magenta]Checking Balance...[/magenta]")
    block = subtensor.get_current_block()
    new_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address, block=block)

    logging.info(
        f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
    )
    is_registered = subtensor.is_hotkey_registered(
        netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address, block=block
    )
    if is_registered:
        message = "Registered"
        logging.info(f":white_heavy_check_mark: [green]{message}[/green]")
        return response

    # neuron not found, try again
    message = "Unknown error. Neuron not found."
    logging.error(f":cross_mark: [red]{message}[/red]")
    response.success = False
    response.message = message
    return response


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
    balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
    burn_cost = subtensor.get_subnet_burn_cost()

    if burn_cost > balance:
        message = f"Insufficient balance {balance} to register subnet. Current burn cost is {burn_cost} TAO."
        logging.error(message)
        return ExtrinsicResponse(False, message, extrinsic_function=get_function_name())

    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="register_network",
        call_params={
            "hotkey": wallet.hotkey.ss58_address,
            "mechid": 1,
        },
    )

    response = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
        calling_function=get_function_name(),
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return response

    if response.success:
        logging.success(
            ":white_heavy_check_mark: [green]Successfully registered subnet[/green]"
        )
        return response

    logging.error(f"Failed to register subnet: {response.message}")
    return response


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

    logging.debug("[magenta]Checking subnet status... [/magenta]")
    block = subtensor.get_current_block()
    if not subtensor.subnet_exists(netuid, block=block):
        message = f"Subnet #{netuid} does not exist."
        logging.error(f":cross_mark: [red]Failed error:[/red] {message}")
        return ExtrinsicResponse(False, message, extrinsic_function=get_function_name())

    logging.info(
        f":satellite: [magenta]Checking Account on subnet[/magenta] [blue]{netuid}[/blue] [magenta]...[/magenta]"
    )
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(
        hotkey_ss58=wallet.hotkey.ss58_address, netuid=netuid, block=block
    )

    if not neuron.is_null:
        message = "Already registered."
        logging.info(f":white_heavy_check_mark: [green]{message}[/green]")
        logging.info(f"\t\tuid: [blue]{neuron.uid}[/blue]")
        logging.info(f"\t\tnetuid: [blue]{neuron.netuid}[/blue]")
        logging.info(f"\t\thotkey: [blue]{neuron.hotkey}[/blue]")
        logging.info(f"\t\tcoldkey: [blue]{neuron.coldkey}[/blue]")
        return ExtrinsicResponse(True, message, extrinsic_function=get_function_name())

    logging.debug(
        f"Registration hotkey: <blue>{wallet.hotkey.ss58_address}</blue>, <green>Public</green> coldkey: "
        f"<blue>{wallet.coldkey.ss58_address}</blue> in the network: <blue>{subtensor.network}</blue>."
    )

    if not torch:
        log_no_torch_error()
        return ExtrinsicResponse(
            False, "No torch installed.", extrinsic_function=get_function_name()
        )

    # Attempt rolling registration.
    attempts = 1

    while True:
        logging.info(
            f":satellite: [magenta]Registering...[/magenta] [blue]({attempts}/{max_allowed_attempts})[/blue]"
        )
        # Solve latest POW.
        if cuda:
            if not torch.cuda.is_available():
                return ExtrinsicResponse(
                    False, "CUDA not available.", extrinsic_function=get_function_name()
                )

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
                message = f"Already registered on netuid: {netuid}"
                logging.info(f":white_heavy_check_mark: [green]{message}[/green]")
                return ExtrinsicResponse(
                    True, message, extrinsic_function=get_function_name()
                )

        # pow successful, proceed to submit pow to chain for registration
        else:
            logging.info(":satellite: [magenta]Submitting POW...[/magenta]")
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
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    period=period,
                    raise_error=raise_error,
                    calling_function=get_function_name(),
                )

                if not response.success:
                    # Look error here
                    # https://github.com/opentensor/subtensor/blob/development/pallets/subtensor/src/errors.rs

                    if "HotKeyAlreadyRegisteredInSubNet" in response.message:
                        logging.info(
                            f":white_heavy_check_mark: [green]Already Registered on subnet:[/green] "
                            f"[blue]{netuid}[/blue]."
                        )
                        return response
                    time.sleep(0.5)

                # Successful registration, final check for neuron and pubkey
                if response.success:
                    logging.info(":satellite: Checking Registration status...")
                    is_registered = subtensor.is_hotkey_registered(
                        netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
                    )
                    if is_registered:
                        logging.success(
                            ":white_heavy_check_mark: [green]Registered.[/green]"
                        )
                        return response

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
            message = "No more attempts."
            logging.error(f"[red]{message}[/red]")
            return ExtrinsicResponse(
                False, message, extrinsic_function=get_function_name()
            )


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

    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return ExtrinsicResponse(
            False, unlock.message, extrinsic_function=get_function_name()
        )

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
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
        calling_function=get_function_name(),
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return response

    if response.success:
        logging.success(
            f":white_heavy_check_mark: [green]Identities for subnet[/green] [blue]{netuid}[/blue] [green]are set.[/green]"
        )
        return response

    message = f"Failed to set identity for subnet #{netuid}"
    logging.error(f":cross_mark: {message}: {response.message}")
    response.message = message
    return response
