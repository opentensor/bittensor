"""
This module provides sync functionalities for registering a wallet with the subtensor network.
"""

import time
from typing import Optional, TYPE_CHECKING

from bittensor.core.errors import BalanceTypeError, RegistrationError
from bittensor.core.extrinsics.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.pallets import SubtensorModule
from bittensor.core.settings import DEFAULT_MEV_PROTECTION
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def burned_register_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """Registers the wallet to chain by recycling TAO.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor wallet object.
        netuid: The ``netuid`` of the subnet to register on.
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
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type="both"
            )
        ).success:
            return unlocked

        block = subtensor.get_current_block()
        if not subtensor.subnet_exists(netuid=netuid, block=block):
            return ExtrinsicResponse(
                False, f"Subnet {netuid} does not exist."
            ).with_log()

        neuron = subtensor.get_neuron_for_pubkey_and_subnet(
            netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address, block=block
        )

        old_balance = subtensor.get_balance(
            address=wallet.coldkeypub.ss58_address, block=block
        )

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

        call = SubtensorModule(subtensor).burned_register(
            netuid=netuid, hotkey=wallet.hotkey.ss58_address
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
        extrinsic_fee = response.extrinsic_fee
        logging.debug(
            f"The registration fee for SN #[blue]{netuid}[/blue] is [blue]{extrinsic_fee}[/blue]."
        )
        if not response.success:
            logging.error(f"[red]{response.message}[/red]")
            time.sleep(0.5)
            return response

        # Successful registration, final check for neuron and pubkey
        new_balance = subtensor.get_balance(address=wallet.coldkeypub.ss58_address)

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
        ).with_log()

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def register_limit_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    limit_price: Balance,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """Registers the wallet to chain by recycling TAO, with a maximum burn price limit.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor wallet object.
        netuid: The ``netuid`` of the subnet to register on.
        limit_price: Maximum acceptable burn price as a Balance instance. If the on-chain burn price exceeds
            this value, the transaction will fail with RegistrationPriceLimitExceeded.
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
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type="both"
            )
        ).success:
            return unlocked

        if not isinstance(limit_price, Balance):
            raise BalanceTypeError("`limit_price` must be an instance of Balance.")

        block = subtensor.get_current_block()
        if not subtensor.subnet_exists(netuid=netuid, block=block):
            return ExtrinsicResponse(
                False, f"Subnet {netuid} does not exist."
            ).with_log()

        neuron = subtensor.get_neuron_for_pubkey_and_subnet(
            netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address, block=block
        )

        old_balance = subtensor.get_balance(
            address=wallet.coldkeypub.ss58_address, block=block
        )

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

        call = SubtensorModule(subtensor).register_limit(
            netuid=netuid,
            hotkey=wallet.hotkey.ss58_address,
            limit_price=limit_price.rao,
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
        extrinsic_fee = response.extrinsic_fee
        logging.debug(
            f"The registration fee for SN #[blue]{netuid}[/blue] is [blue]{extrinsic_fee}[/blue]."
        )
        if not response.success:
            logging.error(f"[red]{response.message}[/red]")
            time.sleep(0.5)
            return response

        new_balance = subtensor.get_balance(address=wallet.coldkeypub.ss58_address)

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

        message = f"Neuron with hotkey {wallet.hotkey.ss58_address} not found in subnet {netuid} after registration."
        return ExtrinsicResponse(
            success=False,
            message=message,
            extrinsic=response.extrinsic,
            error=RegistrationError(message),
        ).with_log()

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def register_subnet_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Registers a new subnetwork on the Bittensor blockchain.

    Parameters:
        subtensor: The subtensor interface to send the extrinsic.
        wallet: The wallet to be used for subnet registration.
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
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type="both"
            )
        ).success:
            return unlocked

        balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        burn_cost = subtensor.get_subnet_burn_cost()

        if burn_cost > balance:
            return ExtrinsicResponse(
                False,
                f"Insufficient balance {balance} to register subnet. Current burn cost is {burn_cost} TAO.",
            ).with_log()

        call = SubtensorModule(subtensor).register_network(
            hotkey=wallet.hotkey.ss58_address
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

        if not wait_for_finalization and not wait_for_inclusion:
            return response

        if response.success:
            logging.debug("[green]Successfully registered subnet.[/green]")
            return response

        logging.error(f"Failed to register subnet: {response.message}")
        return response

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
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
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
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type="both"
            )
        ).success:
            return unlocked

        call = SubtensorModule(subtensor).set_subnet_identity(
            netuid=netuid,
            subnet_name=subnet_name,
            github_repo=github_repo,
            subnet_contact=subnet_contact,
            subnet_url=subnet_url,
            logo_url=logo_url,
            discord=discord,
            description=description,
            additional=additional,
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
