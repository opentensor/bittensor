"""
This module provides functionalities for registering a wallet with the subtensor network using Proof-of-Work (PoW).

Extrinsics:
- register_extrinsic: Registers the wallet to the subnet.
- run_faucet_extrinsic: Runs a continual POW to get a faucet of TAO on the test net.
"""

import asyncio
from typing import Optional, Union, TYPE_CHECKING

from bittensor_wallet import Wallet

from bittensor.utils import format_error_message
from bittensor.utils.btlogging import logging
from bittensor.utils.registration import log_no_torch_error, create_pow_async

# For annotation and lazy import purposes
if TYPE_CHECKING:
    import torch
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor.utils.registration.pow import POWSolution
else:
    from bittensor.utils.registration.pow import LazyLoadedTorch

    torch = LazyLoadedTorch()


class MaxSuccessException(Exception):
    """Raised when the POW Solver has reached the max number of successful solutions."""


class MaxAttemptsException(Exception):
    """Raised when the POW Solver has reached the max number of attempts."""


async def _do_pow_register(
    subtensor: "AsyncSubtensor",
    netuid: int,
    wallet: "Wallet",
    pow_result: "POWSolution",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> tuple[bool, Optional[str]]:
    """Sends a (POW) register extrinsic to the chain.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): The subtensor to send the extrinsic to.
        netuid (int): The subnet to register on.
        wallet (bittensor.wallet): The wallet to register.
        pow_result (POWSolution): The PoW result to register.
        wait_for_inclusion (bool): If ``True``, waits for the extrinsic to be included in a block. Default to `False`.
        wait_for_finalization (bool): If ``True``, waits for the extrinsic to be finalized. Default to `True`.

    Returns:
        success (bool): ``True`` if the extrinsic was included in a block.
        error (Optional[str]): ``None`` on success or not waiting for inclusion/finalization, otherwise the error message.
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
    extrinsic = await subtensor.substrate.create_signed_extrinsic(
        call=call, keypair=wallet.hotkey
    )
    response = await subtensor.substrate.submit_extrinsic(
        extrinsic=extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # We only wait here if we expect finalization.
    if not wait_for_finalization and not wait_for_inclusion:
        return True, None

    # process if registration successful, try again if pow is still valid
    if not await response.is_success:
        return False, format_error_message(error_message=await response.error_message)
    # Successful registration
    else:
        return True, None


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
) -> bool:
    """Registers the wallet to the chain.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): initialized AsyncSubtensor object to use for chain interactions
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        netuid (int): The ``netuid`` of the subnet to register on.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        max_allowed_attempts (int): Maximum number of attempts to register the wallet.
        output_in_place (bool): Whether the POW solving should be outputted to the console as it goes along.
        cuda (bool): If `True`, the wallet should be registered using CUDA device(s).
        dev_id: The CUDA device id to use, or a list of device ids.
        tpb: The number of threads per block (CUDA).
        num_processes: The number of processes to use to register.
        update_interval: The number of nonces to solve between updates.
        log_verbose: If `True`, the registration process will log more information.

    Returns:
        `True` if extrinsic was finalized or included in the block. If we did not wait for finalization/inclusion, the response is `True`.
    """

    logging.debug("[magenta]Checking subnet status... [/magenta]")
    if not await subtensor.subnet_exists(netuid):
        logging.error(
            f":cross_mark: [red]Failed error:[/red] subnet [blue]{netuid}[/blue] does not exist."
        )
        return False

    logging.info(
        f":satellite: [magenta]Checking Account on subnet[/magenta] [blue]{netuid}[/blue] [magenta]...[/magenta]"
    )
    neuron = await subtensor.get_neuron_for_pubkey_and_subnet(
        hotkey_ss58=wallet.hotkey.ss58_address,
        netuid=netuid,
    )

    if not neuron.is_null:
        logging.debug(
            f"Wallet [green]{wallet}[/green] is already registered on subnet [blue]{neuron.netuid}[/blue] with uid[blue]{neuron.uid}[/blue]."
        )
        return True

    logging.debug(
        f"Registration hotkey: <blue>{wallet.hotkey.ss58_address}</blue>, <green>Public</green> coldkey: <blue>{wallet.coldkey.ss58_address}</blue> in the network: <blue>{subtensor.network}</blue>."
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
            # check if pow result is still valid
            while not await pow_result.is_stale_async(subtensor=subtensor):
                result: tuple[bool, Optional[str]] = await _do_pow_register(
                    subtensor=subtensor,
                    netuid=netuid,
                    wallet=wallet,
                    pow_result=pow_result,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )

                success, err_msg = result
                if not success:
                    # Look error here
                    # https://github.com/opentensor/subtensor/blob/development/pallets/subtensor/src/errors.rs

                    if "HotKeyAlreadyRegisteredInSubNet" in err_msg:
                        logging.info(
                            f":white_heavy_check_mark: [green]Already Registered on subnet:[/green] [blue]{netuid}[/blue]."
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
                # continue

        if attempts < max_allowed_attempts:
            # Failed registration, retry pow
            attempts += 1
            logging.error(
                f":satellite: [magenta]Failed registration, retrying pow ...[/magenta] [blue]({attempts}/{max_allowed_attempts})[/blue]"
            )
        else:
            # Failed to register after max attempts.
            logging.error("[red]No more attempts.[/red]")
            return False
