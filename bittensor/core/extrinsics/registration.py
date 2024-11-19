"""
This module provides functionalities for registering a wallet with the subtensor network using Proof-of-Work (PoW).

Extrinsics:
- register_extrinsic: Registers the wallet to the subnet.
- burned_register_extrinsic: Registers the wallet to chain by recycling TAO.
"""

import time
from typing import Union, Optional, TYPE_CHECKING


from bittensor.utils import format_error_message, unlock_key
from bittensor.utils.btlogging import logging
from bittensor.utils.networking import ensure_connected
from bittensor.utils.registration import (
    create_pow,
    torch,
    log_no_torch_error,
)

# For annotation and lazy import purposes
if TYPE_CHECKING:
    import torch
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor
    from bittensor.utils.registration import POWSolution
else:
    from bittensor.utils.registration.pow import LazyLoadedTorch

    torch = LazyLoadedTorch()


@ensure_connected
def _do_pow_register(
    self: "Subtensor",
    netuid: int,
    wallet: "Wallet",
    pow_result: "POWSolution",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> tuple[bool, Optional[str]]:
    """Sends a (POW) register extrinsic to the chain.

    Args:
        self (bittensor.core.subtensor.Subtensor): The subtensor to send the extrinsic to.
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
    call = self.substrate.compose_call(
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
    extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=wallet.hotkey)
    response = self.substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # We only wait here if we expect finalization.
    if not wait_for_finalization and not wait_for_inclusion:
        return True, None

    # process if registration successful, try again if pow is still valid
    response.process_events()
    if not response.is_success:
        return False, format_error_message(
            response.error_message, substrate=self.substrate
        )
    # Successful registration
    else:
        return True, None


def register_extrinsic(
    subtensor: "Subtensor",
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
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor interface.
        wallet (bittensor_wallet.wallet): Bittensor wallet object.
        netuid (int): The ``netuid`` of the subnet to register on.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        max_allowed_attempts (int): Maximum number of attempts to register the wallet.
        output_in_place (bool): If true, prints the progress of the proof of work to the console in-place. Meaning the progress is printed on the same lines. Defaults to `True`.
        cuda (bool): If ``true``, the wallet should be registered using CUDA device(s).
        dev_id (Union[List[int], int]): The CUDA device id to use, or a list of device ids.
        tpb (int): The number of threads per block (CUDA).
        num_processes (int): The number of processes to use to register.
        update_interval (int): The number of nonces to solve between updates.
        log_verbose (bool): If ``true``, the registration process will log more information.

    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if not subtensor.subnet_exists(netuid):
        logging.error(
            f":cross_mark: <red>Failed error:</red> subnet <blue>{netuid}</blue> does not exist."
        )
        return False

    logging.info(
        f":satellite: <magenta>Checking Account on subnet</magenta> <blue>{netuid}</blue><magenta>...</magenta>"
    )
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(
        wallet.hotkey.ss58_address, netuid=netuid
    )

    if not neuron.is_null:
        logging.debug(
            f"Wallet <green>{wallet}</green> is already registered on <blue>{neuron.netuid}</blue> with <blue>{neuron.uid}</blue>."
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
            f":satellite: <magenta>Registering...</magenta> <blue>({attempts}/{max_allowed_attempts})</blue>"
        )
        # Solve latest POW.
        if cuda:
            if not torch.cuda.is_available():
                return False
            pow_result: Optional["POWSolution"] = create_pow(
                subtensor,
                wallet,
                netuid,
                output_in_place,
                cuda=cuda,
                dev_id=dev_id,
                tpb=tpb,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )
        else:
            pow_result: Optional["POWSolution"] = create_pow(
                subtensor,
                wallet,
                netuid,
                output_in_place,
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
                logging.info(
                    f":white_heavy_check_mark: <green>Already registered on netuid:</green> <blue>{netuid}</blue>."
                )
                return True

        # pow successful, proceed to submit pow to chain for registration
        else:
            logging.info(":satellite: <magenta>Submitting POW...</magenta>")
            # check if pow result is still valid
            while not pow_result.is_stale(subtensor=subtensor):
                result: tuple[bool, Optional[str]] = _do_pow_register(
                    self=subtensor,
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
                            f":white_heavy_check_mark: <green>Already Registered on subnet </green><blue>{netuid}</blue>."
                        )
                        return True

                    logging.error(f":cross_mark: <red>Failed:</red> {err_msg}")
                    time.sleep(0.5)

                # Successful registration, final check for neuron and pubkey
                else:
                    logging.info(":satellite: <magenta>Checking Balance...</magenta>")
                    is_registered = subtensor.is_hotkey_registered(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        netuid=netuid,
                    )
                    if is_registered:
                        logging.info(
                            ":white_heavy_check_mark: <green>Registered</green>"
                        )
                        return True
                    else:
                        # neuron not found, try again
                        logging.error(
                            ":cross_mark: <red>Unknown error. Neuron not found.</red>"
                        )
                        # keep commented due to this line brings loop to infinitive one
                        # continue
            else:
                # Exited loop because pow is no longer valid.
                logging.error("<red>POW is stale.</red>")
                # Try again.
                continue

        if attempts < max_allowed_attempts:
            # Failed registration, retry pow
            attempts += 1
            logging.info(
                f":satellite: <magenta>Failed registration, retrying pow ...</magenta> <blue>({attempts}/{max_allowed_attempts})</blue>"
            )
        else:
            # Failed to register after max attempts.
            logging.error("<red>No more attempts.</red>")
            return False


@ensure_connected
def _do_burned_register(
    self,
    netuid: int,
    wallet: "Wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> tuple[bool, Optional[str]]:
    """
    Performs a burned register extrinsic call to the Subtensor chain.

    This method sends a registration transaction to the Subtensor blockchain using the burned register mechanism.

    Args:
        self (bittensor.core.subtensor.Subtensor): Subtensor instance.
        netuid (int): The network unique identifier to register on.
        wallet (bittensor_wallet.Wallet): The wallet to be registered.
        wait_for_inclusion (bool): Whether to wait for the transaction to be included in a block. Default is False.
        wait_for_finalization (bool): Whether to wait for the transaction to be finalized. Default is True.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating success or failure, and an optional error message.
    """

    # create extrinsic call
    call = self.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="burned_register",
        call_params={
            "netuid": netuid,
            "hotkey": wallet.hotkey.ss58_address,
        },
    )
    extrinsic = self.substrate.create_signed_extrinsic(
        call=call, keypair=wallet.coldkey
    )
    response = self.substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # We only wait here if we expect finalization.
    if not wait_for_finalization and not wait_for_inclusion:
        return True, None

    # process if registration successful, try again if pow is still valid
    response.process_events()
    if not response.is_success:
        return False, format_error_message(
            response.error_message, substrate=self.substrate
        )
    # Successful registration
    else:
        return True, None


def burned_register_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> bool:
    """Registers the wallet to chain by recycling TAO.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance.
        wallet (bittensor.wallet): Bittensor wallet object.
        netuid (int): The ``netuid`` of the subnet to register on.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if not subtensor.subnet_exists(netuid):
        logging.error(
            f":cross_mark: <red>Failed error:</red> subnet <blue>{netuid}</blue> does not exist."
        )
        return False

    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    logging.info(
        f":satellite: <magenta>Checking Account on subnet</magenta> <blue>{netuid}</blue><magenta> ...</magenta>"
    )
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(
        wallet.hotkey.ss58_address, netuid=netuid
    )

    old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

    if not neuron.is_null:
        logging.info(":white_heavy_check_mark: <green>Already Registered</green>")
        logging.info(f"\t\tuid: <blue>{neuron.uid}</blue>")
        logging.info(f"\t\tnetuid: <blue>{neuron.netuid}</blue>")
        logging.info(f"\t\thotkey: <blue>{neuron.hotkey}</blue>")
        logging.info(f"\t\tcoldkey: <blue>{neuron.coldkey}</blue>")
        return True

    logging.info(":satellite: <magenta>Recycling TAO for Registration...</magenta>")

    recycle_amount = subtensor.recycle(netuid=netuid)
    logging.info(f"Recycling {recycle_amount} to register on subnet:{netuid}")

    success, err_msg = _do_burned_register(
        self=subtensor,
        netuid=netuid,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if not success:
        logging.error(f":cross_mark: <red>Failed error:</red> {err_msg}")
        time.sleep(0.5)
        return False
    # Successful registration, final check for neuron and pubkey
    else:
        logging.info(":satellite: <magenta>Checking Balance...</magenta>")
        block = subtensor.get_current_block()
        new_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address, block=block)

        logging.info(
            f"Balance: <blue>{old_balance}</blue> :arrow_right: <green>{new_balance}</green>"
        )
        is_registered = subtensor.is_hotkey_registered(
            netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
        )
        if is_registered:
            logging.info(":white_heavy_check_mark: <green>Registered</green>")
            return True
        else:
            # neuron not found, try again
            logging.error(":cross_mark: <red>Unknown error. Neuron not found.</red>")
            return False
