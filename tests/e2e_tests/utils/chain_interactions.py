"""
This module provides functions interacting with the chain for end-to-end testing;
these are not present in btsdk but are required for e2e tests
"""

import asyncio
import unittest.mock
from typing import Union, Optional, TYPE_CHECKING

from bittensor.utils.btlogging import logging

# for typing purposes
if TYPE_CHECKING:
    from bittensor import Wallet
    from bittensor.core.subtensor import Subtensor
    from async_substrate_interface import SubstrateInterface, ExtrinsicReceipt


ANY_BALANCE = unittest.mock.Mock(
    rao=unittest.mock.ANY,
    unit=unittest.mock.ANY,
)


def sudo_set_hyperparameter_bool(
    substrate: "SubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    value: bool,
    netuid: int,
) -> bool:
    """
    Sets boolean hyperparameter value through AdminUtils. Mimics setting hyperparams
    """
    call = substrate.compose_call(
        call_module="AdminUtils",
        call_function=call_function,
        call_params={"netuid": netuid, "enabled": value},
    )
    extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)
    response = substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    return response.is_success


def sudo_set_hyperparameter_values(
    substrate: "SubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    call_params: dict,
    return_error_message: bool = False,
) -> Union[bool, tuple[bool, Optional[str]]]:
    """
    Sets liquid alpha values using AdminUtils. Mimics setting hyperparams
    """
    call = substrate.compose_call(
        call_module="AdminUtils",
        call_function=call_function,
        call_params=call_params,
    )
    extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)
    response = substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    if return_error_message:
        return response.is_success, response.error_message

    return response.is_success


async def wait_epoch(subtensor: "Subtensor", netuid: int = 1, times: int = 1):
    """
    Waits for the next epoch to start on a specific subnet.

    Queries the tempo value from the Subtensor module and calculates the
    interval based on the tempo. Then waits for the next epoch to start
    by monitoring the current block number.

    Raises:
        Exception: If the tempo cannot be determined from the chain.
    """
    q_tempo = [v for (k, v) in subtensor.query_map_subtensor("Tempo") if k == netuid]
    if len(q_tempo) == 0:
        raise Exception("could not determine tempo")
    tempo = q_tempo[0].value
    logging.info(f"tempo = {tempo}")
    await wait_interval(tempo * times, subtensor, netuid)


def next_tempo(current_block: int, tempo: int, netuid: int) -> int:
    """
    Calculates the next tempo block for a specific subnet.

    Args:
        current_block (int): The current block number.
        tempo (int): The tempo value for the subnet.
        netuid (int): The unique identifier of the subnet.

    Returns:
        int: The next tempo block number.
    """
    interval = tempo + 1
    last_epoch = current_block - 1 - (current_block + netuid + 1) % interval
    next_tempo_ = last_epoch + interval
    return next_tempo_


async def wait_interval(
    tempo: int,
    subtensor: "Subtensor",
    netuid: int = 1,
    reporting_interval: int = 1,
    sleep: float = 0.25,
):
    """
    Waits until the next tempo interval starts for a specific subnet.

    Calculates the next tempo block start based on the current block number
    and the provided tempo, then enters a loop where it periodically checks
    the current block number until the next tempo interval starts.
    """
    current_block = subtensor.get_current_block()
    next_tempo_block_start = next_tempo(current_block, tempo, netuid)
    last_reported = None

    while current_block < next_tempo_block_start:
        await asyncio.sleep(
            sleep,
        )  # Wait before checking the block number again
        current_block = subtensor.get_current_block()
        if last_reported is None or current_block - last_reported >= reporting_interval:
            last_reported = current_block
            print(
                f"Current Block: {current_block}  Next tempo for netuid {netuid} at: {next_tempo_block_start}"
            )
            logging.info(
                f"Current Block: {current_block}  Next tempo for netuid {netuid} at: {next_tempo_block_start}"
            )


# Helper to execute sudo wrapped calls on the chain
def sudo_set_admin_utils(
    substrate: "SubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    call_params: dict,
) -> tuple[bool, Optional[dict]]:
    """
    Wraps the call in sudo to set hyperparameter values using AdminUtils.

    Args:
        substrate (SubstrateInterface): Substrate connection.
        wallet (Wallet): Wallet object with the keypair for signing.
        call_function (str): The AdminUtils function to call.
        call_params (dict): Parameters for the AdminUtils function.

    Returns:
        tuple[bool, Optional[dict]]: (success status, error details).
    """
    inner_call = substrate.compose_call(
        call_module="AdminUtils",
        call_function=call_function,
        call_params=call_params,
    )

    sudo_call = substrate.compose_call(
        call_module="Sudo",
        call_function="sudo",
        call_params={"call": inner_call},
    )
    extrinsic = substrate.create_signed_extrinsic(
        call=sudo_call, keypair=wallet.coldkey
    )
    response: "ExtrinsicReceipt" = substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    return response.is_success, response.error_message


async def root_set_subtensor_hyperparameter_values(
    substrate: "SubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    call_params: dict,
    return_error_message: bool = False,
) -> tuple[bool, str]:
    """
    Sets liquid alpha values using AdminUtils. Mimics setting hyperparams
    """
    call = substrate.compose_call(
        call_module="SubtensorModule",
        call_function=call_function,
        call_params=call_params,
    )
    extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)

    response: "ExtrinsicReceipt" = substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    if return_error_message:
        return response.is_success, response.error_message

    return response.is_success, ""


def set_children(subtensor, wallet, netuid, children):
    return subtensor.sign_and_send_extrinsic(
        subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="set_children",
            call_params={
                "children": children,
                "hotkey": wallet.hotkey.ss58_address,
                "netuid": netuid,
            },
        ),
        wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )


def set_identity(
    subtensor,
    wallet,
    name="",
    url="",
    github_repo="",
    image="",
    discord="",
    description="",
    additional="",
):
    return subtensor.sign_and_send_extrinsic(
        subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="set_identity",
            call_params={
                "name": name,
                "url": url,
                "github_repo": github_repo,
                "image": image,
                "discord": discord,
                "description": description,
                "additional": additional,
            },
        ),
        wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )


def propose(subtensor, wallet, proposal, duration):
    return subtensor.sign_and_send_extrinsic(
        subtensor.substrate.compose_call(
            call_module="Triumvirate",
            call_function="propose",
            call_params={
                "proposal": proposal,
                "length_bound": len(proposal.data),
                "duration": duration,
            },
        ),
        wallet,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )


def vote(
    subtensor,
    wallet,
    hotkey,
    proposal,
    index,
    approve,
):
    return subtensor.sign_and_send_extrinsic(
        subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="vote",
            call_params={
                "approve": approve,
                "hotkey": hotkey,
                "index": index,
                "proposal": proposal,
            },
        ),
        wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
