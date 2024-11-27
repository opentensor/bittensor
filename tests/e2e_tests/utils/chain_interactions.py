"""
This module provides functions interacting with the chain for end-to-end testing;
these are not present in btsdk but are required for e2e tests
"""

import asyncio
from typing import Union, Optional, TYPE_CHECKING

from bittensor.utils.btlogging import logging

# for typing purposes
if TYPE_CHECKING:
    from bittensor import Wallet
    from bittensor.core.subtensor import Subtensor
    from bittensor.utils.balance import Balance
    from substrateinterface import SubstrateInterface


def sudo_set_hyperparameter_bool(
    substrate: "SubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    value: bool,
    netuid: int,
    requires_sudo: bool = False,
) -> bool:
    """
    Sets boolean hyperparameter value through AdminUtils. Mimics setting hyperparams
    """
    call = substrate.compose_call(
        call_module="AdminUtils",
        call_function=call_function,
        call_params={"netuid": netuid, "enabled": value},
    )
    if requires_sudo:  # Only sudo can set these values
        call = substrate.compose_call(
            call_module="Sudo",
            call_function="sudo",
            call_params={"call": call},
        )

    extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)
    response = substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    response.process_events()
    return response.is_success


def sudo_set_hyperparameter_values(
    substrate: "SubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    call_params: dict,
    return_error_message: bool = False,
    requires_sudo: bool = False,
) -> Union[bool, tuple[bool, Optional[str]]]:
    """
    Sets liquid alpha values using AdminUtils. Mimics setting hyperparams
    """
    call = substrate.compose_call(
        call_module="AdminUtils",
        call_function=call_function,
        call_params=call_params,
    )
    if requires_sudo:  # Only sudo can set these values
        call = substrate.compose_call(
            call_module="Sudo",
            call_function="sudo",
            call_params={"call": call},
        )

    extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)
    response = substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    response.process_events()

    if return_error_message:
        return response.is_success, response.error_message

    return response.is_success


def add_stake(
    substrate: "SubstrateInterface", wallet: "Wallet", amount: "Balance"
) -> bool:
    """
    Adds stake to a hotkey using SubtensorModule. Mimics command of adding stake
    """
    stake_call = substrate.compose_call(
        call_module="SubtensorModule",
        call_function="add_stake",
        call_params={"hotkey": wallet.hotkey.ss58_address, "amount_staked": amount.rao},
    )
    extrinsic = substrate.create_signed_extrinsic(
        call=stake_call, keypair=wallet.coldkey
    )
    response = substrate.submit_extrinsic(
        extrinsic, wait_for_finalization=True, wait_for_inclusion=True
    )
    response.process_events()
    return response.is_success


def register_subnet(substrate: "SubstrateInterface", wallet: "Wallet") -> bool:
    """
    Registers a subnet on the chain using wallet. Mimics register subnet command.
    """
    register_call = substrate.compose_call(
        call_module="SubtensorModule",
        call_function="register_network",
        call_params={"immunity_period": 0, "reg_allowed": True},
    )
    extrinsic = substrate.create_signed_extrinsic(
        call=register_call, keypair=wallet.coldkey
    )
    response = substrate.submit_extrinsic(
        extrinsic, wait_for_finalization=True, wait_for_inclusion=True
    )
    response.process_events()
    return response.is_success


async def wait_epoch(subtensor: "Subtensor", netuid: int = 1):
    """
    Waits for the next epoch to start on a specific subnet.

    Queries the tempo value from the Subtensor module and calculates the
    interval based on the tempo. Then waits for the next epoch to start
    by monitoring the current block number.

    Raises:
        Exception: If the tempo cannot be determined from the chain.
    """
    q_tempo = [
        v.value
        for [k, v] in subtensor.query_map_subtensor("Tempo")
        if k.value == netuid
    ]
    if len(q_tempo) == 0:
        raise Exception("could not determine tempo")
    tempo = q_tempo[0]
    logging.info(f"tempo = {tempo}")
    await wait_interval(tempo, subtensor, netuid)


async def wait_interval(tempo: int, subtensor: "Subtensor", netuid: int = 1):
    """
    Waits until the next tempo interval starts for a specific subnet.

    Calculates the next tempo block start based on the current block number
    and the provided tempo, then enters a loop where it periodically checks
    the current block number until the next tempo interval starts.
    """
    interval = tempo + 1
    current_block = subtensor.get_current_block()
    last_epoch = current_block - 1 - (current_block + netuid + 1) % interval
    next_tempo_block_start = last_epoch + interval
    last_reported = None

    while current_block < next_tempo_block_start:
        await asyncio.sleep(
            1
        )  # Wait for 1 second before checking the block number again
        current_block = subtensor.get_current_block()
        if last_reported is None or current_block - last_reported >= 10:
            last_reported = current_block
            print(
                f"Current Block: {current_block}  Next tempo for netuid {netuid} at: {next_tempo_block_start}"
            )
            logging.info(
                f"Current Block: {current_block}  Next tempo for netuid {netuid} at: {next_tempo_block_start}"
            )


async def wait_until_block(
    block: int, subtensor: "Subtensor", block_time: int = 250e-3
):
    """
    Waits until a specific block is reached.

    Will return immediately if the block has already been reached.
    """
    current_block = subtensor.get_current_block()

    while current_block < block:
        wait_time = (block - current_block) * block_time
        print(f"Waiting for {wait_time} seconds until block {block}")
        logging.info(f"Waiting for {wait_time} seconds until block {block}")
        await asyncio.sleep(
            wait_time
        )  # Wait for 1 second before checking the block number again
        current_block = subtensor.get_current_block()
