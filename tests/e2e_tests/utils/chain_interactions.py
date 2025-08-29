"""
This module provides functions interacting with the chain for end-to-end testing;
these are not present in btsdk but are required for e2e tests
"""

import asyncio
import functools
import time
from typing import Union, Optional, TYPE_CHECKING

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

# for typing purposes
if TYPE_CHECKING:
    from bittensor import Wallet
    from bittensor.core.subtensor_api import SubtensorApi
    from async_substrate_interface import (
        AsyncSubstrateInterface,
        AsyncExtrinsicReceipt,
        SubstrateInterface,
        ExtrinsicReceipt,
    )


def get_dynamic_balance(rao: int, netuid: int = 0):
    """Returns a Balance object with the given rao and netuid for testing purposes with dynamic values."""
    return Balance.from_rao(rao).set_unit(netuid)


def sudo_set_hyperparameter_bool(
    substrate: "SubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    value: bool,
    netuid: int,
) -> bool:
    """Sets boolean hyperparameter value through AdminUtils. Mimics setting hyperparams."""
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


async def async_sudo_set_hyperparameter_bool(
    substrate: "AsyncSubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    value: bool,
    netuid: int,
) -> bool:
    """Sets boolean hyperparameter value through AdminUtils. Mimics setting hyperparams."""
    call = await substrate.compose_call(
        call_module="AdminUtils",
        call_function=call_function,
        call_params={"netuid": netuid, "enabled": value},
    )
    extrinsic = await substrate.create_signed_extrinsic(
        call=call, keypair=wallet.coldkey
    )
    response = await substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    return await response.is_success


def sudo_set_hyperparameter_values(
    substrate: "SubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    call_params: dict,
    return_error_message: bool = False,
) -> Union[bool, tuple[bool, Optional[str]]]:
    """Sets liquid alpha values using AdminUtils. Mimics setting hyperparams."""
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


async def wait_epoch(subtensor: "SubtensorApi", netuid: int = 1, **kwargs):
    """
    Waits for the next epoch to start on a specific subnet.

    Queries the tempo value from the Subtensor module and calculates the
    interval based on the tempo. Then waits for the next epoch to start
    by monitoring the current block number.

    Raises:
        Exception: If the tempo cannot be determined from the chain.
    """
    q_tempo = [
        v for (k, v) in subtensor.queries.query_map_subtensor("Tempo") if k == netuid
    ]
    if len(q_tempo) == 0:
        raise Exception("could not determine tempo")
    tempo = q_tempo[0].value
    logging.info(f"tempo = {tempo}")
    await wait_interval(tempo, subtensor, netuid, **kwargs)


async def async_wait_epoch(async_subtensor: "SubtensorApi", netuid: int = 1, **kwargs):
    """
    Waits for the next epoch to start on a specific subnet.

    Queries the tempo value from the Subtensor module and calculates the
    interval based on the tempo. Then waits for the next epoch to start
    by monitoring the current block number.

    Raises:
        Exception: If the tempo cannot be determined from the chain.
    """
    q_tempo = [
        v
        async for (k, v) in await async_subtensor.queries.query_map_subtensor("Tempo")
        if k == netuid
    ]
    if len(q_tempo) == 0:
        raise Exception("could not determine tempo")
    tempo = q_tempo[0].value
    logging.info(f"tempo = {tempo}")
    await async_wait_interval(tempo, async_subtensor, netuid, **kwargs)


def next_tempo(current_block: int, tempo: int) -> int:
    """
    Calculates the next tempo block for a specific subnet.

    Args:
        current_block: The current block number.
        tempo: The tempo value for the subnet.

    Returns:
        int: The next tempo block number.
    """
    return ((current_block // tempo) + 1) * tempo + 1


async def wait_interval(
    tempo: int,
    subtensor: "SubtensorApi",
    netuid: int = 1,
    reporting_interval: int = 1,
    sleep: float = 0.25,
    times: int = 1,
):
    """
    Waits until the next tempo interval starts for a specific subnet.

    Calculates the next tempo block start based on the current block number
    and the provided tempo, then enters a loop where it periodically checks
    the current block number until the next tempo interval starts.
    """
    current_block = subtensor.chain.get_current_block()
    next_tempo_block_start = current_block

    for _ in range(times):
        next_tempo_block_start = next_tempo(next_tempo_block_start, tempo)

    last_reported = None

    while current_block < next_tempo_block_start:
        await asyncio.sleep(
            sleep,
        )  # Wait before checking the block number again
        current_block = subtensor.chain.get_current_block()
        if last_reported is None or current_block - last_reported >= reporting_interval:
            last_reported = current_block
            print(
                f"Current Block: {current_block}  Next tempo for netuid {netuid} at: {next_tempo_block_start}"
            )
            logging.info(
                f"Current Block: {current_block}  Next tempo for netuid {netuid} at: {next_tempo_block_start}"
            )


async def async_wait_interval(
    tempo: int,
    subtensor: "SubtensorApi",
    netuid: int = 1,
    reporting_interval: int = 1,
    sleep: float = 0.25,
    times: int = 1,
):
    """
    Waits until the next tempo interval starts for a specific subnet.

    Calculates the next tempo block start based on the current block number
    and the provided tempo, then enters a loop where it periodically checks
    the current block number until the next tempo interval starts.
    """
    current_block = await subtensor.chain.get_current_block()
    next_tempo_block_start = current_block

    for _ in range(times):
        next_tempo_block_start = next_tempo(next_tempo_block_start, tempo)

    last_reported = None

    while current_block < next_tempo_block_start:
        await asyncio.sleep(
            sleep,
        )  # Wait before checking the block number again
        current_block = await subtensor.chain.get_current_block()
        if last_reported is None or current_block - last_reported >= reporting_interval:
            last_reported = current_block
            print(
                f"Current Block: {current_block}  Next tempo for netuid {netuid} at: {next_tempo_block_start}"
            )
            logging.info(
                f"Current Block: {current_block}  Next tempo for netuid {netuid} at: {next_tempo_block_start}"
            )


def execute_and_wait_for_next_nonce(
    subtensor: "SubtensorApi", wallet, sleep=0.25, timeout=60.0, max_retries=3
):
    """Decorator that ensures the nonce has been consumed after a blockchain extrinsic call."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                start_nonce = subtensor.substrate.get_account_next_index(
                    wallet.hotkey.ss58_address
                )

                result = func(*args, **kwargs)

                start_time = time.time()

                while time.time() - start_time < timeout:
                    current_nonce = subtensor.substrate.get_account_next_index(
                        wallet.hotkey.ss58_address
                    )

                    if current_nonce != start_nonce:
                        logging.console.info(
                            f"✅ Nonce changed from {start_nonce} to {current_nonce}"
                        )
                        return result

                    logging.console.info(
                        f"⏳ Waiting for nonce increment. Current: {current_nonce}"
                    )
                    time.sleep(sleep)

                logging.warning(
                    f"⚠️ Attempt {attempt + 1}/{max_retries}: Nonce did not increment."
                )
            raise TimeoutError(f"❌ Nonce did not change after {max_retries} attempts.")

        return wrapper

    return decorator


# Helper to execute sudo wrapped calls on the chain
def sudo_set_admin_utils(
    substrate: "SubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    call_params: dict,
    call_module: str = "AdminUtils",
) -> tuple[bool, Optional[dict]]:
    """
    Wraps the call in sudo to set hyperparameter values using AdminUtils.

    Args:
        substrate: Substrate connection.
        wallet: Wallet object with the keypair for signing.
        call_function: The AdminUtils function to call.
        call_params: Parameters for the AdminUtils function.
        call_module: The AdminUtils module to call. Defaults to "AdminUtils".

    Returns:
        tuple: (success status, error details).
    """
    inner_call = substrate.compose_call(
        call_module=call_module,
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


async def async_sudo_set_admin_utils(
    substrate: "AsyncSubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    call_params: dict,
    call_module: str = "AdminUtils",
) -> tuple[bool, Optional[dict]]:
    """
    Wraps the call in sudo to set hyperparameter values using AdminUtils.

    Parameters:
        substrate: Substrate connection.
        wallet: Wallet object with the keypair for signing.
        call_function: The AdminUtils function to call.
        call_params: Parameters for the AdminUtils function.
        call_module: The AdminUtils module to call. Defaults to "AdminUtils".

    Returns:
        tuple: (success status, error details).
    """
    inner_call = await substrate.compose_call(
        call_module=call_module,
        call_function=call_function,
        call_params=call_params,
    )

    sudo_call = await substrate.compose_call(
        call_module="Sudo",
        call_function="sudo",
        call_params={"call": inner_call},
    )
    extrinsic = await substrate.create_signed_extrinsic(
        call=sudo_call, keypair=wallet.coldkey
    )
    response: "AsyncExtrinsicReceipt" = await substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    return await response.is_success, await response.error_message


def root_set_subtensor_hyperparameter_values(
    substrate: "SubstrateInterface",
    wallet: "Wallet",
    call_function: str,
    call_params: dict,
) -> tuple[bool, Optional[dict]]:
    """Sets liquid alpha values using AdminUtils. Mimics setting hyperparams."""
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

    return response.is_success, response.error_message


def set_identity(
    subtensor: "SubtensorApi",
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


async def async_set_identity(
    subtensor: "SubtensorApi",
    wallet: "Wallet",
    name="",
    url="",
    github_repo="",
    image="",
    discord="",
    description="",
    additional="",
):
    return await subtensor.sign_and_send_extrinsic(
        await subtensor.substrate.compose_call(
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


async def async_propose(
    subtensor: "SubtensorApi",
    wallet: "Wallet",
    proposal,
    duration,
):
    return await subtensor.sign_and_send_extrinsic(
        call=await subtensor.substrate.compose_call(
            call_module="Triumvirate",
            call_function="propose",
            call_params={
                "proposal": proposal,
                "length_bound": len(proposal.data),
                "duration": duration,
            },
        ),
        wallet=wallet,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )


def vote(
    subtensor: "SubtensorApi",
    wallet: "Wallet",
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


async def async_vote(
    subtensor: "SubtensorApi",
    wallet: "Wallet",
    hotkey,
    proposal,
    index,
    approve,
):
    return await subtensor.sign_and_send_extrinsic(
        call=await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="vote",
            call_params={
                "approve": approve,
                "hotkey": hotkey,
                "index": index,
                "proposal": proposal,
            },
        ),
        wallet=wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
