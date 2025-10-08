import functools
import time
from typing import TYPE_CHECKING

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

from bittensor.extras.dev_framework import *  # noqa: F401


if TYPE_CHECKING:
    from bittensor import Wallet
    from bittensor.extras import SubtensorApi


def get_dynamic_balance(rao: int, netuid: int = 0):
    """Returns a Balance object with the given rao and netuid for testing purposes with dynamic values."""
    return Balance.from_rao(rao).set_unit(netuid)


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
        call=subtensor.compose_call(
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
        wallet=wallet,
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
        call=await subtensor.compose_call(
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
        wallet=wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )


def propose(subtensor, wallet, proposal, duration):
    return subtensor.sign_and_send_extrinsic(
        call=subtensor.compose_call(
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


async def async_propose(
    subtensor: "SubtensorApi",
    wallet: "Wallet",
    proposal,
    duration,
):
    return await subtensor.sign_and_send_extrinsic(
        call=await subtensor.compose_call(
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
        call=subtensor.compose_call(
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


async def async_vote(
    subtensor: "SubtensorApi",
    wallet: "Wallet",
    hotkey,
    proposal,
    index,
    approve,
):
    return await subtensor.sign_and_send_extrinsic(
        call=await subtensor.compose_call(
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
