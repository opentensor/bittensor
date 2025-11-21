from bittensor_wallet import Wallet

from bittensor.core.extrinsics.utils import sudo_call_extrinsic
from bittensor.core.extrinsics.asyncex.utils import (
    sudo_call_extrinsic as async_sudo_call_extrinsic,
)
from bittensor.extras.subtensor_api import SubtensorApi


def check_root_sell_flag(subtensor: "SubtensorApi") -> tuple[bool, int]:
    """Checks the current state of root_sell_flag."""
    # Get all non-root subnets
    subnets = [
        netuid for netuid in subtensor.subnets.get_all_subnets_netuid() if netuid != 0
    ]
    total_ema = 0
    for netuid in subnets:
        # Get moving_alpha_price through subnet's Dynamic info data
        moving_price = subtensor.subnets.subnet(netuid).moving_price
        total_ema += moving_price
    root_sell_flag = total_ema > 1.0
    return root_sell_flag, total_ema


async def async_check_root_sell_flag(subtensor: "SubtensorApi") -> tuple[bool, int]:
    """Checks the current state of root_sell_flag."""
    # Get all non-root subnets
    subnets = [
        netuid
        for netuid in await subtensor.subnets.get_all_subnets_netuid()
        if netuid != 0
    ]
    total_ema = 0
    for netuid in subnets:
        # Get moving_alpha_price through subnet's Dynamic info data
        moving_price = (await subtensor.subnets.subnet(netuid)).moving_price
        total_ema += moving_price
    root_sell_flag = total_ema > 1.0
    return root_sell_flag, total_ema


def set_huge_ema_before_test(subtensor: "SubtensorApi", sudo_wallet: "Wallet"):
    """Sets huge EMA value into SubnetMovingPrice storage."""

    encoded_storage_key = (
        "0x658faa385070e074c85bf6b568cf05551abf1b0f4fd14f7b72ee50f9d91d59150200"
    )
    encoded_storage_value = b"003665c4ffc99a3b0000000000000000"
    items = [encoded_storage_key, encoded_storage_value]

    return sudo_call_extrinsic(
        subtensor=subtensor.inner_subtensor,
        wallet=sudo_wallet,
        call_function="set_storage",
        call_params={"items": [items]},
        call_module="System",
    )


async def async_set_huge_ema_before_test(
    subtensor: "SubtensorApi", sudo_wallet: "Wallet"
):
    """Sets huge EMA value into SubnetMovingPrice storage."""

    encoded_storage_key = (
        "0x658faa385070e074c85bf6b568cf05551abf1b0f4fd14f7b72ee50f9d91d59150200"
    )
    encoded_storage_value = b"003665c4ffc99a3b0000000000000000"
    items = [encoded_storage_key, encoded_storage_value]

    return await async_sudo_call_extrinsic(
        subtensor=subtensor.inner_subtensor,
        wallet=sudo_wallet,
        call_function="set_storage",
        call_params={"items": [items]},
        call_module="System",
    )


def increase_subnet_ema(subtensor: "SubtensorApi", sudo_wallet: "Wallet") -> bool:
    """Increases EMA value in SubnetMovingPrice storage."""
    try:
        root_sell_flag, sns_sum_of_ema = check_root_sell_flag(subtensor)
        assert not root_sell_flag
        assert sns_sum_of_ema < 1.0

        response = set_huge_ema_before_test(
            subtensor=subtensor, sudo_wallet=sudo_wallet
        )
        assert response.success, response.message
        subtensor.wait_for_block()

        root_sell_flag, sns_sum_of_ema = check_root_sell_flag(subtensor)
        assert root_sell_flag, "Root sell still false"
        assert sns_sum_of_ema > 1.0, "SNs EMA sum wasn't increased greater than 1.0"
        return True
    except AssertionError:
        return False


async def async_increase_subnet_ema(
    subtensor: "SubtensorApi", sudo_wallet: "Wallet"
) -> bool:
    """Increases EMA value in SubnetMovingPrice storage."""
    try:
        root_sell_flag, sns_sum_of_ema = await async_check_root_sell_flag(subtensor)
        assert not root_sell_flag
        assert sns_sum_of_ema < 1.0

        response = await async_set_huge_ema_before_test(
            subtensor=subtensor, sudo_wallet=sudo_wallet
        )
        assert response.success, response.message
        await subtensor.wait_for_block()

        root_sell_flag, sns_sum_of_ema = await async_check_root_sell_flag(subtensor)
        assert root_sell_flag, "Root sell still false"
        assert sns_sum_of_ema > 1.0, "SNs EMA sum wasn't increased greater than 1.0"
        return True
    except AssertionError:
        return False
