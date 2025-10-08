from dataclasses import dataclass
from typing import TYPE_CHECKING

from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from bittensor.core.chain_data import DynamicInfo


@dataclass
class MoveStakeParams:
    @classmethod
    def move_stake(
        cls,
        origin_netuid: int,
        origin_hotkey_ss58: str,
        destination_netuid: int,
        destination_hotkey_ss58: str,
        amount: Balance,
    ) -> dict:
        """Returns the parameters for the `move_stake`."""
        return {
            "origin_netuid": origin_netuid,
            "origin_hotkey": origin_hotkey_ss58,
            "destination_netuid": destination_netuid,
            "destination_hotkey": destination_hotkey_ss58,
            "alpha_amount": amount.rao,
        }

    @classmethod
    def transfer_stake(
        cls,
        destination_coldkey_ss58: str,
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
    ) -> dict:
        """Returns the parameters for the `transfer_stake`."""
        return {
            "destination_coldkey": destination_coldkey_ss58,
            "hotkey": hotkey_ss58,
            "origin_netuid": origin_netuid,
            "destination_netuid": destination_netuid,
            "alpha_amount": amount.rao,
        }

    @classmethod
    def swap_stake(
        cls,
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
    ) -> dict:
        """Returns the parameters for the `swap_stake`."""
        return {
            "hotkey": hotkey_ss58,
            "origin_netuid": origin_netuid,
            "destination_netuid": destination_netuid,
            "alpha_amount": amount.rao,
        }

    @classmethod
    def swap_stake_limit(
        cls,
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        allow_partial_stake: bool,
        rate_tolerance: float,
        origin_pool: "DynamicInfo",
        destination_pool: "DynamicInfo",
    ) -> dict:
        """Returns the parameters for the `swap_stake_limit`."""
        call_params = cls.swap_stake(
            hotkey_ss58, origin_netuid, destination_netuid, amount
        )

        swap_rate_ratio = origin_pool.price.rao / destination_pool.price.rao
        swap_rate_ratio_with_tolerance = swap_rate_ratio * (1 + rate_tolerance)

        call_params.update(
            {
                "limit_price": swap_rate_ratio_with_tolerance,
                "allow_partial": allow_partial_stake,
            }
        )
        return call_params
