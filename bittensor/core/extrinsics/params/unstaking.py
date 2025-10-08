from dataclasses import dataclass
from typing import Optional

from bittensor.core.chain_data import DynamicInfo
from bittensor.utils.balance import Balance


@dataclass
class UnstakingParams:
    @classmethod
    def remove_stake(cls, netuid: int, hotkey_ss58: str, amount: "Balance") -> dict:
        """Returns the parameters for the `remove_stake`."""
        return {"netuid": netuid, "hotkey": hotkey_ss58, "amount_unstaked": amount.rao}

    @classmethod
    def remove_stake_limit(
        cls,
        netuid: int,
        hotkey_ss58: str,
        amount: "Balance",
        allow_partial_stake: bool,
        rate_tolerance: float,
        pool: "DynamicInfo",
    ) -> dict:
        """Returns the parameters for the `remove_stake_limit`."""
        call_params = cls.remove_stake(netuid, hotkey_ss58, amount)

        base_price = pool.price.tao

        if pool.netuid == 0:
            price_with_tolerance = base_price
        else:
            price_with_tolerance = base_price * (1 - rate_tolerance)

        limit_price = Balance.from_tao(price_with_tolerance).rao
        call_params.update(
            {
                "limit_price": limit_price,
                "allow_partial": allow_partial_stake,
            }
        )
        return call_params

    @classmethod
    def remove_stake_full_limit(
        cls,
        netuid: int,
        hotkey_ss58: str,
        rate_tolerance: Optional[float] = None,
        pool: Optional["DynamicInfo"] = None,
    ) -> dict:
        """Returns the parameters for the `remove_stake_full_limit`."""
        call_params = {
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "limit_price": None,
        }

        if rate_tolerance:
            limit_price = pool.price * (1 - rate_tolerance)
            call_params.update({"limit_price": limit_price})

        return call_params
