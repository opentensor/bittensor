from dataclasses import dataclass
from typing import TYPE_CHECKING

from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from bittensor.core.chain_data import DynamicInfo


@dataclass
class StakingParams:
    @classmethod
    def add_stake(
        cls,
        netuid: int,
        hotkey_ss58: str,
        amount: Balance,
    ) -> dict:
        """Returns the parameters for the `safe` parameters."""
        return {
            "netuid": netuid,
            "hotkey": hotkey_ss58,
            "amount_staked": amount.rao,
        }

    @classmethod
    def add_stake_limit(
        cls,
        netuid: int,
        hotkey_ss58: str,
        amount: "Balance",
        allow_partial_stake: bool,
        rate_tolerance: float,
        pool: "DynamicInfo",
    ) -> dict:
        """Returns the parameters for the `add_stake_limit`."""
        call_params = cls.add_stake(netuid, hotkey_ss58, amount)

        base_price = pool.price.tao
        price_with_tolerance = (
            base_price if pool.netuid == 0 else base_price * (1 + rate_tolerance)
        )
        limit_price = Balance.from_tao(price_with_tolerance).rao

        call_params.update(
            {"limit_price": limit_price, "allow_partial": allow_partial_stake}
        )
        return call_params

    @classmethod
    def set_coldkey_auto_stake_hotkey(
        cls,
        netuid: int,
        hotkey_ss58: str,
    ) -> dict:
        """Returns the parameters for the `set_auto_stake_extrinsic`."""
        return {"hotkey": hotkey_ss58, "netuid": netuid}
