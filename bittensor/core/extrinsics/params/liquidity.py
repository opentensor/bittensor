from dataclasses import dataclass

from bittensor.utils.balance import Balance
from bittensor.utils.liquidity import price_to_tick


@dataclass
class LiquidityParams:
    @classmethod
    def add_liquidity(
        cls,
        netuid: int,
        hotkey_ss58: str,
        liquidity: Balance,
        price_low: Balance,
        price_high: Balance,
    ) -> dict:
        """Returns the parameters for the `add_liquidity`."""
        tick_low = price_to_tick(price_low.tao)
        tick_high = price_to_tick(price_high.tao)

        return {
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "tick_low": tick_low,
            "tick_high": tick_high,
            "liquidity": liquidity.rao,
        }

    @classmethod
    def modify_position(
        cls,
        netuid: int,
        hotkey_ss58: str,
        position_id: int,
        liquidity_delta: Balance,
    ) -> dict:
        """Returns the parameters for the `modify_position`."""
        return {
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "position_id": position_id,
            "liquidity_delta": liquidity_delta.rao,
        }

    @classmethod
    def remove_liquidity(
        cls,
        netuid: int,
        hotkey_ss58: str,
        position_id: int,
    ) -> dict:
        """Returns the parameters for the `remove_liquidity`."""
        return {
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "position_id": position_id,
        }

    @classmethod
    def toggle_user_liquidity(
        cls,
        netuid: int,
        enable: bool,
    ) -> dict:
        """Returns the parameters for the `toggle_user_liquidity`."""
        return {"netuid": netuid, "enable": enable}
