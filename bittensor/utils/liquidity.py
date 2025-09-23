"""
This module provides utilities for managing liquidity positions and price conversions in the Bittensor network. The
module handles conversions between TAO and Alpha tokens while maintaining precise calculations for liquidity
provisioning and fee distribution.
"""

import math
from typing import Any
from dataclasses import dataclass

from bittensor.utils.balance import Balance, fixed_to_float

# These three constants are unchangeable at the level of Uniswap math
MIN_TICK = -887272
MAX_TICK = 887272
PRICE_STEP = 1.0001


@dataclass
class LiquidityPosition:
    id: int
    price_low: Balance  # RAO
    price_high: Balance  # RAO
    liquidity: Balance  # TAO + ALPHA (sqrt by TAO balance * Alpha Balance -> math under the hood)
    fees_tao: Balance  # RAO
    fees_alpha: Balance  # RAO
    netuid: int

    def to_token_amounts(
        self, current_subnet_price: Balance
    ) -> tuple[Balance, Balance]:
        """Convert a position to token amounts.

        Parameters:
            current_subnet_price: current subnet price in Alpha.

        Returns:
            tuple[int, int]:
                Amount of Alpha in liquidity
                Amount of TAO in liquidity

        Liquidity is a combination of TAO and Alpha depending on the price of the subnet at the moment.
        """
        sqrt_price_low = math.sqrt(self.price_low)
        sqrt_price_high = math.sqrt(self.price_high)
        sqrt_current_subnet_price = math.sqrt(current_subnet_price)

        if sqrt_current_subnet_price < sqrt_price_low:
            amount_alpha = self.liquidity * (1 / sqrt_price_low - 1 / sqrt_price_high)
            amount_tao = 0
        elif sqrt_current_subnet_price > sqrt_price_high:
            amount_alpha = 0
            amount_tao = self.liquidity * (sqrt_price_high - sqrt_price_low)
        else:
            amount_alpha = self.liquidity * (
                1 / sqrt_current_subnet_price - 1 / sqrt_price_high
            )
            amount_tao = self.liquidity * (sqrt_current_subnet_price - sqrt_price_low)
        return Balance.from_rao(int(amount_alpha), self.netuid), Balance.from_rao(
            int(amount_tao)
        )


def price_to_tick(price: float) -> int:
    """Converts a float price to the nearest Uniswap V3 tick index."""
    if price <= 0:
        raise ValueError(f"Price must be positive, got `{price}`.")

    tick = int(math.log(price) / math.log(PRICE_STEP))

    if not (MIN_TICK <= tick <= MAX_TICK):
        raise ValueError(
            f"Resulting tick {tick} is out of allowed range ({MIN_TICK} to {MAX_TICK})"
        )
    return tick


def tick_to_price(tick: int) -> float:
    """Convert an integer Uniswap V3 tick index to float price."""
    if not (MIN_TICK <= tick <= MAX_TICK):
        raise ValueError("Tick is out of allowed range")
    return PRICE_STEP**tick


def get_fees(
    current_tick: int,
    tick: dict,
    tick_index: int,
    quote: bool,
    global_fees_tao: float,
    global_fees_alpha: float,
    above: bool,
) -> float:
    """Returns the liquidity fee."""
    tick_fee_key = "fees_out_tao" if quote else "fees_out_alpha"
    tick_fee_value = fixed_to_float(tick.get(tick_fee_key))
    global_fee_value = global_fees_tao if quote else global_fees_alpha

    if above:
        return (
            global_fee_value - tick_fee_value
            if tick_index <= current_tick
            else tick_fee_value
        )
    return (
        tick_fee_value
        if tick_index <= current_tick
        else global_fee_value - tick_fee_value
    )


def get_fees_in_range(
    quote: bool,
    global_fees_tao: float,
    global_fees_alpha: float,
    fees_below_low: float,
    fees_above_high: float,
) -> float:
    """Returns the liquidity fee value in a range."""
    global_fees = global_fees_tao if quote else global_fees_alpha
    return global_fees - fees_below_low - fees_above_high


# Calculate fees for a position
def calculate_fees(
    position: dict[str, Any],
    global_fees_tao: float,
    global_fees_alpha: float,
    tao_fees_below_low: float,
    tao_fees_above_high: float,
    alpha_fees_below_low: float,
    alpha_fees_above_high: float,
    netuid: int,
) -> tuple[Balance, Balance]:
    fee_tao_agg = get_fees_in_range(
        quote=True,
        global_fees_tao=global_fees_tao,
        global_fees_alpha=global_fees_alpha,
        fees_below_low=tao_fees_below_low,
        fees_above_high=tao_fees_above_high,
    )

    fee_alpha_agg = get_fees_in_range(
        quote=False,
        global_fees_tao=global_fees_tao,
        global_fees_alpha=global_fees_alpha,
        fees_below_low=alpha_fees_below_low,
        fees_above_high=alpha_fees_above_high,
    )

    fee_tao = fee_tao_agg - fixed_to_float(position["fees_tao"])
    fee_alpha = fee_alpha_agg - fixed_to_float(position["fees_alpha"])
    liquidity_frac = position["liquidity"]

    fee_tao = liquidity_frac * fee_tao
    fee_alpha = liquidity_frac * fee_alpha

    return Balance.from_rao(int(fee_tao)), Balance.from_rao(int(fee_alpha), netuid)
