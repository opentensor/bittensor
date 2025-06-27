import math
from typing import Any

from bittensor.utils.balance import Balance, fixed_to_float

MIN_TICK = -887272
MAX_TICK = 887272
PRICE_STEP = 1.0001


def price_to_tick(price: float) -> int:
    """Converts a float price to the nearest Uniswap V3 tick index."""
    if price <= 0:
        raise ValueError(f"Price must be positive, got {price}")

    log_base = math.log1p(PRICE_STEP - 1)  # safer for small deltas
    tick = round(math.log(price) / log_base)

    if tick < MIN_TICK or tick > MAX_TICK:
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
    else:
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
) -> list[Balance]:
    """Calculate fees from position and fees values."""
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

    return [Balance.from_rao(fee_tao), Balance.from_rao(fee_alpha, netuid)]
