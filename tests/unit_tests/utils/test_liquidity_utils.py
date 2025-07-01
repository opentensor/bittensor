import math

import pytest

from bittensor.utils.balance import Balance
from bittensor.utils.liquidity import (
    LiquidityPosition,
    price_to_tick,
    tick_to_price,
    get_fees,
    get_fees_in_range,
    calculate_fees,
)


def test_liquidity_position_to_token_amounts():
    """Test conversion of liquidity position to token amounts."""
    # Preps
    pos = LiquidityPosition(
        id=1,
        price_low=Balance.from_tao(10000),
        price_high=Balance.from_tao(40000),
        liquidity=Balance.from_tao(25000),
        fees_tao=Balance.from_tao(0),
        fees_alpha=Balance.from_tao(0),
        netuid=1,
    )
    current_price = Balance.from_tao(20000)
    # Call
    alpha, tao = pos.to_token_amounts(current_price)
    # Asserts
    assert isinstance(alpha, Balance)
    assert isinstance(tao, Balance)
    assert alpha.rao >= 0 and tao.rao >= 0


def test_price_to_tick_and_back():
    """Test price to tick conversion and back."""
    # Preps
    price = 1.25
    # Call
    tick = price_to_tick(price)
    restored_price = tick_to_price(tick)
    # Asserts
    assert math.isclose(restored_price, price, rel_tol=1e-3)


def test_price_to_tick_invalid():
    """Test price to tick conversion with invalid input."""
    with pytest.raises(ValueError):
        price_to_tick(0)


def test_tick_to_price_invalid():
    """Test tick to price conversion with invalid input."""
    with pytest.raises(ValueError):
        tick_to_price(1_000_000)


def test_get_fees_above_true():
    """Test fee calculation for above position."""
    # Preps
    tick = {
        "liquidity_net": 1000000000000,
        "liquidity_gross": 1000000000000,
        "fees_out_tao": {"bits": 0},
        "fees_out_alpha": {"bits": 0},
    }
    # Call
    result = get_fees(
        current_tick=100,
        tick=tick,
        tick_index=90,
        quote=True,
        global_fees_tao=8000,
        global_fees_alpha=6000,
        above=True,
    )
    # Asserts
    assert result == 8000


def test_get_fees_in_range():
    """Test fee calculation within a range."""
    # Call
    value = get_fees_in_range(
        quote=True,
        global_fees_tao=10000,
        global_fees_alpha=5000,
        fees_below_low=2000,
        fees_above_high=1000,
    )
    # Asserts
    assert value == 7000


def test_calculate_fees():
    """Test calculation of fees for a position."""
    # Preps
    position = {
        "id": (2,),
        "netuid": 2,
        "tick_low": (206189,),
        "tick_high": (208196,),
        "liquidity": 1000000000000,
        "fees_tao": {"bits": 0},
        "fees_alpha": {"bits": 0},
    }
    # Call
    result = calculate_fees(
        position=position,
        global_fees_tao=5000,
        global_fees_alpha=8000,
        tao_fees_below_low=1000,
        tao_fees_above_high=1000,
        alpha_fees_below_low=2000,
        alpha_fees_above_high=1000,
        netuid=1,
    )
    # Asserts
    assert isinstance(result[0], Balance)
    assert isinstance(result[1], Balance)
    assert result[0].rao > 0
    assert result[1].rao > 0
