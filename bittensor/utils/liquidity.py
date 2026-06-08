"""Deprecated. User liquidity (Uniswap V3) has been permanently removed from the chain."""

# TODO: remove this module in the next major release (include all references)
from dataclasses import dataclass

from bittensor.utils import ChainFeatureDisabledWarning, deprecated_message
from bittensor.utils.balance import Balance

_DEPRECATED_MSG = (
    "User liquidity (Uniswap V3) has been permanently removed from the chain. "
    "The swap mechanism has been replaced by the Balancer swap."
)


@dataclass
class LiquidityPosition:
    """Deprecated. User liquidity positions no longer exist on-chain."""

    id: int
    price_low: Balance
    price_high: Balance
    liquidity: Balance
    fees_tao: Balance
    fees_alpha: Balance
    netuid: int

    def __post_init__(self):
        deprecated_message(
            message=_DEPRECATED_MSG,
            category=ChainFeatureDisabledWarning,
            stacklevel=3,
        )

    def to_token_amounts(
        self, current_subnet_price: Balance
    ) -> tuple[Balance, Balance]:
        """Deprecated."""
        deprecated_message(
            message=_DEPRECATED_MSG,
            category=ChainFeatureDisabledWarning,
            stacklevel=2,
        )
        return Balance.from_rao(0, self.netuid), Balance.from_rao(0)


def price_to_tick(price: float) -> int:
    """Deprecated."""
    deprecated_message(
        message=_DEPRECATED_MSG,
        category=ChainFeatureDisabledWarning,
        stacklevel=3,
    )
    return 0


def tick_to_price(tick: int) -> float:
    """Deprecated."""
    deprecated_message(
        message=_DEPRECATED_MSG,
        category=ChainFeatureDisabledWarning,
        stacklevel=3,
    )
    return 0.0


def get_fees(
    current_tick: int,
    tick: dict,
    tick_index: int,
    quote: bool,
    global_fees_tao: float,
    global_fees_alpha: float,
    above: bool,
) -> float:
    """Deprecated."""
    deprecated_message(
        message=_DEPRECATED_MSG,
        category=ChainFeatureDisabledWarning,
        stacklevel=3,
    )
    return 0.0


def get_fees_in_range(
    quote: bool,
    global_fees_tao: float,
    global_fees_alpha: float,
    fees_below_low: float,
    fees_above_high: float,
) -> float:
    """Deprecated."""
    deprecated_message(
        message=_DEPRECATED_MSG,
        category=ChainFeatureDisabledWarning,
        stacklevel=3,
    )
    return 0.0


def calculate_fees(
    position: dict,
    global_fees_tao: float,
    global_fees_alpha: float,
    tao_fees_below_low: float,
    tao_fees_above_high: float,
    alpha_fees_below_low: float,
    alpha_fees_above_high: float,
    netuid: int,
) -> tuple[Balance, Balance]:
    """Deprecated."""
    deprecated_message(
        message=_DEPRECATED_MSG,
        category=ChainFeatureDisabledWarning,
        stacklevel=3,
    )
    return Balance.from_rao(0), Balance.from_rao(0, netuid)
