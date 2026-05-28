# TODO: remove this module in the next major release (include all references)
from typing import Optional, TYPE_CHECKING

from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import ChainFeatureDisabledWarning, deprecated_message
from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor

_DEPRECATED_MSG = (
    "User liquidity (Uniswap v3) has been permanently removed from the chain. The swap mechanism has been replaced by "
    "the Balancer swap. This extrinsic is deprecated and will return an error."
)


async def add_liquidity_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    liquidity: Optional[Balance] = None,
    price_low: Optional[Balance] = None,
    price_high: Optional[Balance] = None,
    hotkey_ss58: Optional[str] = None,
    **kwargs,
) -> ExtrinsicResponse:
    """Deprecated. User liquidity has been permanently removed from the chain."""
    deprecated_message(
        message=_DEPRECATED_MSG,
        category=ChainFeatureDisabledWarning,
        stacklevel=3,
    )
    return ExtrinsicResponse(success=False, message=_DEPRECATED_MSG)


async def modify_liquidity_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    position_id: Optional[int] = None,
    liquidity_delta: Optional[Balance] = None,
    hotkey_ss58: Optional[str] = None,
    **kwargs,
) -> ExtrinsicResponse:
    """Deprecated. User liquidity has been permanently removed from the chain."""
    deprecated_message(
        message=_DEPRECATED_MSG,
        category=ChainFeatureDisabledWarning,
        stacklevel=3,
    )
    return ExtrinsicResponse(success=False, message=_DEPRECATED_MSG)


async def remove_liquidity_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    position_id: Optional[int] = None,
    hotkey_ss58: Optional[str] = None,
    **kwargs,
) -> ExtrinsicResponse:
    """Deprecated. User liquidity has been permanently removed from the chain."""
    deprecated_message(
        message=_DEPRECATED_MSG,
        category=ChainFeatureDisabledWarning,
        stacklevel=3,
    )
    return ExtrinsicResponse(success=False, message=_DEPRECATED_MSG)


async def toggle_user_liquidity_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    enable: Optional[bool] = None,
    **kwargs,
) -> ExtrinsicResponse:
    """Deprecated. User liquidity has been permanently removed from the chain."""
    deprecated_message(
        message=_DEPRECATED_MSG,
        category=ChainFeatureDisabledWarning,
        stacklevel=3,
    )
    return ExtrinsicResponse(success=False, message=_DEPRECATED_MSG)
