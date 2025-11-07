from dataclasses import dataclass
from typing import Optional

from .base import CallBuilder, Call


@dataclass
class Swap(CallBuilder):
    """Factory class for creating GenericCall objects for Swap pallet functions.

    This class provides methods to create GenericCall instances for all Swap pallet extrinsics.

    Works with both sync (Subtensor) and async (AsyncSubtensor) instances. For async operations, pass an AsyncSubtensor
    instance and await the result.

    Example:
        # Sync usage
        call = Swap(subtensor).toggle_user_liquidity(netuid=14, enable=True)
        response = subtensor.sign_and_send_extrinsic(call=call, ...)

        # Async usage
        call = await Swap(subtensor).toggle_user_liquidity(netuid=14, enable=True)
        response = await async_subtensor.sign_and_send_extrinsic(call=call, ...)
    """

    def add_liquidity(
        self,
        netuid: int,
        liquidity: int,
        tick_low: int,
        tick_high: int,
        hotkey: Optional[str] = None,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Swap.add_liquidity.

        Parameters:
            netuid: The UID of the target subnet for which the call is being initiated.
            liquidity: The amount of liquidity in RAO to be added.
            tick_low: The lower bound of the price tick range.
            tick_high: The upper bound of the price tick range.
            hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            hotkey=hotkey,
            liquidity=liquidity,
            tick_low=tick_low,
            tick_high=tick_high,
        )

    def modify_position(
        self,
        netuid: int,
        hotkey: str,
        position_id: int,
        liquidity_delta: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Swap.modify_position.

        Parameters:
            netuid: The UID of the target subnet for which the call is being initiated.
            hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
            position_id: The id of the position record in the pool.
            liquidity_delta: The amount of liquidity in RAO to be added or removed (could be positive or negative).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            hotkey=hotkey,
            position_id=position_id,
            liquidity_delta=liquidity_delta,
        )

    def remove_liquidity(
        self,
        netuid: int,
        hotkey: str,
        position_id: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Swap.remove_liquidity.

        Parameters:
            netuid: The UID of the target subnet for which the call is being initiated.
            position_id: The id of the position record in the pool.
            hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            hotkey=hotkey,
            position_id=position_id,
        )

    def toggle_user_liquidity(
        self,
        netuid: int,
        enable: bool,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Swap.toggle_user_liquidity.

        Parameters:
            netuid: The UID of the target subnet for which the call is being initiated.
            enable: Boolean indicating whether to enable user liquidity.
        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            enable=enable,
        )
