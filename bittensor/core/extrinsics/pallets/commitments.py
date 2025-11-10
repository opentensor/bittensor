from dataclasses import dataclass

from .base import CallBuilder as _BasePallet, Call


@dataclass
class Commitments(_BasePallet):
    """Factory class for creating GenericCall objects for Commitments pallet functions.

    This class provides methods to create GenericCall instances for Commitments pallet extrinsics.

    Works with both sync (Subtensor) and async (AsyncSubtensor) instances. For async operations, pass an AsyncSubtensor
    instance and await the result.

    Example:
        # Sync usage
        call = Commitments(subtensor).set_commitment(netuid=14, ...)
        response = subtensor.sign_and_send_extrinsic(call=call, ...)

        # Async usage
        call = await Commitments(async_subtensor).set_commitment(netuid=14, ...)
        response = await async_subtensor.sign_and_send_extrinsic(call=call, ...)
    """

    def set_commitment(
        self,
        netuid: int,
        info: dict,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Commitments.set_commitment.

        Parameters:
            netuid: The netuid of the subnet to set commitment for.
            info: Dictionary of info fields to set.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            info=info,
        )
