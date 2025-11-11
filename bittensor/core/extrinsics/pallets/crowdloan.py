from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional


from .base import CallBuilder, Call

if TYPE_CHECKING:
    from scalecodec import GenericCall


@dataclass
class Crowdloan(CallBuilder):
    """Factory class for creating GenericCall objects for Crowdloan pallet functions.

    This class provides methods to create GenericCall instances for all Crowdloan pallet extrinsics.

    Works with both sync (Subtensor) and async (AsyncSubtensor) instances. For async operations, pass an AsyncSubtensor
    instance and await the result.

    Example:
        # Sync usage
        call = Crowdloan(subtensor).finalize(crowdloan_id=123)
        response = subtensor.sign_and_send_extrinsic(call=call, ...)

        # Async usage
        call = await Crowdloan(subtensor).finalize(crowdloan_id=123)
        response = await async_subtensor.sign_and_send_extrinsic(call=call, ...)
    """

    def contribute(
        self,
        crowdloan_id: int,
        amount: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Crowdloan.contribute.

        Parameters:
            crowdloan_id: The unique identifier of the crowdloan to contribute to.
            amount: Amount in RAO to contribute.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(crowdloan_id=crowdloan_id, amount=amount)

    def create(
        self,
        deposit: int,
        min_contribution: int,
        cap: int,
        end: int,
        call: Optional["GenericCall"] = None,
        target_address: Optional[str] = None,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Crowdloan.create.

        Parameters:
            deposit: Initial deposit in RAO from the creator.
            min_contribution: Minimum contribution amount in RAO.
            cap: Maximum cap to be raised in RAO.
            end: Block number when the campaign ends.
            call: Runtime call data (e.g., subtensor::register_leased_network).
            target_address: SS58 address to transfer funds to on success.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            deposit=deposit,
            min_contribution=min_contribution,
            cap=cap,
            end=end,
            call=call,
            target_address=target_address,
        )

    def dissolve(
        self,
        crowdloan_id: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Crowdloan.dissolve.

        Parameters:
            crowdloan_id: The unique identifier of the crowdloan to dissolve.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(crowdloan_id=crowdloan_id)

    def finalize(
        self,
        crowdloan_id: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Crowdloan.finalize.

        Parameters:
            crowdloan_id: The unique identifier of the crowdloan to finalize.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(crowdloan_id=crowdloan_id)

    def refund(
        self,
        crowdloan_id: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Crowdloan.refund.

        Parameters:
            crowdloan_id: The unique identifier of the crowdloan to refund.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(crowdloan_id=crowdloan_id)

    def update_cap(
        self,
        crowdloan_id: int,
        new_cap: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Crowdloan.update_cap.

        Parameters:
            crowdloan_id: The unique identifier of the crowdloan to update the cap for.
            new_cap: New cap to be raised in RAO.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(crowdloan_id=crowdloan_id, new_cap=new_cap)

    def update_end(
        self,
        crowdloan_id: int,
        new_end: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Crowdloan.update_end.

        Parameters:
            crowdloan_id: The unique identifier of the crowdloan to update the end block number for.
            new_end: New end block number.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(crowdloan_id=crowdloan_id, new_end=new_end)

    def update_min_contribution(
        self,
        crowdloan_id: int,
        new_min_contribution: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Crowdloan.update_min_contribution.

        Parameters:
            crowdloan_id: The unique identifier of the crowdloan to update the minimum contribution amount for.
            new_min_contribution: New minimum contribution amount in RAO.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            crowdloan_id=crowdloan_id,
            new_min_contribution=new_min_contribution,
        )

    def withdraw(
        self,
        crowdloan_id: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Crowdloan.withdraw.

        Parameters:
            crowdloan_id: The unique identifier of the crowdloan to withdraw from.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(crowdloan_id=crowdloan_id)
