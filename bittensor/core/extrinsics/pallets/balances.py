from dataclasses import dataclass

from .base import CallBuilder as _BasePallet, Call


@dataclass
class Balances(_BasePallet):
    """Factory class for creating GenericCall objects for Balances pallet functions.

    This class provides methods to create GenericCall instances for all Balances pallet extrinsics.

    Works with both sync (Subtensor) and async (AsyncSubtensor) instances. For async operations, pass an AsyncSubtensor
    instance and await the result.

    Example:
        # Sync usage
        call = Balances(subtensor).transfer_all(dest="5DE..", keep_alive=True)
        response = subtensor.sign_and_send_extrinsic(call=call, ...)

        # Async usage
        call = await Balances(subtensor).transfer_all(dest="5DE..", keep_alive=True)
        response = await async_subtensor.sign_and_send_extrinsic(call=call, ...)
    """

    def transfer_all(
        self,
        dest: str,
        keep_alive: bool,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Balances.transfer_all.

        Parameters:
            dest: The destination ss58 address.
            keep_alive: A boolean to determine if the transfer_all operation should send all of the funds the account
                has, causing the sender account to be killed (false), or transfer everything except at least the
                existential deposit, which will guarantee to keep the sender account alive (true).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(dest=dest, keep_alive=keep_alive)

    def transfer_allow_death(self, dest: str, value: int) -> Call:
        """Returns GenericCall instance for Subtensor function Balances.transfer_allow_death.

        Parameters:
            dest: The destination ss58 address.
            value: The Balance amount in RAO to transfer.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(dest=dest, value=value)

    def transfer_keep_alive(self, dest: str, value: int) -> Call:
        """Returns GenericCall instance for Subtensor function Balances.transfer_keep_alive.

        Parameters:
            dest: The destination ss58 address.
            value: The Balance amount in RAO to transfer.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(dest=dest, value=value)
