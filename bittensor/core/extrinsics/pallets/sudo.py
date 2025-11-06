from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import CallBuilder, Call

if TYPE_CHECKING:
    from scalecodec import GenericCall


@dataclass
class Sudo(CallBuilder):
    """Factory class for creating GenericCall objects for Sudo pallet functions.

    This class provides methods to create GenericCall instances for all Sudo pallet extrinsics.

    Works with both sync (Subtensor) and async (AsyncSubtensor) instances. For async operations, pass an AsyncSubtensor
    instance and await the result.

    Example:
        # Nested sync calls (e.g., with Sudo)
        inner_call = SubtensorModule(subtensor).set_pending_childkey_cooldown(cooldown=100)
        sudo_call = Sudo(subtensor).sudo(call=inner_call)
        response = subtensor.sign_and_send_extrinsic(call=sudo_call, ...)

        # Nested async calls (e.g., with Sudo)
        inner_call = await SubtensorModule(subtensor).set_pending_childkey_cooldown(cooldown=100)
        sudo_call = await Sudo(subtensor).sudo(call=inner_call)
        response = subtensor.sign_and_send_extrinsic(call=sudo_call, ...)
    """

    def sudo(
        self,
        call: "GenericCall",
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Sudo.sudo.

        Parameters:
            call: The call to be executed as sudo.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(call=call)
