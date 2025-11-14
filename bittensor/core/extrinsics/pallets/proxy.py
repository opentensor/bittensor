from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .base import CallBuilder as _BasePallet, Call

if TYPE_CHECKING:
    from scalecodec import GenericCall


@dataclass
class Proxy(_BasePallet):
    """
    Factory class for creating GenericCall objects for Proxy pallet functions.

    This class provides methods to create GenericCall instances for all Proxy pallet extrinsics.

    Works with both sync (Subtensor) and async (AsyncSubtensor) instances. For async operations, pass an AsyncSubtensor
    instance and await the result.

    Example:
        # Sync usage
        call = Proxy(subtensor).add_proxy(delegate="5DE..", proxy_type="Any", delay=0)
        response = subtensor.sign_and_send_extrinsic(call=call, ...)

        # Async usage
        call = await Proxy(async_subtensor).add_proxy(delegate="5DE..", proxy_type="Any", delay=0)
        response = await async_subtensor.sign_and_send_extrinsic(call=call, ...)
    """

    def add_proxy(
        self,
        delegate: str,
        proxy_type: str,
        delay: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Proxy.add_proxy.

        Parameters:
            delegate: The SS58 address of the delegate proxy account.
            proxy_type: The type of proxy permissions (e.g., "Any", "NonTransfer", "Governance", "Staking").
            delay: The number of blocks before the proxy can be used.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            delegate=delegate,
            proxy_type=proxy_type,
            delay=delay,
        )

    def remove_proxy(
        self,
        delegate: str,
        proxy_type: str,
        delay: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Proxy.remove_proxy.

        Parameters:
            delegate: The SS58 address of the delegate proxy account to remove.
            proxy_type: The type of proxy permissions to remove.
            delay: The number of blocks before the proxy removal takes effect.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            delegate=delegate,
            proxy_type=proxy_type,
            delay=delay,
        )

    def remove_proxies(self) -> Call:
        """Returns GenericCall instance for Proxy.remove_proxies.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call()

    def create_pure(
        self,
        proxy_type: str,
        delay: int,
        index: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Proxy.create_pure.

        Parameters:
            proxy_type: The type of proxy permissions for the pure proxy.
            delay: The number of blocks before the pure proxy can be used.
            index: The index to use for generating the pure proxy account address.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            proxy_type=proxy_type,
            delay=delay,
            index=index,
        )

    def kill_pure(
        self,
        spawner: str,
        proxy_type: str,
        index: int,
        height: int,
        ext_index: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Proxy.kill_pure.

        Parameters:
            spawner: The SS58 address of the account that spawned the pure proxy.
            proxy_type: The type of proxy permissions.
            index: The disambiguation index originally passed to `create_pure`.
            height: The block height at which the pure proxy was created.
            ext_index: The extrinsic index at which the pure proxy was created.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            spawner=spawner,
            proxy_type=proxy_type,
            index=index,
            height=height,
            ext_index=ext_index,
        )

    def proxy(
        self,
        real: str,
        force_proxy_type: Optional[str],
        call: "GenericCall",
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Proxy.proxy.

        Parameters:
            real: The SS58 address of the real account on whose behalf the call is being made.
            force_proxy_type: The type of proxy to use for the call. If None, any proxy type can be used. Otherwise,
                must match one of the allowed proxy types.
            call: The inner call to be executed on behalf of the real account.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            real=real,
            force_proxy_type=force_proxy_type,
            call=call,
        )

    def announce(
        self,
        real: str,
        call_hash: str,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Proxy.announce.

        Parameters:
            real: The SS58 address of the real account on whose behalf the call will be made.
            call_hash: The hash of the call that will be executed in the future.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            real=real,
            call_hash=call_hash,
        )

    def proxy_announced(
        self,
        delegate: str,
        real: str,
        force_proxy_type: Optional[str],
        call: "GenericCall",
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Proxy.proxy_announced.

        Parameters:
            delegate: The SS58 address of the delegate proxy account that made the announcement.
            real: The SS58 address of the real account on whose behalf the call will be made.
            force_proxy_type: The type of proxy to use for the call. If None, any proxy type can be used. Otherwise,
                must match one of the allowed proxy types.
            call: The inner call to be executed on behalf of the real account (must match the announced call_hash).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            delegate=delegate,
            real=real,
            force_proxy_type=force_proxy_type,
            call=call,
        )

    def reject_announcement(
        self,
        delegate: str,
        call_hash: str,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Proxy.reject_announcement.

        Parameters:
            delegate: The SS58 address of the delegate proxy account whose announcement is being rejected.
            call_hash: The hash of the call that was announced and is now being rejected.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            delegate=delegate,
            call_hash=call_hash,
        )

    def remove_announcement(
        self,
        real: str,
        call_hash: str,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function Proxy.remove_announcement.

        Parameters:
            real: The SS58 address of the real account on whose behalf the call was announced.
            call_hash: The hash of the call that was announced and is now being removed.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            real=real,
            call_hash=call_hash,
        )

    def poke_deposit(self) -> Call:
        """Returns GenericCall instance for Proxy.poke_deposit.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call()
