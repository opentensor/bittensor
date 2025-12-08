from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .base import CallBuilder as _BasePallet, Call

if TYPE_CHECKING:
    from scalecodec import GenericCall


@dataclass
class Proxy(_BasePallet):
    """Factory class for creating GenericCall objects for Proxy pallet functions.

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
        """Add a proxy relationship between existing wallets.

        Parameters:
            delegate: The SS58 address of the delegate proxy account.
            proxy_type: The type of proxy permissions (e.g., `Any`, `NonTransfer`, `Staking`). For available
                proxy types and their permissions, see the documentation link in the Notes section below.
            delay: Optionally, include a delay in blocks. The time-lock period for proxy announcements. A delay of `0`
                means immediate execution without announcements.

        Returns:
            GenericCall instance for the `Proxy.addProxy` extrinsic.

        Notes:
            - For available proxy types and their specific permissions, see: <https://docs.learnbittensor.org/keys/proxies#types-of-proxies>
            - See Working with Proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
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
        """Remove a specific proxy relationship.

        Parameters:
            delegate: The SS58 address of the delegate proxy account to remove.
            proxy_type: The type of proxy permissions to remove. Must match the value used when the proxy was added.
            delay: The announcement delay value (in blocks) for the proxy being removed. Must exactly match the delay
                value that was set when the proxy was originally added. This is a required identifier for the specific
                proxy relationship.

        Returns:
            GenericCall instance for the `Proxy.removeProxy` extrinsic.

        Notes:
            See Working with Proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
        """
        return self.create_composed_call(
            delegate=delegate,
            proxy_type=proxy_type,
            delay=delay,
        )

    def remove_proxies(self) -> Call:
        """Remove all proxy relationships for the signing account.

        Returns:
            GenericCall instance for the `Proxy.removeProxies` extrinsic.

        Notes:
            - This removes all proxy relationships in a single call, which is more efficient than removing them one by one.
            - See Working with Proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
        """
        return self.create_composed_call()

    def create_pure(
        self,
        proxy_type: str,
        delay: int,
        index: int,
    ) -> Call:
        """Create a pure proxy account.

        Parameters:
            proxy_type: The type of proxy permissions for the pure proxy (e.g., `Any`, `NonTransfer`,
                `Staking`). For available proxy types and their permissions, see the documentation link in the Notes
                section below.
            delay: Optionally, include a delay in blocks. The time-lock period for proxy announcements. A delay of `0`
                means immediate execution without announcements.
            index: A salt value (u16, range `0-65535`) used to generate unique pure proxy addresses. This should
                generally be left as `0` unless you are creating batches of proxies. Must be preserved for
                `kill_pure`.

        Returns:
            GenericCall instance for the `Proxy.createPure` extrinsic.

        Notes:
            - For available proxy types and their specific permissions, see: <https://docs.learnbittensor.org/keys/proxies#types-of-proxies>
            - See Pure Proxies: <https://docs.learnbittensor.org/keys/proxies/pure-proxies>
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
        """Destroy a pure proxy account.

        Parameters:
            spawner: The SS58 address of the account that spawned the pure proxy (the account that called
                `create_pure`).
            proxy_type: The type of proxy permissions that were used when creating the pure proxy. Must match the value
                used in `create_pure`.
            index: The salt value (u16, range `0-65535`) originally used in `create_pure` to generate this pure
                proxy's address. Must match exactly the index used during creation.
            height: The block number at which the pure proxy was created. This is returned in the `PureCreated`
                event from `create_pure`.
            ext_index: The extrinsic index within the block at which the pure proxy was created. This is returned in the
                `PureCreated` event from `create_pure`.

        Returns:
            GenericCall instance for the `Proxy.killPure` extrinsic.

        Notes:
            See Pure Proxies: <https://docs.learnbittensor.org/keys/proxies/pure-proxies>

        Warning:
            All access to this account will be lost. Any funds remaining in the pure proxy account will become
            permanently inaccessible after this operation.
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
        """Create a call to execute an operation through a proxy relationship.

        Parameters:
            real: The SS58 address of the real account on whose behalf the call is being made.
            force_proxy_type: The type of proxy to use for the call. If `None`, any proxy type can be used. Otherwise,
                must match one of the allowed proxy types that the signing account has for the real account.
            call: The inner call to be executed on behalf of the real account.

        Returns:
            GenericCall instance for the `Proxy.proxy` extrinsic.

        Notes:
            - The call must be permitted by the proxy type. For example, a `NonTransfer` proxy cannot execute transfer
              calls.
            - See Working with Proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
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
        """Create a call to announce a future proxied operation.

        Parameters:
            real: The SS58 address of the real account on whose behalf the call will be made.
            call_hash: The hash of the call that will be executed in the future (hex string with `0x` prefix).

        Returns:
            GenericCall instance for the `Proxy.announce` extrinsic.

        Notes:
            - A deposit is required when making an announcement. The deposit is returned when the announcement is executed,
              rejected, or removed. The announcement can be executed after the delay period has passed.
            - See Working with Proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
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
        """Create a call to execute a previously announced proxied operation.

        Parameters:
            delegate: The SS58 address of the delegate proxy account that made the announcement.
            real: The SS58 address of the real account on whose behalf the call will be made.
            force_proxy_type: The type of proxy to use for the call. If `None`, any proxy type can be used. Otherwise,
                must match one of the allowed proxy types.
            call: The inner call to be executed on behalf of the real account. The hash of this call must match the
                `call_hash` that was announced.

        Returns:
            GenericCall instance for the `Proxy.proxyAnnounced` extrinsic.

        Notes:
            - The `call_hash` of the provided call must match the `call_hash` that was announced. The announcement must
              not have been rejected by the real account, and the delay period must have passed.
            - See Working with Proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
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
        """Reject a proxy announcement.

        Parameters:
            delegate: The SS58 address of the delegate proxy account whose announcement is being rejected.
            call_hash: The hash of the call that was announced and is now being rejected (hex string with `0x`
                prefix).

        Returns:
            GenericCall instance for the `Proxy.rejectAnnouncement` extrinsic.

        Notes:
            - Once rejected, the announcement cannot be executed. The delegate's announcement deposit is returned.
            - See Working with Proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
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
        """Remove an announcement made by the signing proxy account.

        Parameters:
            real: The SS58 address of the real account on whose behalf the call was announced.
            call_hash: The hash of the call that was announced and is now being removed (hex string with `0x`
                prefix).

        Returns:
            GenericCall instance for the `Proxy.removeAnnouncement` extrinsic.

        Notes:
            - Removing an announcement frees up the announcement deposit.
            - See Working with Proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
        """
        return self.create_composed_call(
            real=real,
            call_hash=call_hash,
        )

    def poke_deposit(self) -> Call:
        """Adjust proxy and announcement deposits based on current runtime values.

        Returns:
            GenericCall instance for the `Proxy.pokeDeposit` extrinsic.

        Notes:
            - This can be used by accounts to possibly lower their locked amount. The function automatically recalculates
              deposits for both proxy relationships and announcements for the signing account. The transaction fee is waived
              if the deposit amount has changed.
            - See Working with Proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
        """
        return self.create_composed_call()
