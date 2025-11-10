from typing import TYPE_CHECKING, Optional, Union

from bittensor.core.chain_data.proxy import ProxyType
from bittensor.core.extrinsics.pallets import Proxy
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor
    from scalecodec.types import GenericCall


def add_proxy_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    delegate_ss58: str,
    proxy_type: Union[str, ProxyType],
    delay: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Adds a proxy relationship.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object.
        delegate_ss58: The SS58 address of the delegate proxy account.
        proxy_type: The type of proxy permissions (e.g., "Any", "NonTransfer", "Governance", "Staking"). Can be a
            string or ProxyType enum value.
        delay: The number of blocks before the proxy can be used.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        proxy_type_str = ProxyType.normalize(proxy_type)

        logging.debug(
            f"Adding proxy: delegate=[blue]{delegate_ss58}[/blue], "
            f"type=[blue]{proxy_type_str}[/blue], delay=[blue]{delay}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = Proxy(subtensor).add_proxy(
            delegate=delegate_ss58,
            proxy_type=proxy_type_str,
            delay=delay,
        )

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]Proxy added successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def remove_proxy_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    delegate_ss58: str,
    proxy_type: Union[str, ProxyType],
    delay: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Removes a proxy relationship.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object.
        delegate_ss58: The SS58 address of the delegate proxy account to remove.
        proxy_type: The type of proxy permissions to remove. Can be a string or ProxyType enum value.
        delay: The number of blocks before the proxy removal takes effect.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        proxy_type_str = ProxyType.normalize(proxy_type)

        logging.debug(
            f"Removing proxy: delegate=[blue]{delegate_ss58}[/blue], "
            f"type=[blue]{proxy_type_str}[/blue], delay=[blue]{delay}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = Proxy(subtensor).remove_proxy(
            delegate=delegate_ss58,
            proxy_type=proxy_type_str,
            delay=delay,
        )

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]Proxy removed successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def remove_proxies_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Removes all proxy relationships for the account.

    This removes all proxy relationships in a single call, which is more efficient than removing them one by one. The
    deposit for all proxies will be returned.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object (the account whose proxies will be removed).
        period: The number of blocks during which the transaction will remain valid after it's submitted.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        logging.debug(
            f"Removing all proxies for account [blue]{wallet.coldkey.ss58_address}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = Proxy(subtensor).remove_proxies()

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]All proxies removed successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def create_pure_proxy_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    proxy_type: Union[str, ProxyType],
    delay: int,
    index: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Creates a pure proxy account.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object.
        proxy_type: The type of proxy permissions for the pure proxy. Can be a string or ProxyType enum value.
        delay: The number of blocks before the pure proxy can be used.
        index: The index to use for generating the pure proxy account address.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        proxy_type_str = ProxyType.normalize(proxy_type)

        logging.debug(
            f"Creating pure proxy: type=[blue]{proxy_type_str}[/blue], "
            f"delay=[blue]{delay}[/blue], index=[blue]{index}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = Proxy(subtensor).create_pure(
            proxy_type=proxy_type_str,
            delay=delay,
            index=index,
        )

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]Pure proxy created successfully.[/green]")

            # Extract pure proxy address from PureCreated triggered event
            for event in response.extrinsic_receipt.triggered_events:
                if event.get("event_id") == "PureCreated":
                    # Event structure: PureProxyCreated { disambiguation_index, proxy_type, pure, who }
                    attributes = event.get("attributes", [])
                    if attributes:
                        response.data = {
                            "pure_account": attributes.get("pure"),
                            "spawner": attributes.get("who"),
                            "proxy_type": attributes.get("proxy_type"),
                            "index": attributes.get("disambiguation_index"),
                            "height": subtensor.substrate.get_block_number(
                                response.extrinsic_receipt.block_hash
                            ),
                            "ext_index": response.extrinsic_receipt.extrinsic_idx,
                        }
                    break
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def kill_pure_proxy_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    pure_proxy_ss58: str,
    spawner: str,
    proxy_type: Union[str, "ProxyType"],
    index: int,
    height: int,
    ext_index: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Kills (removes) a pure proxy account.

    This method removes a pure proxy account that was previously created via `create_pure_proxy()`. The `kill_pure` call
    must be executed through the pure proxy account itself, with the spawner acting as an "Any" proxy. This method
    automatically handles this by executing the call via `proxy()`.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object. The wallet.coldkey.ss58_address must be the spawner of the pure proxy (the
            account that created it via `create_pure_proxy()`). The spawner must have an "Any" proxy relationship with
            the pure proxy.
        pure_proxy_ss58: The SS58 address of the pure proxy account to be killed. This is the address that was returned
            in the `create_pure_proxy()` response.
        spawner: The SS58 address of the spawner account (the account that originally created the pure proxy via
            `create_pure_proxy()`). This should match wallet.coldkey.ss58_address.
        proxy_type: The type of proxy permissions. Can be a string or ProxyType enum value. Must match the proxy_type
            used when creating the pure proxy.
        index: The disambiguation index originally passed to `create_pure()`.
        height: The block height at which the pure proxy was created.
        ext_index: The extrinsic index at which the pure proxy was created.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note:
        The `kill_pure` call must be executed through the pure proxy account itself, with the spawner acting as an "Any"
        proxy. This method automatically handles this by executing the call via `proxy()`. The spawner must have an
        "Any" proxy relationship with the pure proxy for this to work.

    Example:
        # After creating a pure proxy
        create_response = subtensor.proxies.create_pure_proxy(
            wallet=spawner_wallet,
            proxy_type=ProxyType.Any,
            delay=0,
            index=0,
        )
        pure_proxy_ss58 = create_response.data["pure_account"]
        spawner = create_response.data["spawner"]
        height = create_response.data["height"]
        ext_index = create_response.data["ext_index"]

        # Kill the pure proxy
        kill_response = subtensor.proxies.kill_pure_proxy(
            wallet=spawner_wallet,
            pure_proxy_ss58=pure_proxy_ss58,
            spawner=spawner,
            proxy_type=ProxyType.Any,
            index=0,
            height=height,
            ext_index=ext_index,
        )
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        proxy_type_str = ProxyType.normalize(proxy_type)

        # Validate that spawner matches wallet
        if wallet.coldkey.ss58_address != spawner:
            error_msg = (
                f"Spawner address {spawner} does not match wallet address "
                f"{wallet.coldkey.ss58_address}"
            )
            if raise_error:
                raise ValueError(error_msg)
            return ExtrinsicResponse(False, error_msg)

        logging.debug(
            f"Killing pure proxy: pure=[blue]{pure_proxy_ss58}[/blue], "
            f"spawner=[blue]{spawner}[/blue], type=[blue]{proxy_type_str}[/blue], "
            f"index=[blue]{index}[/blue], height=[blue]{height}[/blue], "
            f"ext_index=[blue]{ext_index}[/blue] on [blue]{subtensor.network}[/blue]."
        )

        # Create the kill_pure call
        kill_pure_call = Proxy(subtensor).kill_pure(
            spawner=spawner,
            proxy_type=proxy_type_str,
            index=index,
            height=height,
            ext_index=ext_index,
        )

        # Execute kill_pure through proxy() where:
        # - wallet (spawner) signs the transaction
        # - real_account_ss58 (pure_proxy_ss58) is the origin (pure proxy account)
        # - force_proxy_type=Any (spawner acts as Any proxy for pure proxy)
        response = proxy_extrinsic(
            subtensor=subtensor,
            wallet=wallet,
            real_account_ss58=pure_proxy_ss58,
            force_proxy_type=ProxyType.Any,
            call=kill_pure_call,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if response.success:
            logging.debug("[green]Pure proxy killed successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def proxy_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    real_account_ss58: str,
    force_proxy_type: Optional[Union[str, ProxyType]],
    call: "GenericCall",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Executes a call on behalf of the real account through a proxy.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object (should be the proxy account wallet).
        real_account_ss58: The SS58 address of the real account on whose behalf the call is being made.
        force_proxy_type: The type of proxy to use for the call. If None, any proxy type can be used. Otherwise, must
            match one of the allowed proxy types. Can be a string or ProxyType enum value.
        call: The inner call to be executed on behalf of the real account.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        force_proxy_type_str = (
            ProxyType.normalize(force_proxy_type)
            if force_proxy_type is not None
            else None
        )

        logging.debug(
            f"Executing proxy call: real=[blue]{real_account_ss58}[/blue], "
            f"force_type=[blue]{force_proxy_type_str}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        proxy_call = Proxy(subtensor).proxy(
            real=real_account_ss58,
            force_proxy_type=force_proxy_type_str,
            call=call,
        )

        response = subtensor.sign_and_send_extrinsic(
            call=proxy_call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]Proxy call executed successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def proxy_announced_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    delegate_ss58: str,
    real_account_ss58: str,
    force_proxy_type: Optional[Union[str, ProxyType]],
    call: "GenericCall",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Executes an announced call on behalf of the real account through a proxy.

    This extrinsic executes a call that was previously announced via `announce_extrinsic`. The call must match the
    call_hash that was announced, and the delay period must have passed.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object (should be the proxy account wallet that made the announcement).
        delegate_ss58: The SS58 address of the delegate proxy account that made the announcement.
        real_account_ss58: The SS58 address of the real account on whose behalf the call will be made.
        force_proxy_type: The type of proxy to use for the call. If None, any proxy type can be used. Otherwise, must
            match one of the allowed proxy types. Can be a string or ProxyType enum value.
        call: The inner call to be executed on behalf of the real account (must match the announced call_hash).
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        force_proxy_type_str = (
            ProxyType.normalize(force_proxy_type)
            if force_proxy_type is not None
            else None
        )

        logging.debug(
            f"Executing announced proxy call: delegate=[blue]{delegate_ss58}[/blue], "
            f"real=[blue]{real_account_ss58}[/blue], force_type=[blue]{force_proxy_type_str}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        proxy_call = Proxy(subtensor).proxy_announced(
            delegate=delegate_ss58,
            real=real_account_ss58,
            force_proxy_type=force_proxy_type_str,
            call=call,
        )

        response = subtensor.sign_and_send_extrinsic(
            call=proxy_call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]Announced proxy call executed successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def announce_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    real_account_ss58: str,
    call_hash: str,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Announces a future call that will be executed through a proxy.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object (should be the proxy account wallet).
        real_account_ss58: The SS58 address of the real account on whose behalf the call will be made.
        call_hash: The hash of the call that will be executed in the future.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        # make sure the call hash starts with 0x
        call_hash = "0x" + call_hash.lstrip("0x")

        logging.debug(
            f"Announcing proxy call: real=[blue]{real_account_ss58}[/blue], "
            f"call_hash=[blue]{call_hash}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = Proxy(subtensor).announce(
            real=real_account_ss58,
            call_hash=call_hash,
        )

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]Proxy call announced successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def reject_announcement_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    delegate_ss58: str,
    call_hash: str,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Rejects an announcement made by a proxy delegate.

    This extrinsic allows the real account to reject an announcement made by a proxy delegate. This prevents the
    announced call from being executed.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object (should be the real account wallet).
        delegate_ss58: The SS58 address of the delegate proxy account whose announcement is being rejected.
        call_hash: The hash of the call that was announced and is now being rejected.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        # make sure the call hash starts with 0x
        call_hash = "0x" + call_hash.lstrip("0x")

        logging.debug(
            f"Rejecting announcement: delegate=[blue]{delegate_ss58}[/blue], "
            f"call_hash=[blue]{call_hash}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = Proxy(subtensor).reject_announcement(
            delegate=delegate_ss58,
            call_hash=call_hash,
        )

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]Announcement rejected successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def remove_announcement_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    real_account_ss58: str,
    call_hash: str,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Removes an announcement made by a proxy account.

    This extrinsic allows the proxy account to remove its own announcement before it is executed or rejected. This frees
    up the announcement deposit.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object (should be the proxy account wallet that made the announcement).
        real_account_ss58: The SS58 address of the real account on whose behalf the call was announced.
        call_hash: The hash of the call that was announced and is now being removed.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        # make sure the call hash starts with 0x
        call_hash = "0x" + call_hash.lstrip("0x")

        logging.debug(
            f"Removing announcement: real=[blue]{real_account_ss58}[/blue], "
            f"call_hash=[blue]{call_hash}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = Proxy(subtensor).remove_announcement(
            real=real_account_ss58,
            call_hash=call_hash,
        )

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]Announcement removed successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def poke_deposit_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Adjusts deposits made for proxies and announcements based on current values.

    This can be used by accounts to possibly lower their locked amount. The function automatically recalculates deposits
    for both proxy relationships and announcements for the signing account. The transaction fee is waived if the deposit
    amount has changed.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object (the account whose deposits will be adjusted).
        period: The number of blocks during which the transaction will remain valid after it's submitted.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    When to use:
        - After runtime upgrade, if deposit constants have changed.
        - After removing proxies/announcements, to free up excess locked funds.
        - Periodically to optimize locked deposit amounts.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        logging.debug(
            f"Poking deposit for account [blue]{wallet.coldkey.ss58_address}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = Proxy(subtensor).poke_deposit()

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]Deposit poked successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
