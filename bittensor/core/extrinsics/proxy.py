from typing import TYPE_CHECKING, Optional, Union

from bittensor.core.extrinsics.pallets import Proxy
from bittensor.core.chain_data.proxy import ProxyType
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


def announce_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    real_ss58: str,
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
        real_ss58: The SS58 address of the real account on whose behalf the call will be made.
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

        logging.debug(
            f"Announcing proxy call: real=[blue]{real_ss58}[/blue], "
            f"call_hash=[blue]{call_hash}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = Proxy(subtensor).announce(
            real=real_ss58,
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
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def kill_pure_proxy_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    spawner_ss58: str,
    proxy_ss58: str,
    proxy_type: Union[str, ProxyType],
    height: int,
    ext_index: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Kills (removes) a pure proxy account.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object.
        spawner_ss58: The SS58 address of the account that spawned the pure proxy.
        proxy_ss58: The SS58 address of the pure proxy account to kill.
        proxy_type: The type of proxy permissions. Can be a string or ProxyType enum value.
        height: The block height at which the pure proxy was created.
        ext_index: The extrinsic index at which the pure proxy was created.
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
            f"Killing pure proxy: spawner=[blue]{spawner_ss58}[/blue], "
            f"proxy=[blue]{proxy_ss58}[/blue], type=[blue]{proxy_type_str}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = Proxy(subtensor).kill_pure(
            spawner=spawner_ss58,
            proxy=proxy_ss58,
            proxy_type=proxy_type_str,
            height=height,
            ext_index=ext_index,
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
            logging.debug("[green]Pure proxy killed successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def proxy_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    real_ss58: str,
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
        real_ss58: The SS58 address of the real account on whose behalf the call is being made.
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
            ProxyType.normalize(force_proxy_type) if force_proxy_type is not None else None
        )

        logging.debug(
            f"Executing proxy call: real=[blue]{real_ss58}[/blue], "
            f"force_type=[blue]{force_proxy_type_str}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        proxy_call = Proxy(subtensor).proxy(
            real=real_ss58,
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
