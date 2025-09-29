from typing import Optional, TYPE_CHECKING

from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance
from bittensor.utils.liquidity import price_to_tick

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def add_liquidity_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    liquidity: Balance,
    price_low: Balance,
    price_high: Balance,
    hotkey: Optional[str] = None,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Adds liquidity to the specified price range.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked).
        netuid: The UID of the target subnet for which the call is being initiated.
        liquidity: The amount of liquidity to be added.
        price_low: The lower bound of the price tick range.
        price_high: The upper bound of the price tick range.
        hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note: Adding is allowed even when user liquidity is enabled in specified subnet. Call
    `toggle_user_liquidity_extrinsic` to enable/disable user liquidity.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        tick_low = price_to_tick(price_low.tao)
        tick_high = price_to_tick(price_high.tao)

        call = subtensor.substrate.compose_call(
            call_module="Swap",
            call_function="add_liquidity",
            call_params={
                "hotkey": hotkey or wallet.hotkey.ss58_address,
                "netuid": netuid,
                "tick_low": tick_low,
                "tick_high": tick_high,
                "liquidity": liquidity.rao,
            },
        )

        return subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )
    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def modify_liquidity_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    position_id: int,
    liquidity_delta: Balance,
    hotkey: Optional[str] = None,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Modifies liquidity in liquidity position by adding or removing liquidity from it.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked).
        netuid: The UID of the target subnet for which the call is being initiated.
        position_id: The id of the position record in the pool.
        liquidity_delta: The amount of liquidity to be added or removed (add if positive or remove if negative).
        hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note: Modifying is allowed even when user liquidity is enabled in specified subnet. Call
    `toggle_user_liquidity_extrinsic` to enable/disable user liquidity.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        call = subtensor.substrate.compose_call(
            call_module="Swap",
            call_function="modify_position",
            call_params={
                "hotkey": hotkey or wallet.hotkey.ss58_address,
                "netuid": netuid,
                "position_id": position_id,
                "liquidity_delta": liquidity_delta.rao,
            },
        )

        return subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )
    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def remove_liquidity_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    position_id: int,
    hotkey: Optional[str] = None,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Remove liquidity and credit balances back to wallet's hotkey stake.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked).
        netuid: The UID of the target subnet for which the call is being initiated.
        position_id: The id of the position record in the pool.
        hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note: Adding is allowed even when user liquidity is enabled in specified subnet. Call
    `toggle_user_liquidity_extrinsic` to enable/disable user liquidity.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        call = subtensor.substrate.compose_call(
            call_module="Swap",
            call_function="remove_liquidity",
            call_params={
                "hotkey": hotkey or wallet.hotkey.ss58_address,
                "netuid": netuid,
                "position_id": position_id,
            },
        )

        return subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )
    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def toggle_user_liquidity_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    enable: bool,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Allow to toggle user liquidity for specified subnet.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked).
        netuid: The UID of the target subnet for which the call is being initiated.
        enable: Boolean indicating whether to enable user liquidity.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
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

        call = subtensor.substrate.compose_call(
            call_module="Swap",
            call_function="toggle_user_liquidity",
            call_params={"netuid": netuid, "enable": enable},
        )

        return subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )
    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
