from typing import Optional, TYPE_CHECKING

from bittensor.core.extrinsics.asyncex.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.pallets import Swap
from bittensor.core.settings import DEFAULT_MEV_PROTECTION
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance
from bittensor.utils.liquidity import price_to_tick

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor


async def add_liquidity_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    liquidity: Balance,
    price_low: Balance,
    price_high: Balance,
    hotkey_ss58: Optional[str] = None,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
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
        hotkey_ss58: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note: Adding is allowed even when user liquidity is enabled in specified subnet. Call
    `toggle_user_liquidity_extrinsic` to enable/disable user liquidity.
    """
    try:
        unlock_type = "coldkey" if hotkey_ss58 else "both"
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type=unlock_type
            )
        ).success:
            return unlocked

        call = await Swap(subtensor).add_liquidity(
            netuid=netuid,
            liquidity=liquidity.rao,
            tick_low=price_to_tick(price_low.tao),
            tick_high=price_to_tick(price_high.tao),
            hotkey=hotkey_ss58 or wallet.hotkey.ss58_address,
        )

        if mev_protection:
            return await submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            return await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )
    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


async def modify_liquidity_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    position_id: int,
    liquidity_delta: Balance,
    hotkey_ss58: Optional[str] = None,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """Modifies liquidity in liquidity position by adding or removing liquidity from it.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked).
        netuid: The UID of the target subnet for which the call is being initiated.
        position_id: The id of the position record in the pool.
        liquidity_delta: The amount of liquidity to be added or removed (add if positive or remove if negative).
        hotkey_ss58: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note: Modifying is allowed even when user liquidity is enabled in specified subnet.
        Call `toggle_user_liquidity_extrinsic` to enable/disable user liquidity.
    """
    try:
        unlock_type = "coldkey" if hotkey_ss58 else "both"
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type
            )
        ).success:
            return unlocked

        call = await Swap(subtensor).modify_position(
            netuid=netuid,
            hotkey=hotkey_ss58 or wallet.hotkey.ss58_address,
            position_id=position_id,
            liquidity_delta=liquidity_delta.rao,
        )

        if mev_protection:
            return await submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            return await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )
    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


async def remove_liquidity_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    position_id: int,
    hotkey_ss58: Optional[str] = None,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """Remove liquidity and credit balances back to wallet's hotkey stake.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked).
        netuid: The UID of the target subnet for which the call is being initiated.
        position_id: The id of the position record in the pool.
        hotkey_ss58: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note: Adding is allowed even when user liquidity is enabled in specified subnet. Call
    `toggle_user_liquidity_extrinsic` to enable/disable user liquidity.
    """
    try:
        unlock_type = "coldkey" if hotkey_ss58 else "both"
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type
            )
        ).success:
            return unlocked

        call = await Swap(subtensor).remove_liquidity(
            netuid=netuid,
            hotkey=hotkey_ss58 or wallet.hotkey.ss58_address,
            position_id=position_id,
        )

        if mev_protection:
            return await submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            return await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )
    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


async def toggle_user_liquidity_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    enable: bool,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """Allow to toggle user liquidity for specified subnet.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked).
        netuid: The UID of the target subnet for which the call is being initiated.
        enable: Boolean indicating whether to enable user liquidity.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        call = await Swap(subtensor).toggle_user_liquidity(
            netuid=netuid,
            enable=enable,
        )

        if mev_protection:
            return await submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            return await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )
    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
