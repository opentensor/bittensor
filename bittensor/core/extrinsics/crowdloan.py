from typing import TYPE_CHECKING, Optional

from bittensor.core.extrinsics.params import CrowdloanParams
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import check_balance_amount

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor
    from bittensor.utils.balance import Balance
    from scalecodec.types import GenericCall


def contribute_crowdloan_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    crowdloan_id: int,
    amount: "Balance",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Contributes funds to an active crowdloan campaign.

    Parameters:
        subtensor: Active Subtensor connection.
        wallet: Bittensor Wallet instance used to sign the transaction.
        crowdloan_id: The unique identifier of the crowdloan to contribute to.
        amount: Amount to contribute.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        check_balance_amount(amount)

        extrinsic_call = subtensor.compose_call(
            call_module="Crowdloan",
            call_function="contribute",
            call_params=CrowdloanParams.contribute(crowdloan_id, amount),
        )

        return subtensor.sign_and_send_extrinsic(
            call=extrinsic_call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def create_crowdloan_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    deposit: "Balance",
    min_contribution: "Balance",
    cap: "Balance",
    end: int,
    call: Optional["GenericCall"] = None,
    target_address: Optional[str] = None,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> "ExtrinsicResponse":
    """
    Creates a new crowdloan campaign on-chain.

    Parameters:
        subtensor: Active Subtensor connection.
        wallet: Bittensor Wallet instance used to sign the transaction.
        deposit: Initial deposit in RAO from the creator.
        min_contribution: Minimum contribution amount.
        cap: Maximum cap to be raised.
        end: Block number when the campaign ends.
        call: Runtime call data (e.g., subtensor::register_leased_network).
        target_address: SS58 address to transfer funds to on success.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        check_balance_amount(deposit)
        check_balance_amount(min_contribution)
        check_balance_amount(cap)

        extrinsic_call = subtensor.compose_call(
            call_module="Crowdloan",
            call_function="create",
            call_params=CrowdloanParams.create(
                deposit, min_contribution, cap, end, call, target_address
            ),
        )

        return subtensor.sign_and_send_extrinsic(
            call=extrinsic_call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def dissolve_crowdloan_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    crowdloan_id: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Dissolves a completed or failed crowdloan campaign after all refunds are processed.

    This permanently removes the campaign from on-chain storage and refunds the creator's remaining deposit, if
    applicable. Can only be called by the campaign creator.

    Parameters:
        subtensor: Active Subtensor connection.
        wallet: Bittensor Wallet instance used to sign the transaction.
        crowdloan_id: The unique identifier of the crowdloan to dissolve.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Notes:
        - Only the creator can dissolve their own crowdloan.
        - All contributors (except the creator) must have been refunded first.
        - The creator’s remaining contribution (deposit) is returned during dissolution.
        - After this call, the crowdloan is removed from chain storage.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        extrinsic_call = subtensor.compose_call(
            call_module="Crowdloan",
            call_function="dissolve",
            call_params=CrowdloanParams.dissolve(crowdloan_id),
        )

        return subtensor.sign_and_send_extrinsic(
            call=extrinsic_call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def finalize_crowdloan_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    crowdloan_id: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Finalizes a successful crowdloan campaign once the cap has been reached and the end block has passed.

    This executes the stored call or transfers the raised funds to the target address, completing the campaign.

    Parameters:
        subtensor: Active Subtensor connection.
        wallet: Bittensor Wallet instance used to sign the transaction.
        crowdloan_id: The unique identifier of the crowdloan to finalize.
        period: The number of blocks during which the transaction will remain valid after it's submitted.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        extrinsic_call = subtensor.compose_call(
            call_module="Crowdloan",
            call_function="finalize",
            call_params=CrowdloanParams.finalize(crowdloan_id),
        )

        return subtensor.sign_and_send_extrinsic(
            call=extrinsic_call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def refund_crowdloan_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    crowdloan_id: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Refunds contributors from a failed or expired crowdloan campaign.

    This call attempts to refund up to the limit defined by `RefundContributorsLimit` in a single dispatch. If there are
    more contributors than the limit, the call may need to be executed multiple times until all refunds are processed.

    Parameters:
        subtensor: Active Subtensor connection.
        wallet: Bittensor Wallet instance used to sign the transaction.
        crowdloan_id: The unique identifier of the crowdloan to refund.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Notes:
        - Can be called by any signed account (not only the creator).
        - Refunds contributors (excluding the creator) whose funds were locked in a failed campaign.
        - Each call processes a limited number of refunds (`RefundContributorsLimit`).
        - If the campaign has too many contributors, multiple refund calls are required.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        extrinsic_call = subtensor.compose_call(
            call_module="Crowdloan",
            call_function="refund",
            call_params=CrowdloanParams.refund(crowdloan_id),
        )

        return subtensor.sign_and_send_extrinsic(
            call=extrinsic_call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def update_cap_crowdloan_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    crowdloan_id: int,
    new_cap: "Balance",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Updates the fundraising cap (maximum total contribution) of a non-finalized crowdloan.

    Only the creator of the crowdloan can perform this action, and the new cap must be greater than or equal to the
    current amount already raised.

    Parameters:
        subtensor: Active Subtensor connection.
        wallet: Bittensor Wallet instance used to sign the transaction.
        crowdloan_id: The unique identifier of the crowdloan to update.
        new_cap: The new fundraising cap (in TAO or Balance).
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Notes:
        - Only the creator can update the cap.
        - The crowdloan must not be finalized.
        - The new cap must be greater than or equal to the total funds already raised.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        check_balance_amount(new_cap)

        extrinsic_call = subtensor.compose_call(
            call_module="Crowdloan",
            call_function="update_cap",
            call_params=CrowdloanParams.update_cap(crowdloan_id, new_cap),
        )

        return subtensor.sign_and_send_extrinsic(
            call=extrinsic_call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def update_end_crowdloan_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    crowdloan_id: int,
    new_end: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Updates the end block of a non-finalized crowdloan campaign.

    Only the creator of the crowdloan can perform this action. The new end block must be valid — meaning it cannot be in
    the past and must respect the minimum and maximum duration limits enforced by the chain.

    Parameters:
        subtensor: Active Subtensor connection.
        wallet: Bittensor Wallet instance used to sign the transaction.
        crowdloan_id: The unique identifier of the crowdloan to update.
        new_end: The new block number at which the crowdloan will end.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Notes:
        - Only the creator can call this extrinsic.
        - The crowdloan must not be finalized.
        - The new end block must be later than the current block and within valid duration bounds (between
            `MinimumBlockDuration` and `MaximumBlockDuration`).
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        extrinsic_call = subtensor.compose_call(
            call_module="Crowdloan",
            call_function="update_end",
            call_params=CrowdloanParams.update_end(crowdloan_id, new_end),
        )

        return subtensor.sign_and_send_extrinsic(
            call=extrinsic_call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def update_min_contribution_crowdloan_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    crowdloan_id: int,
    new_min_contribution: "Balance",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Updates the minimum contribution amount of a non-finalized crowdloan.

    Only the creator of the crowdloan can perform this action, and the new value must be greater than or equal to the
    absolute minimum contribution defined in the chain configuration.

    Parameters:
        subtensor: Active Subtensor connection.
        wallet: Bittensor Wallet instance used to sign the transaction.
        crowdloan_id: The unique identifier of the crowdloan to update.
        new_min_contribution: The new minimum contribution amount (in TAO or Balance).
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Notes:
        - Can only be called by the creator of the crowdloan.
        - The crowdloan must not be finalized.
        - The new minimum contribution must not fall below the absolute minimum defined in the runtime.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        check_balance_amount(new_min_contribution)

        extrinsic_call = subtensor.compose_call(
            call_module="Crowdloan",
            call_function="update_min_contribution",
            call_params=CrowdloanParams.update_min_contribution(
                crowdloan_id, new_min_contribution
            ),
        )

        return subtensor.sign_and_send_extrinsic(
            call=extrinsic_call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def withdraw_crowdloan_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    crowdloan_id: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Withdraws a contribution from an active (not yet finalized or dissolved) crowdloan.

    Parameters:
        subtensor: Active Subtensor connection.
        wallet: Wallet instance used to sign the transaction (must be unlocked).
        crowdloan_id: The unique identifier of the crowdloan to withdraw from.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note:
        - Regular contributors can fully withdraw their contribution before finalization.
        - The creator cannot withdraw the initial deposit, but may withdraw any amount exceeding his deposit.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        extrinsic_call = subtensor.compose_call(
            call_module="Crowdloan",
            call_function="withdraw",
            call_params=CrowdloanParams.withdraw(crowdloan_id),
        )

        return subtensor.sign_and_send_extrinsic(
            call=extrinsic_call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
