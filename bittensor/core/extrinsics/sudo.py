from typing import Optional, TYPE_CHECKING

from bittensor.core.extrinsics.utils import sudo_call_extrinsic
from bittensor.core.types import Weights as MaybeSplit
from bittensor.utils.weight_utils import convert_maybe_split_to_u16

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.types import ExtrinsicResponse


def sudo_set_admin_freeze_window_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    window: int,
    *,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Sets the admin freeze window length (in blocks) at the end of a tempo.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor Wallet instance.
        window: The amount of blocks to freeze in the end of a tempo.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    call_function = "sudo_set_admin_freeze_window"
    call_params = {"window": window}
    return sudo_call_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        call_function=call_function,
        call_params=call_params,
        period=period,
        raise_error=raise_error,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )


def sudo_set_mechanism_count_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    mech_count: int,
    *,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Sets the number of subnet mechanisms.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor Wallet instance.
        netuid: The subnet unique identifier.
        mech_count: The amount of subnet mechanism to be set.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    call_function = "sudo_set_mechanism_count"
    call_params = {"netuid": netuid, "mechanism_count": mech_count}
    return sudo_call_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        call_function=call_function,
        call_params=call_params,
        period=period,
        raise_error=raise_error,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )


def sudo_set_mechanism_emission_split_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    maybe_split: MaybeSplit,
    *,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> "ExtrinsicResponse":
    """
    Sets the emission split between mechanisms in a provided subnet.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor Wallet instance.
        netuid: The subnet unique identifier.
        maybe_split: List of emission weights (positive integers) for each subnet mechanism.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Note:
        The `maybe_split` list defines the relative emission share for each subnet mechanism.
        Its length must match the number of active mechanisms in the subnet or be shorter, but not equal to zero. For
        example, [3, 1, 1] distributes emissions in a 3:1:1 ratio across subnet mechanisms 0, 1, and 2. Each mechanism's
        emission share is calculated as: share[i] = maybe_split[i] / sum(maybe_split)
    """
    call_function = "sudo_set_mechanism_emission_split"
    call_params = {
        "netuid": netuid,
        "maybe_split": convert_maybe_split_to_u16(maybe_split),
    }
    return sudo_call_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        call_function=call_function,
        call_params=call_params,
        period=period,
        raise_error=raise_error,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
