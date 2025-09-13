from typing import Optional, TYPE_CHECKING

from bittensor.core.extrinsics.utils import sudo_call_extrinsic

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def sudo_set_admin_freez_window(
    subtensor: "Subtensor",
    wallet: "Wallet",
    window: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> tuple[bool, str]:
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
        tuple[bool, str]:
            `True` if the extrinsic executed successfully, `False` otherwise.
            `message` is a string value describing the success or potential error.
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


def sudo_set_sub_subnet_count_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    sub_count: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> tuple[bool, str]:
    """
    Sets the number of sub-subnets in the subnet.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor Wallet instance.
        netuid: The subnet unique identifier.
        sub_count: The amount of sub-subnets in the subnet to be set.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        tuple[bool, str]:
            `True` if the extrinsic executed successfully, `False` otherwise.
            `message` is a string value describing the success or potential error.
    """
    call_function = "sudo_set_subsubnet_count"
    call_params = {"netuid": netuid, "subsub_count": sub_count}
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


def sudo_set_sub_subnet_emission_split(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    maybe_split: list[int],
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> tuple[bool, str]:
    """
    Sets the emission split between sub-subnets in a provided subnet.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor Wallet instance.
        netuid: The subnet unique identifier.
        maybe_split: List of emission weights (positive integers) for each sub-subnet.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        tuple[bool, str]:
            `True` if the extrinsic executed successfully, `False` otherwise.
            `message` is a string value describing the success or potential error.

    Note:
        The `maybe_split` list defines the relative emission share for each sub-subnet.
        Its length must match the number of active sub-subnets in the subnet. For example, [3, 1, 1] distributes
        emissions in a 3:1:1 ratio across sub-subnets 0, 1, and 2. Each sub-subnet's emission share is calculated as:
        share[i] = maybe_split[i] / sum(maybe_split)
    """
    call_function = "sudo_set_subsubnet_emission_split"
    call_params = {"netuid": netuid, "maybe_split": maybe_split}
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
