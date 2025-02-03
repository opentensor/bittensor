from dataclasses import dataclass

from bittensor.core.chain_data.info_base import InfoBase


@dataclass
class DelegateInfoLite(InfoBase):
    """
    Dataclass for `DelegateLiteInfo`. This is a lighter version of :func:``DelegateInfo``.

    Args:
        delegate_ss58 (str): Hotkey of the delegate for which the information is being fetched.
        take (float): Take of the delegate as a percentage.
        nominators (int): Count of the nominators of the delegate.
        owner_ss58 (str): Coldkey of the owner.
        registrations (list[int]): List of subnets that the delegate is registered on.
        validator_permits (list[int]): List of subnets that the delegate is allowed to validate on.
        return_per_1000 (int): Return per 1000 TAO, for the delegate over a day.
        total_daily_return (int): Total daily return of the delegate.
    """

    delegate_ss58: str  # Hotkey of delegate
    take: float  # Take of the delegate as a percentage
    nominators: int  # Count of the nominators of the delegate.
    owner_ss58: str  # Coldkey of owner
    registrations: list[int]  # List of subnets that the delegate is registered on
    validator_permits: list[
        int
    ]  # List of subnets that the delegate is allowed to validate on
    return_per_1000: int  # Return per 1000 tao for the delegate over a day
    total_daily_return: int  # Total daily return of the delegate
