from dataclasses import dataclass
from typing import List


@dataclass
class DelegateInfoLite:
    """
    Dataclass for `DelegateLiteInfo`. This is a lighter version of :func:`DelegateInfo`.

    Args:
        delegate_ss58 (str): Hotkey of the delegate for which the information is being fetched.
        take (float): Take of the delegate as a percentage.
        nominators (int): Count of the nominators of the delegate.
        owner_ss58 (str): Coldkey of the owner.
        registrations (List[int]): List of subnets that the delegate is registered on.
        validator_permits (List[int]): List of subnets that the delegate is allowed to validate on.
        return_per_1000 (int): Return per 1000 TAO for the delegate over a day.
        total_daily_return (int): Total daily return of the delegate.
    """

    delegate_ss58: str  # Hotkey of delegate
    take: float  # Take of the delegate as a percentage
    nominators: int  # Count of the nominators of the delegate
    owner_ss58: str  # Coldkey of owner
    registrations: List[int]  # List of subnets the delegate is registered on
    validator_permits: List[int]  # Subnets the delegate is allowed to validate on
    return_per_1000: int  # Return per 1000 TAO for the delegate over a day
    total_daily_return: int  # Total daily return of the delegate

    @staticmethod
    def from_raw_data(
        delegate_ss58: str,
        take: float,
        nominators: int,
        owner_ss58: str,
        registrations: List,
        validator_permits: List,
        return_per_1000: int,
        total_daily_return: int,
    ) -> "DelegateInfoLite":
        """
        Create a `DelegateInfoLite` instance from raw data with proper type conversions.

        Args:
            delegate_ss58 (str): Delegate's hotkey.
            take (float): Delegate's take percentage.
            nominators (int): Number of nominators.
            owner_ss58 (str): Delegate's coldkey.
            registrations (List): Raw list of registrations (to be converted to List[int]).
            validator_permits (List): Raw list of validator permits (to be converted to List[int]).
            return_per_1000 (int): Raw return per 1000 TAO.
            total_daily_return (int): Raw total daily return.

        Returns:
            DelegateInfoLite: A properly initialized instance.
        """
        # Ensure registrations and validator_permits are lists of integers
        registrations_fixed = [int(r) for r in registrations]
        validator_permits_fixed = [int(v) for v in validator_permits]

        # Create the DelegateInfoLite object
        return DelegateInfoLite(
            delegate_ss58=delegate_ss58,
            take=round(take, 6),  # Ensure take is rounded to 6 decimal places
            nominators=int(nominators),  # Ensure nominators is an integer
            owner_ss58=owner_ss58,
            registrations=registrations_fixed,
            validator_permits=validator_permits_fixed,
            return_per_1000=int(
                return_per_1000
            ),  # Ensure return_per_1000 is an integer
            total_daily_return=int(
                total_daily_return
            ),  # Ensure total_daily_return is an integer
        )
