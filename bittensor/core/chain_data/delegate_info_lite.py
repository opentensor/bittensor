from dataclasses import dataclass

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance


@dataclass
class DelegateInfoLite(InfoBase):
    """
    Dataclass for `DelegateLiteInfo`. This is a lighter version of :func:``DelegateInfo``.

    Parameters:
        delegate_ss58: Hotkey of the delegate for which the information is being fetched.
        take: Take of the delegate as a percentage.
        nominators: Count of the nominators of the delegate.
        owner_ss58: Coldkey of the owner.
        registrations: List of subnets that the delegate is registered on.
        validator_permits: List of subnets that the delegate is allowed to validate on.
        return_per_1000: Return per 1000 TAO, for the delegate over a day.
    """

    delegate_ss58: str  # Hotkey of delegate
    take: float  # Take of the delegate as a percentage
    nominators: int  # Count of the nominators of the delegate.
    owner_ss58: str  # Coldkey of owner
    registrations: list[int]  # List of subnets that the delegate is registered on
    validator_permits: list[
        int
    ]  # List of subnets that the delegate is allowed to validate on
    return_per_1000: Balance  # Return per 1000 tao for the delegate over a day

    @classmethod
    def _from_dict(cls, decoded: dict) -> "DelegateInfoLite":
        return DelegateInfoLite(
            delegate_ss58=decode_account_id(decoded["delegate_ss58"]),
            take=u16_normalized_float(decoded["take"]),
            nominators=decoded["nominators"],
            owner_ss58=decode_account_id(decoded["owner_ss58"]),
            registrations=decoded["registrations"],
            validator_permits=decoded["validator_permits"],
            return_per_1000=Balance.from_rao(decoded["return_per_1000"]),
        )
