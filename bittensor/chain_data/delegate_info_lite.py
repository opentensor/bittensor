from dataclasses import dataclass
from typing import Any, List, Optional

from substrateinterface.utils.ss58 import ss58_encode

from bittensor.chain_data.utils import from_scale_encoding, ChainDataType, SS58_FORMAT
from bittensor.utils import U16_NORMALIZED_FLOAT
from bittensor.utils.balance import Balance


@dataclass
class DelegateInfoLite:
    """
    Dataclass for light delegate information.

    Args:
        hotkey_ss58 (str): Hotkey of the delegate for which the information is being fetched.
        owner_ss58 (str): Coldkey of the owner.
        total_stake (int): Total stake of the delegate.
        owner_stake (int): Own stake of the delegate.
        take (float): Take of the delegate as a percentage. None if custom
    """

    hotkey_ss58: str  # Hotkey of delegate
    owner_ss58: str  # Coldkey of owner
    take: float
    total_stake: Balance  # Total stake of the delegate
    previous_total_stake: Balance  # Total stake of the delegate
    owner_stake: Balance  # Own stake of the delegate

    @classmethod
    def fix_decoded_values(cls, decoded: Any) -> "DelegateInfoLite":
        """Fixes the decoded values."""
        decoded_take = decoded["take"]

        if decoded_take == 65535:
            fixed_take = None
        else:
            fixed_take = U16_NORMALIZED_FLOAT(decoded_take)

        return cls(
            hotkey_ss58=ss58_encode(decoded["delegate_ss58"], SS58_FORMAT),
            owner_ss58=ss58_encode(decoded["owner_ss58"], SS58_FORMAT),
            take=fixed_take,
            total_stake=Balance.from_rao(decoded["total_stake"]),
            owner_stake=Balance.from_rao(decoded["owner_stake"]),
            previous_total_stake=None,
        )

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["DelegateInfoLite"]:
        """Returns a DelegateInfoLight object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.DelegateInfoLight)

        if decoded is None:
            return None

        decoded = DelegateInfoLite.fix_decoded_values(decoded)

        return decoded

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["DelegateInfoLite"]:
        """Returns a list of DelegateInfoLight objects from a ``vec_u8``."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.DelegateInfoLight, is_vec=True
        )

        if decoded is None:
            return []

        decoded = [DelegateInfoLite.fix_decoded_values(d) for d in decoded]

        return decoded
