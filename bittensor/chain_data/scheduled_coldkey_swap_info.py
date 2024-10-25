from dataclasses import dataclass
from typing import Any, List, Optional

from substrateinterface.utils.ss58 import ss58_encode

from bittensor.chain_data.utils import ChainDataType, from_scale_encoding, SS58_FORMAT


@dataclass
class ScheduledColdkeySwapInfo:
    """Dataclass for scheduled coldkey swap information."""

    old_coldkey: str
    new_coldkey: str
    arbitration_block: int

    @classmethod
    def fix_decoded_values(cls, decoded: Any) -> "ScheduledColdkeySwapInfo":
        """Fixes the decoded values."""
        return cls(
            old_coldkey=ss58_encode(decoded["old_coldkey"], SS58_FORMAT),
            new_coldkey=ss58_encode(decoded["new_coldkey"], SS58_FORMAT),
            arbitration_block=decoded["arbitration_block"],
        )

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["ScheduledColdkeySwapInfo"]:
        """Returns a ScheduledColdkeySwapInfo object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.ScheduledColdkeySwapInfo)
        if decoded is None:
            return None

        return ScheduledColdkeySwapInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["ScheduledColdkeySwapInfo"]:
        """Returns a list of ScheduledColdkeySwapInfo objects from a ``vec_u8``."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.ScheduledColdkeySwapInfo, is_vec=True
        )
        if decoded is None:
            return []

        return [ScheduledColdkeySwapInfo.fix_decoded_values(d) for d in decoded]

    @classmethod
    def decode_account_id_list(cls, vec_u8: List[int]) -> Optional[List[str]]:
        """Decodes a list of AccountIds from vec_u8."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.ScheduledColdkeySwapInfo.AccountId, is_vec=True
        )
        if decoded is None:
            return None
        return [ss58_encode(account_id, SS58_FORMAT) for account_id in decoded]
