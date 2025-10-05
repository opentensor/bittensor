from dataclasses import dataclass
from typing import Optional

from bittensor_wallet.utils import SS58_FORMAT
from scalecodec.utils.ss58 import ss58_encode

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.utils import from_scale_encoding, ChainDataType


@dataclass
class ScheduledColdkeySwapInfo(InfoBase):
    """
    The `ScheduledColdkeySwapInfo` class is a dataclass representing information about scheduled cold key swaps.

    Attributes:
        old_coldkey: The old cold key before the swap.
        new_coldkey: The new cold key after the swap.
        arbitration_block: The block number at which the arbitration of the swap will take place.
    """

    old_coldkey: str
    new_coldkey: str
    arbitration_block: int

    @classmethod
    def _from_dict(cls, decoded: dict) -> "ScheduledColdkeySwapInfo":
        """Returns a ScheduledColdkeySwapInfo object from decoded chain data."""
        return cls(
            arbitration_block=decoded["arbitration_block"],
            new_coldkey=ss58_encode(decoded["new_coldkey"], SS58_FORMAT),
            old_coldkey=ss58_encode(decoded["old_coldkey"], SS58_FORMAT),
        )

    @classmethod
    def decode_account_id_list(cls, vec_u8: list[int]) -> Optional[list[str]]:
        """Decodes a list of AccountIds from vec_u8."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.ScheduledColdkeySwapInfo.AccountId, is_vec=True
        )
        if decoded is None:
            return None
        return [ss58_encode(account_id, SS58_FORMAT) for account_id in decoded]
