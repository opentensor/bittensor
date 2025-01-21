from dataclasses import dataclass
from typing import Any, Optional

import bt_decode
from scalecodec.utils.ss58 import ss58_encode

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.utils import (
    decode_account_id,
    from_scale_encoding_using_type_string,
)
from bittensor.core.settings import SS58_FORMAT
from bittensor.utils.balance import Balance


@dataclass
class StakeInfo(InfoBase):
    """
    Dataclass for representing stake information linked to hotkey and coldkey pairs.

    Attributes:
        hotkey_ss58 (str): The SS58 encoded hotkey address.
        coldkey_ss58 (str): The SS58 encoded coldkey address.
        stake (Balance): The stake associated with the hotkey-coldkey pair, represented as a Balance object.
    """

    hotkey_ss58: str  # Hotkey address
    coldkey_ss58: str  # Coldkey address
    netuid: int  # Network UID
    stake: Balance  # Stake for the hotkey-coldkey pair
    locked: Balance  # Stake which is locked.
    emission: Balance  # Emission for the hotkey-coldkey pair
    drain: int
    is_registered: bool

    @classmethod
    def fix_decoded_values(cls, decoded: Any) -> "StakeInfo":
        """Fixes the decoded values."""
        return cls(
            hotkey_ss58=ss58_encode(decoded["hotkey"], SS58_FORMAT),
            coldkey_ss58=ss58_encode(decoded["coldkey"], SS58_FORMAT),
            netuid=int(decoded["netuid"]),
            stake=Balance.from_rao(decoded["stake"]).set_unit(decoded["netuid"]),
            locked=Balance.from_rao(decoded["locked"]).set_unit(decoded["netuid"]),
            emission=Balance.from_rao(decoded["emission"]).set_unit(decoded["netuid"]),
            drain=int(decoded["drain"]),
            is_registered=bool(decoded["is_registered"]),
        )

    @classmethod
    def _fix_decoded(cls, decoded: Any) -> "StakeInfo":
        hotkey = decode_account_id(decoded.hotkey)
        coldkey = decode_account_id(decoded.coldkey)
        stake = Balance.from_rao(decoded.stake)

        return StakeInfo(hotkey, coldkey, stake)

    @classmethod
    def from_vec_u8(cls, vec_u8: list[int]) -> Optional["StakeInfo"]:
        """Returns a StakeInfo object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = bt_decode.StakeInfo.decode(vec_u8)
        if decoded is None:
            return None

        return StakeInfo.fix_decoded_values(decoded)

    @classmethod
    def list_of_tuple_from_vec_u8(
        cls, vec_u8: list[int]
    ) -> dict[str, list["StakeInfo"]]:
        """Returns a list of StakeInfo objects from a ``vec_u8``."""
        decoded: Optional[list[tuple[str, list[object]]]] = (
            from_scale_encoding_using_type_string(
                input_=vec_u8, type_string="Vec<(AccountId, Vec<StakeInfo>)>"
            )
        )

        if decoded is None:
            return {}

        return {
            ss58_encode(address=account_id, ss58_format=SS58_FORMAT): [
                StakeInfo.fix_decoded_values(d) for d in stake_info
            ]
            for account_id, stake_info in decoded
        }

    @classmethod
    def list_from_vec_u8(cls, vec_u8: bytes) -> list["StakeInfo"]:
        """Returns a list of StakeInfo objects from a ``vec_u8``."""
        decoded = bt_decode.StakeInfo.decode_vec(vec_u8)
        if decoded is None:
            return []

        return [StakeInfo.fix_decoded_values(d) for d in decoded]
