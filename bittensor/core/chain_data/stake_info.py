from dataclasses import dataclass

from scalecodec.utils.ss58 import ss58_encode

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.utils import decode_account_id
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
    def fix_decoded_values(cls, decoded: dict) -> "StakeInfo":
        """Fixes the decoded values."""
        netuid = decoded["netuid"]
        return cls(
            hotkey_ss58=ss58_encode(decoded["hotkey"], SS58_FORMAT),
            coldkey_ss58=ss58_encode(decoded["coldkey"], SS58_FORMAT),
            netuid=int(netuid),
            stake=Balance.from_rao(decoded["stake"]).set_unit(netuid),
            locked=Balance.from_rao(decoded["locked"]).set_unit(netuid),
            emission=Balance.from_rao(decoded["emission"]).set_unit(netuid),
            drain=int(decoded["drain"]),
            is_registered=bool(decoded["is_registered"]),
        )

    @classmethod
    def _fix_decoded(cls, decoded: dict) -> "StakeInfo":
        hotkey = decode_account_id(decoded.hotkey)
        coldkey = decode_account_id(decoded.coldkey)
        stake = Balance.from_rao(decoded.stake)

        return StakeInfo(hotkey, coldkey, stake)
