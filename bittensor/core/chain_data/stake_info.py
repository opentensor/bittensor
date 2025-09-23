from dataclasses import dataclass

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils.balance import Balance


@dataclass
class StakeInfo(InfoBase):
    """
    Dataclass for representing stake information linked to hotkey and coldkey pairs.

    Attributes:
        hotkey_ss58: The SS58 encoded hotkey address.
        coldkey_ss58: The SS58 encoded coldkey address.
        stake: The stake associated with the hotkey-coldkey pair, represented as a Balance object.
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
    def from_dict(cls, decoded: dict) -> "StakeInfo":
        """Returns a StakeInfo object from decoded chain data."""
        netuid = decoded["netuid"]
        return cls(
            hotkey_ss58=decode_account_id(decoded["hotkey"]),
            coldkey_ss58=decode_account_id(decoded["coldkey"]),
            netuid=int(netuid),
            stake=Balance.from_rao(decoded["stake"]).set_unit(netuid),
            locked=Balance.from_rao(decoded["locked"]).set_unit(netuid),
            emission=Balance.from_rao(decoded["emission"]).set_unit(netuid),
            drain=int(decoded["drain"]),
            is_registered=bool(decoded["is_registered"]),
        )
