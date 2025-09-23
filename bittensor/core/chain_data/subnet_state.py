"""
This module defines the `SubnetState` data class and associated methods for handling and decoding subnetwork states in
the Bittensor network.
"""

from dataclasses import dataclass

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance


@dataclass
class SubnetState(InfoBase):
    netuid: int
    hotkeys: list[str]
    coldkeys: list[str]
    active: list[bool]
    validator_permit: list[bool]
    pruning_score: list[float]
    last_update: list[int]
    emission: list["Balance"]
    dividends: list[float]
    incentives: list[float]
    consensus: list[float]
    trust: list[float]
    rank: list[float]
    block_at_registration: list[int]
    alpha_stake: list["Balance"]
    tao_stake: list["Balance"]
    total_stake: list["Balance"]
    emission_history: list[list[int]]

    @classmethod
    def _from_dict(cls, decoded: dict) -> "SubnetState":
        """Returns a SubnetState object from decoded chain data."""
        netuid = decoded["netuid"]
        return SubnetState(
            netuid=netuid,
            hotkeys=[decode_account_id(hk) for hk in decoded.get("hotkeys", [])],
            coldkeys=[decode_account_id(ck) for ck in decoded.get("coldkeys", [])],
            active=decoded["active"],
            validator_permit=decoded["validator_permit"],
            pruning_score=[
                u16_normalized_float(val) for val in decoded["pruning_score"]
            ],
            last_update=decoded["last_update"],
            emission=[
                Balance.from_rao(val).set_unit(netuid) for val in decoded["emission"]
            ],
            dividends=[u16_normalized_float(val) for val in decoded["dividends"]],
            incentives=[u16_normalized_float(val) for val in decoded["incentives"]],
            consensus=[u16_normalized_float(val) for val in decoded["consensus"]],
            trust=[u16_normalized_float(val) for val in decoded["trust"]],
            rank=[u16_normalized_float(val) for val in decoded["rank"]],
            block_at_registration=decoded["block_at_registration"],
            alpha_stake=[
                Balance.from_rao(val).set_unit(netuid) for val in decoded["alpha_stake"]
            ],
            tao_stake=[
                Balance.from_rao(val).set_unit(0) for val in decoded["tao_stake"]
            ],
            total_stake=[
                Balance.from_rao(val).set_unit(netuid) for val in decoded["total_stake"]
            ],
            emission_history=decoded["emission_history"],
        )
