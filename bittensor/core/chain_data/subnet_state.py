"""
This module defines the `SubnetState` data class and associated methods for handling and decoding
subnetwork states in the Bittensor network.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from scalecodec.utils.ss58 import ss58_encode

from bittensor.core.chain_data.utils import (
    ChainDataType,
    from_scale_encoding,
    SS58_FORMAT,
)
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance


@dataclass
class SubnetState:
    netuid: int
    hotkeys: List[str]
    coldkeys: List[str]
    active: List[bool]
    validator_permit: List[bool]
    pruning_score: List[float]
    last_update: List[int]
    emission: List[Balance]
    dividends: List[float]
    incentives: List[float]
    consensus: List[float]
    trust: List[float]
    rank: List[float]
    block_at_registration: List[int]
    local_stake: List[Balance]
    global_stake: List[Balance]
    stake_weight: List[float]
    emission_history: List[List[int]]

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["SubnetState"]:
        if len(vec_u8) == 0:
            return None
        decoded = from_scale_encoding(vec_u8, ChainDataType.SubnetState, is_option=True)
        if decoded is None:
            return None
        return SubnetState.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["SubnetState"]:
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.SubnetState, is_vec=True, is_option=True
        )
        if decoded is None:
            return []
        decoded = [SubnetState.fix_decoded_values(d) for d in decoded]
        return decoded

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> "SubnetState":
        netuid = decoded["netuid"]
        return SubnetState(
            netuid=netuid,
            hotkeys=[ss58_encode(val, SS58_FORMAT) for val in decoded["hotkeys"]],
            coldkeys=[ss58_encode(val, SS58_FORMAT) for val in decoded["coldkeys"]],
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
            local_stake=[
                Balance.from_rao(val).set_unit(netuid) for val in decoded["local_stake"]
            ],
            global_stake=[
                Balance.from_rao(val).set_unit(0) for val in decoded["global_stake"]
            ],
            stake_weight=[u16_normalized_float(val) for val in decoded["stake_weight"]],
            emission_history=decoded["emission_history"],
        )
