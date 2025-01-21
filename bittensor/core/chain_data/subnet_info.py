from dataclasses import dataclass
from typing import Any

import bt_decode

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance


@dataclass
class SubnetInfo(InfoBase):
    """Dataclass for subnet info."""

    netuid: int
    rho: int
    kappa: int
    difficulty: int
    immunity_period: int
    max_allowed_validators: int
    min_allowed_weights: int
    max_weight_limit: float
    scaling_law_power: float
    subnetwork_n: int
    max_n: int
    blocks_since_epoch: int
    tempo: int
    modality: int
    connection_requirements: dict[str, float]
    emission_value: float
    burn: Balance
    owner_ss58: str

    @classmethod
    def _fix_decoded(cls, decoded: Any) -> "SubnetInfo":
        return SubnetInfo(
            netuid=decoded.netuid,
            rho=decoded.rho,
            kappa=decoded.kappa,
            difficulty=decoded.difficulty,
            immunity_period=decoded.immunity_period,
            max_allowed_validators=decoded.max_allowed_validators,
            min_allowed_weights=decoded.min_allowed_weights,
            max_weight_limit=decoded.max_weights_limit,
            scaling_law_power=decoded.scaling_law_power,
            subnetwork_n=decoded.subnetwork_n,
            max_n=decoded.max_allowed_uids,
            blocks_since_epoch=decoded.blocks_since_last_step,
            tempo=decoded.tempo,
            modality=decoded.network_modality,
            connection_requirements={
                str(int(netuid)): u16_normalized_float(int(req))
                for (netuid, req) in decoded.network_connect
            },
            emission_value=decoded.emission_values,
            burn=Balance.from_rao(decoded.burn),
            owner_ss58=decode_account_id(decoded.owner),
        )

    @classmethod
    def list_from_vec_u8(cls, vec_u8: bytes) -> list["SubnetInfo"]:
        decoded = bt_decode.SubnetInfo.decode_vec_option(vec_u8)
        return [cls._fix_decoded(d) for d in decoded]
