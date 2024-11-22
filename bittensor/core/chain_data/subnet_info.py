from dataclasses import dataclass

import bt_decode

from bittensor.core.chain_data.utils import decode_account_id
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance


@dataclass
class SubnetInfo:
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
    def list_from_vec_u8(cls, vec_u8: bytes) -> list["SubnetInfo"]:
        decoded = bt_decode.SubnetInfo.decode_vec_option(vec_u8)
        result = []
        for d in decoded:
            result.append(
                SubnetInfo(
                    netuid=d.netuid,
                    rho=d.rho,
                    kappa=d.kappa,
                    difficulty=d.difficulty,
                    immunity_period=d.immunity_period,
                    max_allowed_validators=d.max_allowed_validators,
                    min_allowed_weights=d.min_allowed_weights,
                    max_weight_limit=d.max_weights_limit,
                    scaling_law_power=d.scaling_law_power,
                    subnetwork_n=d.subnetwork_n,
                    max_n=d.max_allowed_uids,
                    blocks_since_epoch=d.blocks_since_last_step,
                    tempo=d.tempo,
                    modality=d.network_modality,
                    connection_requirements={
                        str(int(netuid)): u16_normalized_float(int(req))
                        for (netuid, req) in d.network_connect
                    },
                    emission_value=d.emission_values,
                    burn=Balance.from_rao(d.burn),
                    owner_ss58=decode_account_id(d.owner),
                )
            )
        return result
