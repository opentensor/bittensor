from dataclasses import dataclass
from typing import Any

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
    def _from_dict(cls, decoded: Any) -> "SubnetInfo":
        """Returns a SubnetInfo object from decoded chain data."""
        return SubnetInfo(
            blocks_since_epoch=decoded["blocks_since_last_step"],
            burn=Balance.from_rao(decoded["burn"]),
            connection_requirements={
                str(int(netuid)): u16_normalized_float(int(req))
                for (netuid, req) in decoded["network_connect"]
            },
            difficulty=decoded["difficulty"],
            emission_value=decoded["emission_value"],
            immunity_period=decoded["immunity_period"],
            kappa=decoded["kappa"],
            max_allowed_validators=decoded["max_allowed_validators"],
            max_n=decoded["max_allowed_uids"],
            max_weight_limit=decoded["max_weights_limit"],
            min_allowed_weights=decoded["min_allowed_weights"],
            modality=decoded["network_modality"],
            netuid=decoded["netuid"],
            owner_ss58=decode_account_id(decoded["owner"]),
            rho=decoded["rho"],
            scaling_law_power=decoded["scaling_law_power"],
            subnetwork_n=decoded["subnetwork_n"],
            tempo=decoded["tempo"],
        )
