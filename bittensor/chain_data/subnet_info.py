from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from substrateinterface.utils.ss58 import ss58_encode

from bittensor.chain_data.utils import SS58_FORMAT, ChainDataType, from_scale_encoding
from bittensor.utils import U16_NORMALIZED_FLOAT
from bittensor.utils.balance import Balance
from bittensor.utils.registration import torch, use_torch


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
    # netuid -> topk percentile prunning score requirement (u16:MAX normalized.)
    connection_requirements: Dict[str, float]
    emission_value: float
    burn: Balance
    owner_ss58: str

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["SubnetInfo"]:
        """Returns a SubnetInfo object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.SubnetInfo)
        if decoded is None:
            return None

        return SubnetInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["SubnetInfo"]:
        """Returns a list of SubnetInfo objects from a ``vec_u8``."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.SubnetInfo, is_vec=True, is_option=True
        )

        if decoded is None:
            return []

        return [SubnetInfo.fix_decoded_values(d) for d in decoded]

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> "SubnetInfo":
        """Returns a SubnetInfo object from a decoded SubnetInfo dictionary."""
        return SubnetInfo(
            netuid=decoded["netuid"],
            rho=decoded["rho"],
            kappa=decoded["kappa"],
            difficulty=decoded["difficulty"],
            immunity_period=decoded["immunity_period"],
            max_allowed_validators=decoded["max_allowed_validators"],
            min_allowed_weights=decoded["min_allowed_weights"],
            max_weight_limit=decoded["max_weights_limit"],
            scaling_law_power=decoded["scaling_law_power"],
            subnetwork_n=decoded["subnetwork_n"],
            max_n=decoded["max_allowed_uids"],
            blocks_since_epoch=decoded["blocks_since_last_step"],
            tempo=decoded["tempo"],
            modality=decoded["network_modality"],
            connection_requirements={
                str(int(netuid)): U16_NORMALIZED_FLOAT(int(req))
                for netuid, req in decoded["network_connect"]
            },
            emission_value=decoded["emission_values"],
            burn=Balance.from_rao(decoded["burn"]),
            owner_ss58=ss58_encode(decoded["owner"], SS58_FORMAT),
        )

    def to_parameter_dict(self) -> Union[dict[str, Any], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet info."""
        if use_torch():
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "SubnetInfo":
        if use_torch():
            return cls(**dict(parameter_dict))
        else:
            return cls(**parameter_dict)
