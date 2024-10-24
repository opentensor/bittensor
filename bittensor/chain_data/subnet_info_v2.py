from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from bittensor.utils.balance import Balance
from bittensor.chain_data.utils import SS58_FORMAT, ChainDataType, from_scale_encoding
from .subnet_hyperparameters import SubnetHyperparameters
from bittensor.utils.registration import torch, use_torch
from substrateinterface.utils.ss58 import ss58_encode
from .dynamic_pool import DynamicPool


@dataclass
class SubnetInfoV2:
    """Dataclass for subnet info."""
    netuid: int
    owner_ss58: str
    max_allowed_validators: int
    scaling_law_power: float
    subnetwork_n: int
    max_n: int
    blocks_since_epoch: int
    modality: int
    emission_value: float
    burn: Balance
    tao_locked: Balance
    hyperparameters: "SubnetHyperparameters"

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["SubnetInfoV2"]:
        """Returns a SubnetInfoV2 object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.SubnetInfoV2)

        if decoded is None:
            return None

        return SubnetInfoV2.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["SubnetInfoV2"]:
        """Returns a list of SubnetInfoV2 objects from a ``vec_u8``."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.SubnetInfoV2, is_vec=True, is_option=False
        )

        if decoded is None:
            return []

        decoded = [SubnetInfoV2.fix_decoded_values(d) for d in decoded]

        return decoded

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> "SubnetInfoV2":
        """Returns a SubnetInfoV2 object from a decoded SubnetInfoV2 dictionary."""
       

        return SubnetInfoV2(
            netuid=decoded["netuid"],
            owner_ss58=ss58_encode(decoded["owner"], SS58_FORMAT),
            max_allowed_validators=decoded["max_allowed_validators"],
            scaling_law_power=decoded["scaling_law_power"],
            subnetwork_n=decoded["subnetwork_n"],
            max_n=decoded["max_allowed_uids"],
            blocks_since_epoch=decoded["blocks_since_last_step"],
            modality=decoded["network_modality"],
            emission_value=decoded["emission_values"],
            burn=Balance.from_rao(decoded["burn"]),
            tao_locked=Balance.from_rao(decoded["tao_locked"]),
            hyperparameters=decoded["hyperparameters"],
            rho=decoded["rho"],
            kappa=decoded["kappa"],
            difficulty=decoded["difficulty"],
            immunity_period=decoded["immunity_period"],
            min_allowed_weights=decoded["min_allowed_weights"],
            max_weights_limit=decoded["max_weights_limit"],
        )

    def _to_parameter_dict(self, return_type: str) -> Union[dict[str, Any], "torch.nn.ParameterDict"]:
        if return_type == "torch":
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    def to_parameter_dict(self) -> Union[dict[str, Any], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet info."""
        if use_torch():
            return self._to_parameter_dict("torch")
        else:
            return self._to_parameter_dict("numpy")

    @classmethod
    def _from_parameter_dict_torch(cls, parameter_dict: "torch.nn.ParameterDict") -> "SubnetInfoV2":
        """Returns a SubnetInfoV2 object from a torch parameter_dict."""
        return cls(**dict(parameter_dict))

    @classmethod
    def _from_parameter_dict_numpy(cls, parameter_dict: dict[str, Any]) -> "SubnetInfoV2":
        """Returns a SubnetInfoV2 object from a parameter_dict."""
        return cls(**parameter_dict)

    @classmethod
    def from_parameter_dict(cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]) -> "SubnetInfoV2":
        if use_torch():
            return cls._from_parameter_dict_torch(parameter_dict)
        else:
            return cls._from_parameter_dict_numpy(parameter_dict)
