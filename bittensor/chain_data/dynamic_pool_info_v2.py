from dataclasses import dataclass
from typing import Any, List, Optional, Union

from bittensor.chain_data.utils import ChainDataType, from_scale_encoding
from bittensor.utils.registration import torch, use_torch


@dataclass
class DynamicPoolInfoV2:
    """Dataclass for dynamic pool info."""

    netuid: int
    alpha_issuance: int
    alpha_outstanding: int
    alpha_reserve: int
    tao_reserve: int
    k: int

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["DynamicPoolInfoV2"]:
        """Returns a DynamicPoolInfoV2 object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None
        return from_scale_encoding(vec_u8, ChainDataType.DynamicPoolInfoV2)

    def _to_parameter_dict(
        self, return_type: str
    ) -> Union[dict[str, Any], "torch.nn.ParameterDict"]:
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
    def _from_parameter_dict_torch(
        cls, parameter_dict: "torch.nn.ParameterDict"
    ) -> "DynamicPoolInfoV2":
        """Returns a DynamicPoolInfoV2 object from a torch parameter_dict."""
        return cls(**dict(parameter_dict))

    @classmethod
    def _from_parameter_dict_numpy(
        cls, parameter_dict: dict[str, Any]
    ) -> "DynamicPoolInfoV2":
        """Returns a DynamicPoolInfoV2 object from a parameter_dict."""
        return cls(**parameter_dict)

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "DynamicPoolInfoV2":
        if use_torch():
            return cls._from_parameter_dict_torch(parameter_dict)
        else:
            return cls._from_parameter_dict_numpy(parameter_dict)
