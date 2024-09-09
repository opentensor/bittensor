from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union

import bt_decode

from bittensor.core.chain_data.utils import from_scale_encoding, ChainDataType
from bittensor.utils.registration import torch, use_torch


@dataclass
class SubnetHyperparameters:
    """Dataclass for subnet hyperparameters."""

    rho: int
    kappa: int
    immunity_period: int
    min_allowed_weights: int
    max_weight_limit: float
    tempo: int
    min_difficulty: int
    max_difficulty: int
    weights_version: int
    weights_rate_limit: int
    adjustment_interval: int
    activity_cutoff: int
    registration_allowed: bool
    target_regs_per_interval: int
    min_burn: int
    max_burn: int
    bonds_moving_avg: int
    max_regs_per_block: int
    serving_rate_limit: int
    max_validators: int
    adjustment_alpha: int
    difficulty: int
    commit_reveal_weights_interval: int
    commit_reveal_weights_enabled: bool
    alpha_high: int
    alpha_low: int
    liquid_alpha_enabled: bool

    @classmethod
    def from_vec_u8(cls, vec_u8: bytes) -> Optional["SubnetHyperparameters"]:
        decoded = bt_decode.SubnetHyperparameters.decode(vec_u8)
        return SubnetHyperparameters(
            rho=decoded.rho,
            kappa=decoded.kappa,
            immunity_period=decoded.immunity_period,
            min_allowed_weights=decoded.min_allowed_weights,
            max_weight_limit=decoded.max_weights_limit,
            tempo=decoded.tempo,
            min_difficulty=decoded.min_difficulty,
            max_difficulty=decoded.max_difficulty,
            weights_version=decoded.weights_version,
            weights_rate_limit=decoded.weights_rate_limit,
            adjustment_interval=decoded.adjustment_interval,
            activity_cutoff=decoded.activity_cutoff,
            registration_allowed=decoded.registration_allowed,
            target_regs_per_interval=decoded.target_regs_per_interval,
            min_burn=decoded.min_burn,
            max_burn=decoded.max_burn,
            bonds_moving_avg=decoded.bonds_moving_avg,
            max_regs_per_block=decoded.max_regs_per_block,
            serving_rate_limit=decoded.serving_rate_limit,
            max_validators=decoded.max_validators,
            adjustment_alpha=decoded.adjustment_alpha,
            difficulty=decoded.difficulty,
            commit_reveal_weights_interval=decoded.commit_reveal_weights_interval,
            commit_reveal_weights_enabled=decoded.commit_reveal_weights_enabled,
            alpha_high=decoded.alpha_high,
            alpha_low=decoded.alpha_low,
            liquid_alpha_enabled=decoded.liquid_alpha_enabled,
        )

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["SubnetHyperparameters"]:
        """Returns a list of SubnetHyperparameters objects from a ``vec_u8``."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.SubnetHyperparameters, is_vec=True, is_option=True
        )
        if decoded is None:
            return []

        return [SubnetHyperparameters.fix_decoded_values(d) for d in decoded]

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> "SubnetHyperparameters":
        """Returns a SubnetInfo object from a decoded SubnetInfo dictionary."""
        return SubnetHyperparameters(
            rho=decoded["rho"],
            kappa=decoded["kappa"],
            immunity_period=decoded["immunity_period"],
            min_allowed_weights=decoded["min_allowed_weights"],
            max_weight_limit=decoded["max_weights_limit"],
            tempo=decoded["tempo"],
            min_difficulty=decoded["min_difficulty"],
            max_difficulty=decoded["max_difficulty"],
            weights_version=decoded["weights_version"],
            weights_rate_limit=decoded["weights_rate_limit"],
            adjustment_interval=decoded["adjustment_interval"],
            activity_cutoff=decoded["activity_cutoff"],
            registration_allowed=decoded["registration_allowed"],
            target_regs_per_interval=decoded["target_regs_per_interval"],
            min_burn=decoded["min_burn"],
            max_burn=decoded["max_burn"],
            max_regs_per_block=decoded["max_regs_per_block"],
            max_validators=decoded["max_validators"],
            serving_rate_limit=decoded["serving_rate_limit"],
            bonds_moving_avg=decoded["bonds_moving_avg"],
            adjustment_alpha=decoded["adjustment_alpha"],
            difficulty=decoded["difficulty"],
            commit_reveal_weights_interval=decoded["commit_reveal_weights_interval"],
            commit_reveal_weights_enabled=decoded["commit_reveal_weights_enabled"],
            alpha_high=decoded["alpha_high"],
            alpha_low=decoded["alpha_low"],
            liquid_alpha_enabled=decoded["liquid_alpha_enabled"],
        )

    def to_parameter_dict(
        self,
    ) -> Union[dict[str, Union[int, float, bool]], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet hyperparameters."""
        if use_torch():
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "SubnetHyperparameters":
        """Creates a SubnetHyperparameters instance from a parameter dictionary."""
        if use_torch():
            return cls(**dict(parameter_dict))
        else:
            return cls(**parameter_dict)
