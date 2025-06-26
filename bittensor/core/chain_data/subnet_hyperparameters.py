from dataclasses import dataclass
from bittensor.utils.balance import fixed_to_float
from bittensor.core.chain_data.info_base import InfoBase


@dataclass
class SubnetHyperparameters(InfoBase):
    """
    This class represents the hyperparameters for a subnet.

    Attributes:
        rho (int): The rate of decay of some value.
        kappa (int): A constant multiplier used in calculations.
        immunity_period (int): The period during which immunity is active.
        min_allowed_weights (int): Minimum allowed weights.
        max_weight_limit (float): Maximum weight limit.
        tempo (int): The tempo or rate of operation.
        min_difficulty (int): Minimum difficulty for some operations.
        max_difficulty (int): Maximum difficulty for some operations.
        weights_version (int): The version number of the weights used.
        weights_rate_limit (int): Rate limit for processing weights.
        adjustment_interval (int): Interval at which adjustments are made.
        activity_cutoff (int): Activity cutoff threshold.
        registration_allowed (bool): Indicates if registration is allowed.
        target_regs_per_interval (int): Target number of registrations per interval.
        min_burn (int): Minimum burn value.
        max_burn (int): Maximum burn value.
        bonds_moving_avg (int): Moving average of bonds.
        max_regs_per_block (int): Maximum number of registrations per block.
        serving_rate_limit (int): Limit on the rate of service.
        max_validators (int): Maximum number of validators.
        adjustment_alpha (int): Alpha value for adjustments.
        difficulty (int): Difficulty level.
        commit_reveal_period (int): Interval for commit-reveal weights.
        commit_reveal_weights_enabled (bool): Flag indicating if commit-reveal weights are enabled.
        alpha_high (int): High value of alpha.
        alpha_low (int): Low value of alpha.
        liquid_alpha_enabled (bool): Flag indicating if liquid alpha is enabled.
        alpha_sigmoid_steepness (float):
        yuma_version (int): Version of yuma.
        subnet_is_active (bool): Indicates if subnet is active after START CALL.
        transfers_enabled (bool): Flag indicating if transfers are enabled.
        bonds_reset_enabled (bool): Flag indicating if bonds are reset enabled.
        user_liquidity_enabled (bool): Flag indicating if user liquidity is enabled.
    """

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
    commit_reveal_period: int
    commit_reveal_weights_enabled: bool
    alpha_high: int
    alpha_low: int
    liquid_alpha_enabled: bool
    alpha_sigmoid_steepness: float
    yuma_version: int
    subnet_is_active: bool
    transfers_enabled: bool
    bonds_reset_enabled: bool
    user_liquidity_enabled: bool

    @classmethod
    def _from_dict(cls, decoded: dict) -> "SubnetHyperparameters":
        """Returns a SubnetHyperparameters object from decoded chain data."""
        return SubnetHyperparameters(
            activity_cutoff=decoded["activity_cutoff"],
            adjustment_alpha=decoded["adjustment_alpha"],
            adjustment_interval=decoded["adjustment_interval"],
            alpha_high=decoded["alpha_high"],
            alpha_low=decoded["alpha_low"],
            alpha_sigmoid_steepness=fixed_to_float(
                decoded["alpha_sigmoid_steepness"], frac_bits=32
            ),
            bonds_moving_avg=decoded["bonds_moving_avg"],
            bonds_reset_enabled=decoded["bonds_reset_enabled"],
            commit_reveal_weights_enabled=decoded["commit_reveal_weights_enabled"],
            commit_reveal_period=decoded["commit_reveal_period"],
            difficulty=decoded["difficulty"],
            immunity_period=decoded["immunity_period"],
            kappa=decoded["kappa"],
            liquid_alpha_enabled=decoded["liquid_alpha_enabled"],
            max_burn=decoded["max_burn"],
            max_difficulty=decoded["max_difficulty"],
            max_regs_per_block=decoded["max_regs_per_block"],
            max_validators=decoded["max_validators"],
            max_weight_limit=decoded["max_weights_limit"],
            min_allowed_weights=decoded["min_allowed_weights"],
            min_burn=decoded["min_burn"],
            min_difficulty=decoded["min_difficulty"],
            registration_allowed=decoded["registration_allowed"],
            rho=decoded["rho"],
            serving_rate_limit=decoded["serving_rate_limit"],
            subnet_is_active=decoded["subnet_is_active"],
            target_regs_per_interval=decoded["target_regs_per_interval"],
            tempo=decoded["tempo"],
            transfers_enabled=decoded["transfers_enabled"],
            user_liquidity_enabled=decoded["user_liquidity_enabled"],
            weights_rate_limit=decoded["weights_rate_limit"],
            weights_version=decoded["weights_version"],
            yuma_version=decoded["yuma_version"],
        )
