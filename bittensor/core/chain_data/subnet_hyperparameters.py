from dataclasses import dataclass
from bittensor.utils.balance import fixed_to_float
from bittensor.core.chain_data.info_base import InfoBase


@dataclass
class SubnetHyperparameters(InfoBase):
    """
    This class represents the hyperparameters for a subnet.

    Attributes:
        rho: The rate of decay of some value.
        kappa: A constant multiplier used in calculations.
        immunity_period: The period during which immunity is active.
        min_allowed_weights: Minimum allowed weights.
        max_weight_limit: Maximum weight limit.
        tempo: The tempo or rate of operation.
        min_difficulty: Minimum difficulty for some operations.
        max_difficulty: Maximum difficulty for some operations.
        weights_version: The version number of the weights used.
        weights_rate_limit: Rate limit for processing weights.
        adjustment_interval: Interval at which adjustments are made.
        activity_cutoff: Activity cutoff threshold.
        registration_allowed: Indicates if registration is allowed.
        target_regs_per_interval: Target number of registrations per interval.
        min_burn: Minimum burn value.
        max_burn: Maximum burn value.
        bonds_moving_avg: Moving average of bonds.
        max_regs_per_block: Maximum number of registrations per block.
        serving_rate_limit: Limit on the rate of service.
        max_validators: Maximum number of validators.
        adjustment_alpha: Alpha value for adjustments.
        difficulty: Difficulty level.
        commit_reveal_period: Interval for commit-reveal weights.
        commit_reveal_weights_enabled: Flag indicating if commit-reveal weights are enabled.
        alpha_high: High value of alpha.
        alpha_low: Low value of alpha.
        liquid_alpha_enabled: Flag indicating if liquid alpha is enabled.
        alpha_sigmoid_steepness: Sigmoid steepness parameter for converting miner-validator alignment into alpha.
        yuma_version: Version of yuma.
        subnet_is_active: Indicates if subnet is active after START CALL.
        transfers_enabled: Flag indicating if transfers are enabled.
        bonds_reset_enabled: Flag indicating if bonds are reset enabled.
        user_liquidity_enabled: Flag indicating if user liquidity is enabled.
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
