from dataclasses import dataclass

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

    @classmethod
    def _from_dict(cls, decoded: dict) -> "SubnetHyperparameters":
        """
        Create a `SubnetHyperparameters` instance from a vector of bytes.

        This method decodes the given vector of bytes using the `bt_decode` module and creates a new instance of
            `SubnetHyperparameters` with the decoded values.

        Args:
            vec_u8 (bytes): A vector of bytes to decode into `SubnetHyperparameters`.

        Returns:
            Optional[SubnetHyperparameters]: An instance of `SubnetHyperparameters` if decoding is successful, None
                otherwise.
        """
        return SubnetHyperparameters(
            activity_cutoff=decoded["activity_cutoff"],
            adjustment_alpha=decoded["adjustment_alpha"],
            adjustment_interval=decoded["adjustment_interval"],
            alpha_high=decoded["alpha_high"],
            alpha_low=decoded["alpha_low"],
            bonds_moving_avg=decoded["bonds_moving_avg"],
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
            target_regs_per_interval=decoded["target_regs_per_interval"],
            tempo=decoded["tempo"],
            weights_rate_limit=decoded["weights_rate_limit"],
            weights_version=decoded["weights_version"],
        )
