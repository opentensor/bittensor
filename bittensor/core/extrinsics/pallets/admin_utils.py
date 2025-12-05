"""
WARNING: This module contains administrative utilities that should ONLY be used for local development and testing
purposes. These functions provide direct access to critical network parameters and should never be used in production
environments as they can potentially disrupt network stability.

The AdminUtils module contains powerful administrative functions that can modify core network parameters. Improper use
of these functions outside of development/testing contexts could have severe consequences for network operation.
"""

from dataclasses import dataclass
from typing import Optional

from .base import CallBuilder as _BasePallet, Call


@dataclass
class AdminUtils(_BasePallet):
    """Factory class for creating GenericCall objects for AdminUtils pallet functions.

    This class provides methods to create GenericCall instances for all AdminUtils pallet extrinsics.

    Works with both sync (Subtensor) and async (AsyncSubtensor) instances. For async operations, pass an AsyncSubtensor
    instance and await the result.

    Example:
        # Sync usage
        call = AdminUtils(subtensor).sudo_set_default_take(default_take=100)
        response = subtensor.sign_and_send_extrinsic(call=call, ...)

        # Async usage
        call = await AdminUtils(async_subtensor).sudo_set_default_take(default_take=100)
        response = await async_subtensor.sign_and_send_extrinsic(call=call, ...)
    """

    def swap_authorities(
        self,
        new_authorities: list[str],
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function swap_authorities.

        The extrinsic sets the new authorities for Aura consensus.
        It is only callable by the root account.
        The extrinsic will call the Aura pallet to change the authorities.

        Parameters:
            new_authorities: List of new authority identifiers.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(new_authorities=new_authorities)

    def sudo_set_default_take(
        self,
        default_take: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_default_take.

        The extrinsic sets the default take for the network.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the default take.

        Parameters:
            default_take: The default take value (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(default_take=default_take)

    def sudo_set_tx_rate_limit(
        self,
        tx_rate_limit: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_tx_rate_limit.

        The extrinsic sets the transaction rate limit for the network.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the transaction rate limit.

        Parameters:
            tx_rate_limit: The transaction rate limit (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(tx_rate_limit=tx_rate_limit)

    def sudo_set_serving_rate_limit(
        self,
        netuid: int,
        serving_rate_limit: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_serving_rate_limit.

        The extrinsic sets the serving rate limit for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the serving rate limit.

        Parameters:
            netuid: The network identifier.
            serving_rate_limit: The serving rate limit (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            serving_rate_limit=serving_rate_limit,
        )

    def sudo_set_min_difficulty(
        self,
        netuid: int,
        min_difficulty: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_min_difficulty.

        The extrinsic sets the minimum difficulty for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the minimum difficulty.

        Parameters:
            netuid: The network identifier.
            min_difficulty: The minimum difficulty value (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            min_difficulty=min_difficulty,
        )

    def sudo_set_max_difficulty(
        self,
        netuid: int,
        max_difficulty: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_max_difficulty.

        The extrinsic sets the maximum difficulty for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the maximum difficulty.

        Parameters:
            netuid: The network identifier.
            max_difficulty: The maximum difficulty value (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            max_difficulty=max_difficulty,
        )

    def sudo_set_weights_version_key(
        self,
        netuid: int,
        weights_version_key: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_weights_version_key.

        The extrinsic sets the weights version key for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the weights version key.

        Parameters:
            netuid: The network identifier.
            weights_version_key: The weights version key (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            weights_version_key=weights_version_key,
        )

    def sudo_set_weights_set_rate_limit(
        self,
        netuid: int,
        weights_set_rate_limit: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_weights_set_rate_limit.

        The extrinsic sets the weights set rate limit for a subnet.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the weights set rate limit.

        Parameters:
            netuid: The network identifier.
            weights_set_rate_limit: The weights set rate limit (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            weights_set_rate_limit=weights_set_rate_limit,
        )

    def sudo_set_adjustment_interval(
        self,
        netuid: int,
        adjustment_interval: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_adjustment_interval.

        The extrinsic sets the adjustment interval for a subnet.
        It is only callable by the root account, not changeable by the subnet owner.
        The extrinsic will call the Subtensor pallet to set the adjustment interval.

        Parameters:
            netuid: The network identifier.
            adjustment_interval: The adjustment interval (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            adjustment_interval=adjustment_interval,
        )

    def sudo_set_adjustment_alpha(
        self,
        netuid: int,
        adjustment_alpha: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_adjustment_alpha.

        The extrinsic sets the adjustment alpha for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the adjustment alpha.

        Parameters:
            netuid: The network identifier.
            adjustment_alpha: The adjustment alpha value (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            adjustment_alpha=adjustment_alpha,
        )

    def sudo_set_immunity_period(
        self,
        netuid: int,
        immunity_period: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_immunity_period.

        The extrinsic sets the immunity period for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the immunity period.

        Parameters:
            netuid: The network identifier.
            immunity_period: The immunity period (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            immunity_period=immunity_period,
        )

    def sudo_set_min_allowed_weights(
        self,
        netuid: int,
        min_allowed_weights: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_min_allowed_weights.

        The extrinsic sets the minimum allowed weights for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the minimum allowed weights.

        Parameters:
            netuid: The network identifier.
            min_allowed_weights: The minimum allowed weights (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            min_allowed_weights=min_allowed_weights,
        )

    def sudo_set_max_allowed_uids(
        self,
        netuid: int,
        max_allowed_uids: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_max_allowed_uids.

        The extrinsic sets the maximum allowed UIDs for a subnet.
        It is only callable by the root account and subnet owner.
        The extrinsic will call the Subtensor pallet to set the maximum allowed UIDs for a subnet.

        Parameters:
            netuid: The network identifier.
            max_allowed_uids: The maximum allowed UIDs (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            max_allowed_uids=max_allowed_uids,
        )

    def sudo_set_kappa(
        self,
        netuid: int,
        kappa: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_kappa.

        The extrinsic sets the kappa for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the kappa.

        Parameters:
            netuid: The network identifier.
            kappa: The kappa value (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, kappa=kappa)

    def sudo_set_rho(
        self,
        netuid: int,
        rho: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_rho.

        The extrinsic sets the rho for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the rho.

        Parameters:
            netuid: The network identifier.
            rho: The rho value (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, rho=rho)

    def sudo_set_activity_cutoff(
        self,
        netuid: int,
        activity_cutoff: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_activity_cutoff.

        The extrinsic sets the activity cutoff for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the activity cutoff.

        Parameters:
            netuid: The network identifier.
            activity_cutoff: The activity cutoff value (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            activity_cutoff=activity_cutoff,
        )

    def sudo_set_network_registration_allowed(
        self,
        netuid: int,
        registration_allowed: bool,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_network_registration_allowed.

        The extrinsic sets the network registration allowed for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the network registration allowed.

        Parameters:
            netuid: The network identifier.
            registration_allowed: Whether registration is allowed (bool).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            registration_allowed=registration_allowed,
        )

    def sudo_set_network_pow_registration_allowed(
        self,
        netuid: int,
        registration_allowed: bool,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_network_pow_registration_allowed.

        The extrinsic sets the network PoW registration allowed for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the network PoW registration allowed.

        Parameters:
            netuid: The network identifier.
            registration_allowed: Whether PoW registration is allowed (bool).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            registration_allowed=registration_allowed,
        )

    def sudo_set_target_registrations_per_interval(
        self,
        netuid: int,
        target_registrations_per_interval: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_target_registrations_per_interval.

        The extrinsic sets the target registrations per interval for a subnet.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the target registrations per interval.

        Parameters:
            netuid: The network identifier.
            target_registrations_per_interval: The target registrations per interval (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            target_registrations_per_interval=target_registrations_per_interval,
        )

    def sudo_set_min_burn(
        self,
        netuid: int,
        min_burn: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_min_burn.

        The extrinsic sets the minimum burn for a subnet.
        It is only callable by root and subnet owner.
        The extrinsic will call the Subtensor pallet to set the minimum burn.

        Parameters:
            netuid: The network identifier.
            min_burn: The minimum burn value in RAO (TaoCurrency).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, min_burn=min_burn)

    def sudo_set_max_burn(
        self,
        netuid: int,
        max_burn: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_max_burn.

        The extrinsic sets the maximum burn for a subnet.
        It is only callable by root and subnet owner.
        The extrinsic will call the Subtensor pallet to set the maximum burn.

        Parameters:
            netuid: The network identifier.
            max_burn: The maximum burn value in RAO (TaoCurrency).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, max_burn=max_burn)

    def sudo_set_difficulty(
        self,
        netuid: int,
        difficulty: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_difficulty.

        The extrinsic sets the difficulty for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the difficulty.

        Parameters:
            netuid: The network identifier.
            difficulty: The difficulty value (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, difficulty=difficulty)

    def sudo_set_max_allowed_validators(
        self,
        netuid: int,
        max_allowed_validators: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_max_allowed_validators.

        The extrinsic sets the maximum allowed validators for a subnet.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the maximum allowed validators.

        Parameters:
            netuid: The network identifier.
            max_allowed_validators: The maximum allowed validators (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            max_allowed_validators=max_allowed_validators,
        )

    def sudo_set_bonds_moving_average(
        self,
        netuid: int,
        bonds_moving_average: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_bonds_moving_average.

        The extrinsic sets the bonds moving average for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the bonds moving average.

        Parameters:
            netuid: The network identifier.
            bonds_moving_average: The bonds moving average value (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            bonds_moving_average=bonds_moving_average,
        )

    def sudo_set_bonds_penalty(
        self,
        netuid: int,
        bonds_penalty: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_bonds_penalty.

        The extrinsic sets the bonds penalty for a subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the bonds penalty.

        Parameters:
            netuid: The network identifier.
            bonds_penalty: The bonds penalty value (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            bonds_penalty=bonds_penalty,
        )

    def sudo_set_max_registrations_per_block(
        self,
        netuid: int,
        max_registrations_per_block: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_max_registrations_per_block.

        The extrinsic sets the maximum registrations per block for a subnet.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the maximum registrations per block.

        Parameters:
            netuid: The network identifier.
            max_registrations_per_block: The maximum registrations per block (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            max_registrations_per_block=max_registrations_per_block,
        )

    def sudo_set_subnet_owner_cut(
        self,
        subnet_owner_cut: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_subnet_owner_cut.

        The extrinsic sets the subnet owner cut for a subnet.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the subnet owner cut.

        Parameters:
            subnet_owner_cut: The subnet owner cut value (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(subnet_owner_cut=subnet_owner_cut)

    def sudo_set_network_rate_limit(
        self,
        rate_limit: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_network_rate_limit.

        The extrinsic sets the network rate limit for the network.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the network rate limit.

        Parameters:
            rate_limit: The network rate limit (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(rate_limit=rate_limit)

    def sudo_set_tempo(
        self,
        netuid: int,
        tempo: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_tempo.

        The extrinsic sets the tempo for a subnet.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the tempo.

        Parameters:
            netuid: The network identifier.
            tempo: The tempo value (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, tempo=tempo)

    def sudo_set_total_issuance(
        self,
        total_issuance: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_total_issuance.

        The extrinsic sets the total issuance for the network.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the issuance for the network.

        Parameters:
            total_issuance: The total issuance value in RAO (TaoCurrency).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(total_issuance=total_issuance)

    def sudo_set_network_immunity_period(
        self,
        immunity_period: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_network_immunity_period.

        The extrinsic sets the immunity period for the network.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the immunity period for the network.

        Parameters:
            immunity_period: The immunity period (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(immunity_period=immunity_period)

    def sudo_set_network_min_lock_cost(
        self,
        lock_cost: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_network_min_lock_cost.

        The extrinsic sets the min lock cost for the network.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the min lock cost for the network.

        Parameters:
            lock_cost: The lock cost value in RAO (TaoCurrency).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(lock_cost=lock_cost)

    def sudo_set_subnet_limit(
        self,
        max_subnets: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_subnet_limit.

        The extrinsic sets the subnet limit for the network.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the subnet limit.

        Parameters:
            max_subnets: The maximum number of subnets (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(max_subnets=max_subnets)

    def sudo_set_lock_reduction_interval(
        self,
        interval: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_lock_reduction_interval.

        The extrinsic sets the lock reduction interval for the network.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the lock reduction interval.

        Parameters:
            interval: The lock reduction interval (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(interval=interval)

    def sudo_set_rao_recycled(
        self,
        netuid: int,
        rao_recycled: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_rao_recycled.

        The extrinsic sets the recycled RAO for a subnet.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the recycled RAO.

        Parameters:
            netuid: The network identifier.
            rao_recycled: The recycled RAO value in RAO (TaoCurrency).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, rao_recycled=rao_recycled)

    def sudo_set_stake_threshold(
        self,
        min_stake: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_stake_threshold.

        The extrinsic sets the weights min stake.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the weights min stake.

        Parameters:
            min_stake: The minimum stake value (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(min_stake=min_stake)

    def sudo_set_nominator_min_required_stake(
        self,
        min_stake: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_nominator_min_required_stake.

        The extrinsic sets the minimum stake required for nominators.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the minimum stake required for nominators.

        Parameters:
            min_stake: The minimum stake required for nominators (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(min_stake=min_stake)

    def sudo_set_tx_delegate_take_rate_limit(
        self,
        tx_rate_limit: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_tx_delegate_take_rate_limit.

        The extrinsic sets the rate limit for delegate take transactions.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the rate limit for delegate take transactions.

        Parameters:
            tx_rate_limit: The transaction rate limit (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(tx_rate_limit=tx_rate_limit)

    def sudo_set_min_delegate_take(
        self,
        take: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_min_delegate_take.

        The extrinsic sets the minimum delegate take.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the minimum delegate take.

        Parameters:
            take: The minimum delegate take value (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(take=take)

    def sudo_set_commit_reveal_weights_enabled(
        self,
        netuid: int,
        enabled: bool,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_commit_reveal_weights_enabled.

        The extrinsic enabled/disables commit/reaveal for a given subnet.
        It is only callable by the root account or subnet owner.
        The extrinsic will call the Subtensor pallet to set the value.

        Parameters:
            netuid: The network identifier.
            enabled: Whether commit/reveal weights is enabled (bool).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, enabled=enabled)

    def sudo_set_liquid_alpha_enabled(
        self,
        netuid: int,
        enabled: bool,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_liquid_alpha_enabled.

        Enables or disables Liquid Alpha for a given subnet.

        Parameters:
            netuid: The unique identifier for the subnet.
            enabled: A boolean flag to enable or disable Liquid Alpha.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, enabled=enabled)

    def sudo_set_alpha_values(
        self,
        netuid: int,
        alpha_low: int,
        alpha_high: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_alpha_values.

        Sets values for liquid alpha.

        Parameters:
            netuid: The network identifier.
            alpha_low: The low alpha value (u16).
            alpha_high: The high alpha value (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            alpha_low=alpha_low,
            alpha_high=alpha_high,
        )

    def sudo_set_coldkey_swap_schedule_duration(
        self,
        duration: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_coldkey_swap_schedule_duration.

        Sets the duration of the coldkey swap schedule.

        This extrinsic allows the root account to set the duration for the coldkey swap schedule.
        The coldkey swap schedule determines how long it takes for a coldkey swap operation to complete.

        Parameters:
            duration: The new duration for the coldkey swap schedule, in number of blocks.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(duration=duration)

    def sudo_set_dissolve_network_schedule_duration(
        self,
        duration: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_dissolve_network_schedule_duration.

        Sets the duration of the dissolve network schedule.

        This extrinsic allows the root account to set the duration for the dissolve network schedule.
        The dissolve network schedule determines how long it takes for a network dissolution operation to complete.

        Parameters:
            duration: The new duration for the dissolve network schedule, in number of blocks.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(duration=duration)

    def sudo_set_commit_reveal_weights_interval(
        self,
        netuid: int,
        interval: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_commit_reveal_weights_interval.

        Sets the commit-reveal weights periods for a specific subnet.

        This extrinsic allows the subnet owner or root account to set the duration (in epochs) during which committed weights must be revealed.
        The commit-reveal mechanism ensures that users commit weights in advance and reveal them only within a specified period.

        Parameters:
            netuid: The unique identifier of the subnet for which the periods are being set.
            interval: The number of epochs that define the commit-reveal period.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, interval=interval)

    def sudo_set_evm_chain_id(
        self,
        chain_id: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_evm_chain_id.

        Sets the EVM ChainID.

        Parameters:
            chain_id: The u64 chain ID.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(chain_id=chain_id)

    def schedule_grandpa_change(
        self,
        next_authorities: list[tuple],
        in_blocks: int,
        forced: Optional[int] = None,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function schedule_grandpa_change.

        A public interface for `pallet_grandpa::Pallet::schedule_grandpa_change`.

        Schedule a change in the authorities.

        The change will be applied at the end of execution of the block `in_blocks` after the
        current block. This value may be 0, in which case the change is applied at the end of
        the current block.

        If the `forced` parameter is defined, this indicates that the current set has been
        synchronously determined to be offline and that after `in_blocks` the given change
        should be applied. The given block number indicates the median last finalized block
        number and it should be used as the canon block when starting the new grandpa voter.

        No change should be signaled while any change is pending. Returns an error if a change
        is already pending.

        Parameters:
            next_authorities: The list of next authorities (AuthorityList).
            in_blocks: The number of blocks after which the change is applied.
            forced: Optional block number for forced change.

        Returns:
            GenericCall instance.
        """
        params = {
            "next_authorities": next_authorities,
            "in_blocks": in_blocks,
        }
        if forced is not None:
            params["forced"] = forced
        return self.create_composed_call(**params)

    def sudo_set_toggle_transfer(
        self,
        netuid: int,
        toggle: bool,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_toggle_transfer.

        Enable or disable atomic alpha transfers for a given subnet.

        Parameters:
            netuid: The unique identifier for the subnet.
            toggle: A boolean flag to enable or disable Liquid Alpha.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, toggle=toggle)

    def sudo_set_recycle_or_burn(
        self,
        netuid: int,
        recycle_or_burn: str,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_recycle_or_burn.

        Set the behaviour of the "burn" UID(s) for a given subnet.
        If set to `Burn`, the miner emission sent to the burn UID(s) will be burned.
        If set to `Recycle`, the miner emission sent to the burn UID(s) will be recycled.

        Parameters:
            netuid: The unique identifier for the subnet.
            recycle_or_burn: The desired behaviour of the "burn" UID(s) for the subnet.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            recycle_or_burn=recycle_or_burn,
        )

    def sudo_toggle_evm_precompile(
        self,
        precompile_id: str,
        enabled: bool,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_toggle_evm_precompile.

        Toggles the enablement of an EVM precompile.

        Parameters:
            precompile_id: The identifier of the EVM precompile to toggle.
            enabled: The new enablement state of the precompile.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            precompile_id=precompile_id,
            enabled=enabled,
        )

    def sudo_set_subnet_moving_alpha(
        self,
        alpha: dict,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_subnet_moving_alpha.

        Sets the new moving alpha value for the SubnetMovingAlpha.

        Parameters:
            alpha: The new moving alpha value (I96F32).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(alpha=alpha)

    def sudo_set_subnet_owner_hotkey(
        self,
        netuid: int,
        hotkey: str,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_subnet_owner_hotkey.

        Change the SubnetOwnerHotkey for a given subnet.

        Parameters:
            netuid: The unique identifier for the subnet.
            hotkey: The new hotkey for the subnet owner.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, hotkey=hotkey)

    def sudo_set_ema_price_halving_period(
        self,
        netuid: int,
        ema_halving: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_ema_price_halving_period.

        Sets the number of blocks for EMA price to halve.

        Parameters:
            netuid: The unique identifier for the subnet.
            ema_halving: Number of blocks for EMA price to halve.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, ema_halving=ema_halving)

    def sudo_set_alpha_sigmoid_steepness(
        self,
        netuid: int,
        steepness: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_alpha_sigmoid_steepness.

        Sets the Steepness for the alpha sigmoid function.

        Parameters:
            netuid: The unique identifier for the subnet.
            steepness: The Steepness for the alpha sigmoid function. (range is 0-int16::MAX,
                negative values are reserved for future use).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, steepness=steepness)

    def sudo_set_yuma3_enabled(
        self,
        netuid: int,
        enabled: bool,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_yuma3_enabled.

        Enables or disables Yuma3 for a given subnet.

        Parameters:
            netuid: The unique identifier for the subnet.
            enabled: A boolean flag to enable or disable Yuma3.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, enabled=enabled)

    def sudo_set_bonds_reset_enabled(
        self,
        netuid: int,
        enabled: bool,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_bonds_reset_enabled.

        Enables or disables Bonds Reset for a given subnet.

        Parameters:
            netuid: The unique identifier for the subnet.
            enabled: A boolean flag to enable or disable Bonds Reset.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, enabled=enabled)

    def sudo_set_sn_owner_hotkey(
        self,
        netuid: int,
        hotkey: str,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_sn_owner_hotkey.

        Sets or updates the hotkey account associated with the owner of a specific subnet.

        This function allows either the root origin or the current subnet owner to set or update
        the hotkey for a given subnet. The subnet must already exist. To prevent abuse, the call is
        rate-limited to once per configured interval (default: one week) per subnet.

        Parameters:
            netuid: The unique identifier of the subnet whose owner hotkey is being set.
            hotkey: The new hotkey account to associate with the subnet owner.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, hotkey=hotkey)

    def sudo_set_subtoken_enabled(
        self,
        netuid: int,
        subtoken_enabled: bool,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_subtoken_enabled.

        Enables or disables subtoken trading for a given subnet.

        Parameters:
            netuid: The unique identifier of the subnet.
            subtoken_enabled: A boolean indicating whether subtoken trading should be enabled or disabled.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            subtoken_enabled=subtoken_enabled,
        )

    def sudo_set_commit_reveal_version(
        self,
        version: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_commit_reveal_version.

        Sets the commit-reveal weights version for all subnets.

        Parameters:
            version: The commit-reveal weights version (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(version=version)

    def sudo_set_owner_immune_neuron_limit(
        self,
        netuid: int,
        immune_neurons: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_owner_immune_neuron_limit.

        Sets the number of immune owner neurons.

        Parameters:
            netuid: The network identifier.
            immune_neurons: The number of immune owner neurons (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            immune_neurons=immune_neurons,
        )

    def sudo_set_ck_burn(
        self,
        burn: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_ck_burn.

        Sets the childkey burn for a subnet.
        It is only callable by the root account.
        The extrinsic will call the Subtensor pallet to set the childkey burn.

        Parameters:
            burn: The childkey burn value (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(burn=burn)

    def sudo_set_admin_freeze_window(
        self,
        window: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_admin_freeze_window.

        Sets the admin freeze window length (in blocks) at the end of a tempo.
        Only callable by root.

        Parameters:
            window: The admin freeze window length in blocks (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(window=window)

    def sudo_set_owner_hparam_rate_limit(
        self,
        epochs: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_owner_hparam_rate_limit.

        Sets the owner hyperparameter rate limit in epochs (global multiplier).
        Only callable by root.

        Parameters:
            epochs: The owner hyperparameter rate limit in epochs (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(epochs=epochs)

    def sudo_set_mechanism_count(
        self,
        netuid: int,
        mechanism_count: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_mechanism_count.

        Sets the desired number of mechanisms in a subnet.

        Parameters:
            netuid: The network identifier.
            mechanism_count: The desired number of mechanisms (MechId).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            mechanism_count=mechanism_count,
        )

    def sudo_set_mechanism_emission_split(
        self,
        netuid: int,
        maybe_split: Optional[list[int]] = None,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_mechanism_emission_split.

        Sets the emission split between mechanisms in a subnet.

        Parameters:
            netuid: The network identifier.
            maybe_split: Optional list of emission split values (Option<Vec<u16>>).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, maybe_split=maybe_split)

    def sudo_trim_to_max_allowed_uids(
        self,
        netuid: int,
        max_n: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_trim_to_max_allowed_uids.

        Trims the maximum number of UIDs for a subnet.

        The trimming is done by sorting the UIDs by emission descending and then trimming
        the lowest emitters while preserving temporally and owner immune UIDs. The UIDs are
        then compressed to the left and storage is migrated to the new compressed UIDs.

        Parameters:
            netuid: The network identifier.
            max_n: The maximum number of UIDs (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, max_n=max_n)

    def sudo_set_min_allowed_uids(
        self,
        netuid: int,
        min_allowed_uids: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_min_allowed_uids.

        The extrinsic sets the minimum allowed UIDs for a subnet.
        It is only callable by the root account.

        Parameters:
            netuid: The network identifier.
            min_allowed_uids: The minimum allowed UIDs (u16).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            min_allowed_uids=min_allowed_uids,
        )

    def sudo_set_tao_flow_cutoff(
        self,
        flow_cutoff: dict,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_tao_flow_cutoff.

        Sets TAO flow cutoff value (A).

        Parameters:
            flow_cutoff: The TAO flow cutoff value (I64F64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(flow_cutoff=flow_cutoff)

    def sudo_set_tao_flow_normalization_exponent(
        self,
        exponent: dict,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_tao_flow_normalization_exponent.

        Sets TAO flow normalization exponent (p).

        Parameters:
            exponent: The TAO flow normalization exponent (U64F64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(exponent=exponent)

    def sudo_set_tao_flow_smoothing_factor(
        self,
        smoothing_factor: int,
    ) -> Call:
        """Returns GenericCall instance for AdminUtils function sudo_set_tao_flow_smoothing_factor.

        Sets TAO flow smoothing factor (alpha).

        Parameters:
            smoothing_factor: The TAO flow smoothing factor (u64).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(smoothing_factor=smoothing_factor)
