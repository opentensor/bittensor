from dataclasses import dataclass
from typing import Literal, Optional

from bittensor.core.types import UIDs, Weights, Salt
from bittensor.utils import Certificate
from .base import CallBuilder as _BasePallet, Call


@dataclass
class SubtensorModule(_BasePallet):
    """Factory class for creating GenericCall objects for SubtensorModule pallet functions.

    This class provides methods to create GenericCall instances for all SubtensorModule pallet extrinsics.

    Works with both sync (Subtensor) and async (AsyncSubtensor) instances. For async operations, pass an AsyncSubtensor
    instance and await the result.

    Example:
        # Sync usage
        call = SubtensorModule(subtensor).start_call(netuid=14)
        response = subtensor.sign_and_send_extrinsic(call=call, ...)

        # Async usage
        call = await SubtensorModule(async_subtensor).start_call(netuid=14)
        response = await async_subtensor.sign_and_send_extrinsic(call=call, ...)
    """

    def add_stake(
        self,
        netuid: int,
        hotkey: str,
        amount_staked: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.add_stake.

        Parameters:
            netuid: The netuid of the subnet to add stake to.
            hotkey: The hotkey SS58 address associated with validator.
            amount_staked: Amount of stake in RAO to add.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            hotkey=hotkey,
            amount_staked=amount_staked,
        )

    def add_stake_limit(
        self,
        netuid: int,
        hotkey: str,
        amount_staked: int,
        limit_price: float,
        allow_partial: bool,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.add_stake_limit.

        Parameters:
            netuid: The netuid of the subnet to add stake to.
            hotkey: The hotkey SS58 address associated with validator.
            amount_staked: Amount of stake in RAO to add.
            limit_price: The limit price expressed in units of RAO per one Alpha.
            allow_partial: If True, allows partial unstaking if price tolerance exceeded.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            hotkey=hotkey,
            amount_staked=amount_staked,
            limit_price=limit_price,
            allow_partial=allow_partial,
        )

    def burned_register(
        self,
        netuid: int,
        hotkey: str,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.burned_register.

        Parameters:
            netuid: The netuid of the subnet to register on.
            hotkey: The hotkey SS58 address associated with the neuron.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, hotkey=hotkey)

    def claim_root(self, subnets: list[int]) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.claim_root.

        Parameters:
            subnets: The netuids of the subnets to claim root for. Think about it as netuids.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(subnets=subnets)

    def commit_mechanism_weights(
        self,
        netuid: int,
        mecid: int,
        commit_hash: str,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.commit_mechanism_weights.

        Parameters:
            netuid: The unique identifier of the subnet.
            mecid: The subnet mechanism unique identifier.
            commit_hash: The hash of the commitment.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid, mecid=mecid, commit_hash=commit_hash
        )

    def commit_timelocked_mechanism_weights(
        self,
        netuid: int,
        mecid: int,
        commit: bytes,
        reveal_round: int,
        commit_reveal_version: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.commit_mechanism_weights.

        Parameters:
            netuid: The unique identifier of the subnet.
            mecid: The subnet mechanism unique identifier.
            commit: Raw bytes of the encrypted and compressed uids & weights values for setting weights.
            reveal_round: Drand round number when weights have to be revealed. Based on Drand Quicknet network.
            commit_reveal_version: The version of the commit-reveal in the chain.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            mecid=mecid,
            commit=commit,
            reveal_round=reveal_round,
            commit_reveal_version=commit_reveal_version,
        )

    def decrease_take(self, hotkey: str, take: int) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.decrease_take.

        Parameters:
            hotkey: SS58 address of the hotkey to set take for.
            take: The percentage of rewards that the delegate claims from nominators.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(hotkey=hotkey, take=take)

    def increase_take(self, hotkey: str, take: int) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.increase_take.

        Parameters:
            hotkey: SS58 address of the hotkey to set take for.
            take: The percentage of rewards that the delegate claims from nominators.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(hotkey=hotkey, take=take)

    def move_stake(
        self,
        origin_netuid: int,
        origin_hotkey_ss58: str,
        destination_netuid: int,
        destination_hotkey_ss58: str,
        alpha_amount: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.move_stake.

        Parameters:
            origin_netuid: The netuid of the source subnet.
            origin_hotkey_ss58: The SS58 address of the source hotkey.
            destination_netuid: The netuid of the destination subnet.
            destination_hotkey_ss58: The SS58 address of the destination hotkey.
            alpha_amount: Amount of origin Balance to move.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            origin_netuid=origin_netuid,
            origin_hotkey=origin_hotkey_ss58,
            destination_netuid=destination_netuid,
            destination_hotkey=destination_hotkey_ss58,
            alpha_amount=alpha_amount,
        )

    def register(
        self,
        netuid: int,
        coldkey: str,
        hotkey: str,
        block_number: int,
        nonce: int,
        work: list[int],
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.register.

        Parameters:
            netuid: The netuid of the subnet to register on.
            coldkey: The coldkey SS58 address associated with the neuron.
            hotkey: The hotkey SS58 address associated with the neuron.
            block_number: POW block number.
            nonce: POW nonce.
            work: List representation of POW seal.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            coldkey=coldkey,
            hotkey=hotkey,
            block_number=block_number,
            nonce=nonce,
            work=work,
        )

    def register_network(self, hotkey: str) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.register_network.

        Parameters:
            hotkey: The hotkey SS58 address associated with the subnet owner.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(hotkey=hotkey)

    def remove_stake(
        self,
        netuid: int,
        hotkey: str,
        amount_unstaked: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.remove_stake.

        Parameters:
            netuid: The netuid of the subnet to remove stake from.
            hotkey: The hotkey SS58 address associated with validator.
            amount_unstaked: Amount of stake in RAO to remove/unstake from the validator.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            hotkey=hotkey,
            amount_unstaked=amount_unstaked,
        )

    def remove_stake_limit(
        self,
        netuid: int,
        hotkey: str,
        amount_unstaked: int,
        limit_price: float,
        allow_partial: bool,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.remove_stake_full.

        Parameters:
            netuid: The netuid of the subnet to remove stake from.
            hotkey: The hotkey SS58 address associated with validator.
            amount_unstaked: Amount of stake in RAO to remove/unstake from the validator.
            limit_price: The limit price expressed in units of RAO per one Alpha.
            allow_partial: Allows partial stake execution of the amount.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            hotkey=hotkey,
            amount_unstaked=amount_unstaked,
            limit_price=limit_price,
            allow_partial=allow_partial,
        )

    def remove_stake_full_limit(
        self,
        netuid: int,
        hotkey: str,
        limit_price: Optional[float] = None,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.remove_stake_full_limit.

        Parameters:
            netuid: The netuid of the subnet to remove stake from.
            hotkey: The hotkey SS58 address associated with validator.
            limit_price: The limit price expressed in units of RAO per one Alpha.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid, hotkey=hotkey, limit_price=limit_price
        )

    def reveal_mechanism_weights(
        self,
        netuid: int,
        mecid: int,
        uids: UIDs,
        values: Weights,
        salt: Salt,
        version_key: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.reveal_mechanism_weights.

        Parameters:
            netuid: The unique identifier of the subnet.
            mecid: The subnet mechanism unique identifier.
            uids: List of neuron UIDs for which weights are being revealed. Think like UIDs.
            values: List of weight values corresponding to each UID. Think like Weights.
            salt: The salt used to generate the hash.
            version_key: Version key for compatibility with the network.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            mecid=mecid,
            uids=uids,
            values=values,
            salt=salt,
            version_key=version_key,
        )

    def root_register(self, hotkey: str) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.root_register.

        Parameters:
            hotkey: The hotkey SS58 address associated with the neuron.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(hotkey=hotkey)

    def serve_axon(
        self,
        netuid: int,
        version: int,
        ip: int,
        port: int,
        ip_type: int,
        protocol: int,
        placeholder1: int = 0,
        placeholder2: int = 0,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.serve_axon.

        Parameters:
            netuid: The network uid to serve on.
            version: The bittensor version identifier.
            ip: Integer representation of endpoint ip.
            port: Endpoint port number i.e., ``9221``.
            ip_type: The endpoint ip version.
            protocol: An ``int`` representation of the protocol.
            placeholder1: Placeholder for further extra params.
            placeholder2: Placeholder for further extra params.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            version=version,
            ip=ip,
            port=port,
            ip_type=ip_type,
            protocol=protocol,
            placeholder1=placeholder1,
            placeholder2=placeholder2,
        )

    def serve_axon_tls(
        self,
        netuid: int,
        version: int,
        ip: int,
        port: int,
        ip_type: int,
        protocol: int,
        placeholder1: int = 0,
        placeholder2: int = 0,
        certificate: Optional[Certificate] = None,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.serve_axon_tls.

        Parameters:
            netuid: The network uid to serve on.
            version: The bittensor version identifier.
            ip: Integer representation of endpoint ip.
            port: Endpoint port number i.e., ``9221``.
            ip_type: The endpoint ip version.
            protocol: An ``int`` representation of the protocol.
            placeholder1: Placeholder for further extra params.
            placeholder2: Placeholder for further extra params.
            certificate: Certificate to use for TLS. If ``None``, no TLS will be used.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            version=version,
            ip=ip,
            port=port,
            ip_type=ip_type,
            protocol=protocol,
            placeholder1=placeholder1,
            placeholder2=placeholder2,
            certificate=certificate,
        )

    def set_coldkey_auto_stake_hotkey(self, netuid: int, hotkey: str) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_coldkey_auto_stake_hotkey.

        Parameters:
            netuid: The netuid of the subnet to set auto stake hotkey for.
            hotkey: The hotkey SS58 address associated with the validator neuron.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, hotkey=hotkey)

    def set_children(
        self,
        hotkey: str,
        netuid: int,
        children: list[tuple[float, str]],
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_children.

        Parameters:
            hotkey: The hotkey SS58 address associated with the neuron.
            netuid: The netuid of the subnet to set children for.
            children: List of tuples containing the proportion of stake to assign to each child hotkey.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            hotkey=hotkey,
            netuid=netuid,
            children=children,
        )

    def set_mechanism_weights(
        self,
        netuid: int,
        mecid: int,
        dests: UIDs,
        weights: Weights,
        version_key: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_mechanism_weights.

        Parameters:
            netuid: The unique identifier of the subnet.
            mecid: The subnet mechanism unique identifier.
            dests: List of neuron UIDs for which weights are being revealed. Think like UIDs.
            weights: List of weight values corresponding to each UID.
            version_key: Version key for compatibility with the network.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            mecid=mecid,
            dests=dests,
            weights=weights,
            version_key=version_key,
        )

    def set_pending_childkey_cooldown(
        self,
        cooldown: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_pending_childkey_cooldown.

        Parameters:
            cooldown: The pending childkey cooldown period in seconds.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(cooldown=cooldown)

    def set_root_claim_type(
        self,
        new_root_claim_type: Literal["Swap", "Keep"] | dict,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_root_claim_type.

        Parameters:
            new_root_claim_type: The new root claim type. Can be:
                - String: "Swap" or "Keep"
                - Dict: {"KeepSubnets": {"subnets": [1, 2, 3]}}

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(new_root_claim_type=new_root_claim_type)

    def set_subnet_identity(
        self,
        netuid: int,
        subnet_name: str,
        github_repo: str,
        subnet_contact: str,
        subnet_url: str,
        discord: str,
        description: str,
        logo_url: str,
        additional: str,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_subnet_identity.

        Parameters:
            netuid: The netuid of the subnet to set identity for.
            subnet_name: The name of the subnet.
            github_repo: The GitHub repository URL of the subnet.
            subnet_contact: The contact information of the subnet owner.
            subnet_url: The URL of the subnet.
            logo_url: The URL of the subnet logo.
            discord: The Discord server URL of the subnet.
            description: The description of the subnet.
            additional: Additional information about the subnet.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            subnet_name=subnet_name,
            github_repo=github_repo,
            subnet_contact=subnet_contact,
            subnet_url=subnet_url,
            discord=discord,
            description=description,
            logo_url=logo_url,
            additional=additional,
        )

    def start_call(self, netuid: int) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.start_call.

        Parameters:
            netuid: The netuid of the subnet to to be activated.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid)

    def swap_stake(
        self,
        hotkey: str,
        origin_netuid: int,
        destination_netuid: int,
        alpha_amount: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.swap_stake.

        Parameters:
            hotkey: The hotkey SS58 address associated with the stake.
            origin_netuid: The source subnet UID.
            destination_netuid: The destination subnet UID.
            alpha_amount: Amount of stake in RAO to swap.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            hotkey=hotkey,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            alpha_amount=alpha_amount,
        )

    def swap_stake_limit(
        self,
        hotkey: str,
        origin_netuid: int,
        destination_netuid: int,
        alpha_amount: int,
        limit_price: float,
        allow_partial: bool,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.swap_stake_limit.

        Parameters:
            hotkey: The hotkey SS58 address associated with the stake.
            origin_netuid: The source subnet UID.
            destination_netuid: The destination subnet UID.
            alpha_amount: The amount of stake in RAO to swap.
            allow_partial: If true, allows partial stake swaps when the full amount would exceed the price
                tolerance.
            limit_price: Maximum allowed increase in a price ratio (0.005 = 0.5%).

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            hotkey=hotkey,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            alpha_amount=alpha_amount,
            limit_price=limit_price,
            allow_partial=allow_partial,
        )

    def transfer_stake(
        self,
        destination_coldkey: str,
        hotkey: str,
        origin_netuid: int,
        destination_netuid: int,
        alpha_amount: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.transfer_stake.

        Parameters:
            destination_coldkey: SS58 address of the destination coldkey.
            hotkey: SS58 address of the hotkey associated with the stake.
            origin_netuid: Network UID of the origin subnet.
            destination_netuid: Network UID of the destination subnet.
            alpha_amount: The amount of stake in RAO to transfer as a `Balance` object.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            destination_coldkey=destination_coldkey,
            hotkey=hotkey,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            alpha_amount=alpha_amount,
        )
