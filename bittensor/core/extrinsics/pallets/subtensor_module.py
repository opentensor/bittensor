from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING

from bittensor.core.types import UIDs, Weights, Salt
from bittensor.utils import float_to_u64, Certificate
from bittensor.utils.balance import Balance
from .base import CallBuilder as _BasePallet, Call

if TYPE_CHECKING:
    from bittensor.core.chain_data import DynamicInfo


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
        hotkey_ss58: str,
        amount: "Balance",
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.add_stake.

        Parameters:
            netuid: The netuid of the subnet to add stake to.
            hotkey_ss58: The hotkey SS58 address associated with validator.
            amount: Amount of stake to add.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            hotkey=hotkey_ss58,
            amount_staked=amount.rao,
        )

    def add_stake_limit(
        self,
        netuid: int,
        hotkey_ss58: str,
        amount: "Balance",
        allow_partial_stake: bool,
        rate_tolerance: float,
        pool: "DynamicInfo",
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.add_stake_limit.

        Parameters:
            netuid: The netuid of the subnet to add stake to.
            hotkey_ss58: The hotkey SS58 address associated with validator.
            amount: Amount of stake to add.
            allow_partial_stake: If True, allows partial unstaking if price tolerance exceeded.
            rate_tolerance: Maximum allowed price increase percentage.
            pool: DynamicInfo object of the Subnet with provided netuid.

        Returns:
            GenericCall instance.
        """
        base_price = pool.price.tao
        price_with_tolerance = (
            base_price if pool.netuid == 0 else base_price * (1 + rate_tolerance)
        )
        limit_price = Balance.from_tao(price_with_tolerance).rao
        return self.create_composed_call(
            netuid=netuid,
            hotkey=hotkey_ss58,
            amount_staked=amount.rao,
            limit_price=limit_price,
            allow_partial=allow_partial_stake,
        )

    def burned_register(
        self,
        netuid: int,
        hotkey_ss58: str,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.burned_register.

        Parameters:
            netuid: The netuid of the subnet to register on.
            hotkey_ss58: The hotkey SS58 address associated with the neuron.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, hotkey=hotkey_ss58)

    def claim_root(self, netuids: list[int]) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.claim_root.

        Parameters:
            netuids: The netuids of the subnets to claim root for.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(subnets=netuids)

    def commit_mechanism_weights(
        self,
        netuid: int,
        mechid: int,
        commit_hash: str,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.commit_mechanism_weights.

        Parameters:
            netuid: The unique identifier of the subnet.
            mechid: The subnet mechanism unique identifier.
            commit_hash: The hash of the commitment.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid, mecid=mechid, commit_hash=commit_hash
        )

    def commit_timelocked_mechanism_weights(
        self,
        netuid: int,
        mechid: int,
        commit_for_reveal: bytes,
        reveal_round: int,
        commit_reveal_version: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.commit_mechanism_weights.

        Parameters:
            netuid: The unique identifier of the subnet.
            mechid: The subnet mechanism unique identifier.
            commit_for_reveal: Raw bytes of the encrypted and compressed uids & weights values for setting weights.
            reveal_round: Drand round number when weights have to be revealed. Based on Drand Quicknet network.
            commit_reveal_version: The version of the commit-reveal in the chain.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            mecid=mechid,
            commit=commit_for_reveal,
            reveal_round=reveal_round,
            commit_reveal_version=commit_reveal_version,
        )

    def decrease_take(self, hotkey_ss58: str, take: int) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.decrease_take.

        Parameters:
            hotkey_ss58: SS58 address of the hotkey to set take for.
            take: The percentage of rewards that the delegate claims from nominators.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(hotkey=hotkey_ss58, take=take)

    def increase_take(self, hotkey_ss58: str, take: int) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.increase_take.

        Parameters:
            hotkey_ss58: SS58 address of the hotkey to set take for.
            take: The percentage of rewards that the delegate claims from nominators.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(hotkey=hotkey_ss58, take=take)

    def move_stake(
        self,
        origin_netuid: int,
        origin_hotkey_ss58: str,
        destination_netuid: int,
        destination_hotkey_ss58: str,
        amount: "Balance",
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.move_stake.

        Parameters:
            origin_netuid: The netuid of the source subnet.
            origin_hotkey_ss58: The SS58 address of the source hotkey.
            destination_netuid: The netuid of the destination subnet.
            destination_hotkey_ss58: The SS58 address of the destination hotkey.
            amount: Amount of origin Balance to move.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            origin_netuid=origin_netuid,
            origin_hotkey=origin_hotkey_ss58,
            destination_netuid=destination_netuid,
            destination_hotkey=destination_hotkey_ss58,
            alpha_amount=amount.rao,
        )

    def register(
        self,
        netuid: int,
        coldkey_ss58: str,
        hotkey_ss58: str,
        block_number: int,
        nonce: int,
        work: list[int],
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.register.

        Parameters:
            netuid: The netuid of the subnet to register on.
            coldkey_ss58: The coldkey SS58 address associated with the neuron.
            hotkey_ss58: The hotkey SS58 address associated with the neuron.
            block_number: POW block number.
            nonce: POW nonce.
            work: List representation of POW seal.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            coldkey=coldkey_ss58,
            hotkey=hotkey_ss58,
            block_number=block_number,
            nonce=nonce,
            work=work,
        )

    def register_network(self, hotkey_ss58: str) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.register_network.

        Parameters:
            hotkey_ss58: The hotkey SS58 address associated with the subnet owner.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(hotkey=hotkey_ss58)

    def remove_stake(
        self,
        netuid: int,
        hotkey_ss58: str,
        amount: "Balance",
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.remove_stake.

        Parameters:
            netuid: The netuid of the subnet to remove stake from.
            hotkey_ss58: The hotkey SS58 address associated with validator.
            amount: Amount of stake to remove/unstake from the validator.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            hotkey=hotkey_ss58,
            amount_unstaked=amount.rao,
        )

    def remove_stake_limit(
        self,
        netuid: int,
        hotkey_ss58: str,
        amount: "Balance",
        allow_partial_stake: bool,
        rate_tolerance: float,
        pool: Optional["DynamicInfo"] = None,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.remove_stake_full.

        Parameters:
            netuid: The netuid of the subnet to remove stake from.
            hotkey_ss58: The hotkey SS58 address associated with validator.
            amount: Amount of stake to remove/unstake from the validator.
            allow_partial_stake: If True, allows partial unstaking if price tolerance exceeded.
            rate_tolerance: Maximum allowed price decrease percentage.
            pool: DynamicInfo object of the Subnet with provided netuid.

        Returns:
            GenericCall instance.
        """
        params = {
            "netuid": netuid,
            "hotkey": hotkey_ss58,
            "amount_unstaked": amount.rao,
        }

        if pool.netuid == 0:
            price_with_tolerance = pool.price.tao
        else:
            price_with_tolerance = pool.price.tao * (1 - rate_tolerance)

        limit_price = Balance.from_tao(price_with_tolerance).rao
        params.update(
            {
                "limit_price": limit_price,
                "allow_partial": allow_partial_stake,
            }
        )
        return self.create_composed_call(**params)

    def remove_stake_full_limit(
        self,
        netuid: int,
        hotkey_ss58: str,
        rate_tolerance: Optional[float] = None,
        pool: Optional["DynamicInfo"] = None,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.remove_stake_full_limit.

        Parameters:
            netuid: The netuid of the subnet to remove stake from.
            hotkey_ss58: The hotkey SS58 address associated with validator.
            rate_tolerance: Maximum allowed price decrease percentage.
            pool: DynamicInfo object of the Subnet with provided netuid.

        Returns:
            GenericCall instance.
        """
        params = {
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "limit_price": None,
        }
        if rate_tolerance:
            limit_price = pool.price * (1 - rate_tolerance)
            params.update({"limit_price": limit_price})

        return self.create_composed_call(**params)

    def reveal_mechanism_weights(
        self,
        netuid: int,
        mechid: int,
        uids: UIDs,
        weights: Weights,
        salt: Salt,
        version_key: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.reveal_mechanism_weights."""
        return self.create_composed_call(
            netuid=netuid,
            mecid=mechid,
            uids=uids,
            values=weights,
            salt=salt,
            version_key=version_key,
        )

    def root_register(self, hotkey_ss58: str) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.root_register.

        Parameters:
            hotkey_ss58: The hotkey SS58 address associated with the neuron.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(hotkey=hotkey_ss58)

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

    def set_coldkey_auto_stake_hotkey(self, netuid: int, hotkey_ss58: str) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_coldkey_auto_stake_hotkey.

        Parameters:
            netuid: The netuid of the subnet to set auto stake hotkey for.
            hotkey_ss58: The hotkey SS58 address associated with the validator neuron.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(netuid=netuid, hotkey=hotkey_ss58)

    def set_children(
        self,
        hotkey_ss58: str,
        netuid: int,
        children: list[tuple[float, str]],
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_children.

        Parameters:
            hotkey_ss58: The hotkey SS58 address associated with the neuron.
            netuid: The netuid of the subnet to set children for.
            children: List of tuples containing the proportion of stake to assign to each child hotkey.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            hotkey=hotkey_ss58,
            netuid=netuid,
            children=[
                (float_to_u64(proportion), child_hotkey)
                for proportion, child_hotkey in children
            ],
        )

    def set_mechanism_weights(
        self,
        netuid: int,
        mechid: int,
        uids: UIDs,
        weights: Weights,
        version_key: int,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_mechanism_weights.

        Parameters:
            netuid: The unique identifier of the subnet.
            mechid: The subnet mechanism unique identifier.
            uids: List of neuron UIDs for which weights are being revealed.
            weights: List of weight values corresponding to each UID.
            version_key: Version key for compatibility with the network.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            netuid=netuid,
            mecid=mechid,
            dests=uids,
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
        new_root_claim_type: Literal["Swap", "Keep"],
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_root_claim_type.

        Parameters:
            new_root_claim_type: The new root claim type.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(new_root_claim_type=new_root_claim_type)

    def set_subnet_identity(
        self,
        netuid: int,
        hotkey_ss58: str,
        subnet_name: str,
        github_repo: str,
        subnet_contact: str,
        subnet_url: str,
        logo_url: str,
        discord: str,
        description: str,
        additional: str,
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.set_subnet_identity.

        Parameters:
            netuid: The netuid of the subnet to set identity for.
            hotkey_ss58: The hotkey SS58 address associated with the subnet owner.
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
            hotkey=hotkey_ss58,
            netuid=netuid,
            subnet_name=subnet_name,
            github_repo=github_repo,
            subnet_contact=subnet_contact,
            subnet_url=subnet_url,
            logo_url=logo_url,
            discord=discord,
            description=description,
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
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: "Balance",
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.swap_stake.

        Parameters:
            hotkey_ss58: The hotkey SS58 address associated with the stake.
            origin_netuid: The source subnet UID.
            destination_netuid: The destination subnet UID.
            amount: Amount to swap.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            hotkey=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            alpha_amount=amount.rao,
        )

    def swap_stake_limit(
        self,
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: "Balance",
        allow_partial_stake: bool,
        rate_tolerance: float,
        origin_pool: "DynamicInfo",
        destination_pool: "DynamicInfo",
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.swap_stake_limit.

        Parameters:
            hotkey_ss58: The hotkey SS58 address associated with the stake.
            origin_netuid: The source subnet UID.
            destination_netuid: The destination subnet UID.
            amount: Amount to swap.
            allow_partial_stake: If true, allows partial stake swaps when the full amount would exceed the price
                tolerance.
            rate_tolerance: Maximum allowed increase in a price ratio (0.005 = 0.5%).
            origin_pool: Subnet DynamicInfo object for the origin subnet.
            destination_pool: Subnet DynamicInfo object for the destination subnet.

        Returns:
            GenericCall instance.
        """
        swap_rate_ratio = origin_pool.price.rao / destination_pool.price.rao
        swap_rate_ratio_with_tolerance = swap_rate_ratio * (1 + rate_tolerance)

        return self.create_composed_call(
            hotkey=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            alpha_amount=amount.rao,
            limit_price=swap_rate_ratio_with_tolerance,
            allow_partial=allow_partial_stake,
        )

    def transfer_stake(
        self,
        destination_coldkey_ss58: str,
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: "Balance",
    ) -> Call:
        """Returns GenericCall instance for Subtensor function SubtensorModule.transfer_stake.

        Parameters:
            destination_coldkey_ss58: SS58 address of the destination coldkey.
            hotkey_ss58: SS58 address of the hotkey associated with the stake.
            origin_netuid: Network UID of the origin subnet.
            destination_netuid: Network UID of the destination subnet.
            amount: The amount of stake to transfer as a `Balance` object.

        Returns:
            GenericCall instance.
        """
        return self.create_composed_call(
            destination_coldkey=destination_coldkey_ss58,
            hotkey=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            alpha_amount=amount.rao,
        )
