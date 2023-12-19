# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import copy
import torch
import argparse
import bittensor
import scalecodec

from retry import retry
from loguru import logger
from typing import List, Dict, Union, Optional, Tuple, TypedDict, Any
from substrateinterface.base import QueryMapResult, SubstrateInterface
from scalecodec.base import RuntimeConfiguration
from scalecodec.type_registry import load_type_registry_preset

# Local imports.
from .chain_data import (
    NeuronInfo,
    DelegateInfo,
    PrometheusInfo,
    SubnetInfo,
    SubnetHyperparameters,
    StakeInfo,
    NeuronInfoLite,
    AxonInfo,
    ProposalVoteData,
    ProposalCallData,
    IPInfo,
    custom_rpc_type_registry,
)
from .errors import *
from .extrinsics.network import (
    register_subnetwork_extrinsic,
    set_hyperparameter_extrinsic,
)
from .extrinsics.staking import add_stake_extrinsic, add_stake_multiple_extrinsic
from .extrinsics.unstaking import unstake_extrinsic, unstake_multiple_extrinsic
from .extrinsics.serving import (
    serve_extrinsic,
    serve_axon_extrinsic,
    publish_metadata,
    get_metadata,
)
from .extrinsics.registration import (
    register_extrinsic,
    burned_register_extrinsic,
    run_faucet_extrinsic,
    swap_hotkey_extrinsic,
)
from .extrinsics.transfer import transfer_extrinsic
from .extrinsics.set_weights import set_weights_extrinsic
from .extrinsics.prometheus import prometheus_extrinsic
from .extrinsics.delegation import (
    delegate_extrinsic,
    nominate_extrinsic,
    undelegate_extrinsic,
)
from .extrinsics.senate import (
    register_senate_extrinsic,
    leave_senate_extrinsic,
    vote_senate_extrinsic,
)
from .extrinsics.root import root_register_extrinsic, set_root_weights_extrinsic
from .types import AxonServeCallParams, PrometheusServeCallParams
from .utils import U16_NORMALIZED_FLOAT, ss58_to_vec_u8
from .utils.balance import Balance
from .utils.registration import POWSolution

logger = logger.opt(colors=True)


class ParamWithTypes(TypedDict):
    name: str  # Name of the parameter.
    type: str  # ScaleType string of the parameter.


class subtensor:
    """
    The Subtensor class in Bittensor serves as a crucial interface for interacting with the Bittensor blockchain,
    facilitating a range of operations essential for the decentralized machine learning network. This class
    enables neurons (network participants) to engage in activities such as registering on the network, managing
    staked weights, setting inter-neuronal weights, and participating in consensus mechanisms.

    The Bittensor network operates on a digital ledger where each neuron holds stakes (S) and learns a set
    of inter-peer weights (W). These weights, set by the neurons themselves, play a critical role in determining
    the ranking and incentive mechanisms within the network. Higher-ranked neurons, as determined by their
    contributions and trust within the network, receive more incentives.

    The Subtensor class connects to various Bittensor networks like the main 'finney' network or local test
    networks, providing a gateway to the blockchain layer of Bittensor. It leverages a staked weighted trust
    system and consensus to ensure fair and distributed incentive mechanisms, where incentives (I) are
    primarily allocated to neurons that are trusted by the majority of the network&#8203;``【oaicite:1】``&#8203;.

    Additionally, Bittensor introduces a speculation-based reward mechanism in the form of bonds (B), allowing
    neurons to accumulate bonds in other neurons, speculating on their future value. This mechanism aligns
    with market-based speculation, incentivizing neurons to make judicious decisions in their inter-neuronal
    investments.

    Attributes:
        network (str): The name of the Bittensor network (e.g., 'finney', 'test', 'archive', 'local') the instance
                       is connected to, determining the blockchain interaction context.
        chain_endpoint (str): The blockchain node endpoint URL, enabling direct communication
                              with the Bittensor blockchain for transaction processing and data retrieval.

    Example Usage:
        # Connect to the main Bittensor network (Finney).
        finney_subtensor = subtensor(network='finney')

        # Register a new neuron on the network.
        wallet = bittensor.wallet(...)  # Assuming a wallet instance is created.
        success = finney_subtensor.register(wallet=wallet, netuid=netuid)

        # Set inter-neuronal weights for collaborative learning.
        success = finney_subtensor.set_weights(wallet=wallet, netuid=netuid, uids=[...], weights=[...])

        # Speculate by accumulating bonds in other promising neurons.
        success = finney_subtensor.delegate(wallet=wallet, delegate_ss58=other_neuron_ss58, amount=bond_amount)

        # Get the metagraph for a specific subnet using given subtensor connection
        metagraph = subtensor.metagraph(netuid=netuid)

    By facilitating these operations, the Subtensor class is instrumental in maintaining the decentralized
    intelligence and dynamic learning environment of the Bittensor network, as envisioned in its foundational
    principles and mechanisms described in the NeurIPS paper.
    """

    @staticmethod
    def config() -> "bittensor.config":
        parser = argparse.ArgumentParser()
        subtensor.add_args(parser)
        return bittensor.config(parser, args=[])

    @classmethod
    def help(cls):
        """Print help to stdout"""
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print(cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        prefix_str = "" if prefix == None else prefix + "."
        try:
            default_network = os.getenv("BT_SUBTENSOR_NETWORK") or "finney"
            default_chain_endpoint = (
                os.getenv("BT_SUBTENSOR_CHAIN_ENDPOINT")
                or bittensor.__finney_entrypoint__
            )
            parser.add_argument(
                "--" + prefix_str + "subtensor.network",
                default=default_network,
                type=str,
                help="""The subtensor network flag. The likely choices are:
                                        -- finney (main network)
                                        -- test (test network)
                                        -- archive (archive network +300 blocks)
                                        -- local (local running network)
                                    If this option is set it overloads subtensor.chain_endpoint with
                                    an entry point node from that network.
                                    """,
            )
            parser.add_argument(
                "--" + prefix_str + "subtensor.chain_endpoint",
                default=default_chain_endpoint,
                type=str,
                help="""The subtensor endpoint flag. If set, overrides the --network flag.
                                    """,
            )
            parser.add_argument(
                "--" + prefix_str + "subtensor._mock",
                default=False,
                type=bool,
                help="""If true, uses a mocked connection to the chain.
                                    """,
            )

        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @staticmethod
    def determine_chain_endpoint_and_network(network: str):
        """Determines the chain endpoint and network from the passed network or chain_endpoint.
        Args:
            network (str): The network flag. The likely choices are:
                    -- finney (main network)
                    -- archive (archive network +300 blocks)
                    -- local (local running network)
                    -- test (test network)
            chain_endpoint (str): The chain endpoint flag. If set, overrides the network argument.
        Returns:
            network (str): The network flag. The likely choices are:
            chain_endpoint (str): The chain endpoint flag. If set, overrides the network argument.
        """
        if network == None:
            return None, None
        if network in ["finney", "local", "test", "archive"]:
            if network == "finney":
                # Kiru Finney stagin network.
                return network, bittensor.__finney_entrypoint__
            elif network == "local":
                return network, bittensor.__local_entrypoint__
            elif network == "test":
                return network, bittensor.__finney_test_entrypoint__
            elif network == "archive":
                return network, bittensor.__archive_entrypoint__
        else:
            if (
                network == bittensor.__finney_entrypoint__
                or "entrypoint-finney.opentensor.ai" in network
            ):
                return "finney", bittensor.__finney_entrypoint__
            elif (
                network == bittensor.__finney_test_entrypoint__
                or "test.finney.opentensor.ai" in network
            ):
                return "test", bittensor.__finney_test_entrypoint__
            elif (
                network == bittensor.__archive_entrypoint__
                or "archive.chain.opentensor.ai" in network
            ):
                return "archive", bittensor.__archive_entrypoint__
            elif "127.0.0.1" in network or "localhost" in network:
                return "local", network
            else:
                return "unknown", network

    @staticmethod
    def setup_config(network: str, config: bittensor.config):
        if network != None:
            (
                evaluated_network,
                evaluated_endpoint,
            ) = subtensor.determine_chain_endpoint_and_network(network)
        else:
            if config.get("__is_set", {}).get("subtensor.chain_endpoint"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.chain_endpoint
                )

            elif config.get("__is_set", {}).get("subtensor.network"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.network
                )

            elif config.subtensor.get("chain_endpoint"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.chain_endpoint
                )

            elif config.subtensor.get("network"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.network
                )

            else:
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = subtensor.determine_chain_endpoint_and_network(
                    bittensor.defaults.subtensor.network
                )

        return (
            bittensor.utils.networking.get_formatted_ws_endpoint_url(
                evaluated_endpoint
            ),
            evaluated_network,
        )

    def __init__(
        self,
        network: str = None,
        config: "bittensor.config" = None,
        _mock: bool = False,
        log_verbose: bool = True,
    ) -> None:
        """
        Initializes a Subtensor interface for interacting with the Bittensor blockchain.

        NOTE: Currently subtensor defaults to the finney network. This will change in a future release.

        We strongly encourage users to run their own local subtensor node whenever possible. This increases
        decentralization and resilience of the network. In a future release, local subtensor will become the
        default and the fallback to finney removed. Please plan ahead for this change. We will provide detailed
        instructions on how to run a local subtensor node in the documentation in a subsequent release.

        Args:
            network (str, optional): The network name to connect to (e.g., 'finney', 'local').
                                     Defaults to the main Bittensor network if not specified.
            config (bittensor.config, optional): Configuration object for the subtensor.
                                                 If not provided, a default configuration is used.
            _mock (bool, optional): If set to True, uses a mocked connection for testing purposes.

        This initialization sets up the connection to the specified Bittensor network, allowing for various
        blockchain operations such as neuron registration, stake management, and setting weights.

        """
        # Determine config.subtensor.chain_endpoint and config.subtensor.network config.
        # If chain_endpoint is set, we override the network flag, otherwise, the chain_endpoint is assigned by the network.
        # Argument importance: network > chain_endpoint > config.subtensor.chain_endpoint > config.subtensor.network
        if config == None:
            config = subtensor.config()
        self.config = copy.deepcopy(config)

        # Setup config.subtensor.network and config.subtensor.chain_endpoint
        self.chain_endpoint, self.network = subtensor.setup_config(network, config)

        if (
            self.network == "finney"
            or self.chain_endpoint == bittensor.__finney_entrypoint__
        ) and log_verbose:
            bittensor.logging.info(
                f"You are connecting to {self.network} network with endpoint {self.chain_endpoint}."
            )
            bittensor.logging.warning(
                "We strongly encourage running a local subtensor node whenever possible. "
                "This increases decentralization and resilience of the network."
            )
            bittensor.logging.warning(
                "In a future release, local subtensor will become the default endpoint. "
                "To get ahead of this change, please run a local subtensor node and point to it."
            )

        # Returns a mocked connection with a background chain connection.
        self.config.subtensor._mock = (
            _mock
            if _mock != None
            else self.config.subtensor.get("_mock", bittensor.defaults.subtensor._mock)
        )
        if self.config.subtensor._mock:
            config.subtensor._mock = True
            return bittensor.subtensor_mock.MockSubtensor()

        # Attempt to connect to chosen endpoint. Fallback to finney if local unavailable.
        try:
            # Set up params.
            self.substrate = SubstrateInterface(
                ss58_format=bittensor.__ss58_format__,
                use_remote_preset=True,
                url=self.chain_endpoint,
                type_registry=bittensor.__type_registry__,
            )
        except ConnectionRefusedError as e:
            bittensor.logging.error(
                f"Could not connect to {self.network} network with {self.chain_endpoint} chain endpoint. Exiting..."
            )
            bittensor.logging.info(
                f"You can check if you have connectivity by runing this command: nc -vz localhost {self.chain_endpoint.split(':')[2]}"
            )
            exit(1)
            # TODO (edu/phil): Advise to run local subtensor and point to dev docs.

        try:
            self.substrate.websocket.settimeout(600)
        except:
            bittensor.logging.warning("Could not set websocket timeout.")

        if log_verbose:
            bittensor.logging.info(
                f"Connected to {self.network} network and {self.chain_endpoint}."
            )

    def __str__(self) -> str:
        if self.network == self.chain_endpoint:
            # Connecting to chain endpoint without network known.
            return "subtensor({})".format(self.chain_endpoint)
        else:
            # Connecting to network with endpoint known.
            return "subtensor({}, {})".format(self.network, self.chain_endpoint)

    def __repr__(self) -> str:
        return self.__str__()

    #####################
    #### Delegation #####
    #####################
    def nominate(
        self,
        wallet: "bittensor.wallet",
        wait_for_finalization: bool = False,
        wait_for_inclusion: bool = True,
    ) -> bool:
        """
        Becomes a delegate for the hotkey associated with the given wallet. This method is used to nominate
        a neuron (identified by the hotkey in the wallet) as a delegate on the Bittensor network, allowing it
        to participate in consensus and validation processes.

        Args:
            wallet (bittensor.wallet): The wallet containing the hotkey to be nominated.
            wait_for_finalization (bool, optional): If True, waits until the transaction is finalized on the blockchain.
            wait_for_inclusion (bool, optional): If True, waits until the transaction is included in a block.

        Returns:
            bool: True if the nomination process is successful, False otherwise.

        This function is a key part of the decentralized governance mechanism of Bittensor, allowing for the
        dynamic selection and participation of validators in the network's consensus process.
        """
        return nominate_extrinsic(
            subtensor=self,
            wallet=wallet,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
        )

    def delegate(
        self,
        wallet: "bittensor.wallet",
        delegate_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Becomes a delegate for the hotkey associated with the given wallet. This method is used to nominate
        a neuron (identified by the hotkey in the wallet) as a delegate on the Bittensor network, allowing it
        to participate in consensus and validation processes.

        Args:
            wallet (bittensor.wallet): The wallet containing the hotkey to be nominated.
            wait_for_finalization (bool, optional): If True, waits until the transaction is finalized on the blockchain.
            wait_for_inclusion (bool, optional): If True, waits until the transaction is included in a block.

        Returns:
            bool: True if the nomination process is successful, False otherwise.

        This function is a key part of the decentralized governance mechanism of Bittensor, allowing for the
        dynamic selection and participation of validators in the network's consensus process.
        """
        return delegate_extrinsic(
            subtensor=self,
            wallet=wallet,
            delegate_ss58=delegate_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    def undelegate(
        self,
        wallet: "bittensor.wallet",
        delegate_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Removes a specified amount of stake from a delegate neuron using the provided wallet. This action
        reduces the staked amount on another neuron, effectively withdrawing support or speculation.

        Args:
            wallet (bittensor.wallet): The wallet used for the undelegation process.
            delegate_ss58 (Optional[str]): The SS58 address of the delegate neuron.
            amount (Union[Balance, float]): The amount of TAO to undelegate.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the undelegation is successful, False otherwise.

        This function reflects the dynamic and speculative nature of the Bittensor network, allowing neurons
        to adjust their stakes and investments based on changing perceptions and performances within the network.
        """
        return undelegate_extrinsic(
            subtensor=self,
            wallet=wallet,
            delegate_ss58=delegate_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    #####################
    #### Set Weights ####
    #####################
    def set_weights(
        self,
        wallet: "bittensor.wallet",
        netuid: int,
        uids: Union[torch.LongTensor, list],
        weights: Union[torch.FloatTensor, list],
        version_key: int = bittensor.__version_as_int__,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Sets the inter-neuronal weights for the specified neuron. This process involves specifying the
        influence or trust a neuron places on other neurons in the network, which is a fundamental aspect
        of Bittensor's decentralized learning architecture.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron setting the weights.
            netuid (int): The unique identifier of the subnet.
            uids (Union[torch.LongTensor, list]): The list of neuron UIDs that the weights are being set for.
            weights (Union[torch.FloatTensor, list]): The corresponding weights to be set for each UID.
            version_key (int, optional): Version key for compatibility with the network.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the setting of weights is successful, False otherwise.

        This function is crucial in shaping the network's collective intelligence, where each neuron's
        learning and contribution are influenced by the weights it sets towards others【81†source】.
        """
        return set_weights_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    def _do_set_weights(
        self,
        wallet: "bittensor.wallet",
        uids: List[int],
        vals: List[int],
        netuid: int,
        version_key: int = bittensor.__version_as_int__,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:  # (success, error_message)
        """
        Internal method to send a transaction to the Bittensor blockchain, setting weights
        for specified neurons. This method constructs and submits the transaction, handling
        retries and blockchain communication.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron setting the weights.
            uids (List[int]): List of neuron UIDs for which weights are being set.
            vals (List[int]): List of weight values corresponding to each UID.
            netuid (int): Unique identifier for the network.
            version_key (int, optional): Version key for compatibility with the network.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

        This method is vital for the dynamic weighting mechanism in Bittensor, where neurons adjust their
        trust in other neurons based on observed performance and contributions.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="set_weights",
                    call_params={
                        "dests": uids,
                        "weights": vals,
                        "netuid": netuid,
                        "version_key": version_key,
                    },
                )
                # Period dictates how long the extrinsic will stay as part of waiting pool
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.hotkey, era={"period": 100}
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True, None

                response.process_events()
                if response.is_success:
                    return True, None
                else:
                    return False, response.error_message

        return make_substrate_call_with_retry()

    ######################
    #### Registration ####
    ######################
    def register(
        self,
        wallet: "bittensor.wallet",
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
        max_allowed_attempts: int = 3,
        output_in_place: bool = True,
        cuda: bool = False,
        dev_id: Union[List[int], int] = 0,
        tpb: int = 256,
        num_processes: Optional[int] = None,
        update_interval: Optional[int] = None,
        log_verbose: bool = False,
    ) -> bool:
        """
        Registers a neuron on the Bittensor network using the provided wallet. Registration
        is a critical step for a neuron to become an active participant in the network, enabling
        it to stake, set weights, and receive incentives.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron to be registered.
            netuid (int): The unique identifier of the subnet.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            Other arguments: Various optional parameters to customize the registration process.

        Returns:
            bool: True if the registration is successful, False otherwise.

        This function facilitates the entry of new neurons into the network, supporting the decentralized
        growth and scalability of the Bittensor ecosystem.
        """
        return register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
            max_allowed_attempts=max_allowed_attempts,
            output_in_place=output_in_place,
            cuda=cuda,
            dev_id=dev_id,
            tpb=tpb,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )

    def swap_hotkey(
        self,
        wallet: "bittensor.wallet",
        new_wallet: "bittensor.wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
    ) -> bool:
        """Swaps an old hotkey to a new hotkey."""
        return swap_hotkey_extrinsic(
            subtensor=self,
            wallet=wallet,
            new_wallet=new_wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    def run_faucet(
        self,
        wallet: "bittensor.wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
        max_allowed_attempts: int = 3,
        output_in_place: bool = True,
        cuda: bool = False,
        dev_id: Union[List[int], int] = 0,
        tpb: int = 256,
        num_processes: Optional[int] = None,
        update_interval: Optional[int] = None,
        log_verbose: bool = False,
    ) -> bool:
        """
        Facilitates a faucet transaction, allowing new neurons to receive an initial amount of TAO
        for participating in the network. This function is particularly useful for newcomers to the
        Bittensor network, enabling them to start with a small stake on testnet only.

        Args:
            wallet (bittensor.wallet): The wallet for which the faucet transaction is to be run.
            Other arguments: Various optional parameters to customize the faucet transaction process.

        Returns:
            bool: True if the faucet transaction is successful, False otherwise.

        This function is part of Bittensor's onboarding process, ensuring that new neurons have
        the necessary resources to begin their journey in the decentralized AI network.

        Note: This is for testnet ONLY and is disabled currently. You must build your own
        staging subtensor chain with the `--features pow-faucet` argument to enable this.
        """
        return run_faucet_extrinsic(
            subtensor=self,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
            max_allowed_attempts=max_allowed_attempts,
            output_in_place=output_in_place,
            cuda=cuda,
            dev_id=dev_id,
            tpb=tpb,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )

    def burned_register(
        self,
        wallet: "bittensor.wallet",
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
    ) -> bool:
        """
        Registers a neuron on the Bittensor network by burning TAO. This method of registration
        involves recycling TAO tokens, contributing to the network's deflationary mechanism.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron to be registered.
            netuid (int): The unique identifier of the subnet.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the registration is successful, False otherwise.

        This function offers an alternative registration path, aligning with the network's principles
        of token circulation and value conservation.
        """
        return burned_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    def _do_pow_register(
        self,
        netuid: int,
        wallet: "bittensor.wallet",
        pow_result: POWSolution,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """Sends a (POW) register extrinsic to the chain.
        Args:
            netuid (int): the subnet to register on.
            wallet (bittensor.wallet): the wallet to register.
            pow_result (POWSolution): the pow result to register.
            wait_for_inclusion (bool): if true, waits for the extrinsic to be included in a block.
            wait_for_finalization (bool): if true, waits for the extrinsic to be finalized.
        Returns:
            success (bool): True if the extrinsic was included in a block.
            error (Optional[str]): None on success or not waiting for inclusion/finalization, otherwise the error message.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                # create extrinsic call
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="register",
                    call_params={
                        "netuid": netuid,
                        "block_number": pow_result.block_number,
                        "nonce": pow_result.nonce,
                        "work": [int(byte_) for byte_ in pow_result.seal],
                        "hotkey": wallet.hotkey.ss58_address,
                        "coldkey": wallet.coldkeypub.ss58_address,
                    },
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.hotkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )

                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True, None

                # process if registration successful, try again if pow is still valid
                response.process_events()
                if not response.is_success:
                    return False, response.error_message
                # Successful registration
                else:
                    return True, None

        return make_substrate_call_with_retry()

    def _do_burned_register(
        self,
        netuid: int,
        wallet: "bittensor.wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                # create extrinsic call
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="burned_register",
                    call_params={
                        "netuid": netuid,
                        "hotkey": wallet.hotkey.ss58_address,
                    },
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )

                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True

                # process if registration successful, try again if pow is still valid
                response.process_events()
                if not response.is_success:
                    return False, response.error_message
                # Successful registration
                else:
                    return True, None

        return make_substrate_call_with_retry()

    def _do_swap_hotkey(
        self,
        wallet: "bittensor.wallet",
        new_wallet: "bittensor.wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                # create extrinsic call
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="swap_hotkey",
                    call_params={
                        "hotkey": wallet.hotkey.ss58_address,
                        "new_hotkey": new_wallet.hotkey.ss58_address,
                    },
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )

                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True

                # process if registration successful, try again if pow is still valid
                response.process_events()
                if not response.is_success:
                    return False, response.error_message
                # Successful registration
                else:
                    return True, None

        return make_substrate_call_with_retry()

    ##################
    #### Transfer ####
    ##################
    def transfer(
        self,
        wallet: "bittensor.wallet",
        dest: str,
        amount: Union[Balance, float],
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Executes a transfer of funds from the provided wallet to the specified destination address.
        This function is used to move TAO tokens within the Bittensor network, facilitating transactions
        between neurons.

        Args:
            wallet (bittensor.wallet): The wallet from which funds are being transferred.
            dest (str): The destination public key address.
            amount (Union[Balance, float]): The amount of TAO to be transferred.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the transfer is successful, False otherwise.

        This function is essential for the fluid movement of tokens in the network, supporting
        various economic activities such as staking, delegation, and reward distribution.
        """
        return transfer_extrinsic(
            subtensor=self,
            wallet=wallet,
            dest=dest,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    def get_transfer_fee(
        self, wallet: "bittensor.wallet", dest: str, value: Union[Balance, float, int]
    ) -> Balance:
        """
        Calculates the transaction fee for transferring tokens from a wallet to a specified destination address.
        This function simulates the transfer to estimate the associated cost, taking into account the current
        network conditions and transaction complexity.

        Args:
            wallet (bittensor.wallet): The wallet from which the transfer is initiated.
            dest (str): The SS58 address of the destination account.
            value (Union[Balance, float, int]): The amount of tokens to be transferred, specified as a Balance object,
                                                or in Tao (float) or Rao (int) units.

        Returns:
            Balance: The estimated transaction fee for the transfer, represented as a Balance object.

        Estimating the transfer fee is essential for planning and executing token transactions, ensuring that the
        wallet has sufficient funds to cover both the transfer amount and the associated costs. This function
        provides a crucial tool for managing financial operations within the Bittensor network.
        """
        if isinstance(value, float):
            transfer_balance = Balance.from_tao(value)
        elif isinstance(value, int):
            transfer_balance = Balance.from_rao(value)

        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module="Balances",
                call_function="transfer",
                call_params={"dest": dest, "value": transfer_balance.rao},
            )

            try:
                payment_info = substrate.get_payment_info(
                    call=call, keypair=wallet.coldkeypub
                )
            except Exception as e:
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed to get payment info[/red]:[bold white]\n  {}[/bold white]".format(
                        e
                    )
                )
                payment_info = {"partialFee": 2e7}  # assume  0.02 Tao

        fee = Balance.from_rao(payment_info["partialFee"])
        return fee

    def _do_transfer(
        self,
        wallet: "bittensor.wallet",
        dest: str,
        transfer_balance: Balance,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Sends a transfer extrinsic to the chain.
        Args:
            wallet (:obj:`bittensor.wallet`): Wallet object.
            dest (:obj:`str`): Destination public key address.
            transfer_balance (:obj:`Balance`): Amount to transfer.
            wait_for_inclusion (:obj:`bool`): If true, waits for inclusion.
            wait_for_finalization (:obj:`bool`): If true, waits for finalization.
        Returns:
            success (:obj:`bool`): True if transfer was successful.
            block_hash (:obj:`str`): Block hash of the transfer.
                (On success and if wait_for_ finalization/inclusion is True)
            error (:obj:`str`): Error message if transfer failed.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="Balances",
                    call_function="transfer",
                    call_params={"dest": dest, "value": transfer_balance.rao},
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True, None, None

                # Otherwise continue with finalization.
                response.process_events()
                if response.is_success:
                    block_hash = response.block_hash
                    return True, block_hash, None
                else:
                    return False, None, response.error_message

        return make_substrate_call_with_retry()

    def get_existential_deposit(self, block: Optional[int] = None) -> Optional[Balance]:
        """
        Retrieves the existential deposit amount for the Bittensor blockchain. The existential deposit
        is the minimum amount of TAO required for an account to exist on the blockchain. Accounts with
        balances below this threshold can be reaped to conserve network resources.

        Args:
            block (Optional[int], optional): Block number at which to query the deposit amount. If None,
                                            the current block is used.

        Returns:
            Optional[Balance]: The existential deposit amount, or None if the query fails.

        The existential deposit is a fundamental economic parameter in the Bittensor network, ensuring
        efficient use of storage and preventing the proliferation of dust accounts.
        """
        result = self.query_constant(
            module_name="Balances", constant_name="ExistentialDeposit", block=block
        )

        if result is None:
            return None

        return Balance.from_rao(result.value)

    #################
    #### Network ####
    #################
    def register_subnetwork(
        self,
        wallet: "bittensor.wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization=True,
        prompt: bool = False,
    ) -> bool:
        """
        Registers a new subnetwork on the Bittensor network using the provided wallet. This function
        is used for the creation and registration of subnetworks, which are specialized segments of the
        overall Bittensor network.

        Args:
            wallet (bittensor.wallet): The wallet to be used for registration.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the subnetwork registration is successful, False otherwise.

        This function allows for the expansion and diversification of the Bittensor network, supporting
        its decentralized and adaptable architecture.
        """
        return register_subnetwork_extrinsic(
            self,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    def set_hyperparameter(
        self,
        wallet: "bittensor.wallet",
        netuid: int,
        parameter: str,
        value,
        wait_for_inclusion: bool = False,
        wait_for_finalization=True,
        prompt: bool = False,
    ) -> bool:
        """
        Sets a specific hyperparameter for a given subnetwork on the Bittensor blockchain. This action
        involves adjusting network-level parameters, influencing the behavior and characteristics of the
        subnetwork.

        Args:
            wallet (bittensor.wallet): The wallet used for setting the hyperparameter.
            netuid (int): The unique identifier of the subnetwork.
            parameter (str): The name of the hyperparameter to be set.
            value: The new value for the hyperparameter.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the hyperparameter setting is successful, False otherwise.

        This function plays a critical role in the dynamic governance and adaptability of the Bittensor
        network, allowing for fine-tuning of network operations and characteristics.
        """
        return set_hyperparameter_extrinsic(
            self,
            wallet=wallet,
            netuid=netuid,
            parameter=parameter,
            value=value,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    #################
    #### Serving ####
    #################
    def serve(
        self,
        wallet: "bittensor.wallet",
        ip: str,
        port: int,
        protocol: int,
        netuid: int,
        placeholder1: int = 0,
        placeholder2: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization=True,
        prompt: bool = False,
    ) -> bool:
        """
        Registers a neuron's serving endpoint on the Bittensor network. This function announces the
        IP address and port where the neuron is available to serve requests, facilitating peer-to-peer
        communication within the network.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron being served.
            ip (str): The IP address of the serving neuron.
            port (int): The port number on which the neuron is serving.
            protocol (int): The protocol type used by the neuron (e.g., GRPC, HTTP).
            netuid (int): The unique identifier of the subnetwork.
            Other arguments: Placeholder parameters for future extensions.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the serve registration is successful, False otherwise.

        This function is essential for establishing the neuron's presence in the network, enabling
        it to participate in the decentralized machine learning processes of Bittensor.
        """
        return serve_extrinsic(
            self,
            wallet,
            ip,
            port,
            protocol,
            netuid,
            placeholder1,
            placeholder2,
            wait_for_inclusion,
            wait_for_finalization,
        )

    def serve_axon(
        self,
        netuid: int,
        axon: "bittensor.Axon",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
    ) -> bool:
        """
        Registers an Axon serving endpoint on the Bittensor network for a specific neuron. This function
        is used to set up the Axon, a key component of a neuron that handles incoming queries and data
        processing tasks.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            axon (bittensor.Axon): The Axon instance to be registered for serving.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the Axon serve registration is successful, False otherwise.

        By registering an Axon, the neuron becomes an active part of the network's distributed
        computing infrastructure, contributing to the collective intelligence of Bittensor.
        """
        return serve_axon_extrinsic(
            self, netuid, axon, wait_for_inclusion, wait_for_finalization
        )

    def _do_serve_axon(
        self,
        wallet: "bittensor.wallet",
        call_params: AxonServeCallParams,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Internal method to submit a serve axon transaction to the Bittensor blockchain. This method
        creates and submits a transaction, enabling a neuron's Axon to serve requests on the network.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron.
            call_params (AxonServeCallParams): Parameters required for the serve axon call.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

        This function is crucial for initializing and announcing a neuron's Axon service on the network,
        enhancing the decentralized computation capabilities of Bittensor.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="serve_axon",
                    call_params=call_params,
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.hotkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                if wait_for_inclusion or wait_for_finalization:
                    response.process_events()
                    if response.is_success:
                        return True, None
                    else:
                        return False, response.error_message
                else:
                    return True, None

        return make_substrate_call_with_retry()

    def serve_prometheus(
        self,
        wallet: "bittensor.wallet",
        port: int,
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        return prometheus_extrinsic(
            self,
            wallet=wallet,
            port=port,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def _do_serve_prometheus(
        self,
        wallet: "bittensor.wallet",
        call_params: PrometheusServeCallParams,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Sends a serve prometheus extrinsic to the chain.
        Args:
            wallet (:obj:`bittensor.wallet`): Wallet object.
            call_params (:obj:`PrometheusServeCallParams`): Prometheus serve call parameters.
            wait_for_inclusion (:obj:`bool`): If true, waits for inclusion.
            wait_for_finalization (:obj:`bool`): If true, waits for finalization.
        Returns:
            success (:obj:`bool`): True if serve prometheus was successful.
            error (:obj:`Optional[str]`): Error message if serve prometheus failed, None otherwise.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="serve_prometheus",
                    call_params=call_params,
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.hotkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                if wait_for_inclusion or wait_for_finalization:
                    response.process_events()
                    if response.is_success:
                        return True, None
                    else:
                        return False, response.error_message
                else:
                    return True, None

        return make_substrate_call_with_retry()

    def _do_associate_ips(
        self,
        wallet: "bittensor.wallet",
        ip_info_list: List[IPInfo],
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Sends an associate IPs extrinsic to the chain.

        Args:
            wallet (:obj:`bittensor.wallet`): Wallet object.
            ip_info_list (:obj:`List[IPInfo]`): List of IPInfo objects.
            netuid (:obj:`int`): Netuid to associate IPs to.
            wait_for_inclusion (:obj:`bool`): If true, waits for inclusion.
            wait_for_finalization (:obj:`bool`): If true, waits for finalization.

        Returns:
            success (:obj:`bool`): True if associate IPs was successful.
            error (:obj:`Optional[str]`): Error message if associate IPs failed, None otherwise.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="associate_ips",
                    call_params={
                        "ip_info_list": [ip_info.encode() for ip_info in ip_info_list],
                        "netuid": netuid,
                    },
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.hotkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                if wait_for_inclusion or wait_for_finalization:
                    response.process_events()
                    if response.is_success:
                        return True, None
                    else:
                        return False, response.error_message
                else:
                    return True, None

        return make_substrate_call_with_retry()

    #################
    #### Staking ####
    #################
    def add_stake(
        self,
        wallet: "bittensor.wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Adds the specified amount of stake to a neuron identified by the hotkey SS58 address. Staking
        is a fundamental process in the Bittensor network that enables neurons to participate actively
        and earn incentives.

        Args:
            wallet (bittensor.wallet): The wallet to be used for staking.
            hotkey_ss58 (Optional[str]): The SS58 address of the hotkey associated with the neuron.
            amount (Union[Balance, float]): The amount of TAO to stake.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the staking is successful, False otherwise.

        This function enables neurons to increase their stake in the network, enhancing their influence
        and potential rewards in line with Bittensor's consensus and reward mechanisms.
        """
        return add_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    def add_stake_multiple(
        self,
        wallet: "bittensor.wallet",
        hotkey_ss58s: List[str],
        amounts: List[Union[Balance, float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Adds stakes to multiple neurons identified by their hotkey SS58 addresses. This bulk operation
        allows for efficient staking across different neurons from a single wallet.

        Args:
            wallet (bittensor.wallet): The wallet used for staking.
            hotkey_ss58s (List[str]): List of SS58 addresses of hotkeys to stake to.
            amounts (List[Union[Balance, float]], optional): Corresponding amounts of TAO to stake for each hotkey.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the staking is successful for all specified neurons, False otherwise.

        This function is essential for managing stakes across multiple neurons, reflecting the dynamic
        and collaborative nature of the Bittensor network.
        """
        return add_stake_multiple_extrinsic(
            self,
            wallet,
            hotkey_ss58s,
            amounts,
            wait_for_inclusion,
            wait_for_finalization,
            prompt,
        )

    def _do_stake(
        self,
        wallet: "bittensor.wallet",
        hotkey_ss58: str,
        amount: Balance,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """Sends a stake extrinsic to the chain.
        Args:
            wallet (:obj:`bittensor.wallet`): Wallet object that can sign the extrinsic.
            hotkey_ss58 (:obj:`str`): Hotkey ss58 address to stake to.
            amount (:obj:`Balance`): Amount to stake.
            wait_for_inclusion (:obj:`bool`): If true, waits for inclusion before returning.
            wait_for_finalization (:obj:`bool`): If true, waits for finalization before returning.
        Returns:
            success (:obj:`bool`): True if the extrinsic was successful.
        Raises:
            StakeError: If the extrinsic failed.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="add_stake",
                    call_params={"hotkey": hotkey_ss58, "amount_staked": amount.rao},
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True

                response.process_events()
                if response.is_success:
                    return True
                else:
                    raise StakeError(response.error_message)

        return make_substrate_call_with_retry()

    ###################
    #### Unstaking ####
    ###################
    def unstake_multiple(
        self,
        wallet: "bittensor.wallet",
        hotkey_ss58s: List[str],
        amounts: List[Union[Balance, float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Performs batch unstaking from multiple hotkey accounts, allowing a neuron to reduce its staked amounts
        efficiently. This function is useful for managing the distribution of stakes across multiple neurons.

        Args:
            wallet (bittensor.wallet): The wallet linked to the coldkey from which the stakes are being withdrawn.
            hotkey_ss58s (List[str]): A list of hotkey SS58 addresses to unstake from.
            amounts (List[Union[Balance, float]], optional): The amounts of TAO to unstake from each hotkey.
                                                            If not provided, unstakes all available stakes.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the batch unstaking is successful, False otherwise.

        This function allows for strategic reallocation or withdrawal of stakes, aligning with the dynamic
        stake management aspect of the Bittensor network.
        """
        return unstake_multiple_extrinsic(
            self,
            wallet,
            hotkey_ss58s,
            amounts,
            wait_for_inclusion,
            wait_for_finalization,
            prompt,
        )

    def unstake(
        self,
        wallet: "bittensor.wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting
        individual neuron stakes within the Bittensor network.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron from which the stake is being removed.
            hotkey_ss58 (Optional[str]): The SS58 address of the hotkey account to unstake from.
            amount (Union[Balance, float], optional): The amount of TAO to unstake. If not specified, unstakes all.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the unstaking process is successful, False otherwise.

        This function supports flexible stake management, allowing neurons to adjust their network participation
        and potential reward accruals.
        """
        return unstake_extrinsic(
            self,
            wallet,
            hotkey_ss58,
            amount,
            wait_for_inclusion,
            wait_for_finalization,
            prompt,
        )

    def _do_unstake(
        self,
        wallet: "bittensor.wallet",
        hotkey_ss58: str,
        amount: Balance,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """Sends an unstake extrinsic to the chain.
        Args:
            wallet (:obj:`bittensor.wallet`): Wallet object that can sign the extrinsic.
            hotkey_ss58 (:obj:`str`): Hotkey ss58 address to unstake from.
            amount (:obj:`Balance`): Amount to unstake.
            wait_for_inclusion (:obj:`bool`): If true, waits for inclusion before returning.
            wait_for_finalization (:obj:`bool`): If true, waits for finalization before returning.
        Returns:
            success (:obj:`bool`): True if the extrinsic was successful.
        Raises:
            StakeError: If the extrinsic failed.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="remove_stake",
                    call_params={"hotkey": hotkey_ss58, "amount_unstaked": amount.rao},
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True

                response.process_events()
                if response.is_success:
                    return True
                else:
                    raise StakeError(response.error_message)

        return make_substrate_call_with_retry()

    ################
    #### Senate ####
    ################

    def register_senate(
        self,
        wallet: "bittensor.wallet",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting
        individual neuron stakes within the Bittensor network.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron from which the stake is being removed.
            hotkey_ss58 (Optional[str]): The SS58 address of the hotkey account to unstake from.
            amount (Union[Balance, float], optional): The amount of TAO to unstake. If not specified, unstakes all.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the unstaking process is successful, False otherwise.

        This function supports flexible stake management, allowing neurons to adjust their network participation
        and potential reward accruals.
        """
        return register_senate_extrinsic(
            self, wallet, wait_for_inclusion, wait_for_finalization, prompt
        )

    def leave_senate(
        self,
        wallet: "bittensor.wallet",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting
        individual neuron stakes within the Bittensor network.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron from which the stake is being removed.
            hotkey_ss58 (Optional[str]): The SS58 address of the hotkey account to unstake from.
            amount (Union[Balance, float], optional): The amount of TAO to unstake. If not specified, unstakes all.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the unstaking process is successful, False otherwise.

        This function supports flexible stake management, allowing neurons to adjust their network participation
        and potential reward accruals.
        """
        return leave_senate_extrinsic(
            self, wallet, wait_for_inclusion, wait_for_finalization, prompt
        )

    def vote_senate(
        self,
        wallet: "bittensor.wallet",
        proposal_hash: str,
        proposal_idx: int,
        vote: bool,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting
        individual neuron stakes within the Bittensor network.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron from which the stake is being removed.
            hotkey_ss58 (Optional[str]): The SS58 address of the hotkey account to unstake from.
            amount (Union[Balance, float], optional): The amount of TAO to unstake. If not specified, unstakes all.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the unstaking process is successful, False otherwise.

        This function supports flexible stake management, allowing neurons to adjust their network participation
        and potential reward accruals.
        """
        return vote_senate_extrinsic(
            self,
            wallet,
            proposal_hash,
            proposal_idx,
            vote,
            wait_for_inclusion,
            wait_for_finalization,
            prompt,
        )

    def is_senate_member(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        """
        Checks if a given neuron (identified by its hotkey SS58 address) is a member of the Bittensor senate.
        The senate is a key governance body within the Bittensor network, responsible for overseeing and
        approving various network operations and proposals.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block (Optional[int], optional): The blockchain block number at which to check senate membership.

        Returns:
            bool: True if the neuron is a senate member at the given block, False otherwise.

        This function is crucial for understanding the governance dynamics of the Bittensor network and for
        identifying the neurons that hold decision-making power within the network.
        """
        senate_members = self.query_module(
            module="SenateMembers", name="Members", block=block
        ).serialize()
        return senate_members.count(hotkey_ss58) > 0

    def get_vote_data(
        self, proposal_hash: str, block: Optional[int] = None
    ) -> Optional[ProposalVoteData]:
        """
        Retrieves the voting data for a specific proposal on the Bittensor blockchain. This data includes
        information about how senate members have voted on the proposal.

        Args:
            proposal_hash (str): The hash of the proposal for which voting data is requested.
            block (Optional[int], optional): The blockchain block number to query the voting data.

        Returns:
            Optional[ProposalVoteData]: An object containing the proposal's voting data, or None if not found.

        This function is important for tracking and understanding the decision-making processes within
        the Bittensor network, particularly how proposals are received and acted upon by the governing body.
        """
        vote_data = self.query_module(
            module="Triumvirate", name="Voting", block=block, params=[proposal_hash]
        )
        return vote_data.serialize() if vote_data != None else None

    get_proposal_vote_data = get_vote_data

    def get_senate_members(self, block: Optional[int] = None) -> Optional[List[str]]:
        """
        Retrieves the list of current senate members from the Bittensor blockchain. Senate members are
        responsible for governance and decision-making within the network.

        Args:
            block (Optional[int], optional): The blockchain block number at which to retrieve the senate members.

        Returns:
            Optional[List[str]]: A list of SS58 addresses of current senate members, or None if not available.

        Understanding the composition of the senate is key to grasping the governance structure and
        decision-making authority within the Bittensor network.
        """
        senate_members = self.query_module("SenateMembers", "Members", block=block)

        return senate_members.serialize() if senate_members != None else None

    def get_proposal_call_data(
        self, proposal_hash: str, block: Optional[int] = None
    ) -> Optional["bittensor.ProposalCallData"]:
        """
        Retrieves the call data of a specific proposal on the Bittensor blockchain. This data provides
        detailed information about the proposal, including its purpose and specifications.

        Args:
            proposal_hash (str): The hash of the proposal.
            block (Optional[int], optional): The blockchain block number at which to query the proposal call data.

        Returns:
            Optional[bittensor.ProposalCallData]: An object containing the proposal's call data, or None if not found.

        This function is crucial for analyzing the types of proposals made within the network and the
        specific changes or actions they intend to implement or address.
        """
        proposal_data = self.query_module(
            module="Triumvirate", name="ProposalOf", block=block, params=[proposal_hash]
        )

        return proposal_data.serialize() if proposal_data != None else None

    def get_proposal_hashes(self, block: Optional[int] = None) -> Optional[List[str]]:
        """
        Retrieves the list of proposal hashes currently present on the Bittensor blockchain. Each hash
        uniquely identifies a proposal made within the network.

        Args:
            block (Optional[int], optional): The blockchain block number to query the proposal hashes.

        Returns:
            Optional[List[str]]: A list of proposal hashes, or None if not available.

        This function enables tracking and reviewing the proposals made in the network, offering insights
        into the active governance and decision-making processes.
        """
        proposal_hashes = self.query_module(
            module="Triumvirate", name="Proposals", block=block
        )

        return proposal_hashes.serialize() if proposal_hashes != None else None

    def get_proposals(
        self, block: Optional[int] = None
    ) -> Optional[
        Dict[str, Tuple["bittensor.ProposalCallData", "bittensor.ProposalVoteData"]]
    ]:
        """
        Retrieves all active proposals on the Bittensor blockchain, along with their call and voting data.
        This comprehensive view allows for a thorough understanding of the proposals and their reception
        by the senate.

        Args:
            block (Optional[int], optional): The blockchain block number to query the proposals.

        Returns:
            Optional[Dict[str, Tuple[bittensor.ProposalCallData, bittensor.ProposalVoteData]]]:
                A dictionary mapping proposal hashes to their corresponding call and vote data, or None if not available.

        This function is integral for analyzing the governance activity on the Bittensor network,
        providing a holistic view of the proposals and their impact or potential changes within the network.
        """
        proposals = {}
        proposal_hashes: List = self.get_proposal_hashes(block=block)

        for proposal_hash in proposal_hashes:
            proposals[proposal_hash] = (
                self.get_proposal_call_data(proposal_hash, block=block),
                self.get_proposal_vote_data(proposal_hash, block=block),
            )

        return proposals

    ##############
    #### Root ####
    ##############

    def root_register(
        self,
        wallet: "bittensor.wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
    ) -> bool:
        """
        Registers the neuron associated with the wallet on the root network. This process is integral for
        participating in the highest layer of decision-making and governance within the Bittensor network.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron to be registered on the root network.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the registration on the root network is successful, False otherwise.

        This function enables neurons to engage in the most critical and influential aspects of the network's
        governance, signifying a high level of commitment and responsibility in the Bittensor ecosystem.
        """
        return root_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    def _do_root_register(
        self,
        wallet: "bittensor.wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                # create extrinsic call
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="root_register",
                    call_params={"hotkey": wallet.hotkey.ss58_address},
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )

                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True

                # process if registration successful, try again if pow is still valid
                response.process_events()
                if not response.is_success:
                    return False, response.error_message
                # Successful registration
                else:
                    return True, None

        return make_substrate_call_with_retry()

    def root_set_weights(
        self,
        wallet: "bittensor.wallet",
        netuids: Union[torch.LongTensor, list],
        weights: Union[torch.FloatTensor, list],
        version_key: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Sets the weights for neurons on the root network. This action is crucial for defining the influence
        and interactions of neurons at the root level of the Bittensor network.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron setting the weights.
            netuids (Union[torch.LongTensor, list]): The list of neuron UIDs for which weights are being set.
            weights (Union[torch.FloatTensor, list]): The corresponding weights to be set for each UID.
            version_key (int, optional): Version key for compatibility with the network.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If True, prompts for user confirmation before proceeding.

        Returns:
            bool: True if the setting of root-level weights is successful, False otherwise.

        This function plays a pivotal role in shaping the root network's collective intelligence and decision-making
        processes, reflecting the principles of decentralized governance and collaborative learning in Bittensor.
        """
        return set_root_weights_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )

    ########################
    #### Registry Calls ####
    ########################

    """ Queries subtensor registry named storage with params and block. """

    def query_identity(
        self,
        key: str,
        block: Optional[int] = None,
    ) -> Optional[object]:
        """
        Queries the identity of a neuron on the Bittensor blockchain using the given key. This function retrieves
        detailed identity information about a specific neuron, which is a crucial aspect of the network's decentralized
        identity and governance system.

        NOTE: See the bittensor cli documentation for supported identity parameters.

        Args:
            key (str): The key used to query the neuron's identity, typically the neuron's SS58 address.
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            Optional[object]: An object containing the identity information of the neuron if found, None otherwise.

        The identity information can include various attributes such as the neuron's stake, rank, and other
        network-specific details, providing insights into the neuron's role and status within the Bittensor network.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module="Registry",
                    storage_function="IdentityOf",
                    params=[key],
                    block_hash=None
                    if block == None
                    else substrate.get_block_hash(block),
                )

        identity_info = make_substrate_call_with_retry()
        return bittensor.utils.wallet_utils.decode_hex_identity_dict(
            identity_info.value["info"]
        )

    def update_identity(
        self,
        wallet: "bittensor.wallet",
        identified: str = None,
        params: dict = {},
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Updates the identity of a neuron on the Bittensor blockchain. This function allows neurons to modify their
        identity attributes, reflecting changes in their roles, stakes, or other network-specific parameters.

        NOTE: See the bittensor cli documentation for supported identity parameters.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron whose identity is being updated.
            identified (str, optional): The identified SS58 address of the neuron. Defaults to the wallet's coldkey address.
            params (dict, optional): A dictionary of parameters to update in the neuron's identity.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.

        Returns:
            bool: True if the identity update is successful, False otherwise.

        This function plays a vital role in maintaining the accuracy and currency of neuron identities in the
        Bittensor network, ensuring that the network's governance and consensus mechanisms operate effectively.
        """
        if identified == None:
            identified = wallet.coldkey.ss58_address

        call_params = bittensor.utils.wallet_utils.create_identity_dict(**params)
        call_params["identified"] = identified

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="Registry",
                    call_function="set_identity",
                    call_params=call_params,
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True
                response.process_events()
                if response.is_success:
                    return True
                else:
                    raise IdentityError(response.error_message)

        return make_substrate_call_with_retry()

    """ Make some commitment on-chain about arbitary data """

    def commit(self, wallet, netuid: int, data: str):
        publish_metadata(self, wallet, netuid, f"Raw{len(data)}", data.encode())

    def get_commitment(self, netuid: int, uid: int, block: Optional[int] = None) -> str:
        metagraph = self.metagraph(netuid)
        hotkey = metagraph.hotkeys[uid]

        metadata = get_metadata(self, netuid, hotkey, block)
        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]

        return bytes.fromhex(hex_data).decode()

    ########################
    #### Standard Calls ####
    ########################

    """ Queries subtensor named storage with params and block. """

    def query_subtensor(
        self,
        name: str,
        block: Optional[int] = None,
        params: Optional[List[object]] = [],
    ) -> Optional[object]:
        """
        Queries named storage from the Subtensor module on the Bittensor blockchain. This function is used to retrieve
        specific data or parameters from the blockchain, such as stake, rank, or other neuron-specific attributes.

        Args:
            name (str): The name of the storage function to query.
            block (Optional[int], optional): The blockchain block number at which to perform the query.
            params (Optional[List[object]], optional): A list of parameters to pass to the query function.

        Returns:
            Optional[object]: An object containing the requested data if found, None otherwise.

        This query function is essential for accessing detailed information about the network and its neurons,
        providing valuable insights into the state and dynamics of the Bittensor ecosystem.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module="SubtensorModule",
                    storage_function=name,
                    params=params,
                    block_hash=None
                    if block == None
                    else substrate.get_block_hash(block),
                )

        return make_substrate_call_with_retry()

    """ Queries subtensor map storage with params and block. """

    def query_map_subtensor(
        self,
        name: str,
        block: Optional[int] = None,
        params: Optional[List[object]] = [],
    ) -> QueryMapResult:
        """
        Queries map storage from the Subtensor module on the Bittensor blockchain. This function is designed to
        retrieve a map-like data structure, which can include various neuron-specific details or network-wide attributes.

        Args:
            name (str): The name of the map storage function to query.
            block (Optional[int], optional): The blockchain block number at which to perform the query.
            params (Optional[List[object]], optional): A list of parameters to pass to the query function.

        Returns:
            QueryMapResult: An object containing the map-like data structure, or None if not found.

        This function is particularly useful for analyzing and understanding complex network structures and
        relationships within the Bittensor ecosystem, such as inter-neuronal connections and stake distributions.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query_map(
                    module="SubtensorModule",
                    storage_function=name,
                    params=params,
                    block_hash=None
                    if block == None
                    else substrate.get_block_hash(block),
                )

        return make_substrate_call_with_retry()

    def query_constant(
        self, module_name: str, constant_name: str, block: Optional[int] = None
    ) -> Optional[object]:
        """
        Retrieves a constant from the specified module on the Bittensor blockchain. This function is used to
        access fixed parameters or values defined within the blockchain's modules, which are essential for
        understanding the network's configuration and rules.

        Args:
            module_name (str): The name of the module containing the constant.
            constant_name (str): The name of the constant to retrieve.
            block (Optional[int], optional): The blockchain block number at which to query the constant.

        Returns:
            Optional[object]: The value of the constant if found, None otherwise.

        Constants queried through this function can include critical network parameters such as inflation rates,
        consensus rules, or validation thresholds, providing a deeper understanding of the Bittensor network's
        operational parameters.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.get_constant(
                    module_name=module_name,
                    constant_name=constant_name,
                    block_hash=None
                    if block == None
                    else substrate.get_block_hash(block),
                )

        return make_substrate_call_with_retry()

    """ Queries any module storage with params and block. """

    def query_module(
        self,
        module: str,
        name: str,
        block: Optional[int] = None,
        params: Optional[List[object]] = [],
    ) -> Optional[object]:
        """
        Queries any module storage on the Bittensor blockchain with the specified parameters and block number.
        This function is a generic query interface that allows for flexible and diverse data retrieval from
        various blockchain modules.

        Args:
            module (str): The name of the module from which to query data.
            name (str): The name of the storage function within the module.
            block (Optional[int], optional): The blockchain block number at which to perform the query.
            params (Optional[List[object]], optional): A list of parameters to pass to the query function.

        Returns:
            Optional[object]: An object containing the requested data if found, None otherwise.

        This versatile query function is key to accessing a wide range of data and insights from different
        parts of the Bittensor blockchain, enhancing the understanding and analysis of the network's state and dynamics.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module=module,
                    storage_function=name,
                    params=params,
                    block_hash=None
                    if block == None
                    else substrate.get_block_hash(block),
                )

        return make_substrate_call_with_retry()

    """ Queries any module map storage with params and block. """

    def query_map(
        self,
        module: str,
        name: str,
        block: Optional[int] = None,
        params: Optional[List[object]] = [],
    ) -> Optional[object]:
        """
        Queries map storage from any module on the Bittensor blockchain. This function retrieves data structures
        that represent key-value mappings, essential for accessing complex and structured data within the blockchain modules.

        Args:
            module (str): The name of the module from which to query the map storage.
            name (str): The specific storage function within the module to query.
            block (Optional[int], optional): The blockchain block number at which to perform the query.
            params (Optional[List[object]], optional): Parameters to be passed to the query.

        Returns:
            Optional[object]: A data structure representing the map storage if found, None otherwise.

        This function is particularly useful for retrieving detailed and structured data from various blockchain
        modules, offering insights into the network's state and the relationships between its different components.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query_map(
                    module=module,
                    storage_function=name,
                    params=params,
                    block_hash=None
                    if block == None
                    else substrate.get_block_hash(block),
                )

        return make_substrate_call_with_retry()

    def state_call(
        self,
        method: str,
        data: str,
        block: Optional[int] = None,
    ) -> Optional[object]:
        """
        Makes a state call to the Bittensor blockchain, allowing for direct queries of the blockchain's state.
        This function is typically used for advanced queries that require specific method calls and data inputs.

        Args:
            method (str): The method name for the state call.
            data (str): The data to be passed to the method.
            block (Optional[int], optional): The blockchain block number at which to perform the state call.

        Returns:
            Optional[object]: The result of the state call if successful, None otherwise.

        The state call function provides a more direct and flexible way of querying blockchain data,
        useful for specific use cases where standard queries are insufficient.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash(block)
                params = [method, data]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(method="state_call", params=params)

        return make_substrate_call_with_retry()

    def query_runtime_api(
        self,
        runtime_api: str,
        method: str,
        params: Optional[List[ParamWithTypes]],
        block: Optional[int] = None,
    ) -> Optional[bytes]:
        """
        Queries the runtime API of the Bittensor blockchain, providing a way to interact with the underlying
        runtime and retrieve data encoded in Scale Bytes format. This function is essential for advanced users
        who need to interact with specific runtime methods and decode complex data types.

        Args:
            runtime_api (str): The name of the runtime API to query.
            method (str): The specific method within the runtime API to call.
            params (Optional[List[ParamWithTypes]], optional): The parameters to pass to the method call.
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            Optional[bytes]: The Scale Bytes encoded result from the runtime API call, or None if the call fails.

        This function enables access to the deeper layers of the Bittensor blockchain, allowing for detailed
        and specific interactions with the network's runtime environment.
        """
        call_definition = bittensor.__type_registry__["runtime_api"][runtime_api][
            "methods"
        ][method]

        json_result = self.state_call(
            method=f"{runtime_api}_{method}",
            data="0x"
            if params is None
            else self._encode_params(call_definition=call_definition, params=params),
            block=block,
        )

        if json_result is None:
            return None

        return_type = call_definition["type"]

        as_scale_bytes = scalecodec.ScaleBytes(json_result["result"])

        rpc_runtime_config = RuntimeConfiguration()
        rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
        rpc_runtime_config.update_type_registry(custom_rpc_type_registry)

        obj = rpc_runtime_config.create_scale_object(return_type, as_scale_bytes)
        if obj.data.to_hex() == "0x0400":  # RPC returned None result
            return None

        return obj.decode()

    def _encode_params(
        self,
        call_definition: List[ParamWithTypes],
        params: Union[List[Any], Dict[str, str]],
    ) -> str:
        """
        Returns a hex encoded string of the params using their types.
        """
        param_data = scalecodec.ScaleBytes(b"")

        for i, param in enumerate(call_definition["params"]):
            scale_obj = self.substrate.create_scale_object(param["type"])
            if type(params) is list:
                param_data += scale_obj.encode(params[i])
            else:
                if param["name"] not in params:
                    raise ValueError(f"Missing param {param['name']} in params dict.")

                param_data += scale_obj.encode(params[param["name"]])

        return param_data.to_hex()

    #####################################
    #### Hyper parameter calls. ####
    #####################################

    def rho(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the 'Rho' hyperparameter for a specified subnet within the Bittensor network.
        'Rho' represents the global inflation rate, which directly influences the network's
        token emission rate and economic model.

        Note: This is currently fixed such that the Bittensor blockchain emmits 7200 Tao per day.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number at which to query the parameter.

        Returns:
            Optional[int]: The value of the 'Rho' hyperparameter if the subnet exists, None otherwise.

        Mathematical Context:
            Rho (p) is calculated based on the network's target inflation and actual neuron staking.
            It adjusts the emission rate of the TAO token to balance the network's economy and dynamics.
            The formula for Rho is defined as: p = (Staking_Target / Staking_Actual) * Inflation_Target.
            Here, Staking_Target and Staking_Actual represent the desired and actual total stakes in the network,
            while Inflation_Target is the predefined inflation rate goal&#8203;``【oaicite:0】``&#8203;.

        'Rho' is essential for understanding the network's economic dynamics, affecting the reward distribution
        and incentive structures across the network's neurons.
        """
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("Rho", block, [netuid]).value

    def kappa(self, netuid: int, block: Optional[int] = None) -> Optional[float]:
        """
        Retrieves the 'Kappa' hyperparameter for a specified subnet. 'Kappa' is a critical parameter in
        the Bittensor network that controls the distribution of stake weights among neurons, impacting their
        rankings and incentive allocations.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[float]: The value of the 'Kappa' hyperparameter if the subnet exists, None otherwise.

        Mathematical Context:
            Kappa (κ) is used in the calculation of neuron ranks, which determine their share of network incentives.
            It is derived from the softmax function applied to the inter-neuronal weights set by each neuron.
            The formula for Kappa is: κ_i = exp(w_i) / Σ(exp(w_j)), where w_i represents the weight set by neuron i,
            and the denominator is the sum of exponential weights set by all neurons.
            This mechanism ensures a normalized and probabilistic distribution of ranks based on relative weights&#8203;``【oaicite:0】``&#8203;.

        Understanding 'Kappa' is crucial for analyzing stake dynamics and the consensus mechanism within the network,
        as it plays a significant role in neuron ranking and incentive allocation processes.
        """
        if not self.subnet_exists(netuid, block):
            return None
        return U16_NORMALIZED_FLOAT(
            self.query_subtensor("Kappa", block, [netuid]).value
        )

    def difficulty(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the 'Difficulty' hyperparameter for a specified subnet in the Bittensor network.
        This parameter is instrumental in determining the computational challenge required for neurons
        to participate in consensus and validation processes.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[int]: The value of the 'Difficulty' hyperparameter if the subnet exists, None otherwise.

        The 'Difficulty' parameter directly impacts the network's security and integrity by setting the
        computational effort required for validating transactions and participating in the network's consensus mechanism.
        """
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("Difficulty", block, [netuid]).value

    def burn(self, netuid: int, block: Optional[int] = None) -> Optional[Balance]:
        """
        Retrieves the 'Burn' hyperparameter for a specified subnet. The 'Burn' parameter represents the
        amount of Tao that is effectively removed from circulation within the Bittensor network.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[Balance]: The value of the 'Burn' hyperparameter if the subnet exists, None otherwise.

        Understanding the 'Burn' rate is essential for analyzing the network's economic model, particularly
        how it manages inflation and the overall supply of its native token Tao.
        """
        if not self.subnet_exists(netuid, block):
            return None
        return Balance.from_rao(self.query_subtensor("Burn", block, [netuid]).value)

    """ Returns network ImmunityPeriod hyper parameter """

    def immunity_period(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Retrieves the 'ImmunityPeriod' hyperparameter for a specific subnet. This parameter defines the
        duration during which new neurons are protected from certain network penalties or restrictions.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[int]: The value of the 'ImmunityPeriod' hyperparameter if the subnet exists, None otherwise.

        The 'ImmunityPeriod' is a critical aspect of the network's governance system, ensuring that new
        participants have a grace period to establish themselves and contribute to the network without facing
        immediate punitive actions.
        """
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("ImmunityPeriod", block, [netuid]).value

    def validator_batch_size(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """Returns network ValidatorBatchSize hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("ValidatorBatchSize", block, [netuid]).value

    def validator_prune_len(self, netuid: int, block: Optional[int] = None) -> int:
        """Returns network ValidatorPruneLen hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("ValidatorPruneLen", block, [netuid]).value

    def validator_logits_divergence(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        """Returns network ValidatorLogitsDivergence hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return U16_NORMALIZED_FLOAT(
            self.query_subtensor("ValidatorLogitsDivergence", block, [netuid]).value
        )

    def validator_sequence_length(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """Returns network ValidatorSequenceLength hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("ValidatorSequenceLength", block, [netuid]).value

    def validator_epochs_per_reset(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """Returns network ValidatorEpochsPerReset hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("ValidatorEpochsPerReset", block, [netuid]).value

    def validator_epoch_length(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """Returns network ValidatorEpochLen hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("ValidatorEpochLen", block, [netuid]).value

    def validator_exclude_quantile(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        """Returns network ValidatorEpochLen hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return U16_NORMALIZED_FLOAT(
            self.query_subtensor("ValidatorExcludeQuantile", block, [netuid]).value
        )

    def max_allowed_validators(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """Returns network MaxAllowedValidators hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("MaxAllowedValidators", block, [netuid]).value

    def min_allowed_weights(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """Returns network MinAllowedWeights hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("MinAllowedWeights", block, [netuid]).value

    def max_weight_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        """Returns network MaxWeightsLimit hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return U16_NORMALIZED_FLOAT(
            self.query_subtensor("MaxWeightsLimit", block, [netuid]).value
        )

    def scaling_law_power(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        """Returns network ScalingLawPower hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("ScalingLawPower", block, [netuid]).value / 100.0

    def synergy_scaling_law_power(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        """Returns network SynergyScalingLawPower hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return (
            self.query_subtensor("SynergyScalingLawPower", block, [netuid]).value
            / 100.0
        )

    def subnetwork_n(self, netuid: int, block: Optional[int] = None) -> int:
        """Returns network SubnetworkN hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("SubnetworkN", block, [netuid]).value

    def max_n(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """Returns network MaxAllowedUids hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("MaxAllowedUids", block, [netuid]).value

    def blocks_since_epoch(self, netuid: int, block: Optional[int] = None) -> int:
        """Returns network BlocksSinceLastStep hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("BlocksSinceLastStep", block, [netuid]).value

    def tempo(self, netuid: int, block: Optional[int] = None) -> int:
        """Returns network Tempo hyper parameter"""
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor("Tempo", block, [netuid]).value

    ##########################
    #### Account functions ###
    ##########################

    def get_total_stake_for_hotkey(
        self, ss58_address: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        """Returns the total stake held on a hotkey including delegative"""
        return Balance.from_rao(
            self.query_subtensor("TotalHotkeyStake", block, [ss58_address]).value
        )

    def get_total_stake_for_coldkey(
        self, ss58_address: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        """Returns the total stake held on a coldkey across all hotkeys including delegates"""
        return Balance.from_rao(
            self.query_subtensor("TotalColdkeyStake", block, [ss58_address]).value
        )

    def get_stake_for_coldkey_and_hotkey(
        self, hotkey_ss58: str, coldkey_ss58: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        """Returns the stake under a coldkey - hotkey pairing"""
        return Balance.from_rao(
            self.query_subtensor("Stake", block, [hotkey_ss58, coldkey_ss58]).value
        )

    def get_stake(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> List[Tuple[str, "Balance"]]:
        """Returns a list of stake tuples (coldkey, balance) for each delegating coldkey including the owner"""
        return [
            (r[0].value, Balance.from_rao(r[1].value))
            for r in self.query_map_subtensor("Stake", block, [hotkey_ss58])
        ]

    def does_hotkey_exist(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        """Returns true if the hotkey is known by the chain and there are accounts."""
        return (
            self.query_subtensor("Owner", block, [hotkey_ss58]).value
            != "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        )

    def get_hotkey_owner(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[str]:
        """Returns the coldkey owner of the passed hotkey"""
        if self.does_hotkey_exist(hotkey_ss58, block):
            return self.query_subtensor("Owner", block, [hotkey_ss58]).value
        else:
            return None

    def get_axon_info(
        self, netuid: int, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[AxonInfo]:
        """Returns the axon information for this hotkey account"""
        result = self.query_subtensor("Axons", block, [netuid, hotkey_ss58])
        if result != None:
            return AxonInfo(
                ip=bittensor.utils.networking.int_to_ip(result.value["ip"]),
                ip_type=result.value["ip_type"],
                port=result.value["port"],
                protocol=result.value["protocol"],
                version=result.value["version"],
                placeholder1=result.value["placeholder1"],
                placeholder2=result.value["placeholder2"],
            )
        else:
            return None

    def get_prometheus_info(
        self, netuid: int, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[AxonInfo]:
        """Returns the prometheus information for this hotkey account"""
        result = self.query_subtensor("Prometheus", block, [netuid, hotkey_ss58])
        if result != None:
            return PrometheusInfo(
                ip=bittensor.utils.networking.int_to_ip(result.value["ip"]),
                ip_type=result.value["ip_type"],
                port=result.value["port"],
                version=result.value["version"],
                block=result.value["block"],
            )
        else:
            return None

    ###########################
    #### Global Parameters ####
    ###########################

    @property
    def block(self) -> int:
        r"""Returns current chain block.
        Returns:
            block (int):
                Current chain block.
        """
        return self.get_current_block()

    def total_issuance(self, block: Optional[int] = None) -> "Balance":
        """
        Retrieves the total issuance of the Bittensor network's native token (Tao) as of a specific
        blockchain block. This represents the total amount of currency that has been issued or mined on the network.

        Args:
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            Balance: The total issuance of TAO, represented as a Balance object.

        The total issuance is a key economic indicator in the Bittensor network, reflecting the overall supply
        of the currency and providing insights into the network's economic health and inflationary trends.
        """
        return Balance.from_rao(self.query_subtensor("TotalIssuance", block).value)

    def total_stake(self, block: Optional[int] = None) -> "Balance":
        """
        Retrieves the total amount of TAO staked on the Bittensor network as of a specific blockchain block.
        This represents the cumulative stake across all neurons in the network, indicating the overall level
        of participation and investment by the network's participants.

        Args:
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            Balance: The total amount of TAO staked on the network, represented as a Balance object.

        The total stake is an important metric for understanding the network's security, governance dynamics,
        and the level of commitment by its participants. It is also a critical factor in the network's
        consensus and incentive mechanisms.
        """
        return Balance.from_rao(self.query_subtensor("TotalStake", block).value)

    def serving_rate_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Retrieves the serving rate limit for a specific subnet within the Bittensor network.
        This rate limit determines the maximum number of requests a neuron can serve within a given time frame.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            Optional[int]: The serving rate limit of the subnet if it exists, None otherwise.

        The serving rate limit is a crucial parameter for maintaining network efficiency and preventing
        overuse of resources by individual neurons. It helps ensure a balanced distribution of service
        requests across the network.
        """
        if not self.subnet_exists(netuid, block):
            return None
        return self.query_subtensor(
            "ServingRateLimit", block=block, params=[netuid]
        ).value

    def tx_rate_limit(self, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the transaction rate limit for the Bittensor network as of a specific blockchain block.
        This rate limit sets the maximum number of transactions that can be processed within a given time frame.

        Args:
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            Optional[int]: The transaction rate limit of the network, None if not available.

        The transaction rate limit is an essential parameter for ensuring the stability and scalability
        of the Bittensor network. It helps in managing network load and preventing congestion, thereby
        maintaining efficient and timely transaction processing.
        """
        return self.query_subtensor("TxRateLimit", block).value

    #####################################
    #### Network Parameters ####
    #####################################

    def subnet_exists(self, netuid: int, block: Optional[int] = None) -> bool:
        """
        Checks if a subnet with the specified unique identifier (netuid) exists within the Bittensor network.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number at which to check the subnet's existence.

        Returns:
            bool: True if the subnet exists, False otherwise.

        This function is critical for verifying the presence of specific subnets in the network,
        enabling a deeper understanding of the network's structure and composition.
        """
        return self.query_subtensor("NetworksAdded", block, [netuid]).value

    def get_all_subnet_netuids(self, block: Optional[int] = None) -> List[int]:
        """
        Retrieves the list of all subnet unique identifiers (netuids) currently present in the Bittensor network.

        Args:
            block (Optional[int], optional): The blockchain block number at which to retrieve the subnet netuids.

        Returns:
            List[int]: A list of subnet netuids.

        This function provides a comprehensive view of the subnets within the Bittensor network,
        offering insights into its diversity and scale.
        """
        subnet_netuids = []
        result = self.query_map_subtensor("NetworksAdded", block)
        if result.records:
            for netuid, exists in result:
                if exists:
                    subnet_netuids.append(netuid.value)

        return subnet_netuids

    def get_total_subnets(self, block: Optional[int] = None) -> int:
        """
        Retrieves the total number of subnets within the Bittensor network as of a specific blockchain block.

        Args:
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            int: The total number of subnets in the network.

        Understanding the total number of subnets is essential for assessing the network's growth and
        the extent of its decentralized infrastructure.
        """
        return self.query_subtensor("TotalNetworks", block).value

    def get_subnet_modality(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.query_subtensor("NetworkModality", block, [netuid]).value

    def get_subnet_connection_requirement(
        self, netuid_0: int, netuid_1: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.query_subtensor("NetworkConnect", block, [netuid_0, netuid_1]).value

    def get_emission_value_by_subnet(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        """
        Retrieves the emission value of a specific subnet within the Bittensor network. The emission value
        represents the rate at which the subnet emits or distributes the network's native token (Tao).

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[float]: The emission value of the subnet, None if not available.

        The emission value is a critical economic parameter, influencing the incentive distribution and
        reward mechanisms within the subnet.
        """
        return Balance.from_rao(
            self.query_subtensor("EmissionValues", block, [netuid]).value
        )

    def get_subnet_connection_requirements(
        self, netuid: int, block: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Retrieves the connection requirements for a specific subnet within the Bittensor network. This
        function provides details on the criteria that must be met for neurons to connect to the subnet.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Dict[str, int]: A dictionary detailing the connection requirements for the subnet.

        Understanding these requirements is crucial for neurons looking to participate in or interact
        with specific subnets, ensuring compliance with their connection standards.
        """
        result = self.query_map_subtensor("NetworkConnect", block, [netuid])
        if result.records:
            requirements = {}
            for tuple in result.records:
                requirements[str(tuple[0].value)] = tuple[1].value
        else:
            return {}

    def get_subnets(self, block: Optional[int] = None) -> List[int]:
        """
        Retrieves a list of all subnets currently active within the Bittensor network. This function
        provides an overview of the various subnets and their identifiers.

        Args:
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[int]: A list of network UIDs representing each active subnet.

        This function is valuable for understanding the network's structure and the diversity of subnets
        available for neuron participation and collaboration.
        """
        subnets = []
        result = self.query_map_subtensor("NetworksAdded", block)
        if result.records:
            for network in result.records:
                subnets.append(network[0].value)
            return subnets
        else:
            return []

    def get_all_subnets_info(self, block: Optional[int] = None) -> List[SubnetInfo]:
        """
        Retrieves detailed information about all subnets within the Bittensor network. This function
        provides comprehensive data on each subnet, including its characteristics and operational parameters.

        Args:
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[SubnetInfo]: A list of SubnetInfo objects, each containing detailed information about a subnet.

        Gaining insights into the subnets' details assists in understanding the network's composition,
        the roles of different subnets, and their unique features.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash(block)
                params = []
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="subnetInfo_getSubnetsInfo",  # custom rpc method
                    params=params,
                )

        json_body = make_substrate_call_with_retry()
        result = json_body["result"]

        if result in (None, []):
            return []

        return SubnetInfo.list_from_vec_u8(result)

    def get_subnet_info(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[SubnetInfo]:
        """
        Retrieves detailed information about a specific subnet within the Bittensor network. This function
        provides key data on the subnet, including its operational parameters and network status.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[SubnetInfo]: Detailed information about the subnet, or None if not found.

        This function is essential for neurons and stakeholders interested in the specifics of a particular
        subnet, including its governance, performance, and role within the broader network.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash(block)
                params = [netuid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="subnetInfo_getSubnetInfo",  # custom rpc method
                    params=params,
                )

        json_body = make_substrate_call_with_retry()
        result = json_body["result"]

        if result in (None, []):
            return None

        return SubnetInfo.from_vec_u8(result)

    def get_subnet_hyperparameters(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[SubnetHyperparameters]:
        """
        Retrieves the hyperparameters for a specific subnet within the Bittensor network. These hyperparameters
        define the operational settings and rules governing the subnet's behavior.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[SubnetHyperparameters]: The subnet's hyperparameters, or None if not available.

        Understanding the hyperparameters is crucial for comprehending how subnets are configured and
        managed, and how they interact with the network's consensus and incentive mechanisms.
        """
        hex_bytes_result = self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_hyperparams",
            params=[netuid],
            block=block,
        )

        if hex_bytes_result == None:
            return []

        if hex_bytes_result.startswith("0x"):
            bytes_result = bytes.fromhex(hex_bytes_result[2:])
        else:
            bytes_result = bytes.fromhex(hex_bytes_result)

        return SubnetHyperparameters.from_vec_u8(bytes_result)

    def get_subnet_owner(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[str]:
        """
        Retrieves the owner's address of a specific subnet within the Bittensor network. The owner is
        typically the entity responsible for the creation and maintenance of the subnet.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[str]: The SS58 address of the subnet's owner, or None if not available.

        Knowing the subnet owner provides insights into the governance and operational control of the subnet,
        which can be important for decision-making and collaboration within the network.
        """
        return self.query_subtensor("SubnetOwner", block, [netuid]).value

    ####################
    #### Nomination ####
    ####################
    def is_hotkey_delegate(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        """
        Determines whether a given hotkey (public key) is a delegate on the Bittensor network. This function
        checks if the neuron associated with the hotkey is part of the network's delegation system.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            bool: True if the hotkey is a delegate, False otherwise.

        Being a delegate is a significant status within the Bittensor network, indicating a neuron's
        involvement in consensus and governance processes.
        """
        return hotkey_ss58 in [
            info.hotkey_ss58 for info in self.get_delegates(block=block)
        ]

    def get_delegate_take(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[float]:
        """
        Retrieves the delegate 'take' percentage for a neuron identified by its hotkey. The 'take'
        represents the percentage of rewards that the delegate claims from its nominators' stakes.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[float]: The delegate take percentage, None if not available.

        The delegate take is a critical parameter in the network's incentive structure, influencing
        the distribution of rewards among neurons and their nominators.
        """
        return U16_NORMALIZED_FLOAT(
            self.query_subtensor("Delegates", block, [hotkey_ss58]).value
        )

    def get_nominators_for_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> List[Tuple[str, Balance]]:
        """
        Retrieves a list of nominators and their stakes for a neuron identified by its hotkey.
        Nominators are neurons that stake their tokens on a delegate to support its operations.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[Tuple[str, Balance]]: A list of tuples containing each nominator's address and staked amount.

        This function provides insights into the neuron's support network within the Bittensor ecosystem,
        indicating its trust and collaboration relationships.
        """
        result = self.query_map_subtensor("Stake", block, [hotkey_ss58])
        if result.records:
            return [(record[0].value, record[1].value) for record in result.records]
        else:
            return 0

    def get_delegate_by_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[DelegateInfo]:
        """
        Retrieves detailed information about a delegate neuron based on its hotkey. This function provides
        a comprehensive view of the delegate's status, including its stakes, nominators, and reward distribution.

        Args:
            hotkey_ss58 (str): The SS58 address of the delegate's hotkey.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[DelegateInfo]: Detailed information about the delegate neuron, None if not found.

        This function is essential for understanding the roles and influence of delegate neurons within
        the Bittensor network's consensus and governance structures.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry(encoded_hotkey: List[int]):
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash(block)
                params = [encoded_hotkey]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="delegateInfo_getDelegate",  # custom rpc method
                    params=params,
                )

        encoded_hotkey = ss58_to_vec_u8(hotkey_ss58)
        json_body = make_substrate_call_with_retry(encoded_hotkey)
        result = json_body["result"]

        if result in (None, []):
            return None

        return DelegateInfo.from_vec_u8(result)

    def get_delegates(self, block: Optional[int] = None) -> List[DelegateInfo]:
        """
        Retrieves a list of all delegate neurons within the Bittensor network. This function provides an
        overview of the neurons that are actively involved in the network's delegation system.

        Args:
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[DelegateInfo]: A list of DelegateInfo objects detailing each delegate's characteristics.

        Analyzing the delegate population offers insights into the network's governance dynamics and the
        distribution of trust and responsibility among participating neurons.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash(block)
                params = []
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="delegateInfo_getDelegates",  # custom rpc method
                    params=params,
                )

        json_body = make_substrate_call_with_retry()
        result = json_body["result"]

        if result in (None, []):
            return []

        return DelegateInfo.list_from_vec_u8(result)

    def get_delegated(
        self, coldkey_ss58: str, block: Optional[int] = None
    ) -> List[Tuple[DelegateInfo, Balance]]:
        """
        Retrieves a list of delegates and their associated stakes for a given coldkey. This function
        identifies the delegates that a specific account has staked tokens on.

        Args:
            coldkey_ss58 (str): The SS58 address of the account's coldkey.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[Tuple[DelegateInfo, Balance]]: A list of tuples, each containing a delegate's information and staked amount.

        This function is important for account holders to understand their stake allocations and their
        involvement in the network's delegation and consensus mechanisms.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry(encoded_coldkey: List[int]):
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash(block)
                params = [encoded_coldkey]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="delegateInfo_getDelegated",  # custom rpc method
                    params=params,
                )

        encoded_coldkey = ss58_to_vec_u8(coldkey_ss58)
        json_body = make_substrate_call_with_retry(encoded_coldkey)
        result = json_body["result"]

        if result in (None, []):
            return []

        return DelegateInfo.delegated_list_from_vec_u8(result)

    ###########################
    #### Stake Information ####
    ###########################

    def get_stake_info_for_coldkey(
        self, coldkey_ss58: str, block: Optional[int] = None
    ) -> List[StakeInfo]:
        """
        Retrieves stake information associated with a specific coldkey. This function provides details
        about the stakes held by an account, including the staked amounts and associated delegates.

        Args:
            coldkey_ss58 (str): The SS58 address of the account's coldkey.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[StakeInfo]: A list of StakeInfo objects detailing the stake allocations for the account.

        Stake information is vital for account holders to assess their investment and participation
        in the network's delegation and consensus processes.
        """
        encoded_coldkey = ss58_to_vec_u8(coldkey_ss58)

        hex_bytes_result = self.query_runtime_api(
            runtime_api="StakeInfoRuntimeApi",
            method="get_stake_info_for_coldkey",
            params=[encoded_coldkey],
            block=block,
        )

        if hex_bytes_result == None:
            return None

        if hex_bytes_result.startswith("0x"):
            bytes_result = bytes.fromhex(hex_bytes_result[2:])
        else:
            bytes_result = bytes.fromhex(hex_bytes_result)

        return StakeInfo.list_from_vec_u8(bytes_result)

    def get_stake_info_for_coldkeys(
        self, coldkey_ss58_list: List[str], block: Optional[int] = None
    ) -> Dict[str, List[StakeInfo]]:
        """
        Retrieves stake information for a list of coldkeys. This function aggregates stake data for multiple
        accounts, providing a collective view of their stakes and delegations.

        Args:
            coldkey_ss58_list (List[str]): A list of SS58 addresses of the accounts' coldkeys.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Dict[str, List[StakeInfo]]: A dictionary mapping each coldkey to a list of its StakeInfo objects.

        This function is useful for analyzing the stake distribution and delegation patterns of multiple
        accounts simultaneously, offering a broader perspective on network participation and investment strategies.
        """
        encoded_coldkeys = [
            ss58_to_vec_u8(coldkey_ss58) for coldkey_ss58 in coldkey_ss58_list
        ]

        hex_bytes_result = self.query_runtime_api(
            runtime_api="StakeInfoRuntimeApi",
            method="get_stake_info_for_coldkeys",
            params=[encoded_coldkeys],
            block=block,
        )

        if hex_bytes_result == None:
            return None

        if hex_bytes_result.startswith("0x"):
            bytes_result = bytes.fromhex(hex_bytes_result[2:])
        else:
            bytes_result = bytes.fromhex(hex_bytes_result)

        return StakeInfo.list_of_tuple_from_vec_u8(bytes_result)

    ########################################
    #### Neuron information per subnet ####
    ########################################

    def is_hotkey_registered_any(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> bool:
        """
        Checks if a neuron's hotkey is registered on any subnet within the Bittensor network.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block (Optional[int], optional): The blockchain block number at which to perform the check.

        Returns:
            bool: True if the hotkey is registered on any subnet, False otherwise.

        This function is essential for determining the network-wide presence and participation of a neuron.
        """
        return len(self.get_netuids_for_hotkey(hotkey_ss58, block)) > 0

    def is_hotkey_registered_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> bool:
        """
        Checks if a neuron's hotkey is registered on a specific subnet within the Bittensor network.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number at which to perform the check.

        Returns:
            bool: True if the hotkey is registered on the specified subnet, False otherwise.

        This function helps in assessing the participation of a neuron in a particular subnet,
        indicating its specific area of operation or influence within the network.
        """
        return self.get_uid_for_hotkey_on_subnet(hotkey_ss58, netuid, block) != None

    def is_hotkey_registered(
        self,
        hotkey_ss58: str,
        netuid: Optional[int] = None,
        block: Optional[int] = None,
    ) -> bool:
        """
        Determines whether a given hotkey (public key) is registered in the Bittensor network, either
        globally across any subnet or specifically on a specified subnet. This function checks the registration
        status of a neuron identified by its hotkey, which is crucial for validating its participation and
        activities within the network.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            netuid (Optional[int], optional): The unique identifier of the subnet to check the registration.
                                            If None, the registration is checked across all subnets.
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            bool: True if the hotkey is registered in the specified context (either any subnet or a specific subnet),
                False otherwise.

        This function is important for verifying the active status of neurons in the Bittensor network. It aids
        in understanding whether a neuron is eligible to participate in network processes such as consensus,
        validation, and incentive distribution based on its registration status.
        """
        if netuid == None:
            return self.is_hotkey_registered_any(hotkey_ss58, block)
        else:
            return self.is_hotkey_registered_on_subnet(hotkey_ss58, netuid, block)

    def get_uid_for_hotkey_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Retrieves the unique identifier (UID) for a neuron's hotkey on a specific subnet.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[int]: The UID of the neuron if it is registered on the subnet, None otherwise.

        The UID is a critical identifier within the network, linking the neuron's hotkey to its
        operational and governance activities on a particular subnet.
        """
        return self.query_subtensor("Uids", block, [netuid, hotkey_ss58]).value

    def get_all_uids_for_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> List[int]:
        """
        Retrieves all unique identifiers (UIDs) associated with a given hotkey across different subnets
        within the Bittensor network. This function helps in identifying all the neuron instances that are
        linked to a specific hotkey.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            List[int]: A list of UIDs associated with the given hotkey across various subnets.

        This function is important for tracking a neuron's presence and activities across different
        subnets within the Bittensor ecosystem.
        """
        return [
            self.get_uid_for_hotkey_on_subnet(hotkey_ss58, netuid, block)
            for netuid in self.get_netuids_for_hotkey(hotkey_ss58, block)
        ]

    def get_netuids_for_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> List[int]:
        """
        Retrieves a list of subnet UIDs (netuids) for which a given hotkey is a member. This function
        identifies the specific subnets within the Bittensor network where the neuron associated with
        the hotkey is active.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            List[int]: A list of netuids where the neuron is a member.

        This function provides insights into a neuron's involvement and distribution across different
        subnets, illustrating its role and contributions within the network.
        """
        result = self.query_map_subtensor("IsNetworkMember", block, [hotkey_ss58])
        netuids = []
        for netuid, is_member in result.records:
            if is_member:
                netuids.append(netuid.value)
        return netuids

    def get_neuron_for_pubkey_and_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional[NeuronInfo]:
        """
        Retrieves information about a neuron based on its public key (hotkey SS58 address) and the specific
        subnet UID (netuid). This function provides detailed neuron information for a particular subnet within
        the Bittensor network.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            Optional[NeuronInfo]: Detailed information about the neuron if found, None otherwise.

        This function is crucial for accessing specific neuron data and understanding its status, stake,
        and other attributes within a particular subnet of the Bittensor ecosystem.
        """
        return self.neuron_for_uid(
            self.get_uid_for_hotkey_on_subnet(hotkey_ss58, netuid, block=block),
            netuid,
            block=block,
        )

    def get_all_neurons_for_pubkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> List[NeuronInfo]:
        """
        Retrieves information about all neuron instances associated with a given public key (hotkey SS58
        address) across different subnets of the Bittensor network. This function aggregates neuron data
        from various subnets to provide a comprehensive view of a neuron's presence and status within the network.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[NeuronInfo]: A list of NeuronInfo objects detailing the neuron's presence across various subnets.

        This function is valuable for analyzing a neuron's overall participation, influence, and
        contributions across the Bittensor network.
        """
        netuids = self.get_netuids_for_hotkey(hotkey_ss58, block)
        uids = [self.get_uid_for_hotkey_on_subnet(hotkey_ss58, net) for net in netuids]
        return [self.neuron_for_uid(uid, net) for uid, net in list(zip(uids, netuids))]

    def neuron_has_validator_permit(
        self, uid: int, netuid: int, block: Optional[int] = None
    ) -> Optional[bool]:
        """
        Checks if a neuron, identified by its unique identifier (UID), has a validator permit on a specific
        subnet within the Bittensor network. This function determines whether the neuron is authorized to
        participate in validation processes on the subnet.

        Args:
            uid (int): The unique identifier of the neuron.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[bool]: True if the neuron has a validator permit, False otherwise.

        This function is essential for understanding a neuron's role and capabilities within a specific
        subnet, particularly regarding its involvement in network validation and governance.
        """
        return self.query_subtensor("ValidatorPermit", block, [netuid, uid]).value

    def neuron_for_wallet(
        self, wallet: "bittensor.wallet", netuid: int, block: Optional[int] = None
    ) -> Optional[NeuronInfo]:
        """
        Retrieves information about a neuron associated with a given wallet on a specific subnet.
        This function provides detailed data about the neuron's status, stake, and activities based on
        the wallet's hotkey address.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number at which to perform the query.

        Returns:
            Optional[NeuronInfo]: Detailed information about the neuron if found, None otherwise.

        This function is important for wallet owners to understand and manage their neuron's presence
        and activities within a particular subnet of the Bittensor network.
        """
        return self.get_neuron_for_pubkey_and_subnet(
            wallet.hotkey.ss58_address, netuid=netuid, block=block
        )

    def neuron_for_uid(
        self, uid: int, netuid: int, block: Optional[int] = None
    ) -> Optional[NeuronInfo]:
        """
        Retrieves detailed information about a specific neuron identified by its unique identifier (UID)
        within a specified subnet (netuid) of the Bittensor network. This function provides a comprehensive
        view of a neuron's attributes, including its stake, rank, and operational status.

        Args:
            uid (int): The unique identifier of the neuron.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[NeuronInfo]: Detailed information about the neuron if found, None otherwise.

        This function is crucial for analyzing individual neurons' contributions and status within a specific
        subnet, offering insights into their roles in the network's consensus and validation mechanisms.
        """
        if uid == None:
            return NeuronInfo._null_neuron()

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash(block)
                params = [netuid, uid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="neuronInfo_getNeuron", params=params  # custom rpc method
                )

        json_body = make_substrate_call_with_retry()
        result = json_body["result"]

        if result in (None, []):
            return NeuronInfo._null_neuron()

        return NeuronInfo.from_vec_u8(result)

    def neurons(self, netuid: int, block: Optional[int] = None) -> List[NeuronInfo]:
        """
        Retrieves a list of all neurons within a specified subnet of the Bittensor network. This function
        provides a snapshot of the subnet's neuron population, including each neuron's attributes and network
        interactions.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[NeuronInfo]: A list of NeuronInfo objects detailing each neuron's characteristics in the subnet.

        Understanding the distribution and status of neurons within a subnet is key to comprehending the
        network's decentralized structure and the dynamics of its consensus and governance processes.
        """
        neurons_lite = self.neurons_lite(netuid=netuid, block=block)
        weights = self.weights(block=block, netuid=netuid)
        bonds = self.bonds(block=block, netuid=netuid)

        weights_as_dict = {uid: w for uid, w in weights}
        bonds_as_dict = {uid: b for uid, b in bonds}

        neurons = [
            NeuronInfo.from_weights_bonds_and_neuron_lite(
                neuron_lite, weights_as_dict, bonds_as_dict
            )
            for neuron_lite in neurons_lite
        ]

        return neurons

    def neuron_for_uid_lite(
        self, uid: int, netuid: int, block: Optional[int] = None
    ) -> Optional[NeuronInfoLite]:
        """
        Retrieves a lightweight version of information about a neuron in a specific subnet, identified by
        its UID. The 'lite' version focuses on essential attributes such as stake and network activity.

        Args:
            uid (int): The unique identifier of the neuron.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[NeuronInfoLite]: A simplified version of neuron information if found, None otherwise.

        This function is useful for quick and efficient analyses of neuron status and activities within a
        subnet without the need for comprehensive data retrieval.
        """
        if uid == None:
            return NeuronInfoLite._null_neuron()

        hex_bytes_result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neuron_lite",
            params={
                "netuid": netuid,
                "uid": uid,
            },
            block=block,
        )

        if hex_bytes_result == None:
            return NeuronInfoLite._null_neuron()

        if hex_bytes_result.startswith("0x"):
            bytes_result = bytes.fromhex(hex_bytes_result[2:])
        else:
            bytes_result = bytes.fromhex(hex_bytes_result)

        return NeuronInfoLite.from_vec_u8(bytes_result)

    def neurons_lite(
        self, netuid: int, block: Optional[int] = None
    ) -> List[NeuronInfoLite]:
        """
        Retrieves a list of neurons in a 'lite' format from a specific subnet of the Bittensor network.
        This function provides a streamlined view of the neurons, focusing on key attributes such as stake
        and network participation.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[NeuronInfoLite]: A list of simplified neuron information for the subnet.

        This function offers a quick overview of the neuron population within a subnet, facilitating
        efficient analysis of the network's decentralized structure and neuron dynamics.
        """
        hex_bytes_result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons_lite",
            params=[netuid],
            block=block,
        )

        if hex_bytes_result == None:
            return []

        if hex_bytes_result.startswith("0x"):
            bytes_result = bytes.fromhex(hex_bytes_result[2:])
        else:
            bytes_result = bytes.fromhex(hex_bytes_result)

        return NeuronInfoLite.list_from_vec_u8(bytes_result)

    def metagraph(
        self,
        netuid: int,
        lite: bool = True,
        block: Optional[int] = None,
    ) -> "bittensor.Metagraph":
        """
        Returns a synced metagraph for a specified subnet within the Bittensor network. The metagraph
        represents the network's structure, including neuron connections and interactions.

        Args:
            netuid (int): The network UID of the subnet to query.
            lite (bool, default=True): If true, returns a metagraph using a lightweight sync (no weights, no bonds).
            block (Optional[int]): Block number for synchronization, or None for the latest block.

        Returns:
            bittensor.Metagraph: The metagraph representing the subnet's structure and neuron relationships.

        The metagraph is an essential tool for understanding the topology and dynamics of the Bittensor
        network's decentralized architecture, particularly in relation to neuron interconnectivity and consensus processes.
        """
        metagraph_ = bittensor.metagraph(
            network=self.network, netuid=netuid, lite=lite, sync=False
        )
        metagraph_.sync(block=block, lite=lite, subtensor=self)

        return metagraph_

    def incentive(self, netuid: int, block: Optional[int] = None) -> List[int]:
        """
        Retrieves the list of incentives for neurons within a specific subnet of the Bittensor network.
        This function provides insights into the reward distribution mechanisms and the incentives allocated
        to each neuron based on their contributions and activities.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            List[int]: The list of incentives for neurons within the subnet, indexed by UID.

        Understanding the incentive structure is crucial for analyzing the network's economic model and
        the motivational drivers for neuron participation and collaboration.
        """
        i_map = []
        i_map_encoded = self.query_map_subtensor(name="Incentive", block=block)
        if i_map_encoded.records:
            for netuid_, incentives_map in i_map_encoded:
                if netuid_ == netuid:
                    i_map = incentives_map.serialize()
                    break

        return i_map

    def weights(
        self, netuid: int, block: Optional[int] = None
    ) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """
        Retrieves the weight distribution set by neurons within a specific subnet of the Bittensor network.
        This function maps each neuron's UID to the weights it assigns to other neurons, reflecting the
        network's trust and value assignment mechanisms.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            List[Tuple[int, List[Tuple[int, int]]]]: A list of tuples mapping each neuron's UID to its assigned weights.

        The weight distribution is a key factor in the network's consensus algorithm and the ranking of neurons,
        influencing their influence and reward allocation within the subnet.
        """
        w_map = []
        w_map_encoded = self.query_map_subtensor(
            name="Weights", block=block, params=[netuid]
        )
        if w_map_encoded.records:
            for uid, w in w_map_encoded:
                w_map.append((uid.serialize(), w.serialize()))

        return w_map

    def bonds(
        self, netuid: int, block: Optional[int] = None
    ) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """
        Retrieves the bond distribution set by neurons within a specific subnet of the Bittensor network.
        Bonds represent the investments or commitments made by neurons in one another, indicating a level
        of trust and perceived value. This bonding mechanism is integral to the network's market-based approach
        to measuring and rewarding machine intelligence.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            List[Tuple[int, List[Tuple[int, int]]]]: A list of tuples mapping each neuron's UID to its bonds with other neurons.

        Understanding bond distributions is crucial for analyzing the trust dynamics and market behavior
        within the subnet. It reflects how neurons recognize and invest in each other's intelligence and
        contributions, supporting diverse and niche systems within the Bittensor ecosystem.
        """
        b_map = []
        b_map_encoded = self.query_map_subtensor(
            name="Bonds", block=block, params=[netuid]
        )
        if b_map_encoded.records:
            for uid, b in b_map_encoded:
                b_map.append((uid.serialize(), b.serialize()))

        return b_map

    def associated_validator_ip_info(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[List[IPInfo]]:
        """
        Retrieves the list of all validator IP addresses associated with a specific subnet in the Bittensor
        network. This information is crucial for network communication and the identification of validator nodes.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[List[IPInfo]]: A list of IPInfo objects for validator nodes in the subnet, or None if no validators are associated.

        Validator IP information is key for establishing secure and reliable connections within the network,
        facilitating consensus and validation processes critical for the network's integrity and performance.
        """
        hex_bytes_result = self.query_runtime_api(
            runtime_api="ValidatorIPRuntimeApi",
            method="get_associated_validator_ip_info_for_subnet",
            params=[netuid],
            block=block,
        )

        if hex_bytes_result == None:
            return None

        if hex_bytes_result.startswith("0x"):
            bytes_result = bytes.fromhex(hex_bytes_result[2:])
        else:
            bytes_result = bytes.fromhex(hex_bytes_result)

        return IPInfo.list_from_vec_u8(bytes_result)

    def get_subnet_burn_cost(self, block: Optional[int] = None) -> int:
        """
        Retrieves the burn cost for registering a new subnet within the Bittensor network. This cost
        represents the amount of Tao that needs to be locked or burned to establish a new subnet.

        Args:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            int: The burn cost for subnet registration.

        The subnet burn cost is an important economic parameter, reflecting the network's mechanisms for
        controlling the proliferation of subnets and ensuring their commitment to the network's long-term viability.
        """
        lock_cost = self.query_runtime_api(
            runtime_api="SubnetRegistrationRuntimeApi",
            method="get_network_registration_cost",
            params=[],
            block=block,
        )

        if lock_cost == None:
            return None

        return lock_cost

    ################
    ## Extrinsics ##
    ################

    def _do_delegation(
        self,
        wallet: "bittensor.wallet",
        delegate_ss58: str,
        amount: "Balance",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="add_stake",
                    call_params={"hotkey": delegate_ss58, "amount_staked": amount.rao},
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True
                response.process_events()
                if response.is_success:
                    return True
                else:
                    raise StakeError(response.error_message)

        return make_substrate_call_with_retry()

    def _do_undelegation(
        self,
        wallet: "bittensor.wallet",
        delegate_ss58: str,
        amount: "Balance",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="remove_stake",
                    call_params={
                        "hotkey": delegate_ss58,
                        "amount_unstaked": amount.rao,
                    },
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True
                response.process_events()
                if response.is_success:
                    return True
                else:
                    raise StakeError(response.error_message)

        return make_substrate_call_with_retry()

    def _do_nominate(
        self,
        wallet: "bittensor.wallet",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="become_delegate",
                    call_params={"hotkey": wallet.hotkey.ss58_address},
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )  # sign with coldkey
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    return True
                response.process_events()
                if response.is_success:
                    return True
                else:
                    raise NominationError(response.error_message)

        return make_substrate_call_with_retry()

    ################
    #### Legacy ####
    ################

    def get_balance(self, address: str, block: int = None) -> Balance:
        """
        Retrieves the token balance of a specific address within the Bittensor network. This function queries
        the blockchain to determine the amount of Tao held by a given account.

        Args:
            address (str): The Substrate address in ss58 format.
            block (int, optional): The blockchain block number at which to perform the query.

        Returns:
            Balance: The account balance at the specified block, represented as a Balance object.

        This function is important for monitoring account holdings and managing financial transactions
        within the Bittensor ecosystem. It helps in assessing the economic status and capacity of network participants.
        """
        try:

            @retry(delay=2, tries=3, backoff=2, max_delay=4)
            def make_substrate_call_with_retry():
                with self.substrate as substrate:
                    return substrate.query(
                        module="System",
                        storage_function="Account",
                        params=[address],
                        block_hash=None
                        if block == None
                        else substrate.get_block_hash(block),
                    )

            result = make_substrate_call_with_retry()
        except scalecodec.exceptions.RemainingScaleBytesNotEmptyException:
            bittensor.logging.error(
                "Your wallet it legacy formatted, you need to run btcli stake --ammount 0 to reformat it."
            )
            return Balance(1000)
        return Balance(result.value["data"]["free"])

    def get_current_block(self) -> int:
        """
        Returns the current block number on the Bittensor blockchain. This function provides the latest block
        number, indicating the most recent state of the blockchain.

        Returns:
            int: The current chain block number.

        Knowing the current block number is essential for querying real-time data and performing time-sensitive
        operations on the blockchain. It serves as a reference point for network activities and data synchronization.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.get_block_number(None)

        return make_substrate_call_with_retry()

    def get_balances(self, block: int = None) -> Dict[str, Balance]:
        """
        Retrieves the token balances of all accounts within the Bittensor network as of a specific blockchain block.
        This function provides a comprehensive view of the token distribution among different accounts.

        Args:
            block (int, optional): The blockchain block number at which to perform the query.

        Returns:
            Dict[str, Balance]: A dictionary mapping each account's ss58 address to its balance.

        This function is valuable for analyzing the overall economic landscape of the Bittensor network,
        including the distribution of financial resources and the financial status of network participants.
        """

        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query_map(
                    module="System",
                    storage_function="Account",
                    block_hash=None
                    if block == None
                    else substrate.get_block_hash(block),
                )

        result = make_substrate_call_with_retry()
        return_dict = {}
        for r in result:
            bal = Balance(int(r[1]["data"]["free"].value))
            return_dict[r[0].value] = bal
        return return_dict

    @staticmethod
    def _null_neuron() -> NeuronInfo:
        neuron = NeuronInfo(
            uid=0,
            netuid=0,
            active=0,
            stake="0",
            rank=0,
            emission=0,
            incentive=0,
            consensus=0,
            trust=0,
            validator_trust=0,
            dividends=0,
            last_update=0,
            validator_permit=False,
            weights=[],
            bonds=[],
            prometheus_info=None,
            axon_info=None,
            is_null=True,
            coldkey="000000000000000000000000000000000000000000000000",
            hotkey="000000000000000000000000000000000000000000000000",
        )
        return neuron

    def get_block_hash(self, block_id: int) -> str:
        """
        Retrieves the hash of a specific block on the Bittensor blockchain. The block hash is a unique
        identifier representing the cryptographic hash of the block's content, ensuring its integrity and
        immutability.

        Args:
            block_id (int): The block number for which the hash is to be retrieved.

        Returns:
            str: The cryptographic hash of the specified block.

        The block hash is a fundamental aspect of blockchain technology, providing a secure reference to
        each block's data. It is crucial for verifying transactions, ensuring data consistency, and
        maintaining the trustworthiness of the blockchain.
        """
        return self.substrate.get_block_hash(block_id=block_id)
