# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
The ``bittensor.subtensor`` module in Bittensor serves as a crucial interface for interacting with the Bittensor
blockchain, facilitating a range of operations essential for the decentralized machine learning network.
"""

import argparse
import copy
import socket
import sys
from functools import wraps
from typing import List, Dict, Union, Optional, Tuple, TypedDict, Any

import numpy as np
import scalecodec
from bittensor_wallet import Wallet
from numpy.typing import NDArray
from retry import retry
from scalecodec.base import RuntimeConfiguration
from scalecodec.exceptions import RemainingScaleBytesNotEmptyException
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.types import ScaleType
from substrateinterface.base import QueryMapResult, SubstrateInterface

from bittensor.core import settings
from bittensor.core.axon import Axon
from bittensor.core.chain_data import (
    NeuronInfo,
    PrometheusInfo,
    SubnetHyperparameters,
    NeuronInfoLite,
    custom_rpc_type_registry,
)
from bittensor.core.config import Config
from bittensor.core.metagraph import Metagraph
from bittensor.core.types import AxonServeCallParams, PrometheusServeCallParams
from bittensor.utils import torch, format_error_message
from bittensor.utils import u16_normalized_float, networking
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.deprecated.extrinsics.commit_weights import (
    commit_weights_extrinsic,
    reveal_weights_extrinsic,
)
from bittensor.utils.deprecated.extrinsics.prometheus import (
    prometheus_extrinsic,
)
from bittensor.utils.deprecated.extrinsics.serving import (
    serve_extrinsic,
    serve_axon_extrinsic,
    publish_metadata,
    get_metadata,
)
from bittensor.utils.deprecated.extrinsics.set_weights import (
    set_weights_extrinsic,
)
from bittensor.utils.deprecated.extrinsics.transfer import (
    transfer_extrinsic,
)
from bittensor.utils.weight_utils import generate_weight_hash

KEY_NONCE: Dict[str, int] = {}


class ParamWithTypes(TypedDict):
    name: str  # Name of the parameter.
    type: str  # ScaleType string of the parameter.


def _ensure_connected(func):
    """Decorator ensuring the function executes with an active substrate connection."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check the socket state before method execution
        if (
            self.substrate.websocket.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            != 0
        ):
            logging.info("Reconnection substrate...")
            self._get_substrate()
        # Execute the method if the connection is active or after reconnecting
        return func(self, *args, **kwargs)

    return wrapper


class Subtensor:
    """
    The Subtensor class in Bittensor serves as a crucial interface for interacting with the Bittensor blockchain,
    facilitating a range of operations essential for the decentralized machine learning network.

    This class enables neurons (network participants) to engage in activities such as registering on the network,
    managing staked weights, setting inter-neuronal weights, and participating in consensus mechanisms.

    The Bittensor network operates on a digital ledger where each neuron holds stakes (S) and learns a set
    of inter-peer weights (W). These weights, set by the neurons themselves, play a critical role in determining
    the ranking and incentive mechanisms within the network. Higher-ranked neurons, as determined by their
    contributions and trust within the network, receive more incentives.

    The Subtensor class connects to various Bittensor networks like the main ``finney`` network or local test
    networks, providing a gateway to the blockchain layer of Bittensor. It leverages a staked weighted trust
    system and consensus to ensure fair and distributed incentive mechanisms, where incentives (I) are
    primarily allocated to neurons that are trusted by the majority of the network.

    Additionally, Bittensor introduces a speculation-based reward mechanism in the form of bonds (B), allowing
    neurons to accumulate bonds in other neurons, speculating on their future value. This mechanism aligns
    with market-based speculation, incentivizing neurons to make judicious decisions in their inter-neuronal
    investments.

    Example Usage::

        # Connect to the main Bittensor network (Finney).
        finney_subtensor = subtensor(network='finney')

        # Close websocket connection with the Bittensor network.
        finney_subtensor.close()

        # (Re)creates the websocket connection with the Bittensor network.
        finney_subtensor.connect_websocket()

        # Register a new neuron on the network.
        wallet = bittensor_wallet.wallet(...)  # Assuming a wallet instance is created.
        success = finney_subtensor.register(wallet=wallet, netuid=netuid)

        # Set inter-neuronal weights for collaborative learning.
        success = finney_subtensor.set_weights(wallet=wallet, netuid=netuid, uids=[...], weights=[...])

        # Speculate by accumulating bonds in other promising neurons.
        success = finney_subtensor.delegate(wallet=wallet, delegate_ss58=other_neuron_ss58, amount=bond_amount)

        # Get the metagraph for a specific subnet using given subtensor connection
        metagraph = subtensor.metagraph(netuid=netuid)

    By facilitating these operations, the Subtensor class is instrumental in maintaining the decentralized
    intelligence and dynamic learning environment of the Bittensor network, as envisioned in its foundational
    principles and mechanisms described in the `NeurIPS paper
    <https://bittensor.com/pdfs/academia/NeurIPS_DAO_Workshop_2022_3_3.pdf>`_. paper.
    """

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional["Config"] = None,
        _mock: bool = False,
        log_verbose: bool = True,
    ) -> None:
        """
        Initializes a Subtensor interface for interacting with the Bittensor blockchain.

        NOTE:
            Currently subtensor defaults to the ``finney`` network. This will change in a future release.

        We strongly encourage users to run their own local subtensor node whenever possible. This increases decentralization and resilience of the network. In a future release, local subtensor will become the default and the fallback to ``finney`` removed. Please plan ahead for this change. We will provide detailed instructions on how to run a local subtensor node in the documentation in a subsequent release.

        Args:
            network (str, optional): The network name to connect to (e.g., ``finney``, ``local``). This can also be the chain endpoint (e.g., ``wss://entrypoint-finney.opentensor.ai:443``) and will be correctly parsed into the network and chain endpoint. If not specified, defaults to the main Bittensor network.
            config (bittensor.core.config.Config, optional): Configuration object for the subtensor. If not provided, a default configuration is used.
            _mock (bool, optional): If set to ``True``, uses a mocked connection for testing purposes.

        This initialization sets up the connection to the specified Bittensor network, allowing for various blockchain operations such as neuron registration, stake management, and setting weights.
        """
        # Determine config.subtensor.chain_endpoint and config.subtensor.network config.
        # If chain_endpoint is set, we override the network flag, otherwise, the chain_endpoint is assigned by the
        # network.
        # Argument importance: network > chain_endpoint > config.subtensor.chain_endpoint > config.subtensor.network

        if config is None:
            config = Subtensor.config()
        self._config = copy.deepcopy(config)

        # Setup config.subtensor.network and config.subtensor.chain_endpoint
        self.chain_endpoint, self.network = Subtensor.setup_config(
            network, self._config
        )

        if (
            self.network == "finney"
            or self.chain_endpoint == settings.FINNEY_ENTRYPOINT
        ) and log_verbose:
            logging.info(
                f"You are connecting to {self.network} network with endpoint {self.chain_endpoint}."
            )
            logging.warning(
                "We strongly encourage running a local subtensor node whenever possible. "
                "This increases decentralization and resilience of the network."
            )
            logging.warning(
                "In a future release, local subtensor will become the default endpoint. "
                "To get ahead of this change, please run a local subtensor node and point to it."
            )

        self.log_verbose = log_verbose
        self._get_substrate()

    def __str__(self) -> str:
        if self.network == self.chain_endpoint:
            # Connecting to chain endpoint without network known.
            return f"subtensor({self.chain_endpoint})"
        else:
            # Connecting to network with endpoint known.
            return f"subtensor({self.network}, {self.chain_endpoint})"

    def __repr__(self) -> str:
        return self.__str__()

    def close(self):
        """Cleans up resources for this subtensor instance like active websocket connection and active extensions."""
        self.substrate.close()

    def _get_substrate(self):
        """Establishes a connection to the Substrate node using configured parameters."""
        try:
            # Set up params.
            self.substrate = SubstrateInterface(
                ss58_format=settings.SS58_FORMAT,
                use_remote_preset=True,
                url=self.chain_endpoint,
                type_registry=settings.TYPE_REGISTRY,
            )
            if self.log_verbose:
                logging.info(
                    f"Connected to {self.network} network and {self.chain_endpoint}."
                )

        except ConnectionRefusedError:
            logging.error(
                f"Could not connect to {self.network} network with {self.chain_endpoint} chain endpoint. Exiting...",
            )
            logging.info(
                "You can check if you have connectivity by running this command: nc -vz localhost "
                f"{self.chain_endpoint.split(':')[2]}"
            )
            sys.exit(1)

        try:
            self.substrate.websocket.settimeout(600)
        except AttributeError as e:
            logging.warning(f"AttributeError: {e}")
        except TypeError as e:
            logging.warning(f"TypeError: {e}")
        except (socket.error, OSError) as e:
            logging.warning(f"Socket error: {e}")

    @staticmethod
    def config() -> "Config":
        """
        Creates and returns a Bittensor configuration object.

        Returns:
            config (bittensor.core.config.Config): A Bittensor configuration object configured with arguments added by the `subtensor.add_args` method.
        """
        parser = argparse.ArgumentParser()
        Subtensor.add_args(parser)
        return Config(parser, args=[])

    @staticmethod
    def setup_config(network: Optional[str], config: "Config"):
        """
        Sets up and returns the configuration for the Subtensor network and endpoint.

        This method determines the appropriate network and chain endpoint based on the provided network string or
        configuration object. It evaluates the network and endpoint in the following order of precedence:
        1. Provided network string.
        2. Configured chain endpoint in the `config` object.
        3. Configured network in the `config` object.
        4. Default chain endpoint.
        5. Default network.

        Args:
            network (str): The name of the Subtensor network. If None, the network and endpoint will be determined from the `config` object.
            config (bittensor.core.config.Config): The configuration object containing the network and chain endpoint settings.

        Returns:
            tuple: A tuple containing the formatted WebSocket endpoint URL and the evaluated network name.
        """
        if network is not None:
            (
                evaluated_network,
                evaluated_endpoint,
            ) = Subtensor.determine_chain_endpoint_and_network(network)
        else:
            if config.get("__is_set", {}).get("subtensor.chain_endpoint"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.chain_endpoint
                )

            elif config.get("__is_set", {}).get("subtensor.network"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.network
                )

            elif config.subtensor.get("chain_endpoint"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.chain_endpoint
                )

            elif config.subtensor.get("network"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.network
                )

            else:
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    settings.DEFAULTS.subtensor.network
                )

        return (
            networking.get_formatted_ws_endpoint_url(evaluated_endpoint),
            evaluated_network,
        )

    @classmethod
    def help(cls):
        """Print help to stdout."""
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print(cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: "argparse.ArgumentParser", prefix: Optional[str] = None):
        """
        Adds command-line arguments to the provided ArgumentParser for configuring the Subtensor settings.

        Args:
            parser (argparse.ArgumentParser): The ArgumentParser object to which the Subtensor arguments will be added.
            prefix (Optional[str]): An optional prefix for the argument names. If provided, the prefix is prepended to each argument name.

        Arguments added:
            --subtensor.network: The Subtensor network flag. Possible values are 'finney', 'test', 'archive', and 'local'. Overrides the chain endpoint if set.
            --subtensor.chain_endpoint: The Subtensor chain endpoint flag. If set, it overrides the network flag.
            --subtensor._mock: If true, uses a mocked connection to the chain.

        Example:
            parser = argparse.ArgumentParser()
            Subtensor.add_args(parser)
        """
        prefix_str = "" if prefix is None else f"{prefix}."
        try:
            default_network = settings.DEFAULT_NETWORK
            default_chain_endpoint = settings.FINNEY_ENTRYPOINT

            parser.add_argument(
                f"--{prefix_str}subtensor.network",
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
                f"--{prefix_str}subtensor.chain_endpoint",
                default=default_chain_endpoint,
                type=str,
                help="""The subtensor endpoint flag. If set, overrides the --network flag.""",
            )
            parser.add_argument(
                f"--{prefix_str}subtensor._mock",
                default=False,
                type=bool,
                help="""If true, uses a mocked connection to the chain.""",
            )

        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    # Inner private functions
    @_ensure_connected
    def _encode_params(
        self,
        call_definition: List["ParamWithTypes"],
        params: Union[List[Any], Dict[str, Any]],
    ) -> str:
        """Returns a hex encoded string of the params using their types."""
        param_data = scalecodec.ScaleBytes(b"")

        for i, param in enumerate(call_definition["params"]):  # type: ignore
            scale_obj = self.substrate.create_scale_object(param["type"])
            if type(params) is list:
                param_data += scale_obj.encode(params[i])
            else:
                if param["name"] not in params:
                    raise ValueError(f"Missing param {param['name']} in params dict.")

                param_data += scale_obj.encode(params[param["name"]])

        return param_data.to_hex()

    def _get_hyperparameter(
        self, param_name: str, netuid: int, block: Optional[int] = None
    ) -> Optional[Any]:
        """
        Retrieves a specified hyperparameter for a specific subnet.

        Args:
            param_name (str): The name of the hyperparameter to retrieve.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[Union[int, float]]: The value of the specified hyperparameter if the subnet exists, ``None`` otherwise.
        """
        if not self.subnet_exists(netuid, block):
            return None

        result = self.query_subtensor(param_name, block, [netuid])
        if result is None or not hasattr(result, "value"):
            return None

        return result.value

    # Calls methods
    @_ensure_connected
    def query_subtensor(
        self, name: str, block: Optional[int] = None, params: Optional[list] = None
    ) -> "ScaleType":
        """
        Queries named storage from the Subtensor module on the Bittensor blockchain. This function is used to retrieve specific data or parameters from the blockchain, such as stake, rank, or other neuron-specific attributes.

        Args:
            name (str): The name of the storage function to query.
            block (Optional[int]): The blockchain block number at which to perform the query.
            params (Optional[List[object]], optional): A list of parameters to pass to the query function.

        Returns:
            query_response (ScaleType): An object containing the requested data.

        This query function is essential for accessing detailed information about the network and its neurons, providing valuable insights into the state and dynamics of the Bittensor ecosystem.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry() -> "ScaleType":
            return self.substrate.query(
                module="SubtensorModule",
                storage_function=name,
                params=params,
                block_hash=(
                    None if block is None else self.substrate.get_block_hash(block)
                ),
            )

        return make_substrate_call_with_retry()

    @_ensure_connected
    def query_map_subtensor(
        self, name: str, block: Optional[int] = None, params: Optional[list] = None
    ) -> "QueryMapResult":
        """
        Queries map storage from the Subtensor module on the Bittensor blockchain. This function is designed to retrieve a map-like data structure, which can include various neuron-specific details or network-wide attributes.

        Args:
            name (str): The name of the map storage function to query.
            block (Optional[int]): The blockchain block number at which to perform the query.
            params (Optional[List[object]], optional): A list of parameters to pass to the query function.

        Returns:
            QueryMapResult: An object containing the map-like data structure, or ``None`` if not found.

        This function is particularly useful for analyzing and understanding complex network structures and relationships within the Bittensor ecosystem, such as inter-neuronal connections and stake distributions.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry():
            return self.substrate.query_map(
                module="SubtensorModule",
                storage_function=name,
                params=params,
                block_hash=(
                    None if block is None else self.substrate.get_block_hash(block)
                ),
            )

        return make_substrate_call_with_retry()

    def query_runtime_api(
        self,
        runtime_api: str,
        method: str,
        params: Optional[Union[List[int], Dict[str, int]]],
        block: Optional[int] = None,
    ) -> Optional[str]:
        """
        Queries the runtime API of the Bittensor blockchain, providing a way to interact with the underlying runtime and retrieve data encoded in Scale Bytes format. This function is essential for advanced users who need to interact with specific runtime methods and decode complex data types.

        Args:
            runtime_api (str): The name of the runtime API to query.
            method (str): The specific method within the runtime API to call.
            params (Optional[List[ParamWithTypes]], optional): The parameters to pass to the method call.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            Optional[bytes]: The Scale Bytes encoded result from the runtime API call, or ``None`` if the call fails.

        This function enables access to the deeper layers of the Bittensor blockchain, allowing for detailed and specific interactions with the network's runtime environment.
        """
        call_definition = settings.TYPE_REGISTRY["runtime_api"][runtime_api]["methods"][
            method
        ]

        json_result = self.state_call(
            method=f"{runtime_api}_{method}",
            data=(
                "0x"
                if params is None
                else self._encode_params(call_definition=call_definition, params=params)
            ),
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

    @_ensure_connected
    def state_call(
        self, method: str, data: str, block: Optional[int] = None
    ) -> Dict[Any, Any]:
        """
        Makes a state call to the Bittensor blockchain, allowing for direct queries of the blockchain's state. This function is typically used for advanced queries that require specific method calls and data inputs.

        Args:
            method (str): The method name for the state call.
            data (str): The data to be passed to the method.
            block (Optional[int]): The blockchain block number at which to perform the state call.

        Returns:
            result (Dict[Any, Any]): The result of the rpc call.

        The state call function provides a more direct and flexible way of querying blockchain data, useful for specific use cases where standard queries are insufficient.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry() -> Dict[Any, Any]:
            block_hash = None if block is None else self.substrate.get_block_hash(block)
            return self.substrate.rpc_request(
                method="state_call",
                params=[method, data, block_hash] if block_hash else [method, data],
            )

        return make_substrate_call_with_retry()

    @_ensure_connected
    def query_map(
        self,
        module: str,
        name: str,
        block: Optional[int] = None,
        params: Optional[list] = None,
    ) -> QueryMapResult:
        """
        Queries map storage from any module on the Bittensor blockchain. This function retrieves data structures that represent key-value mappings, essential for accessing complex and structured data within the blockchain modules.

        Args:
            module (str): The name of the module from which to query the map storage.
            name (str): The specific storage function within the module to query.
            block (Optional[int]): The blockchain block number at which to perform the query.
            params (Optional[List[object]], optional): Parameters to be passed to the query.

        Returns:
            result (QueryMapResult): A data structure representing the map storage if found, ``None`` otherwise.

        This function is particularly useful for retrieving detailed and structured data from various blockchain modules, offering insights into the network's state and the relationships between its different components.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry() -> "QueryMapResult":
            return self.substrate.query_map(
                module=module,
                storage_function=name,
                params=params,
                block_hash=(
                    None if block is None else self.substrate.get_block_hash(block)
                ),
            )

        return make_substrate_call_with_retry()

    @_ensure_connected
    def query_constant(
        self, module_name: str, constant_name: str, block: Optional[int] = None
    ) -> Optional["ScaleType"]:
        """
        Retrieves a constant from the specified module on the Bittensor blockchain. This function is used to access fixed parameters or values defined within the blockchain's modules, which are essential for understanding the network's configuration and rules.

        Args:
            module_name (str): The name of the module containing the constant.
            constant_name (str): The name of the constant to retrieve.
            block (Optional[int]): The blockchain block number at which to query the constant.

        Returns:
            Optional[ScaleType]: The value of the constant if found, ``None`` otherwise.

        Constants queried through this function can include critical network parameters such as inflation rates, consensus rules, or validation thresholds, providing a deeper understanding of the Bittensor network's operational parameters.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry():
            return self.substrate.get_constant(
                module_name=module_name,
                constant_name=constant_name,
                block_hash=(
                    None if block is None else self.substrate.get_block_hash(block)
                ),
            )

        return make_substrate_call_with_retry()

    @_ensure_connected
    def query_module(
        self,
        module: str,
        name: str,
        block: Optional[int] = None,
        params: Optional[list] = None,
    ) -> "ScaleType":
        """
        Queries any module storage on the Bittensor blockchain with the specified parameters and block number. This function is a generic query interface that allows for flexible and diverse data retrieval from various blockchain modules.

        Args:
            module (str): The name of the module from which to query data.
            name (str): The name of the storage function within the module.
            block (Optional[int]): The blockchain block number at which to perform the query.
            params (Optional[List[object]], optional): A list of parameters to pass to the query function.

        Returns:
            Optional[ScaleType]: An object containing the requested data if found, ``None`` otherwise.

        This versatile query function is key to accessing a wide range of data and insights from different parts of the Bittensor blockchain, enhancing the understanding and analysis of the network's state and dynamics.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry() -> "ScaleType":
            return self.substrate.query(
                module=module,
                storage_function=name,
                params=params,
                block_hash=(
                    None if block is None else self.substrate.get_block_hash(block)
                ),
            )

        return make_substrate_call_with_retry()

    # Common subtensor methods
    def metagraph(
        self, netuid: int, lite: bool = True, block: Optional[int] = None
    ) -> "Metagraph":  # type: ignore
        """
        Returns a synced metagraph for a specified subnet within the Bittensor network. The metagraph represents the network's structure, including neuron connections and interactions.

        Args:
            netuid (int): The network UID of the subnet to query.
            lite (bool, default=True): If true, returns a metagraph using a lightweight sync (no weights, no bonds).
            block (Optional[int]): Block number for synchronization, or ``None`` for the latest block.

        Returns:
            bittensor.core.metagraph.Metagraph: The metagraph representing the subnet's structure and neuron relationships.

        The metagraph is an essential tool for understanding the topology and dynamics of the Bittensor network's decentralized architecture, particularly in relation to neuron interconnectivity and consensus processes.
        """
        metagraph = Metagraph(
            network=self.network, netuid=netuid, lite=lite, sync=False
        )
        metagraph.sync(block=block, lite=lite, subtensor=self)

        return metagraph

    @staticmethod
    def determine_chain_endpoint_and_network(network: str):
        """Determines the chain endpoint and network from the passed network or chain_endpoint.

        Args:
            network (str): The network flag. The choices are: ``-- finney`` (main network), ``-- archive``
                (archive network +300 blocks), ``-- local`` (local running network), ``-- test`` (test network).
        Returns:
            network (str): The network flag.
            chain_endpoint (str): The chain endpoint flag. If set, overrides the ``network`` argument.
        """
        if network is None:
            return None, None
        if network in ["finney", "local", "test", "archive"]:
            if network == "finney":
                # Kiru Finney staging network.
                return network, settings.FINNEY_ENTRYPOINT
            elif network == "local":
                return network, settings.LOCAL_ENTRYPOINT
            elif network == "test":
                return network, settings.FINNEY_TEST_ENTRYPOINT
            elif network == "archive":
                return network, settings.ARCHIVE_ENTRYPOINT
        else:
            if (
                network == settings.FINNEY_ENTRYPOINT
                or "entrypoint-finney.opentensor.ai" in network
            ):
                return "finney", settings.FINNEY_ENTRYPOINT
            elif (
                network == settings.FINNEY_TEST_ENTRYPOINT
                or "test.finney.opentensor.ai" in network
            ):
                return "test", settings.FINNEY_TEST_ENTRYPOINT
            elif (
                network == settings.ARCHIVE_ENTRYPOINT
                or "archive.chain.opentensor.ai" in network
            ):
                return "archive", settings.ARCHIVE_ENTRYPOINT
            elif "127.0.0.1" in network or "localhost" in network:
                return "local", network
            else:
                return "unknown", network

    def get_netuids_for_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> List[int]:
        """
        Retrieves a list of subnet UIDs (netuids) for which a given hotkey is a member. This function identifies the specific subnets within the Bittensor network where the neuron associated with the hotkey is active.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            List[int]: A list of netuids where the neuron is a member.
        """
        result = self.query_map_subtensor("IsNetworkMember", block, [hotkey_ss58])
        return (
            [record[0].value for record in result.records if record[1]]
            if result and hasattr(result, "records")
            else []
        )

    @_ensure_connected
    def get_current_block(self) -> int:
        """
        Returns the current block number on the Bittensor blockchain. This function provides the latest block number, indicating the most recent state of the blockchain.

        Returns:
            int: The current chain block number.

        Knowing the current block number is essential for querying real-time data and performing time-sensitive operations on the blockchain. It serves as a reference point for network activities and data synchronization.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry():
            return self.substrate.get_block_number(None)  # type: ignore

        return make_substrate_call_with_retry()

    def is_hotkey_registered_any(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> bool:
        """
        Checks if a neuron's hotkey is registered on any subnet within the Bittensor network.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            block (Optional[int]): The blockchain block number at which to perform the check.

        Returns:
            bool: ``True`` if the hotkey is registered on any subnet, False otherwise.

        This function is essential for determining the network-wide presence and participation of a neuron.
        """
        return len(self.get_netuids_for_hotkey(hotkey_ss58, block)) > 0

    def is_hotkey_registered_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> bool:
        """
        Checks if a neuron's hotkey is registered on a specific subnet within the Bittensor network.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number at which to perform the check.

        Returns:
            bool: ``True`` if the hotkey is registered on the specified subnet, False otherwise.

        This function helps in assessing the participation of a neuron in a particular subnet, indicating its specific area of operation or influence within the network.
        """
        return self.get_uid_for_hotkey_on_subnet(hotkey_ss58, netuid, block) is not None

    def is_hotkey_registered(
        self,
        hotkey_ss58: str,
        netuid: Optional[int] = None,
        block: Optional[int] = None,
    ) -> bool:
        """
        Determines whether a given hotkey (public key) is registered in the Bittensor network, either globally across any subnet or specifically on a specified subnet. This function checks the registration status of a neuron identified by its hotkey, which is crucial for validating its participation and activities within the network.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            netuid (Optional[int]): The unique identifier of the subnet to check the registration. If ``None``, the registration is checked across all subnets.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            bool: ``True`` if the hotkey is registered in the specified context (either any subnet or a specific subnet), ``False`` otherwise.

        This function is important for verifying the active status of neurons in the Bittensor network. It aids in understanding whether a neuron is eligible to participate in network processes such as consensus, validation, and incentive distribution based on its registration status.
        """
        if netuid is None:
            return self.is_hotkey_registered_any(hotkey_ss58, block)
        else:
            return self.is_hotkey_registered_on_subnet(hotkey_ss58, netuid, block)

    @_ensure_connected
    def do_set_weights(
        self,
        wallet: "Wallet",
        uids: List[int],
        vals: List[int],
        netuid: int,
        version_key: int = settings.version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> Tuple[bool, Optional[str]]:  # (success, error_message)
        """
        Internal method to send a transaction to the Bittensor blockchain, setting weights for specified neurons. This method constructs and submits the transaction, handling retries and blockchain communication.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron setting the weights.
            uids (List[int]): List of neuron UIDs for which weights are being set.
            vals (List[int]): List of weight values corresponding to each UID.
            netuid (int): Unique identifier for the network.
            version_key (int, optional): Version key for compatibility with the network.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

        This method is vital for the dynamic weighting mechanism in Bittensor, where neurons adjust their trust in other neurons based on observed performance and contributions.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry():
            call = self.substrate.compose_call(
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
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=wallet.hotkey,
                era={"period": 5},
            )
            response = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True, "Not waiting for finalization or inclusion."

            response.process_events()
            if response.is_success:
                return True, "Successfully set weights."
            else:
                return False, format_error_message(response.error_message)

        return make_substrate_call_with_retry()

    # keep backwards compatibility for the community
    _do_set_weights = do_set_weights

    # Not used in Bittensor, but is actively used by the community in almost all subnets
    def set_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: Union[NDArray[np.int64], "torch.LongTensor", list],
        weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
        version_key: int = settings.version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        prompt: bool = False,
        max_retries: int = 5,
    ) -> Tuple[bool, str]:
        """
        Sets the inter-neuronal weights for the specified neuron. This process involves specifying the influence or trust a neuron places on other neurons in the network, which is a fundamental aspect of Bittensor's decentralized learning architecture.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron setting the weights.
            netuid (int): The unique identifier of the subnet.
            uids (Union[NDArray[np.int64], torch.LongTensor, list]): The list of neuron UIDs that the weights are being set for.
            weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The corresponding weights to be set for each UID.
            version_key (int, optional): Version key for compatibility with the network.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If ``True``, prompts for user confirmation before proceeding.
            max_retries (int, optional): The number of maximum attempts to set weights. (Default: 5)

        Returns:
            Tuple[bool, str]: ``True`` if the setting of weights is successful, False otherwise. And `msg`, a string value describing the success or potential error.

        This function is crucial in shaping the network's collective intelligence, where each neuron's learning and contribution are influenced by the weights it sets towards others【81†source】.
        """
        uid = self.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, netuid)
        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to set weights!"
        while (
            self.blocks_since_last_update(netuid, uid) > self.weights_rate_limit(netuid)  # type: ignore
            and retries < max_retries
        ):
            try:
                success, message = set_weights_extrinsic(
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
            except Exception as e:
                logging.error(f"Error setting weights: {e}")
            finally:
                retries += 1

        return success, message

    def serve_axon(
        self,
        netuid: int,
        axon: "Axon",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        """
        Registers an Axon serving endpoint on the Bittensor network for a specific neuron. This function is used to set up the Axon, a key component of a neuron that handles incoming queries and data processing tasks.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            axon (bittensor.core.axon.Axon): The Axon instance to be registered for serving.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.

        Returns:
            bool: ``True`` if the Axon serve registration is successful, False otherwise.

        By registering an Axon, the neuron becomes an active part of the network's distributed computing infrastructure, contributing to the collective intelligence of Bittensor.
        """
        return serve_axon_extrinsic(
            self, netuid, axon, wait_for_inclusion, wait_for_finalization
        )

    # metagraph
    @property
    def block(self) -> int:
        """Returns current chain block.

        Returns:
            block (int): Current chain block.
        """
        return self.get_current_block()

    def blocks_since_last_update(self, netuid: int, uid: int) -> Optional[int]:
        """
        Returns the number of blocks since the last update for a specific UID in the subnetwork.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            uid (int): The unique identifier of the neuron.

        Returns:
            Optional[int]: The number of blocks since the last update, or ``None`` if the subnetwork or UID does not exist.
        """
        call = self._get_hyperparameter(param_name="LastUpdate", netuid=netuid)
        return None if call is None else self.get_current_block() - int(call[uid])

    @_ensure_connected
    def get_block_hash(self, block_id: int) -> str:
        """
        Retrieves the hash of a specific block on the Bittensor blockchain. The block hash is a unique identifier representing the cryptographic hash of the block's content, ensuring its integrity and immutability.

        Args:
            block_id (int): The block number for which the hash is to be retrieved.

        Returns:
            str: The cryptographic hash of the specified block.

        The block hash is a fundamental aspect of blockchain technology, providing a secure reference to each block's data. It is crucial for verifying transactions, ensuring data consistency, and maintaining the trustworthiness of the blockchain.
        """
        return self.substrate.get_block_hash(block_id=block_id)

    def weights_rate_limit(self, netuid: int) -> Optional[int]:
        """
        Returns network WeightsSetRateLimit hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.

        Returns:
            Optional[int]: The value of the WeightsSetRateLimit hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not found.
        """
        call = self._get_hyperparameter(param_name="WeightsSetRateLimit", netuid=netuid)
        return None if call is None else int(call)

    # Keep backwards compatibility for community usage.
    # Make some commitment on-chain about arbitrary data.
    def commit(self, wallet, netuid: int, data: str):
        """
        Commits arbitrary data to the Bittensor network by publishing metadata.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the data.
            netuid (int): The unique identifier of the subnetwork.
            data (str): The data to be committed to the network.
        """
        publish_metadata(self, wallet, netuid, f"Raw{len(data)}", data.encode())

    # Keep backwards compatibility for community usage.
    def subnetwork_n(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Returns network SubnetworkN hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int], optional): The block number to retrieve the parameter from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[int]: The value of the SubnetworkN hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not found.
        """
        call = self._get_hyperparameter(
            param_name="SubnetworkN", netuid=netuid, block=block
        )
        return None if call is None else int(call)

    @_ensure_connected
    def do_transfer(
        self,
        wallet: "Wallet",
        dest: str,
        transfer_balance: "Balance",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Sends a transfer extrinsic to the chain.

        Args:
            wallet (bittensor_wallet.Wallet): Wallet object.
            dest (str): Destination public key address.
            transfer_balance (bittensor.utils.balance.Balance): Amount to transfer.
            wait_for_inclusion (bool): If ``true``, waits for inclusion.
            wait_for_finalization (bool): If ``true``, waits for finalization.

        Returns:
            success (bool): ``True`` if transfer was successful.
            block_hash (str): Block hash of the transfer. On success and if wait_for_ finalization/inclusion is ``True``.
            error (str): Error message if transfer failed.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry():
            call = self.substrate.compose_call(
                call_module="Balances",
                call_function="transfer_allow_death",
                call_params={"dest": dest, "value": transfer_balance.rao},
            )
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call, keypair=wallet.coldkey
            )
            response = self.substrate.submit_extrinsic(
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
                return False, None, format_error_message(response.error_message)

        return make_substrate_call_with_retry()

    # Community uses this method
    def transfer(
        self,
        wallet: "Wallet",
        dest: str,
        amount: Union[Balance, float],
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """
        Executes a transfer of funds from the provided wallet to the specified destination address. This function is used to move TAO tokens within the Bittensor network, facilitating transactions between neurons.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet from which funds are being transferred.
            dest (str): The destination public key address.
            amount (Union[Balance, float]): The amount of TAO to be transferred.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If ``True``, prompts for user confirmation before proceeding.

        Returns:
            transfer_extrinsic (bool): ``True`` if the transfer is successful, False otherwise.

        This function is essential for the fluid movement of tokens in the network, supporting various economic activities such as staking, delegation, and reward distribution.
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

    # Community uses this method via `bittensor.api.extrinsics.prometheus.prometheus_extrinsic`
    def get_neuron_for_pubkey_and_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional[NeuronInfo]:
        """
        Retrieves information about a neuron based on its public key (hotkey SS58 address) and the specific subnet UID (netuid). This function provides detailed neuron information for a particular subnet within the Bittensor network.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            Optional[NeuronInfo]: Detailed information about the neuron if found, ``None`` otherwise.

        This function is crucial for accessing specific neuron data and understanding its status, stake, and other attributes within a particular subnet of the Bittensor ecosystem.
        """
        return self.neuron_for_uid(
            self.get_uid_for_hotkey_on_subnet(hotkey_ss58, netuid, block=block),
            netuid,
            block=block,
        )

    @_ensure_connected
    def neuron_for_uid(
        self, uid: Optional[int], netuid: int, block: Optional[int] = None
    ) -> NeuronInfo:
        """
        Retrieves detailed information about a specific neuron identified by its unique identifier (UID) within a specified subnet (netuid) of the Bittensor network. This function provides a comprehensive view of a neuron's attributes, including its stake, rank, and operational status.

        Args:
            uid (int): The unique identifier of the neuron.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            NeuronInfo: Detailed information about the neuron if found, ``None`` otherwise.

        This function is crucial for analyzing individual neurons' contributions and status within a specific subnet, offering insights into their roles in the network's consensus and validation mechanisms.
        """
        if uid is None:
            return NeuronInfo.get_null_neuron()

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry():
            block_hash = None if block is None else self.substrate.get_block_hash(block)
            params = [netuid, uid]
            if block_hash:
                params = params + [block_hash]
            return self.substrate.rpc_request(
                method="neuronInfo_getNeuron",
                params=params,  # custom rpc method
            )

        json_body = make_substrate_call_with_retry()

        if not (result := json_body.get("result", None)):
            return NeuronInfo.get_null_neuron()

        return NeuronInfo.from_vec_u8(result)

    # Community uses this method via `bittensor.api.extrinsics.prometheus.prometheus_extrinsic`
    @_ensure_connected
    def do_serve_prometheus(
        self,
        wallet: "Wallet",
        call_params: PrometheusServeCallParams,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Sends a serve prometheus extrinsic to the chain.

        Args:
            wallet (:func:`bittensor_wallet.Wallet`): Wallet object.
            call_params (:func:`PrometheusServeCallParams`): Prometheus serve call parameters.
            wait_for_inclusion (bool): If ``true``, waits for inclusion.
            wait_for_finalization (bool): If ``true``, waits for finalization.

        Returns:
            success (bool): ``True`` if serve prometheus was successful.
            error (:func:`Optional[str]`): Error message if serve prometheus failed, ``None`` otherwise.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry():
            call = self.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="serve_prometheus",
                call_params=call_params,
            )
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call, keypair=wallet.hotkey
            )
            response = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            if wait_for_inclusion or wait_for_finalization:
                response.process_events()
                if response.is_success:
                    return True, None
                else:
                    return False, format_error_message(response.error_message)
            else:
                return True, None

        return make_substrate_call_with_retry()

    # Community uses this method name
    _do_serve_prometheus = do_serve_prometheus

    # Community uses this method
    def serve_prometheus(
        self,
        wallet: "Wallet",
        port: int,
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        """
        Serves Prometheus metrics by submitting an extrinsic to a blockchain network via the specified wallet. The function allows configuring whether to wait for the transaction's inclusion in a block and its finalization.

        Args:
            wallet (bittensor_wallet.Wallet): Bittensor wallet instance used for submitting the extrinsic.
            port (int): The port number on which Prometheus metrics are served.
            netuid (int): The unique identifier of the subnetwork.
            wait_for_inclusion (bool, optional): If True, waits for the transaction to be included in a block. Defaults to ``False``.
            wait_for_finalization (bool, optional): If True, waits for the transaction to be finalized. Defaults to ``True``.

        Returns:
            bool: Returns True if the Prometheus extrinsic is successfully processed, otherwise False.
        """
        return prometheus_extrinsic(
            self,
            wallet=wallet,
            port=port,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    # Community uses this method as part of `subtensor.serve_axon`
    @_ensure_connected
    def do_serve_axon(
        self,
        wallet: "Wallet",
        call_params: AxonServeCallParams,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Internal method to submit a serve axon transaction to the Bittensor blockchain. This method creates and submits a transaction, enabling a neuron's Axon to serve requests on the network.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron.
            call_params (AxonServeCallParams): Parameters required for the serve axon call.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

        This function is crucial for initializing and announcing a neuron's Axon service on the network, enhancing the decentralized computation capabilities of Bittensor.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry():
            call = self.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="serve_axon",
                call_params=call_params,
            )
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call, keypair=wallet.hotkey
            )
            response = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            if wait_for_inclusion or wait_for_finalization:
                response.process_events()
                if response.is_success:
                    return True, None
                else:
                    return False, format_error_message(response.error_message)
            else:
                return True, None

        return make_substrate_call_with_retry()

    # keep backwards compatibility for the community
    _do_serve_axon = do_serve_axon

    # Community uses this method
    def serve(
        self,
        wallet: "Wallet",
        ip: str,
        port: int,
        protocol: int,
        netuid: int,
        placeholder1: int = 0,
        placeholder2: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization=True,
    ) -> bool:
        """
        Registers a neuron's serving endpoint on the Bittensor network. This function announces the IP address and port where the neuron is available to serve requests, facilitating peer-to-peer communication within the network.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron being served.
            ip (str): The IP address of the serving neuron.
            port (int): The port number on which the neuron is serving.
            protocol (int): The protocol type used by the neuron (e.g., GRPC, HTTP).
            netuid (int): The unique identifier of the subnetwork.
            placeholder1 (int, optional): Placeholder parameter for future extensions. Default is ``0``.
            placeholder2 (int, optional): Placeholder parameter for future extensions. Default is ``0``.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain. Default is ``True``.

        Returns:
            bool: ``True`` if the serve registration is successful, False otherwise.

        This function is essential for establishing the neuron's presence in the network, enabling it to participate in the decentralized machine learning processes of Bittensor.
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

    # Community uses this method
    def get_subnet_hyperparameters(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[Union[List, SubnetHyperparameters]]:
        """
        Retrieves the hyperparameters for a specific subnet within the Bittensor network. These hyperparameters define the operational settings and rules governing the subnet's behavior.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[SubnetHyperparameters]: The subnet's hyperparameters, or ``None`` if not available.

        Understanding the hyperparameters is crucial for comprehending how subnets are configured and managed, and how they interact with the network's consensus and incentive mechanisms.
        """
        hex_bytes_result = self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_hyperparams",
            params=[netuid],
            block=block,
        )

        if hex_bytes_result is None:
            return []

        if hex_bytes_result.startswith("0x"):
            bytes_result = bytes.fromhex(hex_bytes_result[2:])
        else:
            bytes_result = bytes.fromhex(hex_bytes_result)

        return SubnetHyperparameters.from_vec_u8(bytes_result)  # type: ignore

    # Community uses this method
    # Returns network ImmunityPeriod hyper parameter.
    def immunity_period(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Retrieves the 'ImmunityPeriod' hyperparameter for a specific subnet. This parameter defines the duration during which new neurons are protected from certain network penalties or restrictions.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The value of the 'ImmunityPeriod' hyperparameter if the subnet exists, ``None`` otherwise.

        The 'ImmunityPeriod' is a critical aspect of the network's governance system, ensuring that new participants have a grace period to establish themselves and contribute to the network without facing immediate punitive actions.
        """
        call = self._get_hyperparameter(
            param_name="ImmunityPeriod", netuid=netuid, block=block
        )
        return None if call is None else int(call)

    # Community uses this method
    def get_uid_for_hotkey_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Retrieves the unique identifier (UID) for a neuron's hotkey on a specific subnet.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The UID of the neuron if it is registered on the subnet, ``None`` otherwise.

        The UID is a critical identifier within the network, linking the neuron's hotkey to its operational and governance activities on a particular subnet.
        """
        _result = self.query_subtensor("Uids", block, [netuid, hotkey_ss58])
        return getattr(_result, "value", None)

    # Community uses this method
    def tempo(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Returns network Tempo hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int], optional): The block number to retrieve the parameter from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[int]: The value of the Tempo hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not found.
        """
        call = self._get_hyperparameter(param_name="Tempo", netuid=netuid, block=block)
        return None if call is None else int(call)

    # Community uses this method
    def get_commitment(self, netuid: int, uid: int, block: Optional[int] = None) -> str:
        """
        Retrieves the on-chain commitment for a specific neuron in the Bittensor network.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            uid (int): The unique identifier of the neuron.
            block (Optional[int]): The block number to retrieve the commitment from. If None, the latest block is used. Default is ``None``.

        Returns:
            str: The commitment data as a string.
        """
        metagraph = self.metagraph(netuid)
        hotkey = metagraph.hotkeys[uid]  # type: ignore

        metadata = get_metadata(self, netuid, hotkey, block)
        commitment = metadata["info"]["fields"][0]  # type: ignore
        hex_data = commitment[list(commitment.keys())[0]][2:]  # type: ignore

        return bytes.fromhex(hex_data).decode()

    # Community uses this via `bittensor.utils.weight_utils.process_weights_for_netuid` function.
    def min_allowed_weights(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Returns network MinAllowedWeights hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int], optional): The block number to retrieve the parameter from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[int]: The value of the MinAllowedWeights hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not found.
        """
        call = self._get_hyperparameter(
            param_name="MinAllowedWeights", block=block, netuid=netuid
        )
        return None if call is None else int(call)

    # Community uses this via `bittensor.utils.weight_utils.process_weights_for_netuid` function.
    def max_weight_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        """
        Returns network MaxWeightsLimit hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int], optional): The block number to retrieve the parameter from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[float]: The value of the MaxWeightsLimit hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not found.
        """
        call = self._get_hyperparameter(
            param_name="MaxWeightsLimit", block=block, netuid=netuid
        )
        return None if call is None else u16_normalized_float(int(call))

    # # Community uses this method. It is used in subtensor in neuron_info, and serving.
    def get_prometheus_info(
        self, netuid: int, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[PrometheusInfo]:
        """
        Returns the prometheus information for this hotkey account.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            hotkey_ss58 (str): The SS58 address of the hotkey.
            block (Optional[int], optional): The block number to retrieve the prometheus information from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[PrometheusInfo]: A PrometheusInfo object containing the prometheus information, or ``None`` if the prometheus information is not found.
        """
        result = self.query_subtensor("Prometheus", block, [netuid, hotkey_ss58])
        if result is not None and hasattr(result, "value"):
            return PrometheusInfo(
                ip=networking.int_to_ip(result.value["ip"]),
                ip_type=result.value["ip_type"],
                port=result.value["port"],
                version=result.value["version"],
                block=result.value["block"],
            )
        return None

    # Community uses this method
    def subnet_exists(self, netuid: int, block: Optional[int] = None) -> bool:
        """
        Checks if a subnet with the specified unique identifier (netuid) exists within the Bittensor network.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number at which to check the subnet's existence.

        Returns:
            bool: ``True`` if the subnet exists, False otherwise.

        This function is critical for verifying the presence of specific subnets in the network, enabling a deeper understanding of the network's structure and composition.
        """
        _result = self.query_subtensor("NetworksAdded", block, [netuid])
        return getattr(_result, "value", False)

    # Metagraph uses this method
    def bonds(
        self, netuid: int, block: Optional[int] = None
    ) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """
        Retrieves the bond distribution set by neurons within a specific subnet of the Bittensor network. Bonds represent the investments or commitments made by neurons in one another, indicating a level of trust and perceived value. This bonding mechanism is integral to the network's market-based approach to measuring and rewarding machine intelligence.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            List[Tuple[int, List[Tuple[int, int]]]]: A list of tuples mapping each neuron's UID to its bonds with other neurons.

        Understanding bond distributions is crucial for analyzing the trust dynamics and market behavior within the subnet. It reflects how neurons recognize and invest in each other's intelligence and contributions, supporting diverse and niche systems within the Bittensor ecosystem.
        """
        b_map = []
        b_map_encoded = self.query_map_subtensor(
            name="Bonds", block=block, params=[netuid]
        )
        if b_map_encoded.records:
            for uid, b in b_map_encoded:
                b_map.append((uid.serialize(), b.serialize()))

        return b_map

    # Metagraph uses this method
    def neurons(self, netuid: int, block: Optional[int] = None) -> List[NeuronInfo]:
        """
        Retrieves a list of all neurons within a specified subnet of the Bittensor network. This function provides a snapshot of the subnet's neuron population, including each neuron's attributes and network interactions.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[NeuronInfo]: A list of NeuronInfo objects detailing each neuron's characteristics in the subnet.

        Understanding the distribution and status of neurons within a subnet is key to comprehending the network's decentralized structure and the dynamics of its consensus and governance processes.
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

    # Metagraph uses this method
    def get_total_subnets(self, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the total number of subnets within the Bittensor network as of a specific blockchain block.

        Args:
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            int: The total number of subnets in the network.

        Understanding the total number of subnets is essential for assessing the network's growth and the extent of its decentralized infrastructure.
        """
        _result = self.query_subtensor("TotalNetworks", block)
        return getattr(_result, "value", None)

    # Metagraph uses this method
    def get_subnets(self, block: Optional[int] = None) -> List[int]:
        """
        Retrieves a list of all subnets currently active within the Bittensor network. This function provides an overview of the various subnets and their identifiers.

        Args:
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[int]: A list of network UIDs representing each active subnet.

        This function is valuable for understanding the network's structure and the diversity of subnets available for neuron participation and collaboration.
        """
        result = self.query_map_subtensor("NetworksAdded", block)
        return (
            [network[0].value for network in result.records]
            if result and hasattr(result, "records")
            else []
        )

    # Metagraph uses this method
    def neurons_lite(
        self, netuid: int, block: Optional[int] = None
    ) -> List[NeuronInfoLite]:
        """
        Retrieves a list of neurons in a 'lite' format from a specific subnet of the Bittensor network. This function provides a streamlined view of the neurons, focusing on key attributes such as stake and network participation.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            List[NeuronInfoLite]: A list of simplified neuron information for the subnet.

        This function offers a quick overview of the neuron population within a subnet, facilitating efficient analysis of the network's decentralized structure and neuron dynamics.
        """
        hex_bytes_result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons_lite",
            params=[netuid],
            block=block,
        )

        if hex_bytes_result is None:
            return []

        if hex_bytes_result.startswith("0x"):
            bytes_result = bytes.fromhex(hex_bytes_result[2:])
        else:
            bytes_result = bytes.fromhex(hex_bytes_result)

        return NeuronInfoLite.list_from_vec_u8(bytes_result)  # type: ignore

    # Used in the `neurons` method which is used in metagraph.py
    def weights(
        self, netuid: int, block: Optional[int] = None
    ) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """
        Retrieves the weight distribution set by neurons within a specific subnet of the Bittensor network. This function maps each neuron's UID to the weights it assigns to other neurons, reflecting the network's trust and value assignment mechanisms.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            List[Tuple[int, List[Tuple[int, int]]]]: A list of tuples mapping each neuron's UID to its assigned weights.

        The weight distribution is a key factor in the network's consensus algorithm and the ranking of neurons, influencing their influence and reward allocation within the subnet.
        """
        w_map = []
        w_map_encoded = self.query_map_subtensor(
            name="Weights", block=block, params=[netuid]
        )
        if w_map_encoded.records:
            for uid, w in w_map_encoded:
                w_map.append((uid.serialize(), w.serialize()))

        return w_map

    # Used by community via `transfer_extrinsic`
    @_ensure_connected
    def get_balance(self, address: str, block: Optional[int] = None) -> Balance:
        """
        Retrieves the token balance of a specific address within the Bittensor network. This function queries the blockchain to determine the amount of Tao held by a given account.

        Args:
            address (str): The Substrate address in ``ss58`` format.
            block (int, optional): The blockchain block number at which to perform the query.

        Returns:
            Balance: The account balance at the specified block, represented as a Balance object.

        This function is important for monitoring account holdings and managing financial transactions within the Bittensor ecosystem. It helps in assessing the economic status and capacity of network participants.
        """
        try:

            @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
            def make_substrate_call_with_retry():
                return self.substrate.query(
                    module="System",
                    storage_function="Account",
                    params=[address],
                    block_hash=(
                        None if block is None else self.substrate.get_block_hash(block)
                    ),
                )

            result = make_substrate_call_with_retry()

        except RemainingScaleBytesNotEmptyException:
            logging.error(
                "Received a corrupted message. This likely points to an error with the network or subnet."
            )
            return Balance(1000)
        return Balance(result.value["data"]["free"])

    # Used in community via `bittensor.core.subtensor.Subtensor.transfer`
    @_ensure_connected
    def get_transfer_fee(
        self, wallet: "Wallet", dest: str, value: Union["Balance", float, int]
    ) -> "Balance":
        """
        Calculates the transaction fee for transferring tokens from a wallet to a specified destination address. This function simulates the transfer to estimate the associated cost, taking into account the current network conditions and transaction complexity.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet from which the transfer is initiated.
            dest (str): The ``SS58`` address of the destination account.
            value (Union[Balance, float, int]): The amount of tokens to be transferred, specified as a Balance object, or in Tao (float) or Rao (int) units.

        Returns:
            Balance: The estimated transaction fee for the transfer, represented as a Balance object.

        Estimating the transfer fee is essential for planning and executing token transactions, ensuring that the wallet has sufficient funds to cover both the transfer amount and the associated costs. This function provides a crucial tool for managing financial operations within the Bittensor network.
        """
        if isinstance(value, float):
            value = Balance.from_tao(value)
        elif isinstance(value, int):
            value = Balance.from_rao(value)

        if isinstance(value, Balance):
            call = self.substrate.compose_call(
                call_module="Balances",
                call_function="transfer_allow_death",
                call_params={"dest": dest, "value": value.rao},
            )

            try:
                payment_info = self.substrate.get_payment_info(
                    call=call, keypair=wallet.coldkeypub
                )
            except Exception as e:
                settings.bt_console.print(
                    f":cross_mark: [red]Failed to get payment info[/red]:[bold white]\n  {e}[/bold white]"
                )
                payment_info = {"partialFee": int(2e7)}  # assume  0.02 Tao

            fee = Balance.from_rao(payment_info["partialFee"])
            return fee
        else:
            fee = Balance.from_rao(int(2e7))
            logging.error(
                "To calculate the transaction fee, the value must be Balance, float, or int. Received type: %s. Fee "
                "is %s",
                type(value),
                2e7,
            )
            return fee

    # Used in community via `bittensor.core.subtensor.Subtensor.transfer`
    def get_existential_deposit(
        self, block: Optional[int] = None
    ) -> Optional["Balance"]:
        """
        Retrieves the existential deposit amount for the Bittensor blockchain. The existential deposit is the minimum amount of TAO required for an account to exist on the blockchain. Accounts with balances below this threshold can be reaped to conserve network resources.

        Args:
            block (Optional[int]): Block number at which to query the deposit amount. If ``None``, the current block is used.

        Returns:
            Optional[Balance]: The existential deposit amount, or ``None`` if the query fails.

        The existential deposit is a fundamental economic parameter in the Bittensor network, ensuring efficient use of storage and preventing the proliferation of dust accounts.
        """
        result = self.query_constant(
            module_name="Balances", constant_name="ExistentialDeposit", block=block
        )
        if result is None or not hasattr(result, "value"):
            return None
        return Balance.from_rao(result.value)

    # Community uses this method
    def commit_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        salt: List[int],
        uids: Union[NDArray[np.int64], list],
        weights: Union[NDArray[np.int64], list],
        version_key: int = settings.version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        prompt: bool = False,
        max_retries: int = 5,
    ) -> Tuple[bool, str]:
        """
        Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
        This action serves as a commitment or snapshot of the neuron's current weight distribution.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron committing the weights.
            netuid (int): The unique identifier of the subnet.
            salt (List[int]): list of randomly generated integers as salt to generated weighted hash.
            uids (np.ndarray): NumPy array of neuron UIDs for which weights are being committed.
            weights (np.ndarray): NumPy array of weight values corresponding to each UID.
            version_key (int, optional): Version key for compatibility with the network.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If ``True``, prompts for user confirmation before proceeding.
            max_retries (int, optional): The number of maximum attempts to commit weights. (Default: 5)

        Returns:
            Tuple[bool, str]: ``True`` if the weight commitment is successful, False otherwise. And `msg`, a string
            value describing the success or potential error.

        This function allows neurons to create a tamper-proof record of their weight distribution at a specific point in time,
        enhancing transparency and accountability within the Bittensor network.
        """
        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to commit weights!"

        logging.info(
            f"Committing weights with params: netuid={netuid}, uids={uids}, weights={weights}, version_key={version_key}"
        )

        # Generate the hash of the weights
        commit_hash = generate_weight_hash(
            address=wallet.hotkey.ss58_address,
            netuid=netuid,
            uids=list(uids),
            values=list(weights),
            salt=salt,
            version_key=version_key,
        )

        logging.info(f"Commit Hash: {commit_hash}")

        while retries < max_retries:
            try:
                success, message = commit_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    commit_hash=commit_hash,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    prompt=prompt,
                )
                if success:
                    break
            except Exception as e:
                logging.error(f"Error committing weights: {e}")
            finally:
                retries += 1

        return success, message

    # Community uses this method
    @_ensure_connected
    def _do_commit_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        commit_hash: str,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Internal method to send a transaction to the Bittensor blockchain, committing the hash of a neuron's weights.
        This method constructs and submits the transaction, handling retries and blockchain communication.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron committing the weights.
            netuid (int): The unique identifier of the subnet.
            commit_hash (str): The hash of the neuron's weights to be committed.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

        This method ensures that the weight commitment is securely recorded on the Bittensor blockchain, providing a
        verifiable record of the neuron's weight distribution at a specific point in time.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry():
            call = self.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="commit_weights",
                call_params={
                    "netuid": netuid,
                    "commit_hash": commit_hash,
                },
            )
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=wallet.hotkey,
            )
            response = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if not wait_for_finalization and not wait_for_inclusion:
                return True, None

            response.process_events()
            if response.is_success:
                return True, None
            else:
                return False, format_error_message(response.error_message)

        return make_substrate_call_with_retry()

    # Community uses this method
    def reveal_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: Union[NDArray[np.int64], list],
        weights: Union[NDArray[np.int64], list],
        salt: Union[NDArray[np.int64], list],
        version_key: int = settings.version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        prompt: bool = False,
        max_retries: int = 5,
    ) -> Tuple[bool, str]:
        """
        Reveals the weights for a specific subnet on the Bittensor blockchain using the provided wallet.
        This action serves as a revelation of the neuron's previously committed weight distribution.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron revealing the weights.
            netuid (int): The unique identifier of the subnet.
            uids (np.ndarray): NumPy array of neuron UIDs for which weights are being revealed.
            weights (np.ndarray): NumPy array of weight values corresponding to each UID.
            salt (np.ndarray): NumPy array of salt values corresponding to the hash function.
            version_key (int, optional): Version key for compatibility with the network.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
            prompt (bool, optional): If ``True``, prompts for user confirmation before proceeding.
            max_retries (int, optional): The number of maximum attempts to reveal weights. (Default: 5)

        Returns:
            Tuple[bool, str]: ``True`` if the weight revelation is successful, False otherwise. And `msg`, a string
            value describing the success or potential error.

        This function allows neurons to reveal their previously committed weight distribution, ensuring transparency
        and accountability within the Bittensor network.
        """

        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to reveal weights!"

        while retries < max_retries:
            try:
                success, message = reveal_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    uids=list(uids),
                    weights=list(weights),
                    salt=list(salt),
                    version_key=version_key,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    prompt=prompt,
                )
                if success:
                    break
            except Exception as e:
                logging.error(f"Error revealing weights: {e}")
            finally:
                retries += 1

        return success, message

    # Community uses this method
    @_ensure_connected
    def _do_reveal_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: List[int],
        values: List[int],
        salt: List[int],
        version_key: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Internal method to send a transaction to the Bittensor blockchain, revealing the weights for a specific subnet.
        This method constructs and submits the transaction, handling retries and blockchain communication.

        Args:
            wallet (bittensor.wallet): The wallet associated with the neuron revealing the weights.
            netuid (int): The unique identifier of the subnet.
            uids (List[int]): List of neuron UIDs for which weights are being revealed.
            values (List[int]): List of weight values corresponding to each UID.
            salt (List[int]): List of salt values corresponding to the hash function.
            version_key (int): Version key for compatibility with the network.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

        This method ensures that the weight revelation is securely recorded on the Bittensor blockchain, providing transparency
        and accountability for the neuron's weight distribution.
        """

        @retry(delay=1, tries=3, backoff=2, max_delay=4, logger=logging)
        def make_substrate_call_with_retry():
            call = self.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="reveal_weights",
                call_params={
                    "netuid": netuid,
                    "uids": uids,
                    "values": values,
                    "salt": salt,
                    "version_key": version_key,
                },
            )
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=wallet.hotkey,
            )
            response = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if not wait_for_finalization and not wait_for_inclusion:
                return True, None

            response.process_events()
            if response.is_success:
                return True, None
            else:
                return False, format_error_message(response.error_message)

        return make_substrate_call_with_retry()
