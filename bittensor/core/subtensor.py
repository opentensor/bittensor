"""
The ``bittensor.core.subtensor.Subtensor`` module in Bittensor serves as a crucial interface for interacting with the
Bittensor blockchain, facilitating a range of operations essential for the decentralized machine learning network.
"""

import argparse
import copy
import ssl
from typing import Union, Optional, TypedDict, Any

import numpy as np
import scalecodec
from bittensor_wallet import Wallet
from numpy.typing import NDArray
from scalecodec.base import RuntimeConfiguration
from scalecodec.exceptions import RemainingScaleBytesNotEmptyException
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.types import ScaleType
from substrateinterface.base import QueryMapResult, SubstrateInterface
from websockets.sync import client as ws_client

from bittensor.core import settings
from bittensor.core.axon import Axon
from bittensor.core.chain_data import (
    custom_rpc_type_registry,
    DelegateInfo,
    NeuronInfo,
    NeuronInfoLite,
    PrometheusInfo,
    SubnetHyperparameters,
    SubnetInfo,
)
from bittensor.core.config import Config
from bittensor.core.extrinsics.commit_weights import (
    commit_weights_extrinsic,
    reveal_weights_extrinsic,
)
from bittensor.core.extrinsics.registration import (
    burned_register_extrinsic,
    register_extrinsic,
)
from bittensor.core.extrinsics.root import (
    root_register_extrinsic,
    set_root_weights_extrinsic,
)
from bittensor.core.extrinsics.serving import (
    do_serve_axon,
    serve_axon_extrinsic,
    publish_metadata,
    get_metadata,
)
from bittensor.core.extrinsics.set_weights import set_weights_extrinsic
from bittensor.core.extrinsics.transfer import (
    transfer_extrinsic,
)
from bittensor.core.extrinsics.staking import (
    add_stake_extrinsic,
    add_stake_multiple_extrinsic,
)
from bittensor.core.extrinsics.unstaking import (
    unstake_extrinsic,
    unstake_multiple_extrinsic,
)
from bittensor.core.metagraph import Metagraph
from bittensor.utils import (
    networking,
    torch,
    ss58_to_vec_u8,
    u16_normalized_float,
    hex_to_bytes,
    Certificate,
)
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.registration import legacy_torch_api_compat
from bittensor.utils.weight_utils import generate_weight_hash

KEY_NONCE: dict[str, int] = {}


class ParamWithTypes(TypedDict):
    name: str  # Name of the parameter.
    type: str  # ScaleType string of the parameter.


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

        from bittensor.core.subtensor import Subtensor

        # Connect to the main Bittensor network (Finney).
        finney_subtensor = Subtensor(network='finney')

        # Close websocket connection with the Bittensor network.
        finney_subtensor.close()

        # Register a new neuron on the network.
        wallet = bittensor_wallet.Wallet(...)  # Assuming a wallet instance is created.
        netuid = 1
        success = finney_subtensor.register(wallet=wallet, netuid=netuid)

        # Set inter-neuronal weights for collaborative learning.
        success = finney_subtensor.set_weights(wallet=wallet, netuid=netuid, uids=[...], weights=[...])

        # Get the metagraph for a specific subnet using given subtensor connection
        metagraph = finney_subtensor.metagraph(netuid=netuid)

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
        log_verbose: bool = False,
        connection_timeout: int = 600,
        websocket: Optional[ws_client.ClientConnection] = None,
    ) -> None:
        """
        Initializes a Subtensor interface for interacting with the Bittensor blockchain.

        NOTE:
            Currently subtensor defaults to the ``finney`` network. This will change in a future release.

        We strongly encourage users to run their own local subtensor node whenever possible. This increases decentralization and resilience of the network. In a future release, local subtensor will become the default and the fallback to ``finney`` removed. Please plan ahead for this change. We will provide detailed instructions on how to run a local subtensor node in the documentation in a subsequent release.

        Args:
            network (Optional[str]): The network name to connect to (e.g., ``finney``, ``local``). This can also be the chain endpoint (e.g., ``wss://entrypoint-finney.opentensor.ai:443``) and will be correctly parsed into the network and chain endpoint. If not specified, defaults to the main Bittensor network.
            config (Optional[bittensor.core.config.Config]): Configuration object for the subtensor. If not provided, a default configuration is used.
            _mock (bool): If set to ``True``, uses a mocked connection for testing purposes. Default is ``False``.
            log_verbose (bool): Whether to enable verbose logging. If set to ``True``, detailed log information about the connection and network operations will be provided. Default is ``True``.
            connection_timeout (int): The maximum time in seconds to keep the connection alive. Default is ``600``.
            websocket (websockets.sync.client.ClientConnection): websockets sync (threading) client object connected to the network.

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
            logging.debug(
                "We strongly encourage running a local subtensor node whenever possible. "
                "This increases decentralization and resilience of the network."
            )
            logging.debug(
                "In a future release, local subtensor will become the default endpoint. "
                "To get ahead of this change, please run a local subtensor node and point to it."
            )

        self.log_verbose = log_verbose
        self._connection_timeout = connection_timeout
        self.substrate: "SubstrateInterface" = None
        self.websocket = websocket
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
        if self.substrate:
            self.substrate.close()

    def _get_substrate(self, force: bool = False):
        """
        Establishes a connection to the Substrate node using configured parameters.

        Args:
            force: forces a reconnection if this flag is set

        """
        try:
            # Set up params.
            if force and self.websocket:
                logging.debug("Closing websocket connection")
                self.websocket.close()

            if force or self.websocket is None or self.websocket.close_code is not None:
                self.websocket = ws_client.connect(
                    self.chain_endpoint,
                    open_timeout=self._connection_timeout,
                    max_size=2**32,
                )
            self.substrate = SubstrateInterface(
                ss58_format=settings.SS58_FORMAT,
                use_remote_preset=True,
                type_registry=settings.TYPE_REGISTRY,
                websocket=self.websocket,
            )
            if self.log_verbose:
                logging.info(
                    f"Connected to {self.network} network and {self.chain_endpoint}."
                )

        except (ConnectionRefusedError, ssl.SSLError) as error:
            logging.error(
                f"<red>Could not connect to</red> <blue>{self.network}</blue> <red>network with</red> <blue>{self.chain_endpoint}</blue> <red>chain endpoint.</red>",
            )
            raise ConnectionRefusedError(error.args)
        except ssl.SSLError as e:
            logging.critical(
                "SSL error occurred. To resolve this issue, run the following command in your terminal:"
            )
            logging.critical("[blue]sudo python -m bittensor certifi[/blue]")
            raise RuntimeError(
                "SSL configuration issue, please follow the instructions above."
            ) from e

    @staticmethod
    def config() -> "Config":
        """
        Creates and returns a Bittensor configuration object.

        Returns:
            config (bittensor.core.config.Config): A Bittensor configuration object configured with arguments added by the `subtensor.add_args` method.
        """
        parser = argparse.ArgumentParser()
        Subtensor.add_args(parser)
        return Config(parser)

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
            network (Optional[str]): The name of the Subtensor network. If None, the network and endpoint will be determined from the `config` object.
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
            if config.is_set("subtensor.chain_endpoint"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.chain_endpoint
                )

            elif config.is_set("subtensor.network"):
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
    @networking.ensure_connected
    def _encode_params(
        self,
        call_definition: dict[str, list["ParamWithTypes"]],
        params: Union[list[Any], dict[str, Any]],
    ) -> str:
        """Returns a hex encoded string of the params using their types."""
        param_data = scalecodec.ScaleBytes(b"")

        for i, param in enumerate(call_definition["params"]):
            scale_obj = self.substrate.create_scale_object(param["type"])
            if isinstance(params, list):
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

    # Chain calls methods ==============================================================================================
    @networking.ensure_connected
    def query_subtensor(
        self, name: str, block: Optional[int] = None, params: Optional[list] = None
    ) -> "ScaleType":
        """
        Queries named storage from the Subtensor module on the Bittensor blockchain. This function is used to retrieve specific data or parameters from the blockchain, such as stake, rank, or other neuron-specific attributes.

        Args:
            name (str): The name of the storage function to query.
            block (Optional[int]): The blockchain block number at which to perform the query.
            params (Optional[list[object]]): A list of parameters to pass to the query function.

        Returns:
            query_response (scalecodec.ScaleType): An object containing the requested data.

        This query function is essential for accessing detailed information about the network and its neurons, providing valuable insights into the state and dynamics of the Bittensor ecosystem.
        """

        return self.substrate.query(
            module="SubtensorModule",
            storage_function=name,
            params=params,
            block_hash=(
                None if block is None else self.substrate.get_block_hash(block)
            ),
        )

    @networking.ensure_connected
    def query_map_subtensor(
        self, name: str, block: Optional[int] = None, params: Optional[list] = None
    ) -> "QueryMapResult":
        """
        Queries map storage from the Subtensor module on the Bittensor blockchain. This function is designed to retrieve a map-like data structure, which can include various neuron-specific details or network-wide attributes.

        Args:
            name (str): The name of the map storage function to query.
            block (Optional[int]): The blockchain block number at which to perform the query.
            params (Optional[list[object]]): A list of parameters to pass to the query function.

        Returns:
            QueryMapResult (substrateinterface.base.QueryMapResult): An object containing the map-like data structure, or ``None`` if not found.

        This function is particularly useful for analyzing and understanding complex network structures and relationships within the Bittensor ecosystem, such as inter-neuronal connections and stake distributions.
        """
        return self.substrate.query_map(
            module="SubtensorModule",
            storage_function=name,
            params=params,
            block_hash=(
                None if block is None else self.substrate.get_block_hash(block)
            ),
        )

    def query_runtime_api(
        self,
        runtime_api: str,
        method: str,
        params: Optional[Union[list[int], dict[str, int]]],
        block: Optional[int] = None,
    ) -> Optional[str]:
        """
        Queries the runtime API of the Bittensor blockchain, providing a way to interact with the underlying runtime and retrieve data encoded in Scale Bytes format. This function is essential for advanced users who need to interact with specific runtime methods and decode complex data types.

        Args:
            runtime_api (str): The name of the runtime API to query.
            method (str): The specific method within the runtime API to call.
            params (Optional[list[ParamWithTypes]]): The parameters to pass to the method call.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            Optional[str]: The Scale Bytes encoded result from the runtime API call, or ``None`` if the call fails.

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

    @networking.ensure_connected
    def state_call(
        self, method: str, data: str, block: Optional[int] = None
    ) -> dict[Any, Any]:
        """
        Makes a state call to the Bittensor blockchain, allowing for direct queries of the blockchain's state. This function is typically used for advanced queries that require specific method calls and data inputs.

        Args:
            method (str): The method name for the state call.
            data (str): The data to be passed to the method.
            block (Optional[int]): The blockchain block number at which to perform the state call.

        Returns:
            result (dict[Any, Any]): The result of the rpc call.

        The state call function provides a more direct and flexible way of querying blockchain data, useful for specific use cases where standard queries are insufficient.
        """
        block_hash = None if block is None else self.substrate.get_block_hash(block)
        return self.substrate.rpc_request(
            method="state_call",
            params=[method, data, block_hash] if block_hash else [method, data],
        )

    @networking.ensure_connected
    def query_map(
        self,
        module: str,
        name: str,
        block: Optional[int] = None,
        params: Optional[list] = None,
    ) -> "QueryMapResult":
        """
        Queries map storage from any module on the Bittensor blockchain. This function retrieves data structures that represent key-value mappings, essential for accessing complex and structured data within the blockchain modules.

        Args:
            module (str): The name of the module from which to query the map storage.
            name (str): The specific storage function within the module to query.
            block (Optional[int]): The blockchain block number at which to perform the query.
            params (Optional[list[object]]): Parameters to be passed to the query.

        Returns:
            result (substrateinterface.base.QueryMapResult): A data structure representing the map storage if found, ``None`` otherwise.

        This function is particularly useful for retrieving detailed and structured data from various blockchain modules, offering insights into the network's state and the relationships between its different components.
        """
        return self.substrate.query_map(
            module=module,
            storage_function=name,
            params=params,
            block_hash=(
                None if block is None else self.substrate.get_block_hash(block)
            ),
        )

    @networking.ensure_connected
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
            Optional[scalecodec.ScaleType]: The value of the constant if found, ``None`` otherwise.

        Constants queried through this function can include critical network parameters such as inflation rates, consensus rules, or validation thresholds, providing a deeper understanding of the Bittensor network's operational parameters.
        """
        return self.substrate.get_constant(
            module_name=module_name,
            constant_name=constant_name,
            block_hash=(
                None if block is None else self.substrate.get_block_hash(block)
            ),
        )

    @networking.ensure_connected
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
            params (Optional[list[object]]): A list of parameters to pass to the query function.

        Returns:
            Optional[scalecodec.ScaleType]: An object containing the requested data if found, ``None`` otherwise.

        This versatile query function is key to accessing a wide range of data and insights from different parts of the Bittensor blockchain, enhancing the understanding and analysis of the network's state and dynamics.
        """
        return self.substrate.query(
            module=module,
            storage_function=name,
            params=params,
            block_hash=(
                None if block is None else self.substrate.get_block_hash(block)
            ),
        )

    # Common subtensor methods =========================================================================================
    def metagraph(
        self, netuid: int, lite: bool = True, block: Optional[int] = None
    ) -> "Metagraph":  # type: ignore
        """
        Returns a synced metagraph for a specified subnet within the Bittensor network. The metagraph represents the network's structure, including neuron connections and interactions.

        Args:
            netuid (int): The network UID of the subnet to query.
            lite (bool): If true, returns a metagraph using a lightweight sync (no weights, no bonds). Default is ``True``.
            block (Optional[int]): Block number for synchronization, or ``None`` for the latest block.

        Returns:
            bittensor.core.metagraph.Metagraph: The metagraph representing the subnet's structure and neuron relationships.

        The metagraph is an essential tool for understanding the topology and dynamics of the Bittensor network's decentralized architecture, particularly in relation to neuron interconnectivity and consensus processes.
        """
        metagraph = Metagraph(
            network=self.chain_endpoint,
            netuid=netuid,
            lite=lite,
            sync=False,
            subtensor=self,
        )
        metagraph.sync(block=block, lite=lite, subtensor=self)

        return metagraph

    @staticmethod
    def determine_chain_endpoint_and_network(
        network: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Determines the chain endpoint and network from the passed network or chain_endpoint.

        Args:
            network (str): The network flag. The choices are: ``finney`` (main network), ``archive`` (archive network +300 blocks), ``local`` (local running network), ``test`` (test network).

        Returns:
            tuple[Optional[str], Optional[str]]: The network and chain endpoint flag. If passed, overrides the ``network`` argument.
        """

        if network is None:
            return None, None
        if network in settings.NETWORKS:
            return network, settings.NETWORK_MAP[network]
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
    ) -> list[int]:
        """
        Retrieves a list of subnet UIDs (netuids) for which a given hotkey is a member. This function identifies the specific subnets within the Bittensor network where the neuron associated with the hotkey is active.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            list[int]: A list of netuids where the neuron is a member.
        """
        result = self.query_map_subtensor("IsNetworkMember", block, [hotkey_ss58])
        return (
            [record[0].value for record in result if record[1]]
            if result and hasattr(result, "records")
            else []
        )

    @networking.ensure_connected
    def get_current_block(self) -> int:
        """
        Returns the current block number on the Bittensor blockchain. This function provides the latest block number, indicating the most recent state of the blockchain.

        Returns:
            int: The current chain block number.

        Knowing the current block number is essential for querying real-time data and performing time-sensitive operations on the blockchain. It serves as a reference point for network activities and data synchronization.
        """
        return self.substrate.get_block_number(None)  # type: ignore

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

    @networking.ensure_connected
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

    def commit(self, wallet, netuid: int, data: str):
        """
        Commits arbitrary data to the Bittensor network by publishing metadata.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the data.
            netuid (int): The unique identifier of the subnetwork.
            data (str): The data to be committed to the network.
        """
        publish_metadata(self, wallet, netuid, f"Raw{len(data)}", data.encode())

    def subnetwork_n(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Returns network SubnetworkN hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int]): The block number to retrieve the parameter from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[int]: The value of the SubnetworkN hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not found.
        """
        call = self._get_hyperparameter(
            param_name="SubnetworkN", netuid=netuid, block=block
        )
        return None if call is None else int(call)

    def get_neuron_for_pubkey_and_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional["NeuronInfo"]:
        """
        Retrieves information about a neuron based on its public key (hotkey SS58 address) and the specific subnet UID (netuid). This function provides detailed neuron information for a particular subnet within the Bittensor network.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            Optional[bittensor.core.chain_data.neuron_info.NeuronInfo]: Detailed information about the neuron if found, ``None`` otherwise.

        This function is crucial for accessing specific neuron data and understanding its status, stake, and other attributes within a particular subnet of the Bittensor ecosystem.
        """
        return self.neuron_for_uid(
            self.get_uid_for_hotkey_on_subnet(hotkey_ss58, netuid, block=block),
            netuid,
            block=block,
        )

    def get_neuron_certificate(
        self, hotkey: str, netuid: int, block: Optional[int] = None
    ) -> Optional["Certificate"]:
        """
        Retrieves the TLS certificate for a specific neuron identified by its unique identifier (UID)
        within a specified subnet (netuid) of the Bittensor network.

        Args:
            hotkey (str): The hotkey to query.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int], optional): The blockchain block number for the query.

        Returns:
            Optional[Certificate]: the certificate of the neuron if found, ``None`` otherwise.

        This function is used for certificate discovery for setting up mutual tls communication between neurons
        """

        certificate = self.query_module(
            module="SubtensorModule",
            name="NeuronCertificates",
            block=block,
            params=[netuid, hotkey],
        )
        try:
            serialized_certificate = certificate.serialize()
            if serialized_certificate:
                return (
                    chr(serialized_certificate["algorithm"])
                    + serialized_certificate["public_key"]
                )
        except AttributeError:
            return None
        return None

    @networking.ensure_connected
    def neuron_for_uid(
        self, uid: Optional[int], netuid: int, block: Optional[int] = None
    ) -> "NeuronInfo":
        """
        Retrieves detailed information about a specific neuron identified by its unique identifier (UID) within a specified subnet (netuid) of the Bittensor network. This function provides a comprehensive view of a neuron's attributes, including its stake, rank, and operational status.

        Args:
            uid (Optional[int]): The unique identifier of the neuron.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            bittensor.core.chain_data.neuron_info.NeuronInfo: Detailed information about the neuron if found, ``None`` otherwise.

        This function is crucial for analyzing individual neurons' contributions and status within a specific subnet, offering insights into their roles in the network's consensus and validation mechanisms.
        """
        if uid is None:
            return NeuronInfo.get_null_neuron()

        block_hash = None if block is None else self.substrate.get_block_hash(block)
        params = [netuid, uid]
        if block_hash:
            params = params + [block_hash]

        json_body = self.substrate.rpc_request(
            method="neuronInfo_getNeuron",
            params=params,  # custom rpc method
        )

        if not (result := json_body.get("result", None)):
            return NeuronInfo.get_null_neuron()

        return NeuronInfo.from_vec_u8(result)

    def get_subnet_hyperparameters(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[Union[list, "SubnetHyperparameters"]]:
        """
        Retrieves the hyperparameters for a specific subnet within the Bittensor network. These hyperparameters define the operational settings and rules governing the subnet's behavior.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[bittensor.core.chain_data.subnet_hyperparameters.SubnetHyperparameters]: The subnet's hyperparameters, or ``None`` if not available.

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

        return SubnetHyperparameters.from_vec_u8(hex_to_bytes(hex_bytes_result))

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

    def tempo(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Returns network Tempo hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int]): The block number to retrieve the parameter from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[int]: The value of the Tempo hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not found.
        """
        call = self._get_hyperparameter(param_name="Tempo", netuid=netuid, block=block)
        return None if call is None else int(call)

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
        try:
            commitment = metadata["info"]["fields"][0]  # type: ignore
            hex_data = commitment[list(commitment.keys())[0]][2:]  # type: ignore
            return bytes.fromhex(hex_data).decode()

        except TypeError:
            return ""

    def min_allowed_weights(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Returns network MinAllowedWeights hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int]): The block number to retrieve the parameter from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[int]: The value of the MinAllowedWeights hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not found.
        """
        call = self._get_hyperparameter(
            param_name="MinAllowedWeights", block=block, netuid=netuid
        )
        return None if call is None else int(call)

    def max_weight_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        """
        Returns network MaxWeightsLimit hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int]): The block number to retrieve the parameter from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[float]: The value of the MaxWeightsLimit hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not found.
        """
        call = self._get_hyperparameter(
            param_name="MaxWeightsLimit", block=block, netuid=netuid
        )
        return None if call is None else u16_normalized_float(int(call))

    def get_prometheus_info(
        self, netuid: int, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional["PrometheusInfo"]:
        """
        Returns the prometheus information for this hotkey account.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            hotkey_ss58 (str): The SS58 address of the hotkey.
            block (Optional[int]): The block number to retrieve the prometheus information from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[bittensor.core.chain_data.prometheus_info.PrometheusInfo]: A PrometheusInfo object containing the prometheus information, or ``None`` if the prometheus information is not found.
        """
        result = self.query_subtensor("Prometheus", block, [netuid, hotkey_ss58])
        if result is not None and getattr(result, "value", None) is not None:
            return PrometheusInfo(
                ip=networking.int_to_ip(result.value["ip"]),
                ip_type=result.value["ip_type"],
                port=result.value["port"],
                version=result.value["version"],
                block=result.value["block"],
            )
        return None

    def subnet_exists(self, netuid: int, block: Optional[int] = None) -> bool:
        """
        Checks if a subnet with the specified unique identifier (netuid) exists within the Bittensor network.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number at which to check the subnet's existence.

        Returns:
            bool: ``True`` if the subnet exists, False otherwise.

        This function is critical for verifying the presence of specific subnets in the network, enabling a deeper understanding of the network's structure and composition.
        """
        _result = self.query_subtensor("NetworksAdded", block, [netuid])
        return getattr(_result, "value", False)

    @networking.ensure_connected
    def get_all_subnets_info(self, block: Optional[int] = None) -> list[SubnetInfo]:
        """
        Retrieves detailed information about all subnets within the Bittensor network. This function provides comprehensive data on each subnet, including its characteristics and operational parameters.

        Args:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            list[SubnetInfo]: A list of SubnetInfo objects, each containing detailed information about a subnet.

        Gaining insights into the subnets' details assists in understanding the network's composition, the roles of different subnets, and their unique features.
        """
        hex_bytes_result = self.query_runtime_api(
            "SubnetInfoRuntimeApi", "get_subnets_info", params=[], block=block
        )
        if not hex_bytes_result:
            return []
        else:
            return SubnetInfo.list_from_vec_u8(hex_to_bytes(hex_bytes_result))

    def bonds(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the bond distribution set by neurons within a specific subnet of the Bittensor network. Bonds represent the investments or commitments made by neurons in one another, indicating a level of trust and perceived value. This bonding mechanism is integral to the network's market-based approach to measuring and rewarding machine intelligence.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            list[tuple[int, list[tuple[int, int]]]]: A list of tuples mapping each neuron's UID to its bonds with other neurons.

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

    def get_subnet_burn_cost(self, block: Optional[int] = None) -> Optional[str]:
        """
        Retrieves the burn cost for registering a new subnet within the Bittensor network. This cost represents the amount of Tao that needs to be locked or burned to establish a new subnet.

        Args:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            int: The burn cost for subnet registration.

        The subnet burn cost is an important economic parameter, reflecting the network's mechanisms for controlling the proliferation of subnets and ensuring their commitment to the network's long-term viability.
        """
        lock_cost = self.query_runtime_api(
            runtime_api="SubnetRegistrationRuntimeApi",
            method="get_network_registration_cost",
            params=[],
            block=block,
        )

        if lock_cost is None:
            return None

        return lock_cost

    def neurons(self, netuid: int, block: Optional[int] = None) -> list["NeuronInfo"]:
        """
        Retrieves a list of all neurons within a specified subnet of the Bittensor network. This function provides a snapshot of the subnet's neuron population, including each neuron's attributes and network interactions.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            list[bittensor.core.chain_data.neuron_info.NeuronInfo]: A list of NeuronInfo objects detailing each neuron's characteristics in the subnet.

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

    def get_total_stake_for_coldkey(
        self, ss58_address: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        """Retrieves the total stake held by a coldkey across all associated hotkeys, including delegated stakes.

        Args:
            ss58_address (str): The SS58 address of the coldkey account.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            Optional[Balance]: The total stake amount held by the coldkey, or None if the query fails.
        """
        result = self.query_subtensor("TotalColdkeyStake", block, [ss58_address])
        if not hasattr(result, "value") or result is None:
            return None
        return Balance.from_rao(result.value)

    def get_total_stake_for_hotkey(
        self, ss58_address: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        """Retrieves the total stake associated with a hotkey.

        Args:
            ss58_address (str): The SS58 address of the hotkey account.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            Optional[Balance]: The total stake amount held by the hotkey, or None if the query fails.
        """
        result = self.query_subtensor("TotalHotkeyStake", block, [ss58_address])
        if not hasattr(result, "value") or result is None:
            return None
        return Balance.from_rao(result.value)

    def get_stake_for_coldkey_and_hotkey(
        self, hotkey_ss58: str, coldkey_ss58: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        """Retrieves the stake amount for a specific coldkey-hotkey pair within the Bittensor network.

        Args:
            hotkey_ss58 (str): The SS58 address of the hotkey account.
            coldkey_ss58 (str): The SS58 address of the coldkey account.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            Optional[Balance]: The stake amount for the specific coldkey-hotkey pair,
            or None if the query fails.
        """
        result = self.query_subtensor("Stake", block, [hotkey_ss58, coldkey_ss58])
        if not hasattr(result, "value") or result is None:
            return None
        return Balance.from_rao(result.value)

    def get_total_subnets(self, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the total number of subnets within the Bittensor network as of a specific blockchain block.

        Args:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The total number of subnets in the network.

        Understanding the total number of subnets is essential for assessing the network's growth and the extent of its decentralized infrastructure.
        """
        _result = self.query_subtensor("TotalNetworks", block)
        return getattr(_result, "value", None)

    def get_subnets(self, block: Optional[int] = None) -> list[int]:
        """
        Retrieves a list of all subnets currently active within the Bittensor network. This function provides an overview of the various subnets and their identifiers.

        Args:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            list[int]: A list of network UIDs representing each active subnet.

        This function is valuable for understanding the network's structure and the diversity of subnets available for neuron participation and collaboration.
        """
        result = self.query_map_subtensor("NetworksAdded", block)
        return (
            [network[0].value for network in result.records if network[1]]
            if result and hasattr(result, "records")
            else []
        )

    def neurons_lite(
        self, netuid: int, block: Optional[int] = None
    ) -> list["NeuronInfoLite"]:
        """
        Retrieves a list of neurons in a 'lite' format from a specific subnet of the Bittensor network. This function provides a streamlined view of the neurons, focusing on key attributes such as stake and network participation.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            list[bittensor.core.chain_data.neuron_info_lite.NeuronInfoLite]: A list of simplified neuron information for the subnet.

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

        return NeuronInfoLite.list_from_vec_u8(hex_to_bytes(hex_bytes_result))  # type: ignore

    def weights(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the weight distribution set by neurons within a specific subnet of the Bittensor network. This function maps each neuron's UID to the weights it assigns to other neurons, reflecting the network's trust and value assignment mechanisms.

        Args:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            list[tuple[int, list[tuple[int, int]]]]: A list of tuples mapping each neuron's UID to its assigned weights.

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

    @networking.ensure_connected
    def get_balance(self, address: str, block: Optional[int] = None) -> "Balance":
        """
        Retrieves the token balance of a specific address within the Bittensor network. This function queries the blockchain to determine the amount of Tao held by a given account.

        Args:
            address (str): The Substrate address in ``ss58`` format.
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            bittensor.utils.balance.Balance: The account balance at the specified block, represented as a Balance object.

        This function is important for monitoring account holdings and managing financial transactions within the Bittensor ecosystem. It helps in assessing the economic status and capacity of network participants.
        """
        try:
            result = self.substrate.query(
                module="System",
                storage_function="Account",
                params=[address],
                block_hash=(
                    None if block is None else self.substrate.get_block_hash(block)
                ),
            )

        except RemainingScaleBytesNotEmptyException:
            logging.error(
                "Received a corrupted message. This likely points to an error with the network or subnet."
            )
            return Balance(1000)

        return Balance(result.value["data"]["free"])

    @networking.ensure_connected
    def get_transfer_fee(
        self, wallet: "Wallet", dest: str, value: Union["Balance", float, int]
    ) -> "Balance":
        """
        Calculates the transaction fee for transferring tokens from a wallet to a specified destination address. This function simulates the transfer to estimate the associated cost, taking into account the current network conditions and transaction complexity.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet from which the transfer is initiated.
            dest (str): The ``SS58`` address of the destination account.
            value (Union[bittensor.utils.balance.Balance, float, int]): The amount of tokens to be transferred, specified as a Balance object, or in Tao (float) or Rao (int) units.

        Returns:
            bittensor.utils.balance.Balance: The estimated transaction fee for the transfer, represented as a Balance object.

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
                logging.error(f"[red]Failed to get payment info.[/red] {e}")
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

    def get_existential_deposit(
        self, block: Optional[int] = None
    ) -> Optional["Balance"]:
        """
        Retrieves the existential deposit amount for the Bittensor blockchain. The existential deposit is the minimum amount of TAO required for an account to exist on the blockchain. Accounts with balances below this threshold can be reaped to conserve network resources.

        Args:
            block (Optional[int]): Block number at which to query the deposit amount. If ``None``, the current block is used.

        Returns:
            Optional[bittensor.utils.balance.Balance]: The existential deposit amount, or ``None`` if the query fails.

        The existential deposit is a fundamental economic parameter in the Bittensor network, ensuring efficient use of storage and preventing the proliferation of dust accounts.
        """
        result = self.query_constant(
            module_name="Balances", constant_name="ExistentialDeposit", block=block
        )
        if result is None or not hasattr(result, "value"):
            return None
        return Balance.from_rao(result.value)

    def difficulty(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the 'Difficulty' hyperparameter for a specified subnet in the Bittensor network.

        This parameter is instrumental in determining the computational challenge required for neurons to participate in consensus and validation processes.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The value of the 'Difficulty' hyperparameter if the subnet exists, ``None`` otherwise.

        The 'Difficulty' parameter directly impacts the network's security and integrity by setting the computational effort required for validating transactions and participating in the network's consensus mechanism.
        """
        call = self._get_hyperparameter(
            param_name="Difficulty", netuid=netuid, block=block
        )
        if call is None:
            return None
        return int(call)

    def recycle(self, netuid: int, block: Optional[int] = None) -> Optional["Balance"]:
        """
        Retrieves the 'Burn' hyperparameter for a specified subnet. The 'Burn' parameter represents the amount of Tao that is effectively recycled within the Bittensor network.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[Balance]: The value of the 'Burn' hyperparameter if the subnet exists, None otherwise.

        Understanding the 'Burn' rate is essential for analyzing the network registration usage, particularly how it is correlated with user activity and the overall cost of participation in a given subnet.
        """
        call = self._get_hyperparameter(param_name="Burn", netuid=netuid, block=block)
        return None if call is None else Balance.from_rao(int(call))

    def get_delegate_take(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[float]:
        """
        Retrieves the delegate 'take' percentage for a neuron identified by its hotkey. The 'take' represents the percentage of rewards that the delegate claims from its nominators' stakes.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[float]: The delegate take percentage, None if not available.

        The delegate take is a critical parameter in the network's incentive structure, influencing the distribution of rewards among neurons and their nominators.
        """
        _result = self.query_subtensor("Delegates", block, [hotkey_ss58])
        return (
            None
            if getattr(_result, "value", None) is None
            else u16_normalized_float(_result.value)
        )

    @networking.ensure_connected
    def get_delegate_by_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[DelegateInfo]:
        """
        Retrieves detailed information about a delegate neuron based on its hotkey. This function provides a comprehensive view of the delegate's status, including its stakes, nominators, and reward distribution.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the delegate's hotkey.
            block (Optional[int]): The blockchain block number for the query. Default is ``None``.

        Returns:
            Optional[DelegateInfo]: Detailed information about the delegate neuron, ``None`` if not found.

        This function is essential for understanding the roles and influence of delegate neurons within the Bittensor network's consensus and governance structures.
        """
        encoded_hotkey = ss58_to_vec_u8(hotkey_ss58)

        block_hash = None if block is None else self.substrate.get_block_hash(block)

        json_body = self.substrate.rpc_request(
            method="delegateInfo_getDelegate",  # custom rpc method
            params=([encoded_hotkey, block_hash] if block_hash else [encoded_hotkey]),
        )

        if not (result := json_body.get("result", None)):
            return None

        return DelegateInfo.from_vec_u8(bytes(result))

    def get_stake_for_coldkey_and_hotkey(
        self, hotkey_ss58: str, coldkey_ss58: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        """
        Returns the stake under a coldkey - hotkey pairing.

        Args:
            hotkey_ss58 (str): The SS58 address of the hotkey.
            coldkey_ss58 (str): The SS58 address of the coldkey.
            block (Optional[int]): The block number to retrieve the stake from. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[Balance]: The stake under the coldkey - hotkey pairing, or ``None`` if the pairing does not exist or the stake is not found.
        """
        result = self.query_subtensor("Stake", block, [hotkey_ss58, coldkey_ss58])
        return (
            None
            if getattr(result, "value", None) is None
            else Balance.from_rao(result.value)
        )

    def does_hotkey_exist(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        """
        Returns true if the hotkey is known by the chain and there are accounts.

        Args:
            hotkey_ss58 (str): The SS58 address of the hotkey.
            block (Optional[int]): The block number to check the hotkey against. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            bool: ``True`` if the hotkey is known by the chain and there are accounts, ``False`` otherwise.
        """
        result = self.query_subtensor("Owner", block, [hotkey_ss58])
        return (
            False
            if getattr(result, "value", None) is None
            else result.value != "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        )

    def get_hotkey_owner(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[str]:
        """
        Returns the coldkey owner of the passed hotkey.

        Args:
            hotkey_ss58 (str): The SS58 address of the hotkey.
            block (Optional[int]): The block number to check the hotkey owner against. If ``None``, the latest block is used. Default is ``None``.

        Returns:
            Optional[str]: The SS58 address of the coldkey owner, or ``None`` if the hotkey does not exist or the owner is not found.
        """
        result = self.query_subtensor("Owner", block, [hotkey_ss58])
        return (
            None
            if getattr(result, "value", None) is None
            or not self.does_hotkey_exist(hotkey_ss58, block)
            else result.value
        )

    @networking.ensure_connected
    def get_minimum_required_stake(
        self,
    ) -> Balance:
        """
        Returns the minimum required stake for nominators in the Subtensor network.

        This method retries the substrate call up to three times with exponential backoff in case of failures.

        Returns:
            Balance: The minimum required stake as a Balance object.

        Raises:
            Exception: If the substrate call fails after the maximum number of retries.
        """

        result = self.substrate.query(
            module="SubtensorModule", storage_function="NominatorMinRequiredStake"
        )
        return Balance.from_rao(result.decode())

    def tx_rate_limit(self, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the transaction rate limit for the Bittensor network as of a specific blockchain block.
        This rate limit sets the maximum number of transactions that can be processed within a given time frame.

        Args:
            block (Optional[int]): The blockchain block number at which to perform the query.

        Returns:
            Optional[int]: The transaction rate limit of the network, None if not available.

        The transaction rate limit is an essential parameter for ensuring the stability and scalability of the Bittensor network. It helps in managing network load and preventing congestion, thereby maintaining efficient and timely transaction processing.
        """
        result = self.query_subtensor("TxRateLimit", block)
        return getattr(result, "value", None)

    @networking.ensure_connected
    def get_delegates(self, block: Optional[int] = None) -> list[DelegateInfo]:
        """
        Retrieves a list of all delegate neurons within the Bittensor network. This function provides an overview of the neurons that are actively involved in the network's delegation system.

        Analyzing the delegate population offers insights into the network's governance dynamics and the distribution of trust and responsibility among participating neurons.

        Args:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            list[DelegateInfo]: A list of DelegateInfo objects detailing each delegate's characteristics.

        """
        block_hash = None if block is None else self.substrate.get_block_hash(block)

        json_body = self.substrate.rpc_request(
            method="delegateInfo_getDelegates",
            params=[block_hash] if block_hash else [],
        )

        if not (result := json_body.get("result", None)):
            return []

        return DelegateInfo.list_from_vec_u8(result)

    def is_hotkey_delegate(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        """
        Determines whether a given hotkey (public key) is a delegate on the Bittensor network. This function checks if the neuron associated with the hotkey is part of the network's delegation system.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            bool: ``True`` if the hotkey is a delegate, ``False`` otherwise.

        Being a delegate is a significant status within the Bittensor network, indicating a neuron's involvement in consensus and governance processes.
        """
        return hotkey_ss58 in [
            info.hotkey_ss58 for info in self.get_delegates(block=block)
        ]

    # Extrinsics =======================================================================================================

    def set_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: Union[NDArray[np.int64], "torch.LongTensor", list],
        weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
        version_key: int = settings.version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
    ) -> tuple[bool, str]:
        """
        Sets the inter-neuronal weights for the specified neuron. This process involves specifying the influence or trust a neuron places on other neurons in the network, which is a fundamental aspect of Bittensor's decentralized learning architecture.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron setting the weights.
            netuid (int): The unique identifier of the subnet.
            uids (Union[NDArray[np.int64], torch.LongTensor, list]): The list of neuron UIDs that the weights are being set for.
            weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The corresponding weights to be set for each UID.
            version_key (int): Version key for compatibility with the network.  Default is ``int representation of Bittensor version.``.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is ``False``.
            max_retries (int): The number of maximum attempts to set weights. Default is ``5``.

        Returns:
            tuple[bool, str]: ``True`` if the setting of weights is successful, False otherwise. And `msg`, a string value describing the success or potential error.

        This function is crucial in shaping the network's collective intelligence, where each neuron's learning and contribution are influenced by the weights it sets towards others【81†source】.
        """
        uid = self.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, netuid)
        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to set weights!"
        while (
            self.blocks_since_last_update(netuid, uid) > self.weights_rate_limit(netuid)  # type: ignore
            and retries < max_retries
            and success is False
        ):
            try:
                logging.info(
                    f"Setting weights for subnet #{netuid}. Attempt {retries + 1} of {max_retries}."
                )
                success, message = set_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    uids=uids,
                    weights=weights,
                    version_key=version_key,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
            except Exception as e:
                logging.error(f"Error setting weights: {e}")
            finally:
                retries += 1

        return success, message

    @legacy_torch_api_compat
    def root_set_weights(
        self,
        wallet: "Wallet",
        netuids: Union[NDArray[np.int64], "torch.LongTensor", list],
        weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
        version_key: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Sets the weights for neurons on the root network. This action is crucial for defining the influence and interactions of neurons at the root level of the Bittensor network.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron setting the weights.
            netuids (Union[NDArray[np.int64], torch.LongTensor, list]): The list of neuron UIDs for which weights are being set.
            weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The corresponding weights to be set for each UID.
            version_key (int, optional): Version key for compatibility with the network. Default is ``0``.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block. Defaults to ``False``.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain. Defaults to ``False``.

        Returns:
            bool: ``True`` if the setting of root-level weights is successful, False otherwise.

        This function plays a pivotal role in shaping the root network's collective intelligence and decision-making processes, reflecting the principles of decentralized governance and collaborative learning in Bittensor.
        """
        return set_root_weights_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def register(
        self,
        wallet: "Wallet",
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        max_allowed_attempts: int = 3,
        output_in_place: bool = True,
        cuda: bool = False,
        dev_id: Union[list[int], int] = 0,
        tpb: int = 256,
        num_processes: Optional[int] = None,
        update_interval: Optional[int] = None,
        log_verbose: bool = False,
    ) -> bool:
        """
        Registers a neuron on the Bittensor network using the provided wallet.

        Registration is a critical step for a neuron to become an active participant in the network, enabling it to stake, set weights, and receive incentives.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron to be registered.
            netuid (int): The unique identifier of the subnet.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Defaults to `False`.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Defaults to `True`.
            max_allowed_attempts (int): Maximum number of attempts to register the wallet.
            output_in_place (bool): If true, prints the progress of the proof of work to the console in-place. Meaning the progress is printed on the same lines. Defaults to `True`.
            cuda (bool): If ``true``, the wallet should be registered using CUDA device(s). Defaults to `False`.
            dev_id (Union[List[int], int]): The CUDA device id to use, or a list of device ids. Defaults to `0` (zero).
            tpb (int): The number of threads per block (CUDA). Default to `256`.
            num_processes (Optional[int]): The number of processes to use to register. Default to `None`.
            update_interval (Optional[int]): The number of nonces to solve between updates.  Default to `None`.
            log_verbose (bool): If ``true``, the registration process will log more information.  Default to `False`.

        Returns:
            bool: ``True`` if the registration is successful, False otherwise.

        This function facilitates the entry of new neurons into the network, supporting the decentralized
        growth and scalability of the Bittensor ecosystem.
        """
        return register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            max_allowed_attempts=max_allowed_attempts,
            output_in_place=output_in_place,
            cuda=cuda,
            dev_id=dev_id,
            tpb=tpb,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )

    def root_register(
        self,
        wallet: "Wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        """
        Registers the neuron associated with the wallet on the root network. This process is integral for participating in the highest layer of decision-making and governance within the Bittensor network.

        Args:
            wallet (bittensor_wallet.wallet): The wallet associated with the neuron to be registered on the root network.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Defaults to `False`.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Defaults to `True`.

        Returns:
            bool: ``True`` if the registration on the root network is successful, False otherwise.

        This function enables neurons to engage in the most critical and influential aspects of the network's governance, signifying a high level of commitment and responsibility in the Bittensor ecosystem.
        """
        return root_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def burned_register(
        self,
        wallet: "Wallet",
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        """
        Registers a neuron on the Bittensor network by recycling TAO. This method of registration involves recycling TAO tokens, allowing them to be re-mined by performing work on the network.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron to be registered.
            netuid (int): The unique identifier of the subnet.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block. Defaults to `False`.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain. Defaults to `True`.

        Returns:
            bool: ``True`` if the registration is successful, False otherwise.
        """
        return burned_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def serve_axon(
        self,
        netuid: int,
        axon: "Axon",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        certificate: Optional[Certificate] = None,
    ) -> bool:
        """
        Registers an ``Axon`` serving endpoint on the Bittensor network for a specific neuron. This function is used to set up the Axon, a key component of a neuron that handles incoming queries and data processing tasks.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            axon (bittensor.core.axon.Axon): The Axon instance to be registered for serving.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is ``True``.

        Returns:
            bool: ``True`` if the Axon serve registration is successful, False otherwise.

        By registering an Axon, the neuron becomes an active part of the network's distributed computing infrastructure, contributing to the collective intelligence of Bittensor.
        """
        return serve_axon_extrinsic(
            self, netuid, axon, wait_for_inclusion, wait_for_finalization, certificate
        )

    _do_serve_axon = do_serve_axon

    def transfer(
        self,
        wallet: "Wallet",
        dest: str,
        amount: Union["Balance", float],
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Executes a transfer of funds from the provided wallet to the specified destination address. This function is used to move TAO tokens within the Bittensor network, facilitating transactions between neurons.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet from which funds are being transferred.
            dest (str): The destination public key address.
            amount (Union[bittensor.utils.balance.Balance, float]): The amount of TAO to be transferred.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.  Default is ``True``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.  Default is ``False``.

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
        )

    def commit_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        salt: list[int],
        uids: Union[NDArray[np.int64], list],
        weights: Union[NDArray[np.int64], list],
        version_key: int = settings.version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
    ) -> tuple[bool, str]:
        """
        Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
        This action serves as a commitment or snapshot of the neuron's current weight distribution.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the weights.
            netuid (int): The unique identifier of the subnet.
            salt (list[int]): list of randomly generated integers as salt to generated weighted hash.
            uids (np.ndarray): NumPy array of neuron UIDs for which weights are being committed.
            weights (np.ndarray): NumPy array of weight values corresponding to each UID.
            version_key (int): Version key for compatibility with the network. Default is ``int representation of Bittensor version.``.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is ``False``.
            max_retries (int): The number of maximum attempts to commit weights. Default is ``5``.

        Returns:
            tuple[bool, str]: ``True`` if the weight commitment is successful, False otherwise. And `msg`, a string value describing the success or potential error.

        This function allows neurons to create a tamper-proof record of their weight distribution at a specific point in time, enhancing transparency and accountability within the Bittensor network.
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
                )
                if success:
                    break
            except Exception as e:
                logging.error(f"Error committing weights: {e}")
            finally:
                retries += 1

        return success, message

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
        max_retries: int = 5,
    ) -> tuple[bool, str]:
        """
        Reveals the weights for a specific subnet on the Bittensor blockchain using the provided wallet.
        This action serves as a revelation of the neuron's previously committed weight distribution.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron revealing the weights.
            netuid (int): The unique identifier of the subnet.
            uids (np.ndarray): NumPy array of neuron UIDs for which weights are being revealed.
            weights (np.ndarray): NumPy array of weight values corresponding to each UID.
            salt (np.ndarray): NumPy array of salt values corresponding to the hash function.
            version_key (int): Version key for compatibility with the network. Default is ``int representation of Bittensor version``.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is ``False``.
            max_retries (int): The number of maximum attempts to reveal weights. Default is ``5``.

        Returns:
            tuple[bool, str]: ``True`` if the weight revelation is successful, False otherwise. And `msg`, a string value describing the success or potential error.

        This function allows neurons to reveal their previously committed weight distribution, ensuring transparency and accountability within the Bittensor network.
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
                )
                if success:
                    break
            except Exception as e:
                logging.error(f"Error revealing weights: {e}")
            finally:
                retries += 1

        return success, message

    def add_stake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Optional[Union["Balance", float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Adds the specified amount of stake to a neuron identified by the hotkey ``SS58`` address.
        Staking is a fundamental process in the Bittensor network that enables neurons to participate actively and earn incentives.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet to be used for staking.
            hotkey_ss58 (Optional[str]): The ``SS58`` address of the hotkey associated with the neuron.
            amount (Union[Balance, float]): The amount of TAO to stake.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            bool: ``True`` if the staking is successful, False otherwise.

        This function enables neurons to increase their stake in the network, enhancing their influence and potential rewards in line with Bittensor's consensus and reward mechanisms.
        """
        return add_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def add_stake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        amounts: Optional[list[Union["Balance", float]]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Adds stakes to multiple neurons identified by their hotkey SS58 addresses.
        This bulk operation allows for efficient staking across different neurons from a single wallet.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet used for staking.
            hotkey_ss58s (list[str]): List of ``SS58`` addresses of hotkeys to stake to.
            amounts (list[Union[Balance, float]]): Corresponding amounts of TAO to stake for each hotkey.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            bool: ``True`` if the staking is successful for all specified neurons, False otherwise.

        This function is essential for managing stakes across multiple neurons, reflecting the dynamic and collaborative nature of the Bittensor network.
        """
        return add_stake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def unstake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Optional[Union["Balance", float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting individual neuron stakes within the Bittensor network.

        Args:
            wallet (bittensor_wallet.wallet): The wallet associated with the neuron from which the stake is being removed.
            hotkey_ss58 (Optional[str]): The ``SS58`` address of the hotkey account to unstake from.
            amount (Union[Balance, float]): The amount of TAO to unstake. If not specified, unstakes all.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            bool: ``True`` if the unstaking process is successful, False otherwise.

        This function supports flexible stake management, allowing neurons to adjust their network participation and potential reward accruals.
        """
        return unstake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def unstake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        amounts: Optional[list[Union["Balance", float]]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Performs batch unstaking from multiple hotkey accounts, allowing a neuron to reduce its staked amounts efficiently. This function is useful for managing the distribution of stakes across multiple neurons.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet linked to the coldkey from which the stakes are being withdrawn.
            hotkey_ss58s (List[str]): A list of hotkey ``SS58`` addresses to unstake from.
            amounts (List[Union[Balance, float]]): The amounts of TAO to unstake from each hotkey. If not provided, unstakes all available stakes.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            bool: ``True`` if the batch unstaking is successful, False otherwise.

        This function allows for strategic reallocation or withdrawal of stakes, aligning with the dynamic stake management aspect of the Bittensor network.
        """
        return unstake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
