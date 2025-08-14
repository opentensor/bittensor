import asyncio
import copy
import ssl
from datetime import datetime, timezone
from functools import partial
from typing import cast, Optional, Any, Union, Iterable, TYPE_CHECKING

import asyncstdlib as a
import numpy as np
import scalecodec
from async_substrate_interface import AsyncSubstrateInterface
from async_substrate_interface.substrate_addons import RetryAsyncSubstrate
from async_substrate_interface.utils.storage import StorageKey
from bittensor_drand import get_encrypted_commitment
from bittensor_wallet.utils import SS58_FORMAT
from numpy.typing import NDArray
from scalecodec import GenericCall

from bittensor.core.chain_data import (
    DelegateInfo,
    DynamicInfo,
    MetagraphInfo,
    NeuronInfoLite,
    NeuronInfo,
    ProposalVoteData,
    SelectiveMetagraphIndex,
    StakeInfo,
    SubnetHyperparameters,
    SubnetIdentity,
    SubnetInfo,
    WeightCommitInfo,
    decode_account_id,
)
from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.delegate_info import DelegatedInfo
from bittensor.core.chain_data.utils import (
    decode_block,
    decode_metadata,
    decode_revealed_commitment,
    decode_revealed_commitment_with_hotkey,
)
from bittensor.core.config import Config
from bittensor.core.errors import ChainError, SubstrateRequestException
from bittensor.core.extrinsics.asyncex.children import (
    root_set_pending_childkey_cooldown_extrinsic,
    set_children_extrinsic,
)
from bittensor.core.extrinsics.asyncex.commit_reveal import commit_reveal_v3_extrinsic
from bittensor.core.extrinsics.asyncex.move_stake import (
    transfer_stake_extrinsic,
    swap_stake_extrinsic,
    move_stake_extrinsic,
)
from bittensor.core.extrinsics.asyncex.registration import (
    burned_register_extrinsic,
    register_extrinsic,
    register_subnet_extrinsic,
    set_subnet_identity_extrinsic,
)
from bittensor.core.extrinsics.asyncex.root import (
    set_root_weights_extrinsic,
    root_register_extrinsic,
)
from bittensor.core.extrinsics.asyncex.serving import (
    get_last_bonds_reset,
    publish_metadata,
    get_metadata,
)
from bittensor.core.extrinsics.asyncex.serving import serve_axon_extrinsic
from bittensor.core.extrinsics.asyncex.staking import (
    add_stake_extrinsic,
    add_stake_multiple_extrinsic,
)
from bittensor.core.extrinsics.asyncex.start_call import start_call_extrinsic
from bittensor.core.extrinsics.asyncex.take import (
    decrease_take_extrinsic,
    increase_take_extrinsic,
)
from bittensor.core.extrinsics.asyncex.transfer import transfer_extrinsic
from bittensor.core.extrinsics.asyncex.unstaking import (
    unstake_all_extrinsic,
    unstake_extrinsic,
    unstake_multiple_extrinsic,
)
from bittensor.core.extrinsics.asyncex.weights import (
    commit_weights_extrinsic,
    set_weights_extrinsic,
    reveal_weights_extrinsic,
)
from bittensor.core.metagraph import AsyncMetagraph
from bittensor.core.settings import version_as_int, TYPE_REGISTRY
from bittensor.core.types import ParamWithTypes, SubtensorMixin
from bittensor.utils import (
    Certificate,
    decode_hex_identity_dict,
    format_error_message,
    is_valid_ss58_address,
    torch,
    u16_normalized_float,
    u64_normalized_float,
    get_transfer_fn_params,
)
from bittensor.core.extrinsics.asyncex.liquidity import (
    add_liquidity_extrinsic,
    modify_liquidity_extrinsic,
    remove_liquidity_extrinsic,
    toggle_user_liquidity_extrinsic,
)
from bittensor.utils.balance import (
    Balance,
    fixed_to_float,
    check_and_convert_to_balance,
)
from bittensor.utils import deprecated_message
from bittensor.utils.btlogging import logging
from bittensor.utils.liquidity import (
    calculate_fees,
    get_fees,
    tick_to_price,
    price_to_tick,
    LiquidityPosition,
)
from bittensor.utils.weight_utils import (
    generate_weight_hash,
    convert_uids_and_weights,
    U16_MAX,
)

if TYPE_CHECKING:
    from async_substrate_interface.types import ScaleObj
    from bittensor_wallet import Wallet
    from bittensor.core.axon import Axon
    from async_substrate_interface import AsyncQueryMapResult


class AsyncSubtensor(SubtensorMixin):
    """Asynchronous interface for interacting with the Bittensor blockchain.

    This class provides a thin layer over the Substrate Interface, offering a collection of frequently-used calls for
    querying blockchain data, managing stakes, registering neurons, and interacting with the Bittensor network.


    """

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional["Config"] = None,
        log_verbose: bool = False,
        fallback_endpoints: Optional[list[str]] = None,
        retry_forever: bool = False,
        _mock: bool = False,
        archive_endpoints: Optional[list[str]] = None,
        websocket_shutdown_timer: float = 5.0,
    ):
        """Initializes an AsyncSubtensor instance for blockchain interaction.

        Arguments:
            network: The network name or type to connect to (e.g., "finney", "test"). If ``None``, uses the default
                network from config.
            config: Configuration object for the AsyncSubtensor instance. If ``None``, uses the default configuration.
            log_verbose: Enables or disables verbose logging. Defaults to ``False``.
            fallback_endpoints: List of fallback endpoints to use if default or provided network is not available.
                Defaults to ``None``.
            retry_forever: Whether to retry forever on connection errors. Defaults to ``False``.
            _mock: Whether this is a mock instance. Mainly for testing purposes. Defaults to ``False``.
            archive_endpoints: Similar to fallback_endpoints, but specifically only archive nodes. Will be used in
                cases where you are requesting a block that is too old for your current (presumably lite) node.
                Defaults to ``None``.
            websocket_shutdown_timer: Amount of time, in seconds, to wait after the last response from the chain to
                close the connection. Defaults to ``5.0``.
        Returns:
            None

        Raises:
            ConnectionError: If unable to connect to the specified network.
            ValueError: If invalid network or configuration parameters are provided.
            Exception: Any other exceptions raised during setup or configuration.

        Typical usage example:

            import bittensor as bt
            import asyncio

            async def main():
                async with bt.AsyncSubtensor(network="finney") as subtensor:
                    block_hash = await subtensor.get_block_hash()

            asyncio.run(main())
        """
        if config is None:
            config = AsyncSubtensor.config()
        self._config = copy.deepcopy(config)
        self.chain_endpoint, self.network = AsyncSubtensor.setup_config(
            network, self._config
        )

        self.log_verbose = log_verbose
        self._check_and_log_network_settings()

        logging.debug(
            f"Connecting to network: [blue]{self.network}[/blue], "
            f"chain_endpoint: [blue]{self.chain_endpoint}[/blue]..."
        )
        self.substrate = self._get_substrate(
            fallback_endpoints=fallback_endpoints,
            retry_forever=retry_forever,
            _mock=_mock,
            archive_endpoints=archive_endpoints,
            ws_shutdown_timer=websocket_shutdown_timer,
        )
        if self.log_verbose:
            logging.info(
                f"Connected to {self.network} network and {self.chain_endpoint}."
            )

    async def close(self):
        """Closes the connection to the blockchain.

        Use this to explicitly clean up resources and close the network connection instead of waiting for garbage
        collection.

        Returns:
            None

        Example:
            subtensor = AsyncSubtensor(network="finney")
            await subtensor.initialize()

            # Use the subtensor...
            balance = await subtensor.get_balance(address="5F...")

            # Close when done
            await subtensor.close()
        """
        if self.substrate:
            await self.substrate.close()

    async def initialize(self):
        """Initializes the connection to the blockchain.

        This method establishes the connection to the Bittensor blockchain and should be called after creating an
        AsyncSubtensor instance before making any queries.

        Returns:
            AsyncSubtensor: The initialized instance (self) for method chaining.

        Raises:
            ConnectionError: If unable to connect to the blockchain due to timeout or connection refusal.

        Example:
            subtensor = AsyncSubtensor(network="finney")

            # Initialize the connection
            await subtensor.initialize()

            # Now you can make queries
            balance = await subtensor.get_balance(address="5F...")

            # Or chain the initialization
            subtensor = await AsyncSubtensor(network="finney").initialize()
        """
        logging.info(
            f"[magenta]Connecting to Substrate:[/magenta] [blue]{self}[/blue][magenta]...[/magenta]"
        )
        try:
            await self.substrate.initialize()
            return self
        except TimeoutError:
            logging.error(
                f"[red]Error[/red]: Timeout occurred connecting to substrate."
                f" Verify your chain and network settings: {self}"
            )
            raise ConnectionError
        except (ConnectionRefusedError, ssl.SSLError) as error:
            logging.error(
                f"[red]Error[/red]: Connection refused when connecting to substrate. "
                f"Verify your chain and network settings: {self}. Error: {error}"
            )
            raise ConnectionError

    async def __aenter__(self):
        logging.info(
            f"[magenta]Connecting to Substrate:[/magenta] [blue]{self}[/blue][magenta]...[/magenta]"
        )
        try:
            await self.substrate.initialize()
            return self
        except TimeoutError:
            logging.error(
                f"[red]Error[/red]: Timeout occurred connecting to substrate."
                f" Verify your chain and network settings: {self}"
            )
            raise ConnectionError
        except (ConnectionRefusedError, ssl.SSLError) as error:
            logging.error(
                f"[red]Error[/red]: Connection refused when connecting to substrate. "
                f"Verify your chain and network settings: {self}. Error: {error}"
            )
            raise ConnectionError

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.substrate.close()

    async def determine_block_hash(
        self,
        block: Optional[int],
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[str]:
        """Determine the appropriate block hash based on the provided parameters.

        Ensures that only one of the block specification parameters is used and returns the appropriate block hash
        for blockchain queries.

        Arguments:
            block: The block number to get the hash for. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the blockchain block. Do not specify if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block or reuse_block.

        Returns:
            Optional[str]: The block hash if one can be determined, None otherwise.

        Raises:
            ValueError: If more than one of block, block_hash, or reuse_block is specified.

        Example:
            # Get hash for specific block
            block_hash = await subtensor.determine_block_hash(block=1000000)

            # Use provided block hash
            hash = await subtensor.determine_block_hash(block_hash="0x1234...")

            # Reuse last block hash
            hash = await subtensor.determine_block_hash(reuse_block=True)
        """
        # Ensure that only one of the parameters is specified.
        if sum(bool(x) for x in [block, block_hash, reuse_block]) > 1:
            raise ValueError(
                "Only one of ``block``, ``block_hash``, or ``reuse_block`` can be specified."
            )

        # Return the appropriate value.
        if block_hash:
            return block_hash
        if block:
            return await self.get_block_hash(block)
        return None

    async def encode_params(
        self,
        call_definition: dict[str, list["ParamWithTypes"]],
        params: Union[list[Any], dict[str, Any]],
    ) -> str:
        """Encodes parameters into a hex string using their type definitions.

        This method takes a call definition (which specifies parameter types) and actual parameter values, then
        encodes them into a hex string that can be used for blockchain transactions.

        Arguments:
            call_definition: A dictionary containing parameter type definitions. Should have a "params" key with a
                list of parameter definitions.
            params: The actual parameter values to encode. Can be either a list (for positional parameters) or a
                dictionary (for named parameters).

        Returns:
            str: A hex-encoded string representation of the parameters.

        Raises:
            ValueError: If a required parameter is missing from the params dictionary.

        Example:
            # Define parameter types
            call_def = {
                "params": [
                    {"name": "amount", "type": "u64"},
                    {"name": "coldkey_ss58", "type": "str"}
                ]
            }

            # Encode parameters as a dictionary
            params_dict = {
                "amount": 1000000,
                "coldkey_ss58": "5F..."
            }
            encoded = await subtensor.encode_params(call_definition=call_def, params=params_dict)

            # Or encode as a list (positional)
            params_list = [1000000, "5F..."]
            encoded = await subtensor.encode_params(call_definition=call_def, params=params_list)
        """
        param_data = scalecodec.ScaleBytes(b"")

        for i, param in enumerate(call_definition["params"]):
            scale_obj = await self.substrate.create_scale_object(param["type"])
            if isinstance(params, list):
                param_data += scale_obj.encode(params[i])
            else:
                if param["name"] not in params:
                    raise ValueError(f"Missing param {param['name']} in params dict.")

                param_data += scale_obj.encode(params[param["name"]])

        return param_data.to_hex()

    async def get_hyperparameter(
        self,
        param_name: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Any]:
        """Retrieves a specified hyperparameter for a specific subnet.

        This method queries the blockchain for subnet-specific hyperparameters such as difficulty, tempo, immunity
        period, and other network configuration values.

        Arguments:
            param_name: The name of the hyperparameter to retrieve (e.g., "Difficulty", "Tempo", "ImmunityPeriod").
            netuid: The unique identifier of the subnet.
            block: The block number at which to retrieve the hyperparameter. Do not specify if using block_hash or
                reuse_block.
            block_hash: The hash of the blockchain block for the query. Do not specify if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The value of the specified hyperparameter if the subnet exists, None otherwise.

        Example:
            # Get difficulty for subnet 1
            difficulty = await subtensor.get_hyperparameter(param_name="Difficulty", netuid=1)

            # Get tempo at a specific block
            tempo = await subtensor.get_hyperparameter(param_name="Tempo", netuid=1, block=1000000)

            # Get immunity period using block hash
            immunity = await subtensor.get_hyperparameter(param_name="ImmunityPeriod", netuid=1, block_hash="0x1234...")
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not await self.subnet_exists(
            netuid, block_hash=block_hash, reuse_block=reuse_block
        ):
            logging.error(f"subnet {netuid} does not exist")
            return None

        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function=param_name,
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        return getattr(result, "value", result)

    def _get_substrate(
        self,
        fallback_endpoints: Optional[list[str]] = None,
        retry_forever: bool = False,
        _mock: bool = False,
        archive_endpoints: Optional[list[str]] = None,
        ws_shutdown_timer: float = 5.0,
    ) -> Union[AsyncSubstrateInterface, RetryAsyncSubstrate]:
        """Creates the Substrate instance based on provided arguments.

        This internal method creates either a standard AsyncSubstrateInterface or a RetryAsyncSubstrate depending on
         the configuration parameters.

        Arguments:
            fallback_endpoints: List of fallback endpoints to use if default or provided network is not available.
                Defaults to ``None``.
            retry_forever: Whether to retry forever on connection errors. Defaults to ``False``.
            _mock: Whether this is a mock instance. Mainly for testing purposes. Defaults to ``False``.
            archive_endpoints: Similar to fallback_endpoints, but specifically only archive nodes. Will be used in
                cases where you are requesting a block that is too old for your current (presumably lite) node. Defaults
                to ``None``.
            ws_shutdown_timer: Amount of time, in seconds, to wait after the last response from the chain to close the
                connection.

        Returns:
            Either AsyncSubstrateInterface or RetryAsyncSubstrate.
        """
        if fallback_endpoints or retry_forever or archive_endpoints:
            return RetryAsyncSubstrate(
                url=self.chain_endpoint,
                fallback_chains=fallback_endpoints,
                ss58_format=SS58_FORMAT,
                type_registry=TYPE_REGISTRY,
                retry_forever=retry_forever,
                use_remote_preset=True,
                chain_name="Bittensor",
                _mock=_mock,
                archive_nodes=archive_endpoints,
                ws_shutdown_timer=ws_shutdown_timer,
            )
        return AsyncSubstrateInterface(
            url=self.chain_endpoint,
            ss58_format=SS58_FORMAT,
            type_registry=TYPE_REGISTRY,
            use_remote_preset=True,
            chain_name="Bittensor",
            _mock=_mock,
            ws_shutdown_timer=ws_shutdown_timer,
        )

    # Subtensor queries ===========================================================================================

    async def query_constant(
        self,
        module_name: str,
        constant_name: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["ScaleObj"]:
        """Retrieves a constant from the specified module on the Bittensor blockchain.

        This function is used to access fixed values defined within the blockchain's modules, which are essential for
        understanding the network's configuration and rules. These include include critical network parameters such as
        inflation rates, consensus rules, or validation thresholds, providing a deeper understanding of the Bittensor
        network's operational parameters.

        Arguments:
            module_name: The name of the module containing the constant (e.g., "Balances", "SubtensorModule").
            constant_name: The name of the constant to retrieve (e.g., "ExistentialDeposit").
            block: The blockchain block number at which to query the constant. Do not specify if using block_hash or
                reuse_block.
            block_hash: The hash of the blockchain block at which to query the constant. Do not specify if using
                block or reuse_block.
            reuse_block: Whether to reuse the blockchain block at which to query the constant. Defaults to ``False``.

        Returns:
            Optional[async_substrate_interface.types.ScaleObj]: The value of the constant if found, ``None`` otherwise.

        Example:
            # Get existential deposit constant
            existential_deposit = await subtensor.query_constant(
                module_name="Balances",
                constant_name="ExistentialDeposit"
            )

            # Get constant at specific block
            constant = await subtensor.query_constant(
                module_name="SubtensorModule",
                constant_name="SomeConstant",
                block=1000000
            )
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.get_constant(
            module_name=module_name,
            constant_name=constant_name,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def query_map(
        self,
        module: str,
        name: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
        params: Optional[list] = None,
    ) -> "AsyncQueryMapResult":
        """Queries map storage from any module on the Bittensor blockchain.

        This function retrieves data structures that represent key-value mappings, essential for accessing complex and
          structured data within the blockchain modules.

        Arguments:
            module: The name of the module from which to query the map storage (e.g., "SubtensorModule", "System").
            name: The specific storage function within the module to query (e.g., "Bonds", "Weights").
            block: The blockchain block number at which to perform the query. Defaults to ``None`` (latest block).
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block. Defaults to
                ``False``.
            params: Parameters to be passed to the query. Defaults to ``None``.

        Returns:
            AsyncQueryMapResult: A data structure representing the map storage if found, None otherwise.

        Example:
            # Query bonds for subnet 1
            bonds = await subtensor.query_map(module="SubtensorModule", name="Bonds", params=[1])

            # Query weights at specific block
            weights = await subtensor.query_map(module="SubtensorModule", name="Weights", params=[1], block=1000000)
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query_map(
            module=module,
            storage_function=name,
            params=params,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return result

    async def query_map_subtensor(
        self,
        name: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
        params: Optional[list] = None,
    ) -> "AsyncQueryMapResult":
        """Queries map storage from the Subtensor module on the Bittensor blockchain. This function is designed to
        retrieve a map-like data structure, which can include various neuron-specific details or network-wide
        attributes.

        Arguments:
            name: The name of the map storage function to query.
            block: The blockchain block number at which to perform the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.
            params: A list of parameters to pass to the query function.

        Returns:
            An object containing the map-like data structure, or ``None`` if not found.

        This function is particularly useful for analyzing and understanding complex network structures and
        relationships within the Bittensor ecosystem, such as interneuronal connections and stake distributions.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.query_map(
            module="SubtensorModule",
            storage_function=name,
            params=params,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def query_module(
        self,
        module: str,
        name: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
        params: Optional[list] = None,
    ) -> Optional[Union["ScaleObj", Any]]:
        """Queries any module storage on the Bittensor blockchain with the specified parameters and block number.
        This function is a generic query interface that allows for flexible and diverse data retrieval from various
        blockchain modules.

        Arguments:
            module: The name of the module from which to query data.
            name: The name of the storage function within the module.
            block: The blockchain block number at which to perform the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.
            params: A list of parameters to pass to the query function.

        Returns:
            An object containing the requested data if found, ``None`` otherwise.

        This versatile query function is key to accessing a wide range of data and insights from different parts of the
        Bittensor blockchain, enhancing the understanding and analysis of the network's state and dynamics.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.query(
            module=module,
            storage_function=name,
            params=params,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def query_runtime_api(
        self,
        runtime_api: str,
        method: str,
        params: Optional[Union[list[Any], dict[str, Any]]],
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Any]:
        """Queries the runtime API of the Bittensor blockchain, providing a way to interact with the underlying runtime
        and retrieve data encoded in Scale Bytes format. This function is essential for advanced users who need to
        interact with specific runtime methods and decode complex data types.

        Arguments:
            runtime_api: The name of the runtime API to query.
            method: The specific method within the runtime API to call.
            params: The parameters to pass to the method call.
            block: the block number for this query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the blockchain block number at which to perform the query. Do not specify if using
                block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The decoded result from the runtime API call, or ``None`` if the call fails.

        This function enables access to the deeper layers of the Bittensor blockchain, allowing for detailed and
        specific interactions with the network's runtime environment.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash
        result = await self.substrate.runtime_call(
            runtime_api, method, params, block_hash
        )
        return result.value

    async def query_subtensor(
        self,
        name: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
        params: Optional[list] = None,
    ) -> Optional[Union["ScaleObj", Any]]:
        """Queries named storage from the Subtensor module on the Bittensor blockchain. This function is used to
        retrieve specific data or parameters from the blockchain, such as stake, rank, or other neuron-specific
        attributes.

        Arguments:
            name: The name of the storage function to query.
            block: The blockchain block number at which to perform the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.
            params: A list of parameters to pass to the query function.

        Returns:
            query_response: An object containing the requested data.

        This query function is essential for accessing detailed information about the network and its neurons, providing
        valuable insights into the state and dynamics of the Bittensor ecosystem.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.query(
            module="SubtensorModule",
            storage_function=name,
            params=params,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def state_call(
        self,
        method: str,
        data: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[Any, Any]:
        """Makes a state call to the Bittensor blockchain, allowing for direct queries of the blockchain's state.
        This function is typically used for advanced queries that require specific method calls and data inputs.

        Arguments:
            method: The method name for the state call.
            data: The data to be passed to the method.
            block: The blockchain block number at which to perform the state call.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            result (dict[Any, Any]): The result of the rpc call.

        The state call function provides a more direct and flexible way of querying blockchain data, useful for specific
        use cases where standard queries are insufficient.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.rpc_request(
            method="state_call",
            params=[method, data],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    # Common subtensor methods =========================================================================================

    @property
    async def block(self):
        """Provides an asynchronous property to retrieve the current block."""
        return await self.get_current_block()

    async def all_subnets(
        self,
        block_number: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[list[DynamicInfo]]:
        """Queries the blockchain for comprehensive information about all subnets, including their dynamic parameters
        and operational status.

        Arguments:
            block_number: The block number to query the subnet information from. Do not specify if using block_hash or
                reuse_block.
            block_hash: The hash of the blockchain block number for the query. Do not specify if using reuse_block or
                block.
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            Optional[list[DynamicInfo]]: A list of DynamicInfo objects, each containing detailed information about a
            subnet, or None if the query fails.

        Example:
            # Get all subnets at current block
            subnets = await subtensor.all_subnets()
        """
        block_hash = await self.determine_block_hash(
            block=block_number, block_hash=block_hash, reuse_block=reuse_block
        )
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash

        query, subnet_prices = await asyncio.gather(
            self.substrate.runtime_call(
                api="SubnetInfoRuntimeApi",
                method="get_all_dynamic_info",
                block_hash=block_hash,
            ),
            self.get_subnet_prices(block_hash=block_hash),
            return_exceptions=True,
        )

        decoded = query.decode()

        if not isinstance(subnet_prices, (SubstrateRequestException, ValueError)):
            for sn in decoded:
                sn.update(
                    {"price": subnet_prices.get(sn["netuid"], Balance.from_tao(0))}
                )
        else:
            logging.warning(
                f"Unable to fetch subnet prices for block {block_number}, block hash {block_hash}: {subnet_prices}"
            )
        return DynamicInfo.list_from_dicts(decoded)

    async def blocks_since_last_step(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Queries the blockchain to determine how many blocks have passed since the last epoch step for a specific
        subnet.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The block number for this query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the blockchain block number for the query. Do not specify if using reuse_block or
                block.
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            The number of blocks since the last step in the subnet, or None if the query fails.

        Example:
            # Get blocks since last step for subnet 1
            blocks = await subtensor.blocks_since_last_step(netuid=1)

            # Get blocks since last step at specific block
            blocks = await subtensor.blocks_since_last_step(netuid=1, block=1000000)
        """
        query = await self.query_subtensor(
            name="BlocksSinceLastStep",
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[netuid],
        )
        return query.value if query is not None and hasattr(query, "value") else query

    async def blocks_since_last_update(self, netuid: int, uid: int) -> Optional[int]:
        """Returns the number of blocks since the last update, or ``None`` if the subnetwork or UID does not exist.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.

        Returns:
            Optional[int]: The number of blocks since the last update, or None if the subnetwork or UID does not exist.

        Example:
            # Get blocks since last update for UID 5 in subnet 1
            blocks = await subtensor.blocks_since_last_update(netuid=1, uid=5)

            # Check if neuron needs updating
            blocks_since_update = await subtensor.blocks_since_last_update(netuid=1, uid=10)
        """
        call = await self.get_hyperparameter(param_name="LastUpdate", netuid=netuid)
        return None if call is None else await self.get_current_block() - int(call[uid])

    async def bonds(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """Retrieves the bond distribution set by subnet validators within a specific subnet.

        Bonds represent the "investment" a subnet validator has made in evaluating a specific subnet miner. This
        bonding mechanism is integral to the Yuma Consensus' design intent of incentivizing high-quality performance
        by subnet miners, and honest evaluation by subnet validators.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The block number for this query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the block for the query. Do not specify if using reuse_block or block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            List of tuples mapping each neuron's UID to its bonds with other neurons.

        Example:
            # Get bonds for subnet 1 at block 1000000
            bonds = await subtensor.bonds(netuid=1, block=1000000)

        Notes:
            - See <https://docs.learnbittensor.org/glossary#validator-miner-bonds>
            - See <https://docs.learnbittensor.org/glossary#yuma-consensus>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        b_map_encoded = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Bonds",
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        b_map = []
        async for uid, b in b_map_encoded:
            if b.value is not None:
                b_map.append((uid, b.value))

        return b_map

    async def commit(
        self, wallet: "Wallet", netuid: int, data: str, period: Optional[int] = None
    ) -> bool:
        """Commits arbitrary data to the Bittensor network by publishing metadata.

        This method allows neurons to publish arbitrary data to the blockchain, which can be used for various purposes
        such as sharing model updates, configuration data, or other network-relevant information.

        Arguments:
            wallet: The wallet associated with the neuron committing the data.
            netuid: The unique identifier of the subnet.
            data: The data to be committed to the network.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.

        Returns:
            bool: True if the commit was successful, False otherwise.

        Example:
            # Commit some data to subnet 1
            success = await subtensor.commit(wallet=my_wallet, netuid=1, data="Hello Bittensor!")

            # Commit with custom period
            success = await subtensor.commit(wallet=my_wallet, netuid=1, data="Model update v2.0", period=100)

        Note: See <https://docs.learnbittensor.org/glossary#commit-reveal>
        """
        return await publish_metadata(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            data_type=f"Raw{len(data)}",
            data=data.encode(),
            period=period,
        )

    set_commitment = commit

    async def commit_reveal_enabled(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """Check if commit-reveal mechanism is enabled for a given subnet at a specific block.

        The commit reveal feature is designed to solve the weight-copying problem by giving would-be weight-copiers
        access only to stale weights. Copying stale weights should result in subnet validators departing from consensus.

        Arguments:
            netuid: The unique identifier of the subnet for which to check the commit-reveal mechanism.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            bool: True if commit-reveal mechanism is enabled, False otherwise.

        Example:
            # Check if commit-reveal is enabled for subnet 1
            enabled = await subtensor.commit_reveal_enabled(netuid=1)

            # Check at specific block
            enabled = await subtensor.commit_reveal_enabled(netuid=1, block=1000000)

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#commit-reveal>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="CommitRevealWeightsEnabled",
            block_hash=block_hash,
            netuid=netuid,
            reuse_block=reuse_block,
        )
        return True if call is True else False

    async def difficulty(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Retrieves the 'Difficulty' hyperparameter for a specified subnet in the Bittensor network.

        This parameter determines the computational challenge required for neurons to participate in consensus and
         validation processes. The difficulty directly impacts the network's security and integrity by setting the
         computational effort required for validating transactions and participating in the network's consensus
         mechanism.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The block number for the query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            Optional[int]: The value of the 'Difficulty' hyperparameter if the subnet exists, None otherwise.

        Example:
            # Get difficulty for subnet 1
            difficulty = await subtensor.difficulty(netuid=1)

            # Get difficulty at specific block
            difficulty = await subtensor.difficulty(netuid=1, block=1000000)

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#difficulty>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="Difficulty",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if call is None:
            return None
        return int(call)

    async def does_hotkey_exist(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """Returns true if the hotkey is known by the chain and there are accounts.

        This method queries the SubtensorModule's Owner storage function to determine if the hotkey is registered.

        Arguments:
            hotkey_ss58: The SS58 address of the hotkey.
            block: The block number for this query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the block number to check the hotkey against. Do not specify if using reuse_block
                or block.
            reuse_block: Whether to reuse the last-used blockchain hash. Do not set if using block_hash or block.

        Returns:
            bool: True if the hotkey is known by the chain and there are accounts, False otherwise.

        Example:
            # Check if hotkey exists
            exists = await subtensor.does_hotkey_exist(hotkey_ss58="5F...")

            # Check at specific block
            exists = await subtensor.does_hotkey_exist(hotkey_ss58="5F...", block=1000000)
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Owner",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return_val = (
            False
            if result is None
            # not the default key (0x0)
            else result != "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        )
        return return_val

    async def get_all_subnets_info(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list["SubnetInfo"]:
        """Retrieves detailed information about all subnets within the Bittensor network.

        This function provides comprehensive data on each subnet, including its characteristics and operational
        parameters.

        Arguments:
            block: The block number for the query.
            block_hash: The block hash for the query.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            list[SubnetInfo]: A list of SubnetInfo objects, each containing detailed information about a subnet.

        Example:
            # Get all subnet information
            subnets = await subtensor.get_all_subnets_info()

            # Get at specific block
            subnets = await subtensor.get_all_subnets_info(block=1000000)

            # Iterate over subnet information
            for subnet in subnets:
                print(f"Subnet {subnet.netuid}: {subnet.name}")

        Note:
            Gaining insights into the subnets' details assists in understanding the network's composition, the roles
            of different subnets, and their unique features.

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#subnet>
        """
        result, prices = await asyncio.gather(
            self.query_runtime_api(
                runtime_api="SubnetInfoRuntimeApi",
                method="get_subnets_info_v2",
                params=[],
                block=block,
                block_hash=block_hash,
                reuse_block=reuse_block,
            ),
            self.get_subnet_prices(
                block=block, block_hash=block_hash, reuse_block=reuse_block
            ),
            return_exceptions=True,
        )
        if not result:
            return []

        if not isinstance(prices, (SubstrateRequestException, ValueError)):
            for subnet in result:
                subnet.update({"price": prices.get(subnet["netuid"], 0)})
        else:
            logging.warning(
                f"Unable to fetch subnet prices for block {block}, block hash {block_hash}: {prices}"
            )

        return SubnetInfo.list_from_dicts(result)

    async def get_balance(
        self,
        address: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """Retrieves the balance for given coldkey.

        This method queries the System module's Account storage to get the current balance of a coldkey address. The
        balance represents the amount of TAO tokens held by the specified address.

        Arguments:
            address: The coldkey address in SS58 format.
            block: The block number for the query.
            block_hash: The block hash for the query.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Balance: The balance object containing the account's TAO balance.

        Example:
            # Get balance for an address
            balance = await subtensor.get_balance(address="5F...")
            print(f"Balance: {balance.tao} TAO")

            # Get balance at specific block
            balance = await subtensor.get_balance(address="5F...", block=1000000)
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        balance = await self.substrate.query(
            module="System",
            storage_function="Account",
            params=[address],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return Balance(balance["data"]["free"])

    async def get_balances(
        self,
        *addresses: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, Balance]:
        """Retrieves the balance for given coldkey(s).

        This method efficiently queries multiple coldkey addresses in a single batch operation, returning a dictionary
        mapping each address to its corresponding balance. This is more efficient than calling get_balance multiple
        times.

        Arguments:
            *addresses: Variable number of coldkey addresses in SS58 format.
            block: The block number for the query.
            block_hash: The block hash for the query.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            dict[str, Balance]: A dictionary mapping each address to its Balance object.

        Example:
            # Get balances for multiple addresses
            balances = await subtensor.get_balances("5F...", "5G...", "5H...")
        """
        if reuse_block:
            block_hash = self.substrate.last_block_hash
        elif not block_hash:
            block_hash = await self.get_block_hash()
        else:
            block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        calls = [
            (
                await self.substrate.create_storage_key(
                    "System", "Account", [address], block_hash=block_hash
                )
            )
            for address in addresses
        ]
        batch_call = await self.substrate.query_multi(calls, block_hash=block_hash)
        results = {}
        for item in batch_call:
            value = item[1] or {"data": {"free": 0}}
            results.update({item[0].params[0]: Balance(value["data"]["free"])})
        return results

    async def get_current_block(self) -> int:
        """Returns the current block number on the Bittensor blockchain.

        This function provides the latest block number, indicating the most recent state of the blockchain. Knowing
        the current block number is essential for querying real-time data and performing time-sensitive operations on
        the blockchain. It serves as a reference point for network activities and data synchronization.

        Returns:
            int: The current chain block number.

        Example:
            # Get current block number
            current_block = await subtensor.get_current_block()
            print(f"Current block: {current_block}")

            block = await subtensor.get_current_block()
            if block > 1000000:
                print("Network has progressed past block 1M")

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#block>
        """
        return await self.substrate.get_block_number(None)

    @a.lru_cache(maxsize=128)
    async def _get_block_hash(self, block_id: int):
        return await self.substrate.get_block_hash(block_id)

    async def get_block_hash(self, block: Optional[int] = None) -> str:
        """Retrieves the hash of a specific block on the Bittensor blockchain.

        The block hash is a unique identifier representing the cryptographic hash of the block's content, ensuring its
        integrity and immutability. It is a fundamental aspect of blockchain technology, providing a secure reference
        to each block's data. It is crucial for verifying transactions, ensuring data consistency, and maintaining the
        trustworthiness of the blockchain.

        Arguments:
            block: The block number for which the hash is to be retrieved. If ``None``, returns the latest block hash.

        Returns:
            str: The cryptographic hash of the specified block.

        Example:
            # Get hash for specific block
            block_hash = await subtensor.get_block_hash(block=1000000)
            print(f"Block 1000000 hash: {block_hash}")

            # Get latest block hash
            latest_hash = await subtensor.get_block_hash()
            print(f"Latest block hash: {latest_hash}")

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#block>
        """
        if block:
            return await self._get_block_hash(block)
        else:
            return await self.substrate.get_chain_head()

    async def get_parents(
        self,
        hotkey: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[float, str]]:
        """This method retrieves the parent of a given hotkey and netuid. It queries the SubtensorModule's ParentKeys
        storage function to get the children and formats them before returning as a tuple.

        Arguments:
            hotkey: The child hotkey SS58.
            netuid: The netuid value.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            A list of formatted parents [(proportion, parent)]
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        parents = await self.substrate.query(
            module="SubtensorModule",
            storage_function="ParentKeys",
            params=[hotkey, netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        if parents:
            formatted_parents = []
            for proportion, parent in parents.value:
                # Convert U64 to int
                formatted_child = decode_account_id(parent[0])
                normalized_proportion = u64_normalized_float(proportion)
                formatted_parents.append((normalized_proportion, formatted_child))
            return formatted_parents

        return []

    async def get_children(
        self,
        hotkey: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> tuple[bool, list[tuple[float, str]], str]:
        """Retrieves the children of a given hotkey and netuid.

        This method queries the SubtensorModule's ChildKeys storage function to get the children and formats them before
        returning as a tuple. It provides information about the child neurons that a validator has set for weight
        distribution.

        Arguments:
            hotkey: The hotkey value.
            netuid: The netuid value.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            tuple[bool, list[tuple[float, str]], str]: A tuple containing a boolean indicating success or failure, a
            list of formatted children with their proportions, and an error message (if applicable).

        Example:
            # Get children for a hotkey in subnet 1
            success, children, error = await subtensor.get_children(hotkey="5F...", netuid=1)

            if success:
                for proportion, child_hotkey in children:
                    print(f"Child {child_hotkey}: {proportion}")
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        try:
            children = await self.substrate.query(
                module="SubtensorModule",
                storage_function="ChildKeys",
                params=[hotkey, netuid],
                block_hash=block_hash,
                reuse_block_hash=reuse_block,
            )
            if children:
                formatted_children = []
                for proportion, child in children.value:
                    # Convert U64 to int
                    formatted_child = decode_account_id(child[0])
                    normalized_proportion = u64_normalized_float(proportion)
                    formatted_children.append((normalized_proportion, formatted_child))
                return True, formatted_children, ""
            else:
                return True, [], ""
        except SubstrateRequestException as e:
            return False, [], format_error_message(e)

    async def get_children_pending(
        self,
        hotkey: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> tuple[
        list[tuple[float, str]],
        int,
    ]:
        """Retrieves the pending children of a given hotkey and netuid.

        This method queries the SubtensorModule's PendingChildKeys storage function to get children that are pending
        approval or in a cooldown period. These are children that have been proposed but not yet finalized.

        Arguments:
            hotkey: The hotkey value.
            netuid: The netuid value.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            list[tuple[float, str]]: A list of children with their proportions.
            int: The cool-down block number.
        """

        response = await self.substrate.query(
            module="SubtensorModule",
            storage_function="PendingChildKeys",
            params=[netuid, hotkey],
            block_hash=await self.determine_block_hash(
                block,
                block_hash,
                reuse_block,
            ),
            reuse_block_hash=reuse_block,
        )
        children, cooldown = response.value

        return (
            [
                (
                    u64_normalized_float(proportion),
                    decode_account_id(child[0]),
                )
                for proportion, child in children
            ],
            cooldown,
        )

    async def get_commitment(
        self,
        netuid: int,
        uid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> str:
        """Retrieves the on-chain commitment for a specific neuron in the Bittensor network.

        This method retrieves the commitment data that a neuron has published to the blockchain. Commitments are used in
        the commit-reveal mechanism for secure weight setting and other network operations.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.
            block: The block number to retrieve the commitment from. If None, the latest block is used.
                Default is None.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            str: The commitment data as a string.

        Example:
            # Get commitment for UID 5 in subnet 1
            commitment = await subtensor.get_commitment(netuid=1, uid=5)
            print(f"Commitment: {commitment}")

            # Get commitment at specific block
            commitment = await subtensor.get_commitment(
                netuid=1,
                uid=5,
                block=1000000
            )
        """
        metagraph = await self.metagraph(netuid)
        try:
            hotkey = metagraph.hotkeys[uid]  # type: ignore
        except IndexError:
            logging.error(
                "Your uid is not in the hotkeys. Please double-check your UID."
            )
            return ""

        metadata = await get_metadata(
            self, netuid, hotkey, block, block_hash, reuse_block
        )
        try:
            return decode_metadata(metadata)
        except TypeError:
            return ""

    async def get_last_commitment_bonds_reset_block(
        self, netuid: int, uid: int
    ) -> Optional[int]:
        """
        Retrieves the last block number when the bonds reset were triggered by publish_metadata for a specific neuron.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.

        Returns:
            Optional[int]: The block number when the bonds were last reset, or None if not found.
        """

        metagraph = await self.metagraph(netuid)
        try:
            hotkey = metagraph.hotkeys[uid]
        except IndexError:
            logging.error(
                "Your uid is not in the hotkeys. Please double-check your UID."
            )
            return None
        block = await get_last_bonds_reset(self, netuid, hotkey)
        try:
            return decode_block(block)
        except TypeError:
            return None

    async def get_all_commitments(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, str]:
        """Retrieves the on-chain commitments for a specific subnet in the Bittensor network.

        This method retrieves all commitment data for all neurons in a specific subnet. This is useful for analyzing the
        commit-reveal patterns across an entire subnet.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The block number to retrieve the commitment from. If None, the latest block is used.
                Default is None.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            dict[str, str]: A mapping of the ss58:commitment with the commitment as a string.

        Example:
            # Get all commitments for subnet 1
            commitments = await subtensor.get_all_commitments(netuid=1)

            # Iterate over all commitments
            for hotkey, commitment in commitments.items():
                print(f"Hotkey {hotkey}: {commitment}")
        """
        query = await self.query_map(
            module="Commitments",
            name="CommitmentOf",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        result = {}
        async for id_, value in query:
            result[decode_account_id(id_[0])] = decode_metadata(value.value)
        return result

    async def get_revealed_commitment_by_hotkey(
        self,
        netuid: int,
        hotkey_ss58_address: Optional[str] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[tuple[tuple[int, str], ...]]:
        """Returns hotkey related revealed commitment for a given netuid.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The block number to retrieve the commitment from. Default is ``None``.
            hotkey_ss58_address: The ss58 address of the committee member.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            result (tuple[int, str): A tuple of reveal block and commitment message.
        """
        if not is_valid_ss58_address(address=hotkey_ss58_address):
            raise ValueError(f"Invalid ss58 address {hotkey_ss58_address} provided.")

        query = await self.query_module(
            module="Commitments",
            name="RevealedCommitments",
            params=[netuid, hotkey_ss58_address],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if query is None:
            return None
        return tuple(decode_revealed_commitment(pair) for pair in query)

    async def get_revealed_commitment(
        self,
        netuid: int,
        uid: int,
        block: Optional[int] = None,
    ) -> Optional[tuple[tuple[int, str], ...]]:
        """Returns uid related revealed commitment for a given netuid.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            uid: The neuron uid to retrieve the commitment from.
            block: The block number to retrieve the commitment from. Default is ``None``.

        Returns:
            result (Optional[tuple[int, str]]: A tuple of reveal block and commitment message.

        Example of result:
            ( (12, "Alice message 1"), (152, "Alice message 2") )
            ( (12, "Bob message 1"), (147, "Bob message 2") )
        """
        try:
            meta_info = await self.get_metagraph_info(netuid, block=block)
            if meta_info:
                hotkey_ss58_address = meta_info.hotkeys[uid]
            else:
                raise ValueError(f"Subnet with netuid {netuid} does not exist.")
        except IndexError:
            raise ValueError(f"Subnet {netuid} does not have a neuron with uid {uid}.")

        return await self.get_revealed_commitment_by_hotkey(
            netuid=netuid, hotkey_ss58_address=hotkey_ss58_address, block=block
        )

    async def get_all_revealed_commitments(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, tuple[tuple[int, str], ...]]:
        """Returns all revealed commitments for a given netuid.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The block number to retrieve the commitment from. Default is ``None``.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            result: A dictionary of all revealed commitments in view {ss58_address: (reveal block, commitment message)}.

        Example of result:
        {
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY": ( (12, "Alice message 1"), (152, "Alice message 2") ),
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty": ( (12, "Bob message 1"), (147, "Bob message 2") ),
        }
        """
        query = await self.query_map(
            module="Commitments",
            name="RevealedCommitments",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        result = {}
        async for pair in query:
            hotkey_ss58_address, commitment_message = (
                decode_revealed_commitment_with_hotkey(pair)
            )
            result[hotkey_ss58_address] = commitment_message
        return result

    async def get_current_weight_commit_info(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[str, str, int]]:
        """
        Retrieves CRV3 weight commit information for a specific subnet.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query. Default is ``None``.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            A list of commit details, where each item contains:
                - ss58_address: The address of the committer.
                - commit_message: The commit message.
                - reveal_round: The round when the commitment was revealed.

            The list may be empty if there are no commits found.
        """
        deprecated_message(
            message="The method `get_current_weight_commit_info` is deprecated and will be removed in version 10.0.0. "
            "Use `get_current_weight_commit_info_v2` instead."
        )
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="CRV3WeightCommits",
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        commits = result.records[0][1] if result.records else []
        return [WeightCommitInfo.from_vec_u8(commit) for commit in commits]

    async def get_current_weight_commit_info_v2(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[str, int, str, int]]:
        """
        Retrieves CRV3 weight commit information for a specific subnet.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query. Default is ``None``.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            A list of commit details, where each item contains:
                - ss58_address: The address of the committer.
                - commit_block: The block number when the commitment was made.
                - commit_message: The commit message.
                - reveal_round: The round when the commitment was revealed.

            The list may be empty if there are no commits found.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="CRV3WeightCommitsV2",
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        commits = result.records[0][1] if result.records else []
        return [WeightCommitInfo.from_vec_u8_v2(commit) for commit in commits]

    async def get_delegate_by_hotkey(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[DelegateInfo]:
        """
        Retrieves detailed information about a delegate neuron based on its hotkey. This function provides a
        comprehensive view of the delegate's status, including its stakes, nominators, and reward distribution.

        Arguments:
            hotkey_ss58: The ``SS58`` address of the delegate's hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Optional[DelegateInfo]: Detailed information about the delegate neuron, ``None`` if not found.

        This function is essential for understanding the roles and influence of delegate neurons within the Bittensor
        network's consensus and governance structures.
        """

        result = await self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegate",
            params=[hotkey_ss58],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return None

        return DelegateInfo.from_dict(result)

    async def get_delegate_identities(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, ChainIdentity]:
        """
        Fetches delegates identities from the chain.

        Arguments:
            block: The blockchain block number for the query.
            block_hash: the hash of the blockchain block for the query
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Dict {ss58: ChainIdentity, ...}

        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        identities = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="IdentitiesV2",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        return {
            decode_account_id(ss58_address[0]): ChainIdentity.from_dict(
                decode_hex_identity_dict(identity.value),
            )
            async for ss58_address, identity in identities
        }

    async def get_delegate_take(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> float:
        """
        Retrieves the delegate 'take' percentage for a neuron identified by its hotkey. The 'take' represents the
        percentage of rewards that the delegate claims from its nominators' stakes.

        Arguments:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            float: The delegate take percentage.

        The delegate take is a critical parameter in the network's incentive structure, influencing the distribution of
        rewards among neurons and their nominators.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.query_subtensor(
            name="Delegates",
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[hotkey_ss58],
        )

        return u16_normalized_float(result.value)  # type: ignore

    async def get_delegated(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[DelegateInfo]:
        """
        Retrieves a list of delegates and their associated stakes for a given coldkey. This function identifies the
        delegates that a specific account has staked tokens on.

        Arguments:
            coldkey_ss58: The ``SS58`` address of the account's coldkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A list containing the delegated information for the specified coldkey.

        This function is important for account holders to understand their stake allocations and their involvement in
        the network's delegation and consensus mechanisms.
        """

        result = await self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegated",
            params=[coldkey_ss58],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return []

        return DelegatedInfo.list_from_dicts(result)

    async def get_delegates(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[DelegateInfo]:
        """
        Fetches all delegates on the chain

        Arguments:
            block: The blockchain block number for the query.
            block_hash: hash of the blockchain block number for the query.
            reuse_block: whether to reuse the last-used block hash.

        Returns:
            List of DelegateInfo objects, or an empty list if there are no delegates.
        """
        result = await self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegates",
            params=[],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if result:
            return DelegateInfo.list_from_dicts(result)
        else:
            return []

    async def get_existential_deposit(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """
        Retrieves the existential deposit amount for the Bittensor blockchain.
        The existential deposit is the minimum amount of TAO required for an account to exist on the blockchain.
        Accounts with balances below this threshold can be reaped to conserve network resources.

        Arguments:
            block: The blockchain block number for the query.
            block_hash: Block hash at which to query the deposit amount. If ``None``, the current block is used.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            The existential deposit amount.

        The existential deposit is a fundamental economic parameter in the Bittensor network, ensuring efficient use of
        storage and preventing the proliferation of dust accounts.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.get_constant(
            module_name="Balances",
            constant_name="ExistentialDeposit",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        if result is None:
            raise Exception("Unable to retrieve existential deposit amount.")

        return Balance.from_rao(getattr(result, "value", 0))

    async def get_hotkey_owner(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[str]:
        """
        Retrieves the owner of the given hotkey at a specific block hash.
        This function queries the blockchain for the owner of the provided hotkey. If the hotkey does not exist at the
        specified block hash, it returns None.

        Arguments:
            hotkey_ss58: The SS58 address of the hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the hotkey ownership.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            Optional[str]: The SS58 address of the owner if the hotkey exists, or None if it doesn't.

        Notes:
            See also:
            - <https://docs.learnbittensor.org/glossary#hotkey>
            - <https://docs.learnbittensor.org/glossary#subnet>
            - <https://docs.learnbittensor.org/glossary#neuron>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        hk_owner_query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Owner",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        exists = False
        if hk_owner_query:
            exists = await self.does_hotkey_exist(hotkey_ss58, block_hash=block_hash)
        hotkey_owner = hk_owner_query if exists else None
        return hotkey_owner

    async def get_minimum_required_stake(self):
        """
        Returns the minimum required stake for nominators in the Subtensor network.

        Returns:
            Balance: The minimum required stake as a Balance object.
        """
        result = await self.substrate.query(
            module="SubtensorModule", storage_function="NominatorMinRequiredStake"
        )

        return Balance.from_rao(getattr(result, "value", 0))

    async def get_metagraph_info(
        self,
        netuid: int,
        field_indices: Optional[Union[list[SelectiveMetagraphIndex], list[int]]] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[MetagraphInfo]:
        """
        Retrieves full or partial metagraph information for the specified subnet (netuid).

        A metagraph is a data structure that contains comprehensive information about the current state of a subnet,
        including detailed information on all the nodes (neurons) such as subnet validator stakes and subnet weights
        in the subnet. Metagraph aids in calculating emissions.

        Arguments:
            netuid: The unique identifier of the subnet to query.
            field_indices: An optional list of SelectiveMetagraphIndex or int values specifying which fields to
                retrieve. If not provided, all available fields will be returned.
            block: the block number at which to retrieve the hyperparameter. Do not specify if using block_hash or
                reuse_block
            block_hash: The hash of blockchain block number for the query. Do not specify if using
                block or reuse_block
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Optional[MetagraphInfo]: A MetagraphInfo object containing the requested subnet data, or None if the subnet
                with the given netuid does not exist.

        Example:
            meta_info = await subtensor.get_metagraph_info(netuid=2)

            partial_meta_info = await subtensor.get_metagraph_info(
                netuid=2,
                field_indices=[SelectiveMetagraphIndex.Name, SelectiveMetagraphIndex.OwnerHotkeys]
            )

        Notes:
            See also:
            - <https://docs.learnbittensor.org/glossary#metagraph>
            - <https://docs.learnbittensor.org/glossary#emission>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash

        if field_indices:
            if isinstance(field_indices, list) and all(
                isinstance(f, (SelectiveMetagraphIndex, int)) for f in field_indices
            ):
                indexes = [
                    f.value if isinstance(f, SelectiveMetagraphIndex) else f
                    for f in field_indices
                ]
            else:
                raise ValueError(
                    "`field_indices` must be a list of SelectiveMetagraphIndex enums or ints."
                )

            query = await self.substrate.runtime_call(
                "SubnetInfoRuntimeApi",
                "get_selective_metagraph",
                params=[netuid, indexes if 0 in indexes else [0] + indexes],
                block_hash=block_hash,
            )
        else:
            query = await self.substrate.runtime_call(
                "SubnetInfoRuntimeApi",
                "get_metagraph",
                params=[netuid],
                block_hash=block_hash,
            )

        if query.value is None:
            logging.error(f"Subnet {netuid} does not exist.")
            return None

        return MetagraphInfo.from_dict(query.value)

    async def get_all_metagraphs_info(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[MetagraphInfo]:
        """
        Retrieves a list of MetagraphInfo objects for all subnets

        Arguments:
            block: the block number at which to retrieve the hyperparameter. Do not specify if using block_hash or
                reuse_block
            block_hash: The hash of blockchain block number for the query. Do not specify if using
                block or reuse_block
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            MetagraphInfo dataclass

        Notes:
            See also: See <https://docs.learnbittensor.org/glossary#metagraph>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash
        query = await self.substrate.runtime_call(
            "SubnetInfoRuntimeApi",
            "get_all_metagraphs",
            block_hash=block_hash,
        )
        return MetagraphInfo.list_from_dicts(query.decode())

    async def get_netuids_for_hotkey(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[int]:
        """
        Retrieves a list of subnet UIDs (netuids) for which a given hotkey is a member. This function identifies the
        specific subnets within the Bittensor network where the neuron associated with the hotkey is active.

        Arguments:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used block hash when retrieving info.

        Returns:
            A list of netuids where the neuron is a member.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="IsNetworkMember",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        netuids = []
        if result.records:
            async for record in result:
                if record[1].value:
                    netuids.append(record[0])
        return netuids

    async def get_neuron_certificate(
        self,
        hotkey: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Certificate]:
        """
        Retrieves the TLS certificate for a specific neuron identified by its unique identifier (UID) within a
        specified subnet (netuid) of the Bittensor network.

        Arguments:
            hotkey: The hotkey to query.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            the certificate of the neuron if found, ``None`` otherwise.

        This function is used for certificate discovery for setting up mutual tls communication between neurons.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        certificate = cast(
            Union[str, dict],
            await self.query_module(
                module="SubtensorModule",
                name="NeuronCertificates",
                block_hash=block_hash,
                reuse_block=reuse_block,
                params=[netuid, hotkey],
            ),
        )
        try:
            if certificate:
                return Certificate(certificate)

        except AttributeError:
            return None
        return None

    async def get_all_neuron_certificates(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, Certificate]:
        """
        Retrieves the TLS certificates for neurons within a specified subnet (netuid) of the Bittensor network.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            {ss58: Certificate} for the key/Certificate pairs on the subnet

        This function is used for certificate discovery for setting up mutual tls communication between neurons.
        """
        query_certificates = await self.query_map(
            module="SubtensorModule",
            name="NeuronCertificates",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        output = {}
        async for key, item in query_certificates:
            output[decode_account_id(key)] = Certificate(item.value)
        return output

    async def get_liquidity_list(
        self,
        wallet: "Wallet",
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[list[LiquidityPosition]]:
        """
        Retrieves all liquidity positions for the given wallet on a specified subnet (netuid).
        Calculates associated fee rewards based on current global and tick-level fee data.

        Arguments:
            wallet: Wallet instance to fetch positions for.
            netuid: Subnet unique id.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            List of liquidity positions, or None if subnet does not exist.
        """
        if not await self.subnet_exists(netuid=netuid):
            logging.debug(f"Subnet {netuid} does not exist.")
            return None

        if not await self.is_subnet_active(netuid=netuid):
            logging.debug(f"Subnet {netuid} is not active.")
            return None

        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )

        # Fetch global fees and current price
        fee_global_tao_query_sk = await self.substrate.create_storage_key(
            pallet="Swap",
            storage_function="FeeGlobalTao",
            params=[netuid],
            block_hash=block_hash,
        )
        fee_global_alpha_query_sk = await self.substrate.create_storage_key(
            pallet="Swap",
            storage_function="FeeGlobalAlpha",
            params=[netuid],
            block_hash=block_hash,
        )
        sqrt_price_query_sk = await self.substrate.create_storage_key(
            pallet="Swap",
            storage_function="AlphaSqrtPrice",
            params=[netuid],
            block_hash=block_hash,
        )
        (
            (fee_global_tao_query, fee_global_alpha_query, sqrt_price_query),
            positions_response,
        ) = await asyncio.gather(
            self.substrate.query_multi(
                [
                    fee_global_tao_query_sk,
                    fee_global_alpha_query_sk,
                    sqrt_price_query_sk,
                ],
                block_hash=block_hash,
            ),
            self.query_map(
                module="Swap",
                name="Positions",
                block=block,
                params=[netuid, wallet.coldkeypub.ss58_address],
            ),
        )
        # convert to floats
        fee_global_tao = fixed_to_float(fee_global_tao_query[1])
        fee_global_alpha = fixed_to_float(fee_global_alpha_query[1])
        sqrt_price = fixed_to_float(sqrt_price_query[1])

        # Fetch global fees and current price
        current_tick = price_to_tick(sqrt_price**2)

        # Fetch positions
        positions_values: list[tuple[dict, int, int]] = []
        positions_storage_keys: list[StorageKey] = []
        async for _, p in positions_response:
            position = p.value

            tick_low_idx = position.get("tick_low")[0]
            tick_high_idx = position.get("tick_high")[0]
            positions_values.append((position, tick_low_idx, tick_high_idx))
            tick_low_sk = await self.substrate.create_storage_key(
                pallet="Swap",
                storage_function="Ticks",
                params=[netuid, tick_low_idx],
                block_hash=block_hash,
            )
            tick_high_sk = await self.substrate.create_storage_key(
                pallet="Swap",
                storage_function="Ticks",
                params=[netuid, tick_high_idx],
                block_hash=block_hash,
            )
            positions_storage_keys.extend([tick_low_sk, tick_high_sk])

        # query all our ticks at once
        ticks_query = await self.substrate.query_multi(
            positions_storage_keys, block_hash=block_hash
        )
        # iterator with just the values
        ticks = iter([x[1] for x in ticks_query])
        positions = []
        for position, tick_low_idx, tick_high_idx in positions_values:
            tick_low = next(ticks)
            tick_high = next(ticks)
            # Calculate fees above/below range for both tokens
            tao_below = get_fees(
                current_tick=current_tick,
                tick=tick_low,
                tick_index=tick_low_idx,
                quote=True,
                global_fees_tao=fee_global_tao,
                global_fees_alpha=fee_global_alpha,
                above=False,
            )
            tao_above = get_fees(
                current_tick=current_tick,
                tick=tick_high,
                tick_index=tick_high_idx,
                quote=True,
                global_fees_tao=fee_global_tao,
                global_fees_alpha=fee_global_alpha,
                above=True,
            )
            alpha_below = get_fees(
                current_tick=current_tick,
                tick=tick_low,
                tick_index=tick_low_idx,
                quote=False,
                global_fees_tao=fee_global_tao,
                global_fees_alpha=fee_global_alpha,
                above=False,
            )
            alpha_above = get_fees(
                current_tick=current_tick,
                tick=tick_high,
                tick_index=tick_high_idx,
                quote=False,
                global_fees_tao=fee_global_tao,
                global_fees_alpha=fee_global_alpha,
                above=True,
            )

            # Calculate fees earned by position
            fees_tao, fees_alpha = calculate_fees(
                position=position,
                global_fees_tao=fee_global_tao,
                global_fees_alpha=fee_global_alpha,
                tao_fees_below_low=tao_below,
                tao_fees_above_high=tao_above,
                alpha_fees_below_low=alpha_below,
                alpha_fees_above_high=alpha_above,
                netuid=netuid,
            )

            positions.append(
                LiquidityPosition(
                    **{
                        "id": position.get("id")[0],
                        "price_low": Balance.from_tao(
                            tick_to_price(position.get("tick_low")[0])
                        ),
                        "price_high": Balance.from_tao(
                            tick_to_price(position.get("tick_high")[0])
                        ),
                        "liquidity": Balance.from_rao(position.get("liquidity")),
                        "fees_tao": fees_tao,
                        "fees_alpha": fees_alpha,
                        "netuid": position.get("netuid"),
                    }
                )
            )

        return positions

    async def get_neuron_for_pubkey_and_subnet(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> "NeuronInfo":
        """
        Retrieves information about a neuron based on its public key (hotkey SS58 address) and the specific subnet UID
        (netuid). This function provides detailed neuron information for a particular subnet within the Bittensor
        network.

        Arguments:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Optional[bittensor.core.chain_data.neuron_info.NeuronInfo]: Detailed information about the neuron if found,
                ``None`` otherwise.

        This function is crucial for accessing specific neuron data and understanding its status, stake, and other
        attributes within a particular subnet of the Bittensor ecosystem.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        uid_query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        if (uid := getattr(uid_query, "value", None)) is None:
            return NeuronInfo.get_null_neuron()
        else:
            return await self.neuron_for_uid(
                uid=uid,
                netuid=netuid,
                block=block,
                block_hash=block_hash,
                reuse_block=reuse_block,
            )

    async def get_next_epoch_start_block(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Calculates the first block number of the next epoch for the given subnet.

        If ``block`` is not provided, the current chain block will be used. Epochs are determined based on the subnet's
        tempo (i.e., blocks per epoch). The result is the block number at which the next epoch will begin.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The reference block to calculate from. If None, uses the current chain block height.
            block_hash: The blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            int: The block number at which the next epoch will start.

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#tempo>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        blocks_since_last_step = await self.blocks_since_last_step(
            netuid=netuid, block=block, block_hash=block_hash, reuse_block=reuse_block
        )
        tempo = await self.tempo(
            netuid=netuid, block=block, block_hash=block_hash, reuse_block=reuse_block
        )

        if block and blocks_since_last_step is not None and tempo:
            return block - blocks_since_last_step + tempo + 1
        return None

    async def get_owned_hotkeys(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[str]:
        """
        Retrieves all hotkeys owned by a specific coldkey address.

        Arguments:
            coldkey_ss58: The SS58 address of the coldkey to query.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            list[str]: A list of hotkey SS58 addresses owned by the coldkey.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        owned_hotkeys = await self.substrate.query(
            module="SubtensorModule",
            storage_function="OwnedHotkeys",
            params=[coldkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        return [decode_account_id(hotkey[0]) for hotkey in owned_hotkeys or []]

    async def get_stake(
        self,
        coldkey_ss58: str,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """
        Returns the stake under a coldkey - hotkey pairing.

        Arguments:
            hotkey_ss58: The SS58 address of the hotkey.
            coldkey_ss58: The SS58 address of the coldkey.
            netuid: The subnet ID.
            block: The block number at which to query the stake information.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block
                or reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            Balance: The stake under the coldkey - hotkey pairing.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        sub_query = partial(
            self.query_subtensor,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        alpha_shares, hotkey_alpha_result, hotkey_shares = await asyncio.gather(
            sub_query(
                name="Alpha",
                params=[hotkey_ss58, coldkey_ss58, netuid],
            ),
            sub_query(
                name="TotalHotkeyAlpha",
                params=[hotkey_ss58, netuid],
            ),
            sub_query(
                name="TotalHotkeyShares",
                params=[hotkey_ss58, netuid],
            ),
        )

        hotkey_alpha: int = getattr(hotkey_alpha_result, "value", 0)
        alpha_shares_as_float = fixed_to_float(alpha_shares)
        hotkey_shares_as_float = fixed_to_float(hotkey_shares)

        if hotkey_shares_as_float == 0:
            return Balance.from_rao(0).set_unit(netuid=netuid)

        stake = alpha_shares_as_float / hotkey_shares_as_float * hotkey_alpha

        return Balance.from_rao(int(stake)).set_unit(netuid=netuid)

    # TODO: remove unused parameters in SDK.v10
    async def get_stake_add_fee(
        self,
        amount: Balance,
        netuid: int,
        coldkey_ss58: str,
        hotkey_ss58: str,
        block: Optional[int] = None,
    ) -> Balance:
        """
        Calculates the fee for adding new stake to a hotkey.

        Arguments:
            amount: Amount of stake to add in TAO
            netuid: Netuid of subnet
            coldkey_ss58: SS58 address of source coldkey
            hotkey_ss58: SS58 address of destination hotkey
            block: Block number at which to perform the calculation

        Returns:
            The calculated stake fee as a Balance object
        """
        return await self.get_stake_operations_fee(
            amount=amount, netuid=netuid, block=block
        )

    async def get_subnet_info(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["SubnetInfo"]:
        """
        Retrieves detailed information about subnet within the Bittensor network.
        This function provides comprehensive data on subnet, including its characteristics and operational parameters.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block
                or reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            SubnetInfo: A SubnetInfo objects, each containing detailed information about a subnet.

        Gaining insights into the subnet's details assists in understanding the network's composition, the roles of
        different subnets, and their unique features.
        """
        result = await self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_info_v2",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if not result:
            return None
        return SubnetInfo.from_dict(result)

    async def get_subnet_price(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """Gets the current Alpha price in TAO for all subnets.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block
                or reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The current Alpha price in TAO units for the specified subnet.
        """
        # SN0 price is always 1 TAO
        if netuid == 0:
            return Balance.from_tao(1)

        block_hash = await self.determine_block_hash(block=block)
        current_sqrt_price = await self.substrate.query(
            module="Swap",
            storage_function="AlphaSqrtPrice",
            params=[netuid],
            block_hash=block_hash,
        )

        current_sqrt_price = fixed_to_float(current_sqrt_price)
        current_price = current_sqrt_price * current_sqrt_price
        return Balance.from_rao(int(current_price * 1e9))

    async def get_subnet_prices(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[int, Balance]:
        """Gets the current Alpha price in TAO for a specified subnet.

        Arguments:
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block
                or reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            dict:
                - subnet unique ID
                - The current Alpha price in TAO units for the specified subnet.
        """
        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )

        current_sqrt_prices = await self.substrate.query_map(
            module="Swap",
            storage_function="AlphaSqrtPrice",
            block_hash=block_hash,
            page_size=129,  # total number of subnets
        )

        prices = {}
        async for id_, current_sqrt_price in current_sqrt_prices:
            current_sqrt_price = fixed_to_float(current_sqrt_price)
            current_price = current_sqrt_price * current_sqrt_price
            current_price_in_tao = Balance.from_rao(int(current_price * 1e9))
            prices.update({id_: current_price_in_tao})

        # SN0 price is always 1 TAO
        prices.update({0: Balance.from_tao(1)})
        return prices

    # TODO: remove unused parameters in SDK.v10
    async def get_unstake_fee(
        self,
        amount: Balance,
        netuid: int,
        coldkey_ss58: str,
        hotkey_ss58: str,
        block: Optional[int] = None,
    ) -> Balance:
        """
        Calculates the fee for unstaking from a hotkey.

        Arguments:
            amount: Amount of stake to unstake in TAO
            netuid: Netuid of subnet
            coldkey_ss58: SS58 address of source coldkey
            hotkey_ss58: SS58 address of destination hotkey
            block: Block number at which to perform the calculation

        Returns:
            The calculated stake fee as a Balance object
        """
        return await self.get_stake_operations_fee(
            amount=amount, netuid=netuid, block=block
        )

    # TODO: remove unused parameters in SDK.v10
    async def get_stake_movement_fee(
        self,
        amount: Balance,
        origin_netuid: int,
        origin_hotkey_ss58: str,
        origin_coldkey_ss58: str,
        destination_netuid: int,
        destination_hotkey_ss58: str,
        destination_coldkey_ss58: str,
        block: Optional[int] = None,
    ) -> Balance:
        """
        Calculates the fee for moving stake between hotkeys/subnets/coldkeys.

        Arguments:
            amount: Amount of stake to move in TAO
            origin_netuid: Netuid of source subnet
            origin_hotkey_ss58: SS58 address of source hotkey
            origin_coldkey_ss58: SS58 address of source coldkey
            destination_netuid: Netuid of destination subnet
            destination_hotkey_ss58: SS58 address of destination hotkey
            destination_coldkey_ss58: SS58 address of destination coldkey
            block: Block number at which to perform the calculation

        Returns:
            The calculated stake fee as a Balance object
        """
        return await self.get_stake_operations_fee(
            amount=amount, netuid=origin_netuid, block=block
        )

    async def get_stake_for_coldkey_and_hotkey(
        self,
        coldkey_ss58: str,
        hotkey_ss58: str,
        netuids: Optional[list[int]] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[int, StakeInfo]:
        """
        Retrieves all coldkey-hotkey pairing stake across specified (or all) subnets

        Arguments:
            coldkey_ss58: The SS58 address of the coldkey.
            hotkey_ss58: The SS58 address of the hotkey.
            netuids: The subnet IDs to query for. Set to ``None`` for all subnets.
            block: The block number at which to query the stake information.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block
                or reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            A {netuid: StakeInfo} pairing of all stakes across all subnets.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash
        elif not block_hash:
            block_hash = await self.substrate.get_chain_head()
        if netuids is None:
            all_netuids = await self.get_subnets(block_hash=block_hash)
        else:
            all_netuids = netuids
        results = await asyncio.gather(
            *[
                self.query_runtime_api(
                    "StakeInfoRuntimeApi",
                    "get_stake_info_for_hotkey_coldkey_netuid",
                    params=[hotkey_ss58, coldkey_ss58, netuid],
                    block_hash=block_hash,
                )
                for netuid in all_netuids
            ]
        )
        return {
            netuid: StakeInfo.from_dict(result)
            for (netuid, result) in zip(all_netuids, results)
        }

    async def get_stake_for_coldkey(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[list["StakeInfo"]]:
        """
        Retrieves the stake information for a given coldkey.

        Arguments:
            coldkey_ss58: The SS58 address of the coldkey.
            block: The block number at which to query the stake information.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            An optional list of StakeInfo objects, or ``None`` if no stake information is found.
        """
        result = await self.query_runtime_api(
            runtime_api="StakeInfoRuntimeApi",
            method="get_stake_info_for_coldkey",
            params=[coldkey_ss58],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if result is None:
            return []

        stakes = StakeInfo.list_from_dicts(result)  # type: ignore
        return [stake for stake in stakes if stake.stake > 0]

    get_stake_info_for_coldkey = get_stake_for_coldkey

    async def get_stake_for_hotkey(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """
        Retrieves the stake information for a given hotkey.

        Arguments:
            hotkey_ss58: The SS58 address of the hotkey.
            netuid: The subnet ID to query for.
            block: The block number at which to query the stake information. Do not specify if also specifying
                block_hash or reuse_block.
            block_hash: The hash of the blockchain block number for the query. Do not specify if also specifying block
                or reuse_block.
            reuse_block: Whether to reuse for this query the last-used block. Do not specify if also specifying block
                or block_hash.
        """
        hotkey_alpha_query = await self.query_subtensor(
            name="TotalHotkeyAlpha",
            params=[hotkey_ss58, netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        balance = Balance.from_rao(hotkey_alpha_query.value)
        balance.set_unit(netuid=netuid)
        return balance

    get_hotkey_stake = get_stake_for_hotkey

    async def get_stake_operations_fee(
        self,
        amount: Balance,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ):
        """Returns fee for any stake operation in specified subnet.

        Args:
            amount: Amount of stake to add in Alpha/TAO.
            netuid: Netuid of subnet.
            block: The block number at which to query the stake information. Do not specify if also specifying
                block_hash or reuse_block.
            block_hash: The hash of the blockchain block number for the query. Do not specify if also specifying block
                or reuse_block.
            reuse_block: Whether to reuse for this query the last-used block. Do not specify if also specifying block
                or block_hash.

        Returns:
            The calculated stake fee as a Balance object.
        """
        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )
        result = await self.substrate.query(
            module="Swap",
            storage_function="FeeRate",
            params=[netuid],
            block_hash=block_hash,
        )
        return amount * (result.value / U16_MAX)

    async def get_stake_weight(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[float]:
        """
        Retrieves the stake weight for all hotkeys in a given subnet.

        Arguments:
            netuid: Netuid of subnet.
            block: Block number at which to perform the calculation.
            block_hash: The hash of the blockchain block number for the query. Do not specify if also specifying block
                or reuse_block.
            reuse_block: Whether to reuse for this query the last-used block. Do not specify if also specifying block
                or block_hash.

        Returns:
            A list of stake weights for all hotkeys in the specified subnet.
        """
        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="StakeWeight",
            params=[netuid],
            block_hash=block_hash,
        )
        return [u16_normalized_float(w) for w in result]

    async def get_subnet_burn_cost(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Balance]:
        """
        Retrieves the burn cost for registering a new subnet within the Bittensor network. This cost represents the
            amount of Tao that needs to be locked or burned to establish a new

        Arguments:
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash of the block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            int: The burn cost for subnet registration.

        The subnet burn cost is an important economic parameter, reflecting the network's mechanisms for controlling
            the proliferation of subnets and ensuring their commitment to the network's long-term viability.
        """
        lock_cost = await self.query_runtime_api(
            runtime_api="SubnetRegistrationRuntimeApi",
            method="get_network_registration_cost",
            params=[],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if lock_cost is not None:
            return Balance.from_rao(lock_cost)
        else:
            return lock_cost

    async def get_subnet_hyperparameters(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["SubnetHyperparameters"]:
        """
        Retrieves the hyperparameters for a specific subnet within the Bittensor network. These hyperparameters define
        the operational settings and rules governing the subnet's behavior.

        Arguments:
            netuid: The network UID of the subnet to query.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            The subnet's hyperparameters, or ``None`` if not available.

        Understanding the hyperparameters is crucial for comprehending how subnets are configured and managed, and how
        they interact with the network's consensus and incentive mechanisms.
        """
        result = await self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_hyperparams_v2",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return None

        return SubnetHyperparameters.from_dict(result)

    async def get_subnet_reveal_period_epochs(
        self, netuid: int, block: Optional[int] = None, block_hash: Optional[str] = None
    ) -> int:
        """Retrieve the SubnetRevealPeriodEpochs hyperparameter."""
        block_hash = await self.determine_block_hash(block, block_hash)
        return await self.get_hyperparameter(
            param_name="RevealPeriodEpochs", block_hash=block_hash, netuid=netuid
        )

    async def get_subnets(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[int]:
        """
        Retrieves the list of all subnet unique identifiers (netuids) currently present in the Bittensor network.

        Arguments:
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            A list of subnet netuids.

        This function provides a comprehensive view of the subnets within the Bittensor network, offering insights into
        its diversity and scale.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        subnets = []
        if result.records:
            async for netuid, exists in result:
                if exists:
                    subnets.append(netuid)
        return subnets

    async def get_total_subnets(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the total number of subnets within the Bittensor network as of a specific blockchain block.

        Arguments:
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Optional[str]: The total number of subnets in the network.

        Understanding the total number of subnets is essential for assessing the network's growth and the extent of its
        decentralized infrastructure.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="TotalNetworks",
            params=[],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return getattr(result, "value", None)

    async def get_transfer_fee(
        self, wallet: "Wallet", dest: str, value: Balance, keep_alive: bool = True
    ) -> Balance:
        """
        Calculates the transaction fee for transferring tokens from a wallet to a specified destination address. This
        function simulates the transfer to estimate the associated cost, taking into account the current network
        conditions and transaction complexity.

        Arguments:
            wallet: The wallet from which the transfer is initiated.
            dest: The ``SS58`` address of the destination account.
            value: The amount of tokens to be transferred, specified as a Balance object, or in Tao (float) or Rao
                (int) units.
            keep_alive: Whether the transfer fee should be calculated based on keeping the wallet alive (existential
                deposit) or not.

        Returns:
            bittensor.utils.balance.Balance: The estimated transaction fee for the transfer, represented as a Balance
                object.

        Estimating the transfer fee is essential for planning and executing token transactions, ensuring that the
        wallet has sufficient funds to cover both the transfer amount and the associated costs. This function provides
        a crucial tool for managing financial operations within the Bittensor network.
        """
        if value is not None:
            value = check_and_convert_to_balance(value)
        call_params: dict[str, Union[int, str, bool]]
        call_function, call_params = get_transfer_fn_params(value, dest, keep_alive)

        call = await self.substrate.compose_call(
            call_module="Balances",
            call_function=call_function,
            call_params=call_params,
        )

        try:
            payment_info = await self.substrate.get_payment_info(
                call=call, keypair=wallet.coldkeypub
            )
        except Exception as e:
            logging.error(f":cross_mark: [red]Failed to get payment info: [/red]{e}")
            payment_info = {"partial_fee": int(2e7)}  # assume  0.02 Tao

        return Balance.from_rao(payment_info["partial_fee"])

    async def get_vote_data(
        self,
        proposal_hash: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["ProposalVoteData"]:
        """
        Retrieves the voting data for a specific proposal on the Bittensor blockchain. This data includes information
        about how senate members have voted on the proposal.

        Arguments:
            proposal_hash: The hash of the proposal for which voting data is requested.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number to query the voting data.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            An object containing the proposal's voting data, or ``None`` if not found.

        This function is important for tracking and understanding the decision-making processes within the Bittensor
        network, particularly how proposals are received and acted upon by the governing body.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        vote_data: dict[str, Any] = await self.substrate.query(
            module="Triumvirate",
            storage_function="Voting",
            params=[proposal_hash],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        if vote_data is None:
            return None

        return ProposalVoteData.from_dict(vote_data)

    async def get_uid_for_hotkey_on_subnet(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the unique identifier (UID) for a neuron's hotkey on a specific subnet.

        Arguments:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Optional[int]: The UID of the neuron if it is registered on the subnet, ``None`` otherwise.

        The UID is a critical identifier within the network, linking the neuron's hotkey to its operational and
        governance activities on a particular subnet.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return getattr(result, "value", result)

    async def filter_netuids_by_registered_hotkeys(
        self,
        all_netuids: Iterable[int],
        filter_for_netuids: Iterable[int],
        all_hotkeys: Iterable["Wallet"],
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[int]:
        """
        Filters a given list of all netuids for certain specified netuids and hotkeys

        Arguments:
            all_netuids: A list of netuids to filter.
            filter_for_netuids: A subset of all_netuids to filter from the main list.
            all_hotkeys: Hotkeys to filter from the main list.
            block: The blockchain block number for the query.
            block_hash: hash of the blockchain block number at which to perform the query.
            reuse_block: whether to reuse the last-used blockchain hash when retrieving info.

        Returns:
            The filtered list of netuids.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        netuids_with_registered_hotkeys = [
            item
            for sublist in await asyncio.gather(
                *[
                    self.get_netuids_for_hotkey(
                        wallet.hotkey.ss58_address,
                        reuse_block=reuse_block,
                        block_hash=block_hash,
                    )
                    for wallet in all_hotkeys
                ]
            )
            for item in sublist
        ]

        if not filter_for_netuids:
            all_netuids = netuids_with_registered_hotkeys

        else:
            filtered_netuids = [
                netuid for netuid in all_netuids if netuid in filter_for_netuids
            ]

            registered_hotkeys_filtered = [
                netuid
                for netuid in netuids_with_registered_hotkeys
                if netuid in filter_for_netuids
            ]

            # Combine both filtered lists
            all_netuids = filtered_netuids + registered_hotkeys_filtered

        return list(set(all_netuids))

    async def immunity_period(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the 'ImmunityPeriod' hyperparameter for a specific subnet. This parameter defines the duration during
        which new neurons are protected from certain network penalties or restrictions.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Optional[int]: The value of the 'ImmunityPeriod' hyperparameter if the subnet exists, ``None`` otherwise.

        The 'ImmunityPeriod' is a critical aspect of the network's governance system, ensuring that new participants
        have a grace period to establish themselves and contribute to the network without facing immediate punitive
        actions.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="ImmunityPeriod",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def is_fast_blocks(self):
        """Returns True if the node is running with fast blocks. False if not."""
        return (
            await self.query_constant("SubtensorModule", "DurationOfStartCall")
        ).value == 10

    async def is_hotkey_delegate(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Determines whether a given hotkey (public key) is a delegate on the Bittensor network. This function checks if
        the neuron associated with the hotkey is part of the network's delegation system.

        Arguments:
            hotkey_ss58: The SS58 address of the neuron's hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            ``True`` if the hotkey is a delegate, ``False`` otherwise.

        Being a delegate is a significant status within the Bittensor network, indicating a neuron's involvement in
        consensus and governance processes.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        delegates = await self.get_delegates(
            block_hash=block_hash, reuse_block=reuse_block
        )
        return hotkey_ss58 in [info.hotkey_ss58 for info in delegates]

    async def is_hotkey_registered(
        self,
        hotkey_ss58: str,
        netuid: Optional[int] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Determines whether a given hotkey (public key) is registered in the Bittensor network, either globally across
        any subnet or specifically on a specified subnet. This function checks the registration status of a neuron
        identified by its hotkey, which is crucial for validating its participation and activities within the network.

        Arguments:
            hotkey_ss58: The SS58 address of the neuron's hotkey.
            netuid: The unique identifier of the subnet to check the registration. If ``None``, the
                registration is checked across all subnets.
            block: The blockchain block number at which to perform the query.
            block_hash: The blockchain block_hash representation of the block id. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or
                reuse_block.

        Returns:
            bool: ``True`` if the hotkey is registered in the specified context (either any subnet or a specific subnet),
                ``False`` otherwise.

        This function is important for verifying the active status of neurons in the Bittensor network. It aids in
        understanding whether a neuron is eligible to participate in network processes such as consensus, validation,
        and incentive distribution based on its registration status.
        """
        if netuid is None:
            return await self.is_hotkey_registered_any(
                hotkey_ss58, block, block_hash, reuse_block
            )
        else:
            return await self.is_hotkey_registered_on_subnet(
                hotkey_ss58, netuid, block, block_hash, reuse_block
            )

    async def is_hotkey_registered_any(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Checks if a neuron's hotkey is registered on any subnet within the Bittensor network.

        Arguments:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            bool: ``True`` if the hotkey is registered on any subnet, False otherwise.

        This function is essential for determining the network-wide presence and participation of a neuron.
        """
        hotkeys = await self.get_netuids_for_hotkey(
            hotkey_ss58, block, block_hash, reuse_block
        )
        return len(hotkeys) > 0

    async def is_hotkey_registered_on_subnet(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """Checks if the hotkey is registered on a given netuid."""
        return (
            await self.get_uid_for_hotkey_on_subnet(
                hotkey_ss58,
                netuid,
                block=block,
                block_hash=block_hash,
                reuse_block=reuse_block,
            )
            is not None
        )

    async def is_subnet_active(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """Verify if subnet with provided netuid is active.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            ``True`` if subnet is active, ``False`` otherwise.

        Note: This means whether the ``start_call`` was initiated or not.
        """
        query = await self.query_subtensor(
            name="FirstEmissionBlockNumber",
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[netuid],
        )
        return True if query and query.value > 0 else False

    async def last_drand_round(self) -> Optional[int]:
        """
        Retrieves the last drand round emitted in bittensor. This corresponds when committed weights will be revealed.

        Returns:
            int: The latest Drand round emitted in bittensor.
        """
        result = await self.substrate.query(
            module="Drand", storage_function="LastStoredRound"
        )
        return getattr(result, "value", None)

    async def max_weight_limit(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[float]:
        """
        Returns network MaxWeightsLimit hyperparameter.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Optional[float]: The value of the MaxWeightsLimit hyperparameter, or ``None`` if the subnetwork does not
                exist or the parameter is not found.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="MaxWeightsLimit",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else u16_normalized_float(int(call))

    async def metagraph(
        self, netuid: int, lite: bool = True, block: Optional[int] = None
    ) -> "AsyncMetagraph":
        """
        Returns a synced metagraph for a specified subnet within the Bittensor network. The metagraph represents the
        network's structure, including neuron connections and interactions.

        Arguments:
            netuid: The network UID of the subnet to query.
            lite: If true, returns a metagraph using a lightweight sync (no weights, no bonds). Default is
                ``True``.
            block: Block number for synchronization, or `None` for the latest block.

        Returns:
            bittensor.core.metagraph.Metagraph: The metagraph representing the subnet's structure and neuron
                relationships.

        The metagraph is an essential tool for understanding the topology and dynamics of the Bittensor network's
        decentralized architecture, particularly in relation to neuron interconnectivity and consensus processes.
        """
        metagraph = AsyncMetagraph(
            network=self.chain_endpoint,
            netuid=netuid,
            lite=lite,
            sync=False,
            subtensor=self,
        )
        await metagraph.sync(block=block, lite=lite, subtensor=self)

        return metagraph

    async def min_allowed_weights(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Returns network MinAllowedWeights hyperparameter.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Optional[int]: The value of the MinAllowedWeights hyperparameter, or ``None`` if the subnetwork does not
                exist or the parameter is not found.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="MinAllowedWeights",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def neuron_for_uid(
        self,
        uid: Optional[int],
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> NeuronInfo:
        """
        Retrieves detailed information about a specific neuron identified by its unique identifier (UID) within a
        specified subnet (netuid) of the Bittensor network. This function provides a comprehensive view of a neuron's
        attributes, including its stake, rank, and operational status.

        Arguments:
            uid: The unique identifier of the neuron.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Detailed information about the neuron if found, a null neuron otherwise

        This function is crucial for analyzing individual neurons' contributions and status within a specific subnet,
        offering insights into their roles in the network's consensus and validation mechanisms.
        """
        if uid is None:
            return NeuronInfo.get_null_neuron()

        result = await self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neuron",
            params=[netuid, uid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return NeuronInfo.get_null_neuron()

        return NeuronInfo.from_dict(result)

    async def neurons(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[NeuronInfo]:
        """
        Retrieves a list of all neurons within a specified subnet of the Bittensor network.
        This function provides a snapshot of the subnet's neuron population, including each neuron's attributes and
        network interactions.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of NeuronInfo objects detailing each neuron's characteristics in the subnet.

        Understanding the distribution and status of neurons within a subnet is key to comprehending the network's
        decentralized structure and the dynamics of its consensus and governance processes.
        """
        result = await self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return []

        return NeuronInfo.list_from_dicts(result)

    async def neurons_lite(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[NeuronInfoLite]:
        """
        Retrieves a list of neurons in a 'lite' format from a specific subnet of the Bittensor network.
        This function provides a streamlined view of the neurons, focusing on key attributes such as stake and network
        participation.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of simplified neuron information for the subnet.

        This function offers a quick overview of the neuron population within a subnet, facilitating efficient analysis
        of the network's decentralized structure and neuron dynamics.
        """
        result = await self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons_lite",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return []

        return NeuronInfoLite.list_from_dicts(result)

    async def query_identity(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[ChainIdentity]:
        """
        Queries the identity of a neuron on the Bittensor blockchain using the given key. This function retrieves
        detailed identity information about a specific neuron, which is a crucial aspect of the network's decentralized
        identity and governance system.

        Arguments:
            coldkey_ss58: The coldkey used to query the neuron's identity (technically the neuron's coldkey SS58
                address).
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            An object containing the identity information of the neuron if found, ``None`` otherwise.

        The identity information can include various attributes such as the neuron's stake, rank, and other
        network-specific details, providing insights into the neuron's role and status within the Bittensor network.

        Note:
            See the ``Bittensor CLI documentation <https://docs.bittensor.com/reference/btcli>``_ for supported identity
                parameters.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        identity_info = cast(
            dict,
            await self.substrate.query(
                module="SubtensorModule",
                storage_function="IdentitiesV2",
                params=[coldkey_ss58],
                block_hash=block_hash,
                reuse_block_hash=reuse_block,
            ),
        )

        if not identity_info:
            return None

        try:
            return ChainIdentity.from_dict(
                decode_hex_identity_dict(identity_info),
            )
        except TypeError:
            return None

    async def recycle(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Balance]:
        """
        Retrieves the 'Burn' hyperparameter for a specified subnet. The 'Burn' parameter represents the amount of Tao
        that is effectively recycled within the Bittensor network.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Optional[Balance]: The value of the 'Burn' hyperparameter if the subnet exists, ``None`` otherwise.

        Understanding the 'Burn' rate is essential for analyzing the network registration usage, particularly how it is
        correlated with user activity and the overall cost of participation in a given subnet.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="Burn",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else Balance.from_rao(int(call))

    async def set_reveal_commitment(
        self,
        wallet,
        netuid: int,
        data: str,
        blocks_until_reveal: int = 360,
        block_time: Union[int, float] = 12,
        period: Optional[int] = None,
    ) -> tuple[bool, int]:
        """
        Commits arbitrary data to the Bittensor network by publishing metadata.

        Arguments:
            wallet: The wallet associated with the neuron committing the data.
            netuid: The unique identifier of the subnetwork.
            data: The data to be committed to the network.
            blocks_until_reveal: The number of blocks from now after which the data will be revealed.
                Defaults to ``360`` (the number of blocks in one epoch).
            block_time: The number of seconds between each block. Defaults to ``12``.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            bool: ``True`` if the commitment was successful, ``False`` otherwise.

        Note: A commitment can be set once per subnet epoch and is reset at the next epoch in the chain automatically.
        """

        encrypted, reveal_round = get_encrypted_commitment(
            data, blocks_until_reveal, block_time
        )

        # increase reveal_round in return + 1 because we want to fetch data from the chain after that round was revealed
        # and stored.
        data_ = {"encrypted": encrypted, "reveal_round": reveal_round}
        return await publish_metadata(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            data_type="TimelockEncrypted",
            data=data_,
            period=period,
        ), reveal_round

    async def subnet(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[DynamicInfo]:
        """
        Retrieves the subnet information for a single subnet in the Bittensor network.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The block number to get the subnets at.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Optional[DynamicInfo]: A DynamicInfo object, containing detailed information about a subnet.
        """
        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )

        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash

        query, price = await asyncio.gather(
            self.substrate.runtime_call(
                "SubnetInfoRuntimeApi",
                "get_dynamic_info",
                params=[netuid],
                block_hash=block_hash,
            ),
            self.get_subnet_price(
                netuid=netuid,
                block=block,
                block_hash=block_hash,
                reuse_block=reuse_block,
            ),
            return_exceptions=True,
        )

        if isinstance(decoded := query.decode(), dict):
            if isinstance(price, (SubstrateRequestException, ValueError)):
                price = None
            return DynamicInfo.from_dict({**decoded, "price": price})
        return None

    async def subnet_exists(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Checks if a subnet with the specified unique identifier (netuid) exists within the Bittensor network.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to check the subnet existence.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            ``True`` if the subnet exists, ``False`` otherwise.

        This function is critical for verifying the presence of specific subnets in the network, enabling a deeper
        understanding of the network's structure and composition.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return getattr(result, "value", False)

    async def subnetwork_n(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Returns network SubnetworkN hyperparameter.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to check the subnet existence.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Optional[int]: The value of the SubnetworkN hyperparameter, or ``None`` if the subnetwork does not exist or
                the parameter is not found.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="SubnetworkN",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def tempo(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Returns network Tempo hyperparameter.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to check the subnet existence.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Optional[int]: The value of the Tempo hyperparameter, or ``None`` if the subnetwork does not exist or the
                parameter is not found.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="Tempo",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def tx_rate_limit(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the transaction rate limit for the Bittensor network as of a specific blockchain block.
        This rate limit sets the maximum number of transactions that can be processed within a given time frame.

        Arguments:
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to check the subnet existence.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Optional[int]: The transaction rate limit of the network, ``None`` if not available.

        The transaction rate limit is an essential parameter for ensuring the stability and scalability of the Bittensor
        network. It helps in managing network load and preventing congestion, thereby maintaining efficient and timely
        transaction processing.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.query_subtensor(
            "TxRateLimit", block_hash=block_hash, reuse_block=reuse_block
        )
        return getattr(result, "value", None)

    async def wait_for_block(self, block: Optional[int] = None):
        """
        Waits until a specific block is reached on the chain. If no block is specified, waits for the next block.

        Arguments:
            block: The block number to wait for. If ``None``, waits for the next block.

        Returns:
            bool: ``True`` if the target block was reached, ``False`` if timeout occurred.

        Example:
            import bittensor as bt
            subtensor = bt.Subtensor()

            await subtensor.wait_for_block() # Waits for next block
            await subtensor.wait_for_block(block=1234) # Waits for a specific block
        """

        async def handler(block_data: dict):
            logging.debug(
                f"reached block {block_data['header']['number']}. Waiting for block {target_block}"
            )
            if block_data["header"]["number"] >= target_block:
                return True
            return None

        current_block = await self.substrate.get_block()
        current_block_hash = current_block.get("header", {}).get("hash")

        if block is not None:
            target_block = block
        else:
            target_block = current_block["header"]["number"] + 1

        await self.substrate.get_block_handler(
            current_block_hash, header_only=True, subscription_handler=handler
        )
        return True

    async def weights(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the weight distribution set by neurons within a specific subnet of the Bittensor network.
        This function maps each neuron's UID to the weights it assigns to other neurons, reflecting the network's trust
        and value assignment mechanisms.

        Arguments:
            netuid: The network UID of the subnet to query.
            block: Block number for synchronization, or `None` for the latest block.
            block_hash: The hash of the blockchain block for the query.
            reuse_block: reuse the last-used blockchain block hash.

        Returns:
            A list of tuples mapping each neuron's UID to its assigned weights.

        The weight distribution is a key factor in the network's consensus algorithm and the ranking of neurons,
        influencing their influence and reward allocation within the subnet.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        # TODO look into seeing if we can speed this up with storage query
        w_map_encoded = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Weights",
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        w_map = []
        async for uid, w in w_map_encoded:
            w_map.append((uid, w.value))

        return w_map

    async def weights_rate_limit(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Returns network WeightsSetRateLimit hyperparameter.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Optional[int]: The value of the WeightsSetRateLimit hyperparameter, or ``None`` if the subnetwork does not
                exist or the parameter is not found.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="WeightsSetRateLimit",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def get_timestamp(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> datetime:
        """
        Retrieves the datetime timestamp for a given block.

        Arguments:
            block: The blockchain block number for the query. Do not specify if specifying block_hash or reuse_block.
            block_hash: The blockchain block_hash representation of the block id. Do not specify if specifying block
                or reuse_block.
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not specify if specifying block or
                block_hash.

        Returns:
            datetime object for the timestamp of the block.
        """
        res = await self.query_module(
            "Timestamp",
            "Now",
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        unix = res.value
        return datetime.fromtimestamp(unix / 1000, tz=timezone.utc)

    async def get_subnet_owner_hotkey(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[str]:
        """
        Retrieves the hotkey of the subnet owner for a given network UID.

        This function queries the subtensor network to fetch the hotkey of the owner of a subnet specified by its
        netuid. If no data is found or the query fails, the function returns None.

        Arguments:
            netuid: The network UID of the subnet to fetch the owner's hotkey for.
            block: The specific block number to query the data from.

        Returns:
            The hotkey of the subnet owner if available; None otherwise.
        """
        return await self.query_subtensor(
            name="SubnetOwnerHotkey", params=[netuid], block=block
        )

    async def get_subnet_validator_permits(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[list[bool]]:
        """
        Retrieves the list of validator permits for a given subnet as boolean values.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.

        Returns:
            A list of boolean values representing validator permits, or None if not available.
        """
        query = await self.query_subtensor(
            name="ValidatorPermit",
            params=[netuid],
            block=block,
        )
        return query.value if query is not None and hasattr(query, "value") else query

    # Extrinsics helper ================================================================================================

    async def sign_and_send_extrinsic(
        self,
        call: "GenericCall",
        wallet: "Wallet",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        sign_with: str = "coldkey",
        use_nonce: bool = False,
        period: Optional[int] = None,
        nonce_key: str = "hotkey",
        raise_error: bool = False,
    ) -> tuple[bool, str]:
        """
        Helper method to sign and submit an extrinsic call to chain.

        Arguments:
            call: a prepared Call object
            wallet: the wallet whose coldkey will be used to sign the extrinsic
            wait_for_inclusion: whether to wait until the extrinsic call is included on the chain
            wait_for_finalization: whether to wait until the extrinsic call is finalized on the chain
            sign_with: the wallet's keypair to use for the signing. Options are "coldkey", "hotkey", "coldkeypub"
            use_nonce: unique identifier for the transaction related with hot/coldkey.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            nonce_key: the type on nonce to use. Options are "hotkey" or "coldkey".
            nonce_key: the type on nonce to use. Options are "hotkey", "coldkey", or "coldkeypub".
            raise_error: raises a relevant exception rather than returning ``False`` if unsuccessful.

        Returns:
            (success, error message)

        Raises:
            SubstrateRequestException: Substrate request exception.
        """
        possible_keys = ("coldkey", "hotkey", "coldkeypub")
        if sign_with not in possible_keys:
            raise AttributeError(
                f"'sign_with' must be either 'coldkey', 'hotkey' or 'coldkeypub', not '{sign_with}'"
            )
        signing_keypair = getattr(wallet, sign_with)
        extrinsic_data = {"call": call, "keypair": signing_keypair}
        if use_nonce:
            if nonce_key not in possible_keys:
                raise AttributeError(
                    f"'nonce_key' must be either 'coldkey', 'hotkey' or 'coldkeypub', not '{nonce_key}'"
                )
            next_nonce = await self.substrate.get_account_next_index(
                getattr(wallet, nonce_key).ss58_address
            )
            extrinsic_data["nonce"] = next_nonce
        if period is not None:
            extrinsic_data["era"] = {"period": period}

        extrinsic = await self.substrate.create_signed_extrinsic(**extrinsic_data)
        try:
            response = await self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                message = "Not waiting for finalization or inclusion."
                logging.debug(f"{message}. Extrinsic: {extrinsic}")
                return True, message

            if await response.is_success:
                return True, ""

            if raise_error:
                raise ChainError.from_error(await response.error_message)

            return False, format_error_message(await response.error_message)

        except SubstrateRequestException as e:
            if raise_error:
                raise

            return False, format_error_message(e)

    # Extrinsics =======================================================================================================

    async def add_stake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        netuid: Optional[int] = None,
        amount: Optional[Balance] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        safe_staking: bool = False,
        allow_partial_stake: bool = False,
        rate_tolerance: float = 0.005,
        period: Optional[int] = None,
    ) -> bool:
        """
        Adds a stake from the specified wallet to the neuron identified by the SS58 address of its hotkey in specified
        subnet. Staking is a fundamental process in the Bittensor network that enables neurons to participate actively
        and earn incentives.

        Arguments:
            wallet: The wallet to be used for staking.
            hotkey_ss58: The SS58 address of the hotkey associated with the neuron to which you intend to delegate your
                stake. If not specified, the wallet's hotkey will be used. Defaults to ``None``.
            netuid: The unique identifier of the subnet to which the neuron belongs.
            amount: The amount of TAO to stake.
            wait_for_inclusion: Waits for the transaction to be included in a block. Defaults to `True`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Defaults to `False`.
            safe_staking: If true, enables price safety checks to protect against fluctuating prices. The stake will
                only execute if the price change doesn't exceed the rate tolerance. Default is ``False``.
            allow_partial_stake: If true and safe_staking is enabled, allows partial staking when the full amount would
                exceed the price tolerance. If false, the entire stake fails if it would exceed the tolerance.
                Default is ``False``.
            rate_tolerance: The maximum allowed price change ratio when staking. For example, 0.005 = 0.5% maximum price
                increase. Only used when safe_staking is True. Default is ``0.005``.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction. Defaults to ``None``.

        Returns:
            bool: ``True`` if the staking is successful, ``False`` otherwise.

        This function enables neurons to increase their stake in the network, enhancing their influence and potential.
        When safe_staking is enabled, it provides protection against price fluctuations during the time stake is
        executed and the time it is actually processed by the chain.
        """
        amount = check_and_convert_to_balance(amount)
        return await add_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            safe_staking=safe_staking,
            allow_partial_stake=allow_partial_stake,
            rate_tolerance=rate_tolerance,
            period=period,
        )

    async def add_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        liquidity: Balance,
        price_low: Balance,
        price_high: Balance,
        hotkey: Optional[str] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """
        Adds liquidity to the specified price range.

        Arguments:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            liquidity: The amount of liquidity to be added.
            price_low: The lower bound of the price tick range. In TAO.
            price_high: The upper bound of the price tick range. In TAO.
            hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used. Defaults to
                `None`.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block. Defaults to True.
            wait_for_finalization: Whether to wait for finalization of the extrinsic. Defaults to False.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.

        Returns:
            Tuple[bool, str]:
                - True and a success message if the extrinsic is successfully submitted or processed.
                - False and an error message if the submission fails or the wallet cannot be unlocked.

        Note: Adding is allowed even when user liquidity is enabled in specified subnet. Call ``toggle_user_liquidity``
            method to enable/disable user liquidity.
        """
        return await add_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            liquidity=liquidity,
            price_low=price_low,
            price_high=price_high,
            hotkey=hotkey,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def add_stake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        netuids: list[int],
        amounts: Optional[list[Balance]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Adds stakes to multiple neurons identified by their hotkey SS58 addresses.
        This bulk operation allows for efficient staking across different neurons from a single wallet.

        Arguments:
            wallet: The wallet used for staking.
            hotkey_ss58s: List of ``SS58`` addresses of hotkeys to stake to.
            netuids: list of subnet UIDs.
            amounts: Corresponding amounts of TAO to stake for each hotkey.
            wait_for_inclusion: Waits for the transaction to be included in a block. Defaults to `True`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Defaults to `False`.

        Returns:
            bool: ``True`` if the staking is successful for all specified neurons, ``False`` otherwise.

        This function is essential for managing stakes across multiple neurons, reflecting the dynamic and collaborative
        nature of the Bittensor network.
        """
        return await add_stake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            netuids=netuids,
            amounts=amounts,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def burned_register(
        self,
        wallet: "Wallet",
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        period: Optional[int] = None,
    ) -> bool:
        """
        Registers a neuron on the Bittensor network by recycling TAO. This method of registration involves recycling
        TAO tokens, allowing them to be re-mined by performing work on the network.

        Arguments:
            wallet: The wallet associated with the neuron to be registered.
            netuid: The unique identifier of the subnet.
            wait_for_inclusion: Waits for the transaction to be included in a block. Defaults to
                ``False``.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            bool: `True` if the registration is successful, False otherwise.
        """
        async with self:
            if netuid == 0:
                return await root_register_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    period=period,
                )

            return await burned_register_extrinsic(
                subtensor=self,
                wallet=wallet,
                netuid=netuid,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
            )

    async def commit_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        salt: list[int],
        uids: Union[NDArray[np.int64], list],
        weights: Union[NDArray[np.int64], list],
        version_key: int = version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
        period: Optional[int] = 16,
    ) -> tuple[bool, str]:
        """
        Commits a hash of the subnet validator's weight vector to the Bittensor blockchain using the provided wallet.
        This action serves as a commitment or snapshot of the validator's current weight distribution.

        Arguments:
            wallet: The wallet associated with the subnet validator committing the weights.
            netuid: The unique identifier of the subnet.
            salt: list of randomly generated integers as salt to generated weighted hash.
            uids: NumPy array of subnet miner neuron UIDs for which weights are being committed.
            weights: of weight values corresponding toon_key
            version_key: Integer representation of version key for compatibility with the network.
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is
                `False`.
            max_retries: The number of maximum attempts to commit weights. Default is `5`.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            tuple[bool, str]:
                `True` if the weight commitment is successful, False otherwise.
                `msg` is a string value describing the success or potential error.

        This function allows subnet validators to create a tamper-proof record of their weight vector at a specific
        point in time, creating a foundation of transparency and accountability for the Bittensor network.

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#commit-reveal>,
        """
        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to commit weights!"

        logging.info(
            f"Committing weights with params: "
            f"netuid=[blue]{netuid}[/blue], uids=[blue]{uids}[/blue], weights=[blue]{weights}[/blue], "
            f"version_key=[blue]{version_key}[/blue]"
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

        while retries < max_retries and success is False:
            try:
                success, message = await commit_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    commit_hash=commit_hash,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    period=period,
                )
                if success:
                    break
            except Exception as e:
                logging.error(f"Error committing weights: {e}")
                retries += 1

        return success, message

    async def modify_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        position_id: int,
        liquidity_delta: Balance,
        hotkey: Optional[str] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Modifies liquidity in liquidity position by adding or removing liquidity from it.

        Arguments:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            position_id: The id of the position record in the pool.
            liquidity_delta: The amount of liquidity to be added or removed (add if positive or remove if negative).
            hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used. Defaults to
                `None`.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block. Defaults to True.
            wait_for_finalization: Whether to wait for finalization of the extrinsic. Defaults to False.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.

        Returns:
            Tuple[bool, str]:
                - True and a success message if the extrinsic is successfully submitted or processed.
                - False and an error message if the submission fails or the wallet cannot be unlocked.

        Example:
            import bittensor as bt

            subtensor = bt.AsyncSubtensor(network="local")
            await subtensor.initialize()

            my_wallet = bt.Wallet()

            # if `liquidity_delta` is negative
            my_liquidity_delta = Balance.from_tao(100) * -1
            await subtensor.modify_liquidity(
                wallet=my_wallet,
                netuid=123,
                position_id=2,
                liquidity_delta=my_liquidity_delta
            )

            # if `liquidity_delta` is positive
            my_liquidity_delta = Balance.from_tao(120)
            await subtensor.modify_liquidity(
                wallet=my_wallet,
                netuid=123,
                position_id=2,
                liquidity_delta=my_liquidity_delta
            )

        Note: Modifying is allowed even when user liquidity is enabled in specified subnet. Call `toggle_user_liquidity`
            to enable/disable user liquidity.
        """
        return await modify_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            position_id=position_id,
            liquidity_delta=liquidity_delta,
            hotkey=hotkey,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def move_stake(
        self,
        wallet: "Wallet",
        origin_hotkey: str,
        origin_netuid: int,
        destination_hotkey: str,
        destination_netuid: int,
        amount: Balance,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        period: Optional[int] = None,
    ) -> bool:
        """
        Moves stake to a different hotkey and/or subnet.

        Arguments:
            wallet: The wallet to move stake from.
            origin_hotkey: The SS58 address of the source hotkey.
            origin_netuid: The netuid of the source subnet.
            destination_hotkey: The SS58 address of the destination hotkey.
            destination_netuid: The netuid of the destination subnet.
            amount: Amount of stake to move.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            success: True if the stake movement was successful.
        """
        amount = check_and_convert_to_balance(amount)
        return await move_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            origin_hotkey=origin_hotkey,
            origin_netuid=origin_netuid,
            destination_hotkey=destination_hotkey,
            destination_netuid=destination_netuid,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def register(
        self: "AsyncSubtensor",
        wallet: "Wallet",
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        max_allowed_attempts: int = 3,
        output_in_place: bool = False,
        cuda: bool = False,
        dev_id: Union[list[int], int] = 0,
        tpb: int = 256,
        num_processes: Optional[int] = None,
        update_interval: Optional[int] = None,
        log_verbose: bool = False,
        period: Optional[int] = None,
    ):
        """
        Registers a neuron on the Bittensor network using the provided wallet.

        Registration is a critical step for a neuron to become an active participant in the network, enabling it to
        stake, set weights, and receive incentives.

        Arguments:
            wallet: The wallet associated with the neuron to be registered.
            netuid: unique identifier of the subnet.
            wait_for_inclusion: Waits for the transaction to be included in a block. Defaults to `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Defaults to
            max_allowed_attempts: Maximum number of attempts to register the wallet.
            output_in_place: If true, prints the progress of the proof of work to the console in-place. Meaning
                the progress is printed on the same lines. Defaults to `True`.
            cuda: If `true`, the wallet should be registered using CUDA device(s). Defaults to `False`.
            dev_id: The CUDA device id to use, or a list of device ids. Defaults to `0` (zero).
            tpb: The number of threads per block (CUDA). Default to `256`.
            num_processes: The number of processes to use to register. Default to `None`.
            update_interval: The number of nonces to solve between updates.  Default to `None`.
            log_verbose: If `true`, the registration process will log more information.  Default to `False`.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            bool: `True` if the registration is successful, False otherwise.

        This function facilitates the entry of new neurons into the network, supporting the decentralized growth and
        scalability of the Bittensor ecosystem.
        """
        return await register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            max_allowed_attempts=max_allowed_attempts,
            tpb=tpb,
            update_interval=update_interval,
            num_processes=num_processes,
            cuda=cuda,
            dev_id=dev_id,
            output_in_place=output_in_place,
            log_verbose=log_verbose,
            period=period,
        )

    async def register_subnet(
        self: "AsyncSubtensor",
        wallet: "Wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        period: Optional[int] = None,
    ) -> bool:
        """
        Registers a new subnetwork on the Bittensor network.

        Arguments:
            wallet: The wallet to be used for subnet registration.
            wait_for_inclusion: If set, waits for the extrinsic to enter a block before returning `True`,
                os `False` if the extrinsic fails to enter the block within the timeout. Default is `False`.
            wait_for_finalization: If set, waits for the extrinsic to be finalized on the chain before returning
                true, or returns false if the extrinsic fails to be finalized within the timeout. Default is `False`.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            bool: True if the subnet registration was successful, False otherwise.
        """
        return await register_subnet_extrinsic(
            subtensor=self,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def remove_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        position_id: int,
        hotkey: Optional[str] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Remove liquidity and credit balances back to wallet's hotkey stake.

        Arguments:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            position_id: The id of the position record in the pool.
            hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used. Defaults to
                `None`.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block. Defaults to True.
            wait_for_finalization: Whether to wait for finalization of the extrinsic. Defaults to False.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.

        Returns:
            Tuple[bool, str]:
                - True and a success message if the extrinsic is successfully submitted or processed.
                - False and an error message if the submission fails or the wallet cannot be unlocked.

        Note:
            - Adding is allowed even when user liquidity is enabled in specified subnet. Call `toggle_user_liquidity`
                extrinsic to enable/disable user liquidity.
            - To get the `position_id` use `get_liquidity_list` method.
        """
        return await remove_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            position_id=position_id,
            hotkey=hotkey,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def reveal_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: Union[NDArray[np.int64], list],
        weights: Union[NDArray[np.int64], list],
        salt: Union[NDArray[np.int64], list],
        version_key: int = version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """
        Reveals the weight vector for a specific subnet on the Bittensor blockchain using the provided wallet.
        This action serves as a revelation of the subnet validator's previously committed weight distribution as part
        of the commit-reveal mechanism.

        Arguments:
            wallet: The wallet associated with the subnet validator revealing the weights.
            netuid: unique identifier of the subnet.
            uids: NumPy array of subnet miner neuron UIDs for which weights are being revealed.
            weights: NumPy array of weight values corresponding to each UID.
            salt: NumPy array of salt values
            version_key: Version key for compatibility with the network. Default is `int representation of
                the Bittensor version`.
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is
                `False`.
            max_retries: The number of maximum attempts to reveal weights. Default is `5`.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            tuple[bool, str]: `True` if the weight revelation is successful, False otherwise. And `msg`, a string
                value describing the success or potential error.

        This function allows subnet validators to reveal their previously committed weight vector.

        See also: <https://docs.learnbittensor.org/glossary#commit-reveal>,
        """
        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to reveal weights!"

        while retries < max_retries and success is False:
            try:
                success, message = await reveal_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    uids=list(uids),
                    weights=list(weights),
                    salt=list(salt),
                    version_key=version_key,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    period=period,
                )
                if success:
                    break
            except Exception as e:
                logging.error(f"Error revealing weights: {e}")
                retries += 1

        return success, message

    async def root_set_pending_childkey_cooldown(
        self,
        wallet: "Wallet",
        cooldown: int,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Sets the pending childkey cooldown.

        Arguments:
            wallet: bittensor wallet instance.
            cooldown: the number of blocks to setting pending childkey cooldown.
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is
                `False`.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            tuple[bool, str]: A tuple where the first element is a boolean indicating success or failure of the
                operation, and the second element is a message providing additional information.

        Note: This operation can only be successfully performed if your wallet has root privileges.
        """
        return await root_set_pending_childkey_cooldown_extrinsic(
            subtensor=self,
            wallet=wallet,
            cooldown=cooldown,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    # TODO: remove `block_hash` argument
    async def root_register(
        self,
        wallet: "Wallet",
        block_hash: Optional[str] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        period: Optional[int] = None,
    ) -> bool:
        """
        Register neuron by recycling some TAO.

        Arguments:
            wallet: Bittensor wallet instance.
            block_hash: This argument will be removed in Bittensor v10
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is
                `False`.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            `True` if registration was successful, otherwise `False`.
        """

        return await root_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def root_set_weights(
        self,
        wallet: "Wallet",
        netuids: list[int],
        weights: list[float],
        version_key: int = 0,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        period: Optional[int] = None,
    ) -> bool:
        """
        Set weights for the root network.

        Arguments:
            wallet: bittensor wallet instance.
            netuids: The list of subnet uids.
            weights: The list of weights to be set.
            version_key: Version key for compatibility with the network. Default is `0`.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Defaults to `False`.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

        Returns:
            `True` if the setting of weights is successful, `False` otherwise.
        """
        netuids_, weights_ = convert_uids_and_weights(netuids, weights)
        logging.info(f"Setting weights in network: [blue]{self.network}[/blue]")
        # Run the set weights operation.
        return await set_root_weights_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids_,
            weights=weights_,
            version_key=version_key,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
            period=period,
        )

    async def set_children(
        self,
        wallet: "Wallet",
        hotkey: str,
        netuid: int,
        children: list[tuple[float, str]],
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        raise_error: bool = False,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """
        Allows a coldkey to set children-keys.

        Arguments:
            wallet: bittensor wallet instance.
            hotkey: The `SS58` address of the neuron's hotkey.
            netuid: The netuid value.
            children: A list of children with their proportions.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            tuple[bool, str]: A tuple where the first element is a boolean indicating success or failure of the
             operation, and the second element is a message providing additional information.

        Raises:
            DuplicateChild: There are duplicates in the list of children.
            InvalidChild: Child is the hotkey.
            NonAssociatedColdKey: The coldkey does not own the hotkey or the child is the same as the hotkey.
            NotEnoughStakeToSetChildkeys: Parent key doesn't have minimum own stake.
            ProportionOverflow: The sum of the proportions does exceed uint64.
            RegistrationNotPermittedOnRootSubnet: Attempting to register a child on the root network.
            SubNetworkDoesNotExist: Attempting to register to a non-existent network.
            TooManyChildren: Too many children in request.
            TxRateLimitExceeded: Hotkey hit the rate limit.
            bittensor_wallet.errors.KeyFileError: Failed to decode keyfile data.
            bittensor_wallet.errors.PasswordError: Decryption failed or wrong password for decryption provided.
        """
        return await set_children_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey=hotkey,
            netuid=netuid,
            children=children,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            raise_error=raise_error,
            period=period,
        )

    async def set_delegate_take(
        self,
        wallet: "Wallet",
        hotkey_ss58: str,
        take: float,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        raise_error: bool = False,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """
        Sets the delegate 'take' percentage for a neuron identified by its hotkey.
        The 'take' represents the percentage of rewards that the delegate claims from its nominators' stakes.

        Arguments:
            wallet: bittensor wallet instance.
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            take: Percentage reward for the delegate.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on_error: Raises a relevant exception
                rather than returning ``False`` if unsuccessful.
            raise_error: raises a relevant exception rather than returning ``False`` if unsuccessful.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            tuple[bool, str]: A tuple where the first element is a boolean indicating success or failure of the
             operation, and the second element is a message providing additional information.

        Raises:
            DelegateTakeTooHigh: Delegate take is too high.
            DelegateTakeTooLow: Delegate take is too low.
            DelegateTxRateLimitExceeded: A transactor exceeded the rate limit for delegate transaction.
            HotKeyAccountNotExists: The hotkey does not exist.
            NonAssociatedColdKey: Request to stake, unstake, or subscribe is made by a coldkey that is not associated
                with the hotkey account.
            bittensor_wallet.errors.PasswordError: Decryption failed or wrong password for decryption provided.
            bittensor_wallet.errors.KeyFileError: Failed to decode keyfile data.

        The delegate take is a critical parameter in the network's incentive structure, influencing the distribution of
        rewards among neurons and their nominators.
        """

        # u16 representation of the take
        take_u16 = int(take * 0xFFFF)

        current_take = await self.get_delegate_take(hotkey_ss58)
        current_take_u16 = int(current_take * 0xFFFF)

        if current_take_u16 == take_u16:
            logging.info(":white_heavy_check_mark: [green]Already Set[/green]")
            return True, ""

        logging.info(f"Updating {hotkey_ss58} take: current={current_take} new={take}")

        if current_take_u16 < take_u16:
            success, error = await increase_take_extrinsic(
                self,
                wallet,
                hotkey_ss58,
                take_u16,
                wait_for_finalization=wait_for_finalization,
                wait_for_inclusion=wait_for_inclusion,
                raise_error=raise_error,
                period=period,
            )
        else:
            success, error = await decrease_take_extrinsic(
                self,
                wallet,
                hotkey_ss58,
                take_u16,
                wait_for_finalization=wait_for_finalization,
                wait_for_inclusion=wait_for_inclusion,
                raise_error=raise_error,
                period=period,
            )

        if success:
            logging.info(":white_heavy_check_mark: [green]Take Updated[/green]")

        return success, error

    async def set_subnet_identity(
        self,
        wallet: "Wallet",
        netuid: int,
        subnet_identity: SubnetIdentity,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """
        Sets the identity of a subnet for a specific wallet and network.

        Arguments:
            wallet: The wallet instance that will authorize the transaction.
            netuid: The unique ID of the network on which the operation takes place.
            subnet_identity: The identity data of the subnet including attributes like name, GitHub
                repository, contact, URL, discord, description, and any additional metadata.
            wait_for_inclusion: Indicates if the function should wait for the transaction to be included in the
                block.
            wait_for_finalization: Indicates if the function should wait for the transaction to reach
                finalization.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.

        Returns:
            tuple[bool, str]: A tuple where the first element is a boolean indicating success or failure of the
             operation, and the second element is a message providing additional information.
        """
        return await set_subnet_identity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            subnet_name=subnet_identity.subnet_name,
            github_repo=subnet_identity.github_repo,
            subnet_contact=subnet_identity.subnet_contact,
            subnet_url=subnet_identity.subnet_url,
            logo_url=subnet_identity.logo_url,
            discord=subnet_identity.discord,
            description=subnet_identity.description,
            additional=subnet_identity.additional,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def set_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: Union[NDArray[np.int64], "torch.LongTensor", list],
        weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
        version_key: int = version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
        block_time: float = 12.0,
        period: Optional[int] = 8,
    ):
        """
        Sets the weight vector for a neuron acting as a validator, specifying the weights assigned to subnet miners
        based on their performance evaluation.

        This method allows subnet validators to submit their weight vectors, which rank the value of each subnet miner's
        work. These weight vectors are used by the Yuma Consensus algorithm to compute emissions for both validators and
        miners.

        Arguments:
            wallet: The wallet associated with the subnet validator setting the weights.
            netuid: The unique identifier of the subnet.
            uids: The list of subnet miner neuron UIDs that the weights are being set for.
            weights: The corresponding weights to be set for each UID, representing the validator's evaluation of each
                miner's performance.
            version_key: Version key for compatibility with the network.  Default is int representation of
                the Bittensor version.
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is
                `False`.
            max_retries: The number of maximum attempts to set weights. Default is `5`.
            block_time: The number of seconds for block duration. Default is 12.0 seconds.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction. Default is 8.

        Returns:
            tuple[bool, str]: `True` if the setting of weights is successful, False otherwise. And `msg`, a string
                value describing the success or potential error.

        This function is crucial in the Yuma Consensus mechanism, where each validator's weight vector contributes to
        the overall weight matrix used to calculate emissions and maintain network consensus.

        Notes:
            See <https://docs.learnbittensor.org/glossary#yuma-consensus>
        """

        async def _blocks_weight_limit() -> bool:
            bslu, wrl = await asyncio.gather(
                self.blocks_since_last_update(netuid, uid),
                self.weights_rate_limit(netuid),
            )
            return bslu > wrl

        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to set weights!"
        if (
            uid := await self.get_uid_for_hotkey_on_subnet(
                wallet.hotkey.ss58_address, netuid
            )
        ) is None:
            return (
                False,
                f"Hotkey {wallet.hotkey.ss58_address} not registered in subnet {netuid}",
            )

        if await self.commit_reveal_enabled(netuid=netuid):
            # go with `commit reveal v3` extrinsic

            while (
                retries < max_retries
                and success is False
                and await _blocks_weight_limit()
            ):
                logging.info(
                    f"Committing weights for subnet #{netuid}. Attempt {retries + 1} of {max_retries}."
                )
                success, message = await commit_reveal_v3_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    uids=uids,
                    weights=weights,
                    version_key=version_key,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    block_time=block_time,
                    period=period,
                )
                retries += 1
            return success, message
        else:
            # go with classic `set weights extrinsic`

            while (
                retries < max_retries
                and success is False
                and await _blocks_weight_limit()
            ):
                try:
                    logging.info(
                        f"Setting weights for subnet #[blue]{netuid}[/blue]. "
                        f"Attempt [blue]{retries + 1}[/blue] of [green]{max_retries}[/green]."
                    )
                    success, message = await set_weights_extrinsic(
                        subtensor=self,
                        wallet=wallet,
                        netuid=netuid,
                        uids=uids,
                        weights=weights,
                        version_key=version_key,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                        period=period,
                    )
                except Exception as e:
                    logging.error(f"Error setting weights: {e}")
                    retries += 1

            return success, message

    async def serve_axon(
        self,
        netuid: int,
        axon: "Axon",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        certificate: Optional[Certificate] = None,
        period: Optional[int] = None,
    ) -> bool:
        """
        Registers an ``Axon`` serving endpoint on the Bittensor network for a specific neuron. This function is used to
        set up the Axon, a key component of a neuron that handles incoming queries and data processing tasks.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            axon: The Axon instance to be registered for serving.
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is `True`.
            certificate: Certificate to use for TLS. If `None`, no TLS will be used. Defaults to `None`.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.

        Returns:
            bool: `True` if the Axon serve registration is successful, False otherwise.

        By registering an Axon, the neuron becomes an active part of the network's distributed computing infrastructure,
        contributing to the collective intelligence of Bittensor.
        """
        return await serve_axon_extrinsic(
            subtensor=self,
            netuid=netuid,
            axon=axon,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            certificate=certificate,
            period=period,
        )

    async def start_call(
        self,
        wallet: "Wallet",
        netuid: int,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """
        Submits a start_call extrinsic to the blockchain, to trigger the start call process for a subnet (used to start
        a new subnet's emission mechanism).

        Arguments:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block. Defaults to `True`.
            wait_for_finalization: Whether to wait for finalization of the extrinsic. Defaults to `False`.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.

        Returns:
            Tuple[bool, str]:
                - True and a success message if the extrinsic is successfully submitted or processed.
                - False and an error message if the submission fails or the wallet cannot be unlocked.
        """
        return await start_call_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def swap_stake(
        self,
        wallet: "Wallet",
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        safe_staking: bool = False,
        allow_partial_stake: bool = False,
        rate_tolerance: float = 0.005,
        period: Optional[int] = None,
    ) -> bool:
        """
        Moves stake between subnets while keeping the same coldkey-hotkey pair ownership.
        Like subnet hopping - same owner, same hotkey, just changing which subnet the stake is in.

        Arguments:
            wallet: The wallet to swap stake from.
            hotkey_ss58: The SS58 address of the hotkey whose stake is being swapped.
            origin_netuid: The netuid from which stake is removed.
            destination_netuid: The netuid to which stake is added.
            amount: The amount to swap.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            safe_staking: If true, enables price safety checks to protect against fluctuating prices. The swap will only
                execute if the price ratio between subnets doesn't exceed the rate tolerance. Default is False.
            allow_partial_stake: If true and safe_staking is enabled, allows partial stake swaps when the full amount
                would exceed the price threshold. If false, the entire swap fails if it would exceed the threshold.
                Default is False.
            rate_tolerance: The maximum allowed increase in the price ratio between subnets
                (origin_price/destination_price). For example, 0.005 = 0.5% maximum increase. Only used when
                safe_staking is True. Default is 0.005.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.

        Returns:
            success: True if the extrinsic was successful.

        The price ratio for swap_stake in safe mode is calculated as: origin_subnet_price / destination_subnet_price
        When safe_staking is enabled, the swap will only execute if:
            - With allow_partial_stake=False: The entire swap amount can be executed without the price ratio increasing
            more than rate_tolerance.
            - With allow_partial_stake=True: A partial amount will be swapped up to the point where the price ratio
            would increase by rate_tolerance.
        """
        amount = check_and_convert_to_balance(amount)
        return await swap_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            safe_staking=safe_staking,
            allow_partial_stake=allow_partial_stake,
            rate_tolerance=rate_tolerance,
            period=period,
        )

    async def toggle_user_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        enable: bool,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Allow to toggle user liquidity for specified subnet.

        Arguments:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            enable: Boolean indicating whether to enable user liquidity.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block. Defaults to True.
            wait_for_finalization: Whether to wait for finalization of the extrinsic. Defaults to False.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.

        Returns:
            Tuple[bool, str]:
                - True and a success message if the extrinsic is successfully submitted or processed.
                - False and an error message if the submission fails or the wallet cannot be unlocked.

        Note: The call can be executed successfully by the subnet owner only.
        """
        return await toggle_user_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            enable=enable,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def transfer(
        self,
        wallet: "Wallet",
        dest: str,
        amount: Optional[Balance],
        transfer_all: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        keep_alive: bool = True,
        period: Optional[int] = None,
    ) -> bool:
        """
        Transfer token of amount to destination.

        Arguments:
            wallet: Source wallet for the transfer.
            dest: Destination address for the transfer.
            amount: Number of tokens to transfer. `None` is transferring all.
            transfer_all: Flag to transfer all tokens. Default is `False`.
            wait_for_inclusion: Waits for the transaction to be included in a block. Defaults to `True`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Defaults to `False`.
            keep_alive: Flag to keep the connection alive. Default is `True`.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
        Returns:
            `True` if the transferring was successful, otherwise `False`.
        """
        if amount is not None:
            amount = check_and_convert_to_balance(amount)
        return await transfer_extrinsic(
            subtensor=self,
            wallet=wallet,
            dest=dest,
            amount=amount,
            transfer_all=transfer_all,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            keep_alive=keep_alive,
            period=period,
        )

    async def transfer_stake(
        self,
        wallet: "Wallet",
        destination_coldkey_ss58: str,
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        period: Optional[int] = None,
    ) -> bool:
        """
        Transfers stake from one subnet to another while changing the coldkey owner.

        Arguments:
            wallet: The wallet to transfer stake from.
            destination_coldkey_ss58: The destination coldkey SS58 address.
            hotkey_ss58: The hotkey SS58 address associated with the stake.
            origin_netuid: The source subnet UID.
            destination_netuid: The destination subnet UID.
            amount: Amount to transfer.
            wait_for_inclusion: If true, waits for inclusion before returning.
            wait_for_finalization: If true, waits for finalization before returning.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.

        Returns:
            success: True if the transfer was successful.
        """
        amount = check_and_convert_to_balance(amount)
        return await transfer_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            destination_coldkey_ss58=destination_coldkey_ss58,
            hotkey_ss58=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def unstake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        netuid: Optional[int] = None,
        amount: Optional[Balance] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        safe_staking: bool = False,
        allow_partial_stake: bool = False,
        rate_tolerance: float = 0.005,
        period: Optional[int] = None,
        unstake_all: bool = False,
    ) -> bool:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting
        individual neuron stakes within the Bittensor network.

        Arguments:
            wallet: The wallet associated with the neuron from which the stake is being
                removed.
            hotkey_ss58: The `SS58` address of the hotkey account to unstake from.
            netuid: The unique identifier of the subnet.
            amount: The amount of alpha to unstake. If not specified, unstakes all.
            wait_for_inclusion: Waits for the transaction to be included in a block. Defaults to `True`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Defaults to `False`.
            safe_staking: If true, enables price safety checks to protect against fluctuating prices. The unstake will
                only execute if the price change doesn't exceed the rate tolerance. Default is False.
            allow_partial_stake: If true and safe_staking is enabled, allows partial unstaking when
                the full amount would exceed the price threshold. If false, the entire unstake fails if it would exceed
                the threshold. Default is False.
            rate_tolerance: The maximum allowed price change ratio when unstaking. For example, 0.005 = 0.5% maximum
                price decrease. Only used when safe_staking is True. Default is 0.005.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            unstake_all: If `True`, unstakes all tokens and `amount` is ignored. Default is `False`

        Returns:
            bool: `True` if the unstaking process is successful, False otherwise.

        This function supports flexible stake management, allowing neurons to adjust their network participation and
        potential reward accruals.
        """
        amount = check_and_convert_to_balance(amount)
        return await unstake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            safe_staking=safe_staking,
            allow_partial_stake=allow_partial_stake,
            rate_tolerance=rate_tolerance,
            period=period,
            unstake_all=unstake_all,
        )

    async def unstake_all(
        self,
        wallet: "Wallet",
        hotkey: str,
        netuid: int,
        rate_tolerance: Optional[float] = 0.005,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        period: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Unstakes all TAO/Alpha associated with a hotkey from the specified subnets on the Bittensor network.

        Arguments:
            wallet: The wallet of the stake owner.
            hotkey: The SS58 address of the hotkey to unstake from.
            netuid: The unique identifier of the subnet.
            rate_tolerance: The maximum allowed price change ratio when unstaking. For example, 0.005 = 0.5% maximum
                price decrease. If not passed (None), then unstaking goes without price limit. Default is 0.005.
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `True`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is `False`.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction. Default is `None`.

        Returns:
            tuple[bool, str]:
                A tuple containing:
                - `True` and a success message if the unstake operation succeeded;
                - `False` and an error message otherwise.

        Example:
            # If you would like to unstake all stakes in all subnets safely, use default `rate_tolerance` or pass your
                value:
            import bittensor as bt

            subtensor = bt.AsyncSubtensor()
            wallet = bt.Wallet("my_wallet")
            netuid = 14
            hotkey = "5%SOME_HOTKEY_WHERE_IS_YOUR_STAKE_NOW%"

            wallet_stakes = await subtensor.get_stake_info_for_coldkey(coldkey_ss58=wallet.coldkey.ss58_address)

            for stake in wallet_stakes:
                result = await subtensor.unstake_all(
                    wallet=wallet,
                    hotkey_ss58=stake.hotkey_ss58,
                    netuid=stake.netuid,
                )
                print(result)

            # If you would like to unstake all stakes in all subnets unsafely, use `rate_tolerance=None`:
                        import bittensor as bt

            subtensor = bt.AsyncSubtensor()
            wallet = bt.Wallet("my_wallet")
            netuid = 14
            hotkey = "5%SOME_HOTKEY_WHERE_IS_YOUR_STAKE_NOW%"

            wallet_stakes = await subtensor.get_stake_info_for_coldkey(coldkey_ss58=wallet.coldkey.ss58_address)

            for stake in wallet_stakes:
                result = await subtensor.unstake_all(
                    wallet=wallet,
                    hotkey_ss58=stake.hotkey_ss58,
                    netuid=stake.netuid,
                    rate_tolerance=None,
                )
                print(result)
        """
        if netuid != 0:
            logging.debug(
                f"Unstaking without Alpha price control from subnet [blue]#{netuid}[/blue]."
            )
        return await unstake_all_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey=hotkey,
            netuid=netuid,
            rate_tolerance=rate_tolerance,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

    async def unstake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        netuids: list[int],
        amounts: Optional[list[Balance]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        period: Optional[int] = None,
        unstake_all: bool = False,
    ) -> bool:
        """
        Performs batch unstaking from multiple hotkey accounts, allowing a neuron to reduce its staked amounts
        efficiently. This function is useful for managing the distribution of stakes across multiple neurons.

        Arguments:
            wallet: The wallet linked to the coldkey from which the stakes are being withdrawn.
            hotkey_ss58s: A list of hotkey `SS58` addresses to unstake from.
            netuids: Subnets unique IDs.
            amounts: The amounts of TAO to unstake from each hotkey. If not provided, unstakes all.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            unstake_all: If true, unstakes all tokens. Default is `False`. If `True` amounts are ignored.

        Returns:
            bool: `True` if the batch unstaking is successful, False otherwise.

        This function allows for strategic reallocation or withdrawal of stakes, aligning with the dynamic stake
        management aspect of the Bittensor network.
        """
        return await unstake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            netuids=netuids,
            amounts=amounts,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            unstake_all=unstake_all,
        )


async def get_async_subtensor(
    network: Optional[str] = None,
    config: Optional["Config"] = None,
    _mock: bool = False,
    log_verbose: bool = False,
) -> "AsyncSubtensor":
    """Factory method to create an initialized AsyncSubtensor.
    Mainly useful for when you don't want to run `await subtensor.initialize()` after instantiation.
    """
    sub = AsyncSubtensor(
        network=network, config=config, _mock=_mock, log_verbose=log_verbose
    )
    await sub.initialize()
    return sub
