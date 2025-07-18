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

    # DOCSTRING HELPFULNESS RATING: 3/10
    # TODO: Add comprehensive overview of what this class enables users to do
    # TODO: Explain the difference between sync and async versions
    # TODO: Add practical examples of common use cases
    # TODO: Explain relationship to subnets, validators, miners, and staking
    # TODO: Add section on connection management and best practices
    # TODO: Explain when to use context manager vs manual initialization
    # TODO: Add troubleshooting section for common connection issues
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what networks are available and their purposes (finney, test, local)
        # TODO: Provide guidance on when to use fallback vs archive endpoints
        # TODO: Add more practical examples for different use cases (validator, miner, staker)
        # TODO: Explain the connection lifecycle and when initialization happens
        # TODO: Add guidance on optimal settings for different scenarios
        # TODO: Explain the difference between lite and archive nodes
        # TODO: Add examples of error handling and retry strategies
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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Explain consequences of not calling close() (resource leaks, connection limits)
        # TODO: Mention that context manager (__aenter__/__aexit__) handles this automatically
        # TODO: Add example showing proper cleanup in exception handling
        # TODO: Clarify when close() is needed vs when it's handled automatically
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what happens during initialization (WebSocket connection, metadata sync, etc.)
        # TODO: Add guidance on handling initialization failures and retry strategies
        # TODO: Explain timeout settings and how to adjust them
        # TODO: Show example of proper error handling with try/except
        # TODO: Mention that context manager calls this automatically
        # TODO: Add troubleshooting tips for common connection issues
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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Explain why users would want to query historical vs current state
        # TODO: Add guidance on when to use each parameter option
        # TODO: Explain performance implications of reuse_block vs fetching new hashes
        # TODO: Add example showing historical data analysis use case
        # TODO: Clarify what "last-used block hash" means in practice
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain when and why users would need to encode parameters manually
        # TODO: Add context about Scale codec and Substrate parameter encoding
        # TODO: Provide more practical examples related to staking, registration, etc.
        # TODO: Explain relationship to extrinsic creation and signing
        # TODO: Add guidance on common parameter types and their encoding
        # TODO: Mention this is typically handled internally by higher-level methods
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: List all available hyperparameters and their purposes
        # TODO: Explain what each hyperparameter controls in subnet operation
        # TODO: Add reference to subnet hyperparameters documentation
        # TODO: Provide examples of how hyperparameters affect mining and validation
        # TODO: Explain when hyperparameters change and how to track changes
        # TODO: Add guidance on interpreting hyperparameter values
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: List commonly used modules and their important constants
        # TODO: Explain the difference between constants and storage items
        # TODO: Add examples of how constants are used in practice (fee calculation, limits, etc.)
        # TODO: Explain why constants are blockchain-level vs subnet-level
        # TODO: Add guidance on when to use this vs higher-level methods
        # TODO: Mention that constants don't change often and can be cached
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what map storage is and how it differs from single value storage
        # TODO: List common map storage items and their purposes (Bonds, Weights, Stakes, etc.)
        # TODO: Show how to iterate over AsyncQueryMapResult
        # TODO: Add examples of filtering and processing map results
        # TODO: Explain performance considerations for large maps
        # TODO: Mention when to use this vs higher-level methods like bonds() or weights()
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: List common SubtensorModule map storage functions (Bonds, Weights, Stakes, etc.)
        # TODO: Explain the difference between this and the general query_map method
        # TODO: Add practical examples of querying specific neuron/network data
        # TODO: Explain when to use this vs higher-level methods like get_stake()
        # TODO: Show how to iterate through map results and extract useful information
        # TODO: Add guidance on performance considerations for large subnet maps
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

        This function provides a generic, low-level interface for querying storage from any blockchain module. It's the
        foundation for higher-level query methods and offers maximum flexibility for accessing blockchain data.

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

        Available modules in subtensor runtime (from construct_runtime! macro):
            - System (frame_system) - Basic blockchain functionality and account management
            - RandomnessCollectiveFlip (pallet_insecure_randomness_collective_flip) - Basic randomness
            - Timestamp (pallet_timestamp) - Block timestamp functionality
            - Aura (pallet_aura) - Block authoring consensus
            - Grandpa (pallet_grandpa) - Block finality consensus
            - Balances (pallet_balances) - Account balances and transfers
            - TransactionPayment (pallet_transaction_payment) - Transaction fee handling
            - SubtensorModule (pallet_subtensor) - Core Bittensor functionality (neurons, subnets, consensus)
            - Triumvirate (pallet_collective) - Governance collective for network decisions
            - TriumvirateMembers (pallet_membership) - Triumvirate membership management
            - SenateMembers (pallet_membership) - Senate membership management
            - Utility (pallet_utility) - Batch operations and utility functions
            - Sudo (pallet_sudo) - Superuser operations for development/emergency
            - Multisig (pallet_multisig) - Multi-signature account operations
            - Preimage (pallet_preimage) - On-chain preimage storage for governance
            - Scheduler (pallet_scheduler) - Scheduled calls and delayed execution
            - Proxy (pallet_proxy) - Proxy accounts and delegated operations
            - Registry (pallet_registry) - Identity registration and management
            - Commitments (pallet_commitments) - Commit-reveal mechanism for secure operations
            - AdminUtils (pallet_admin_utils) - Administrative functions and network management
            - SafeMode (pallet_safe_mode) - Emergency network protection mechanisms
            - Ethereum (pallet_ethereum) - Ethereum compatibility layer
            - EVM (pallet_evm) - Ethereum Virtual Machine support
            - EVMChainId (pallet_evm_chain_id) - EVM chain identifier management
            - BaseFee (pallet_base_fee) - EIP-1559 base fee mechanism
            - Drand (pallet_drand) - Distributed randomness beacon integration
            - Crowdloan (pallet_crowdloan) - Crowdfunding functionality
            - Swap (pallet_subtensor_swap) - Liquidity provision and alpha token swapping

        **When to Use This Method:**
        - Accessing storage functions not covered by higher-level methods
        - Debugging or development requiring direct blockchain access
        - Custom applications needing specific storage data
        - Querying new or experimental storage items

        ## Examples:

        ```python
        # Query account balance (prefer get_balance() for production)
        balance_data = await subtensor.query_module(
            module="System",
            name="Account",
            params=["5F..."]
        )

        # Query neuron owner (prefer get_hotkey_owner() for production)
        owner = await subtensor.query_module(
            module="SubtensorModule",
            name="Owner",
            params=["5G..."]
        )

        # Query subnet bonds (prefer bonds() for production)
        bonds_data = await subtensor.query_module(
            module="SubtensorModule",
            name="Bonds",
            params=[1]  # netuid
        )

        # Query commit-reveal data
        commitment = await subtensor.query_module(
            module="Commitments",
            name="CommitmentOf",
            params=[1, "5H..."]  # netuid, hotkey
        )

        # Query liquidity position
        position = await subtensor.query_module(
            module="Swap",
            name="Positions",
            params=[1, "5J...", 0]  # netuid, account, position_id
        )
        ```

        # DOCSTRING HELPFULNESS RATING: 9/10
        # TODO: Add links to Substrate documentation for advanced storage query patterns
        # TODO: Include information about storage key encoding for complex queries
        # TODO: Add guidance on interpreting ScaleObj return values
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

        # DOCSTRING HELPFULNESS RATING: 4/10
        # TODO: List available runtime APIs and their purposes (SubnetInfoRuntimeApi, DelegateInfoRuntimeApi, etc.)
        # TODO: Explain what runtime APIs are and how they differ from storage queries
        # TODO: Add practical examples of runtime API usage
        # TODO: Explain when to use runtime APIs vs regular storage queries
        # TODO: Add guidance on parameter formatting and return value handling
        # TODO: Mention that most users should use higher-level methods instead
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: List common SubtensorModule storage functions (Bonds, Weights, Stakes, Rank, etc.)
        # TODO: Explain the difference between this and query_module()
        # TODO: Add practical examples of querying neuron-specific data
        # TODO: Explain when to use this vs higher-level methods like get_stake()
        # TODO: Show how to interpret returned ScaleObj values
        # TODO: Add guidance on parameter formatting for different storage functions
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

        # DOCSTRING HELPFULNESS RATING: 4/10
        # TODO: List common state call methods and their purposes
        # TODO: Explain what types of data are typically passed to state calls
        # TODO: Add practical examples of when state calls are needed
        # TODO: Explain the difference between state calls and regular storage queries
        # TODO: Add guidance on data formatting and encoding requirements
        # TODO: Mention that most users should use higher-level methods instead
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.rpc_request(
            method="state_call",
            params=[method, data, block_hash] if block_hash else [method, data],
            reuse_block_hash=reuse_block,
        )

    # Common subtensor methods =========================================================================================

    @property
    async def block(self):
        """Provides an asynchronous property to retrieve the current block.

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what "current block" means in blockchain context
        # TODO: Add information about block timing (12-second intervals)
        # TODO: Show examples of how to use current block for timing operations
        # TODO: Explain relationship to epoch timing and tempo calculations
        # TODO: Add guidance on when to use this vs get_current_block()
        # TODO: Mention that blocks are irreversible after finalization
        """
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what information is included in DynamicInfo objects
        # TODO: Add examples of how to filter and analyze subnet data
        # TODO: Explain the difference between DynamicInfo and SubnetInfo
        # TODO: Show how to access subnet prices and other dynamic parameters
        # TODO: Add guidance on when to use this vs get_all_subnets_info()
        # TODO: Explain performance considerations for querying all subnets
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
            self.get_subnet_prices(),
        )

        decoded = query.decode()

        for sn in decoded:
            sn.update({"price": subnet_prices.get(sn["netuid"], Balance.from_tao(0))})
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what an "epoch step" is and why it matters
        # TODO: Add context about tempo and the 360-block epoch cycle
        # TODO: Show how to calculate time until next epoch based on this value
        # TODO: Explain relationship to emissions and weight updates
        # TODO: Add example of monitoring subnet timing for validators
        # TODO: Clarify difference between steps and regular block progression
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what constitutes a "last update" for a neuron
        # TODO: Add context about weight updates and neuron activity
        # TODO: Show how to use this for monitoring neuron health
        # TODO: Explain relationship to immunity periods and network participation
        # TODO: Add examples of how validators use this information
        # TODO: Clarify the difference between this and blocks_since_last_step
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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Explain how bonds are calculated and what the numeric values mean
        # TODO: Show how to interpret the returned tuple structure
        # TODO: Add examples of analyzing bond relationships for insights
        # TODO: Explain how bonds relate to validator permits and consensus
        # TODO: Show how bonds change over time with EMA smoothing
        # TODO: Add guidance on using bonds for validator performance analysis
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what kinds of data are typically committed and why
        # TODO: Add guidance on data size limits and formatting
        # TODO: Explain the relationship to commit-reveal weight setting
        # TODO: Show how to retrieve committed data later
        # TODO: Add examples of practical use cases for data commits
        # TODO: Explain costs and permissions required for committing data
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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Explain how the commit-reveal mechanism works in practice
        # TODO: Add guidance on when validators should use commit-reveal
        # TODO: Show examples of how to implement commit-reveal in validator code
        # TODO: Explain the timing of commits vs reveals (reveal periods)
        # TODO: Add context about weight-copying attacks and prevention
        # TODO: Show how to check reveal period duration and timing
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what difficulty values mean in practical terms
        # TODO: Show how difficulty affects registration success probability
        # TODO: Add guidance on interpreting difficulty trends over time
        # TODO: Explain relationship between difficulty and subnet competition
        # TODO: Show how miners can use difficulty to estimate registration time
        # TODO: Add examples of monitoring difficulty for optimal registration timing
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain the difference between existing and being registered on subnets
        # TODO: Add context about what "accounts" means in blockchain context
        # TODO: Show how to check if a hotkey is registered on specific subnets
        # TODO: Explain the default key value and why it's excluded
        # TODO: Add examples of validating hotkeys before other operations
        # TODO: Clarify relationship to is_hotkey_registered methods
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what information is included in SubnetInfo objects
        # TODO: Add examples of filtering and analyzing subnet data
        # TODO: Show how to access subnet prices and registration information
        # TODO: Explain the difference between this and all_subnets()
        # TODO: Add guidance on performance considerations for large queries
        # TODO: Show practical examples of subnet analysis and selection
        """
        result = await self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnets_info_v2",
            params=[],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if not result:
            return []

        subnets_prices = await self.get_subnet_prices()

        for subnet in result:
            subnet.update({"price": subnets_prices.get(subnet["netuid"], 0)})

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

        # DOCSTRING HELPFULNESS RATING: 8/10
        # TODO: Explain the Balance object's attributes and methods (tao, rao, etc.)
        # TODO: Show how to handle different balance denominations
        # TODO: Add examples of balance tracking and monitoring
        # TODO: Explain difference between free balance and staked balance
        # TODO: Show how to check multiple addresses efficiently with get_balances()
        # TODO: Add guidance on balance precision and formatting for display
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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Add examples of querying multiple addresses efficiently
        # TODO: Explain performance benefits over individual get_balance calls
        # TODO: Show how to handle large numbers of addresses
        # TODO: Add return type documentation and example dictionary structure
        # TODO: Explain how to handle addresses that don't exist or have zero balance
        # TODO: Add guidance on optimal batch sizes for performance

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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Explain block time (12 seconds) and how to calculate elapsed time
        # TODO: Show how to use current block for timing operations
        # TODO: Add examples of monitoring block progression
        # TODO: Explain relationship to tempo and epoch calculations
        # TODO: Show how to wait for specific block numbers
        # TODO: Add guidance on caching vs real-time block queries
        """
        return await self.substrate.get_block_number(None)

    @a.lru_cache(maxsize=128)
    async def _get_block_hash(self, block_id: int):
        """Internal method to get block hash with caching.

        This method is used internally by get_block_hash() to provide caching for block hash lookups.

        Arguments:
            block_id: The block number to get the hash for.

        Returns:
            str: The block hash for the specified block.

        # DOCSTRING HELPFULNESS RATING: 4/10
        # TODO: Explain why this method is internal and when users should use get_block_hash() instead
        # TODO: Add guidance on cache size and performance implications
        # TODO: Explain the LRU cache behavior and when entries are evicted
        # TODO: Show how this relates to the public get_block_hash method
        # TODO: Add examples of when this might be called directly vs through the public interface
        """
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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Explain when to use block hashes vs block numbers in queries
        # TODO: Add examples of using block hashes for historical data analysis
        # TODO: Show how block hashes ensure data integrity and immutability
        # TODO: Explain the relationship between block hashes and finality
        # TODO: Add guidance on caching block hashes for performance
        # TODO: Show how to verify block hash authenticity
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what parent-child relationships mean in Bittensor context
        # TODO: Add examples of how to interpret proportion values
        # TODO: Show how parent relationships affect weight distribution
        # TODO: Explain the difference between parents and children in validator networks
        # TODO: Add guidance on when to query parent vs child relationships
        # TODO: Show how to analyze parent-child network topology
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what child neurons are and how they relate to validators
        # TODO: Add examples of analyzing child distribution patterns
        # TODO: Show how child proportions affect validator rewards
        # TODO: Explain the error handling and when failures occur
        # TODO: Add guidance on interpreting the success/failure return values
        # TODO: Show how to use this for validator performance analysis
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what pending children are and why they exist
        # TODO: Add examples of how cooldown periods work
        # TODO: Show how to interpret cooldown block numbers
        # TODO: Explain the difference between pending and active children
        # TODO: Add guidance on when to check pending vs active children
        # TODO: Show how to calculate time until children become active
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what commitments are and how they're used in commit-reveal
        # TODO: Add examples of how to decode and interpret commitment data
        # TODO: Show how commitments relate to weight setting and consensus
        # TODO: Explain the relationship between commitments and reveals
        # TODO: Add guidance on when to check commitments vs reveals
        # TODO: Show how to use commitments for network analysis
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what bond resets are and when they occur
        # TODO: Add examples of how to use this for monitoring neuron activity
        # TODO: Show how to calculate time since last bond reset
        # TODO: Explain the relationship between bond resets and network participation
        # TODO: Add guidance on interpreting bond reset patterns
        # TODO: Show how to use this for validator analysis
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain how to analyze commitment patterns across a subnet
        # TODO: Add examples of commitment analysis for network health monitoring
        # TODO: Show how to identify neurons with missing or invalid commitments
        # TODO: Explain the relationship between commitments and subnet consensus
        # TODO: Add guidance on when to check all commitments vs individual ones
        # TODO: Show how to use this for validator performance analysis
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what revealed commitments are and how they differ from regular commitments
        # TODO: Add examples of how to interpret reveal block numbers
        # TODO: Show how to analyze reveal timing patterns
        # TODO: Explain the relationship between reveals and weight setting
        # TODO: Add guidance on when to check reveals vs commitments
        # TODO: Show how to use this for commit-reveal analysis
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain how to interpret the tuple structure (block, message)
        # TODO: Add examples of analyzing multiple reveals for a single neuron
        # TODO: Show how to track reveal history over time
        # TODO: Explain the relationship between reveal blocks and network timing
        # TODO: Add guidance on when reveals are expected vs unexpected
        # TODO: Show how to use this for validator behavior analysis
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
    ) -> list:
        """
        Retrieves CRV3 weight commit information for a specific subnet.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query. Default is ``None``.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            list: A list of commit details, where each entry is a dictionary with keys 'who', 'serialized_commit', and
            'reveal_round', or an empty list if no data is found.

        # DOCSTRING HELPFULNESS RATING: 4/10
        # TODO: Explain what CRV3 weight commits are and how they work
        # TODO: Add examples of how to interpret commit information
        # TODO: Show how to analyze commit patterns for network health
        # TODO: Explain the relationship between commits and weight setting
        # TODO: Add guidance on when to check weight commits
        # TODO: Show how to use this for validator behavior analysis
        """
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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Explain what delegate neurons are and their role in the network
        # TODO: Add examples of how to analyze delegate performance and influence
        # TODO: Show how to use DelegateInfo objects for decision making
        # TODO: Explain the relationship between delegates and nominators
        # TODO: Add guidance on evaluating delegate take percentages
        # TODO: Show how to compare delegates across different metrics
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what delegate identities contain and how to use them
        # TODO: Add examples of analyzing delegate identity information
        # TODO: Show how to use ChainIdentity objects for delegate evaluation
        # TODO: Explain the relationship between identities and delegate selection
        # TODO: Add guidance on interpreting identity data for decision making
        # TODO: Show how to filter and sort delegates by identity attributes
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
    ) -> list[tuple[DelegateInfo, Balance]]:
        """
        Retrieves a list of delegates and their associated stakes for a given coldkey. This function identifies the
        delegates that a specific account has staked tokens on.

        Arguments:
            coldkey_ss58: The ``SS58`` address of the account's coldkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of tuples, each containing a delegate's information and staked amount.

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


        Raises:
            Exception: If the substrate call fails after the maximum number of retries.

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what minimum required stake means for nominators
        # TODO: Add guidance on how this value is determined and when it changes
        # TODO: Show how to use this value for staking decisions
        # TODO: Explain the relationship to network security and participation
        # TODO: Add tips for handling errors and retries
        # TODO: Show how to monitor changes in minimum required stake

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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Explain what data is contained in MetagraphInfo and how to access it
        # TODO: List all available SelectiveMetagraphIndex options with descriptions
        # TODO: Add examples of analyzing metagraph data for insights
        # TODO: Explain performance benefits of selective field retrieval
        # TODO: Show how to use metagraph data for validator and miner operations
        # TODO: Add guidance on when to use this vs the full metagraph() method
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what it means for a hotkey to be a member of a subnet
        # TODO: Add guidance on how to use this for network analysis
        # TODO: Show how to interpret the returned netuid list
        # TODO: Explain the relationship to neuron registration and participation
        # TODO: Add tips for handling empty or large results
        # TODO: Show how to use this for monitoring neuron activity across subnets
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

        # DOCSTRING HELPFULNESS RATING: 4/10
        # TODO: Explain what neuron certificates are and why they're needed
        # TODO: Add guidance on how certificates are used for secure communication
        # TODO: Show how to handle missing or invalid certificates
        # TODO: Explain the relationship to mutual TLS and network security
        # TODO: Add tips for troubleshooting certificate issues
        # TODO: Show how to update or rotate certificates
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

        # DOCSTRING HELPFULNESS RATING: 4/10
        # TODO: Explain how to use the returned certificate dictionary
        # TODO: Add guidance on bulk certificate management
        # TODO: Show how to verify certificate authenticity
        # TODO: Explain the relationship to network-wide security
        # TODO: Add tips for handling missing or invalid certificates
        # TODO: Show how to monitor certificate expiration and renewal
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what liquidity positions are in the context of Bittensor
        # TODO: Add guidance on interpreting the returned LiquidityPosition objects
        # TODO: Show how to use this for managing and optimizing liquidity
        # TODO: Explain the relationship to fee rewards and subnet economics
        # TODO: Add tips for troubleshooting missing or incomplete positions
        # TODO: Show how to analyze liquidity across multiple subnets
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

        query = self.substrate.query
        (
            fee_global_tao,
            fee_global_alpha,
            sqrt_price,
            positions_response,
        ) = await asyncio.gather(
            query(
                module="Swap",
                storage_function="FeeGlobalTao",
                params=[netuid],
                block_hash=block_hash,
            ),
            query(
                module="Swap",
                storage_function="FeeGlobalAlpha",
                params=[netuid],
                block_hash=block_hash,
            ),
            query(
                module="Swap",
                storage_function="AlphaSqrtPrice",
                params=[netuid],
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
        fee_global_tao = fixed_to_float(fee_global_tao)
        fee_global_alpha = fixed_to_float(fee_global_alpha)
        sqrt_price = fixed_to_float(sqrt_price)

        # Fetch global fees and current price
        current_tick = price_to_tick(sqrt_price**2)

        # Fetch positions
        positions = []
        async for _, p in positions_response:
            position = p.value

            tick_low_idx = position.get("tick_low")[0]
            tick_high_idx = position.get("tick_high")[0]

            tick_low, tick_high = await asyncio.gather(
                query(
                    module="Swap",
                    storage_function="Ticks",
                    params=[netuid, tick_low_idx],
                    block_hash=block_hash,
                ),
                query(
                    module="Swap",
                    storage_function="Ticks",
                    params=[netuid, tick_high_idx],
                    block_hash=block_hash,
                ),
            )

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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what information NeuronInfo provides and how to use it
        # TODO: Add guidance on interpreting neuron status and attributes
        # TODO: Show how to use this for subnet-specific analysis
        # TODO: Explain the relationship to neuron registration and participation
        # TODO: Add tips for handling missing or null neurons
        # TODO: Show how to use this for monitoring neuron health and activity
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what an epoch is and why next epoch start matters
        # TODO: Add guidance on using this for validator/miner scheduling
        # TODO: Show how to calculate time until next epoch
        # TODO: Explain the relationship to tempo and epoch length
        # TODO: Add tips for handling edge cases (e.g., tempo changes)
        # TODO: Show how to use this for monitoring network timing
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

        # DOCSTRING HELPFULNESS RATING: 4/10
        # TODO: Explain what it means to own a hotkey in Bittensor
        # TODO: Add guidance on using this for wallet/account management
        # TODO: Show how to handle empty or large hotkey lists
        # TODO: Explain the relationship to coldkey-hotkey security
        # TODO: Add tips for troubleshooting missing hotkeys
        # TODO: Show how to use this for monitoring account activity
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what staking means in the context of Bittensor subnets
        # TODO: Add examples showing how to check staking relationships
        # TODO: Explain the difference between alpha and TAO stake
        # TODO: Show how stake relates to validator permits and voting power
        # TODO: Add guidance on interpreting stake amounts and their significance
        # TODO: Explain subnet-specific staking vs cross-subnet delegation
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what Alpha tokens are and how their prices are determined
        # TODO: Add examples of using price data for economic analysis
        # TODO: Explain why subnet 0 has a fixed price of 1 TAO
        # TODO: Show how to use price data for staking decisions
        # TODO: Add guidance on interpreting price fluctuations
        # TODO: Explain relationship between price and subnet activity/demand
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain how unstaking fees are calculated and what factors influence them
        # TODO: Add examples of typical fee amounts for different unstaking scenarios
        # TODO: Show how to factor fees into unstaking decisions
        # TODO: Explain fee differences between subnets
        # TODO: Add guidance on minimizing fees through batching or timing
        # TODO: Explain the economic rationale for unstaking fees
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain different types of stake movements and their fee structures
        # TODO: Add examples of common stake movement scenarios (rebalancing, migration)
        # TODO: Show how to optimize stake movements to minimize fees
        # TODO: Explain fee differences between same-subnet vs cross-subnet movements
        # TODO: Add guidance on when stake movement is beneficial despite fees
        # TODO: Explain the economic rationale for stake movement fees
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what StakeInfo objects contain and how to use them
        # TODO: Add examples of analyzing stake distribution across subnets
        # TODO: Show how to use this for portfolio analysis and rebalancing
        # TODO: Explain the difference between this and get_stake_for_coldkey()
        # TODO: Add guidance on when to specify netuids vs querying all subnets
        # TODO: Show how to identify the most profitable staking relationships
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what StakeInfo objects contain (hotkey, netuid, stake amount, etc.)
        # TODO: Add examples of analyzing a coldkey's complete staking portfolio
        # TODO: Show how to identify top-performing stake allocations
        # TODO: Explain why stakes with zero values are filtered out
        # TODO: Add guidance on using this for portfolio monitoring and rebalancing
        # TODO: Show how to calculate total stake across all subnets
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

        Returns:
            Balance: The total stake amount for the hotkey in the specified subnet.

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what total hotkey stake represents (sum of all coldkey stakes)
        # TODO: Add examples of using stake amounts for validator analysis
        # TODO: Show how to compare hotkey stakes across different subnets
        # TODO: Explain relationship between stake and validator influence/rewards
        # TODO: Add guidance on interpreting stake amounts for network health
        # TODO: Show how to use this for identifying top validators by stake
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what subnet registration costs and why they exist
        # TODO: Add guidance on when costs change and how to monitor them
        # TODO: Show how to budget for subnet creation
        # TODO: Explain the economic rationale for burn costs
        # TODO: Add examples of typical cost ranges
        # TODO: Show how to check if you have sufficient funds
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: List all available hyperparameters and their meanings
        # TODO: Add examples of how to interpret hyperparameter values
        # TODO: Show how hyperparameters affect subnet behavior and economics
        # TODO: Explain when hyperparameters change and who controls them
        # TODO: Add guidance on using hyperparameters for subnet analysis
        # TODO: Show how to monitor hyperparameter changes over time
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
        """Retrieve the reveal period, in epochs, for commit-reveal weight setting.

        This method returns the same value as the `CommitRevealPeriod` hyperparameter visible in btcli,
        but queries the actual storage name `RevealPeriodEpochs` in the subtensor blockchain.

        This chain parameter determines how many epochs must pass after a validator commits their weights
        before they can reveal them. This is part of the commit-reveal mechanism designed to prevent weight
        copying attacks by ensuring validators can only see stale weights from other validators.

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The block number for the query. Do not specify if using block_hash.
            block_hash: The hash of the block for the query. Do not specify if using block.

        Returns:
            int: The number of epochs that must elapse between weight commit and reveal.

        Example:
            # Get reveal period for subnet 1
            reveal_period = await subtensor.get_subnet_reveal_period_epochs(netuid=1)
            print(f"Validators must wait {reveal_period} epochs before revealing weights")

            # Calculate blocks until reveal is allowed
            current_block = await subtensor.get_current_block()
            tempo = await subtensor.tempo(netuid=1)
            blocks_per_epoch = tempo + 1
            reveal_delay_blocks = reveal_period * blocks_per_epoch

        Notes:
            - Storage name: `RevealPeriodEpochs` (btcli shows as `commit_reveal_period`)
            - Default value is 1 epoch (meaning reveal in the next epoch after commit)
            - Must be less than the immunity period to prevent miner deregistration
            - Only applies when commit-reveal is enabled for the subnet
            - See commit-reveal documentation for timing details

        # DOCSTRING HELPFULNESS RATING: 8/10
        # TODO: Add examples of how this interacts with validator weight setting schedules
        # TODO: Show how to calculate exact reveal timing windows
        # TODO: Explain edge cases when tempo or reveal period changes mid-commit
        """
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what subnet netuids represent and how they're assigned
        # TODO: Add examples of how to use the netuid list for subnet analysis
        # TODO: Show how to filter active vs inactive subnets
        # TODO: Explain the relationship between netuids and subnet registration
        # TODO: Add guidance on iterating over subnets for bulk operations
        # TODO: Show how to use this for network topology analysis
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain how total subnets differs from active subnets
        # TODO: Add examples of tracking network growth over time
        # TODO: Show how to use this for capacity planning and analysis
        # TODO: Explain the relationship to subnet registration limits
        # TODO: Add guidance on monitoring network expansion
        # TODO: Show how to calculate subnet density and distribution
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
        self, wallet: "Wallet", dest: str, value: Balance
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

        Returns:
            bittensor.utils.balance.Balance: The estimated transaction fee for the transfer, represented as a Balance
                object.

        Estimating the transfer fee is essential for planning and executing token transactions, ensuring that the
        wallet has sufficient funds to cover both the transfer amount and the associated costs. This function provides
        a crucial tool for managing financial operations within the Bittensor network.

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain factors that affect transfer fees (network congestion, transaction complexity)
        # TODO: Add examples of typical fee ranges for different transfer amounts
        # TODO: Show how to optimize transfers to minimize fees
        # TODO: Explain the difference between transfer and transfer_keep_alive fees
        # TODO: Add guidance on handling fee estimation failures
        # TODO: Show how to budget for transfers including fees
        """
        value = check_and_convert_to_balance(value)

        call = await self.substrate.compose_call(
            call_module="Balances",
            call_function="transfer_keep_alive",
            call_params={"dest": dest, "value": value.rao},
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what proposals are and how they're submitted
        # TODO: Add examples of how to interpret ProposalVoteData objects
        # TODO: Show how to analyze voting patterns and participation
        # TODO: Explain the relationship between senate members and governance
        # TODO: Add guidance on tracking proposal lifecycle and outcomes
        # TODO: Show how to use this for governance analysis and monitoring
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what UIDs represent and how they're assigned
        # TODO: Add examples of using UIDs for neuron identification and tracking
        # TODO: Show how UIDs relate to subnet registration and participation
        # TODO: Explain when UIDs change and how to handle registration status
        # TODO: Add guidance on using UIDs for metagraph analysis
        # TODO: Show how to check if a neuron is registered before operations
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

        # DOCSTRING HELPFULNESS RATING: 4/10
        # TODO: Explain the practical use cases for filtering netuids by registered hotkeys
        # TODO: Add examples of how to use this for multi-subnet operations
        # TODO: Show how to handle empty filter lists and edge cases
        # TODO: Explain the relationship between hotkey registration and subnet participation
        # TODO: Add guidance on performance considerations for large hotkey lists
        # TODO: Show how to use this for portfolio management across subnets
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what specific penalties/restrictions immunity protects against
        # TODO: Add examples of how immunity periods work in practice
        # TODO: Show how to calculate remaining immunity time for a neuron
        # TODO: Explain the relationship to registration timing and network health
        # TODO: Add guidance on when immunity periods expire and what happens
        # TODO: Show how immunity affects validator behavior and subnet dynamics
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
        """Returns True if the node is running with fast blocks. False if not.

        # DOCSTRING HELPFULNESS RATING: 3/10
        # TODO: Explain what fast blocks are and how they differ from normal blocks
        # TODO: Add information about block timing differences (fast vs normal)
        # TODO: Show how this affects network operations and timing calculations
        # TODO: Explain when fast blocks are used and why
        # TODO: Add guidance on how this affects validator and miner operations
        # TODO: Include return type and argument documentation
        """
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what being a delegate means and the responsibilities involved
        # TODO: Add examples of how to check delegate status before operations
        # TODO: Show how to become a delegate and the requirements
        # TODO: Explain the relationship between delegates and nominators
        # TODO: Add guidance on delegate take percentages and rewards
        # TODO: Show how to analyze delegate performance and reputation
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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Add examples of checking registration before operations
        # TODO: Explain what registration means (having a UID slot in a subnet)
        # TODO: Show how to handle the case where netuid is None vs specific subnet
        # TODO: Add guidance on registration timing and when to check status
        # TODO: Explain relationship to registration cost and difficulty
        # TODO: Show how to use this for monitoring multiple hotkeys
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Add examples of checking network-wide registration
        # TODO: Explain the difference between this and subnet-specific registration
        # TODO: Show how to get list of specific subnets where hotkey is registered
        # TODO: Add guidance on when to use this vs is_hotkey_registered_on_subnet
        # TODO: Explain performance implications vs checking individual subnets
        # TODO: Add examples of monitoring multiple hotkeys across the network
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
        """Checks if the hotkey is registered on a given netuid.

        # DOCSTRING HELPFULNESS RATING: 4/10
        # TODO: Add comprehensive description of what registration means
        # TODO: Include arguments documentation
        # TODO: Add return value documentation
        # TODO: Provide examples of checking specific subnet registration
        # TODO: Explain relationship to UID assignment in subnets
        # TODO: Add guidance on when to use this vs general registration checks
        """
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what "active" means beyond just start_call
        # TODO: Add examples of checking subnet activation status
        # TODO: Explain the difference between existing and active subnets
        # TODO: Show how to handle inactive subnets in applications
        # TODO: Add context about subnet lifecycle and activation process
        # TODO: Explain relationship to subnet emissions and operations
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what drand is and its purpose in Bittensor
        # TODO: Add context about commit-reveal mechanism and timing
        # TODO: Show how to use this for weight reveal timing
        # TODO: Explain relationship to consensus and weight-copying prevention
        # TODO: Add examples of monitoring drand rounds for validator operations
        # TODO: Clarify return type (says int but signature says Optional[int])
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what MaxWeightsLimit controls in weight setting
        # TODO: Add examples showing how this affects validator weight constraints
        # TODO: Explain relationship to weight normalization and validation
        # TODO: Show how to use this when setting weights programmatically
        # TODO: Add context about why weight limits exist (spam prevention, etc.)
        # TODO: Explain typical values and their implications
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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Explain what data is available in AsyncMetagraph vs regular Metagraph
        # TODO: Add examples of using metagraph for analysis and monitoring
        # TODO: Explain the trade-offs between lite and full sync modes
        # TODO: Show how to access specific neuron data from the metagraph
        # TODO: Add guidance on when to sync vs reuse existing metagraph
        # TODO: Explain performance implications of different sync modes
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what MinAllowedWeights controls in weight setting
        # TODO: Add examples showing minimum weight requirements for validators
        # TODO: Explain relationship to weight sparsity and validation requirements
        # TODO: Show how to use this when setting weights programmatically
        # TODO: Add context about why minimum weights exist (consensus requirements)
        # TODO: Explain typical values and their implications for subnet operation
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what information is included in NeuronInfo object
        # TODO: Add examples of accessing neuron attributes (stake, rank, trust, etc.)
        # TODO: Show how to handle null neurons (when UID doesn't exist)
        # TODO: Explain the difference between this and get_neuron_for_pubkey_and_subnet
        # TODO: Add guidance on when to use this vs neurons() for bulk queries
        # TODO: Show examples of neuron analysis and monitoring use cases
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what information is included in each NeuronInfo object
        # TODO: Add examples of filtering and analyzing neuron data
        # TODO: Show how to access specific neuron attributes (stake, rank, trust, etc.)
        # TODO: Explain when to use this vs neurons_lite() for performance
        # TODO: Add guidance on handling large subnet populations
        # TODO: Show examples of subnet analysis and monitoring use cases
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what information is included vs excluded in NeuronInfoLite
        # TODO: Add examples of when to use lite vs full neuron data
        # TODO: Show performance comparisons and use cases for large subnets
        # TODO: Explain what "key attributes" are specifically included
        # TODO: Add guidance on when lite format is sufficient for analysis
        # TODO: Show examples of efficient subnet monitoring with lite data
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what information is included in ChainIdentity object
        # TODO: Add examples of accessing identity attributes (display name, website, etc.)
        # TODO: Show how to handle cases where identity is not set
        # TODO: Explain the difference between identity and neuron registration
        # TODO: Add guidance on when identity information is useful
        # TODO: Show examples of identity verification and display in applications
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what "recycle" means in the context of subnet registration
        # TODO: Add examples of using recycle cost for registration planning
        # TODO: Explain the relationship between recycle cost and subnet difficulty
        # TODO: Show how recycle cost affects subnet economics and barriers to entry
        # TODO: Add guidance on when recycle cost is used vs proof-of-work registration
        # TODO: Explain how recycle cost changes based on subnet demand
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what information is included in DynamicInfo object
        # TODO: Add examples of accessing subnet attributes (registration cost, emissions, etc.)
        # TODO: Show how to handle case where subnet doesn't exist
        # TODO: Explain the difference between this and get_all_subnets_info()
        # TODO: Add guidance on when to use this vs other subnet query methods
        # TODO: Show examples of subnet monitoring and analysis use cases
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
            if isinstance(price, SubstrateRequestException):
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

        # DOCSTRING HELPFULNESS RATING: 7/10
        # TODO: Add examples of using this for error handling before subnet operations
        # TODO: Explain what determines when a subnet exists vs is deregistered
        # TODO: Show practical examples of subnet validation patterns
        # TODO: Add guidance on handling edge cases (subnet creation/destruction)
        # TODO: Explain relationship to subnet registration and lifecycle
        # TODO: Add timing considerations for subnet existence checks
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
        Retrieves the current number of registered neurons (UIDs) in a specific subnet.

        SubnetworkN represents the actual count of neurons currently registered and active within a subnet. This value
        starts at 0 when a subnet is created and increments by 1 each time a new neuron successfully registers. It
        represents the "filled slots" in the subnet and determines the next available UID for new registrations.

        This is different from the maximum capacity (MaxAllowedUids) - SubnetworkN shows how many neurons are actually
        registered, while MaxAllowedUids shows the theoretical limit. When SubnetworkN reaches MaxAllowedUids, new
        registrations require pruning (replacing) existing neurons with lower performance scores.

        Arguments:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to check the subnet existence.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Optional[int]: The current number of registered neurons in the subnet, or ``None`` if the subnetwork does
                not exist.

        Example:
            # Get current neuron count for subnet 1
            neuron_count = await subtensor.subnetwork_n(netuid=1)
            print(f"Subnet 1 has {neuron_count} registered neurons")

            # Check if subnet is at capacity
            max_uids = await subtensor.max_allowed_uids(netuid=1)
            current_count = await subtensor.subnetwork_n(netuid=1)

            if current_count >= max_uids:
                print("Subnet is at capacity - new registrations will require pruning")
            else:
                print(f"Subnet has {max_uids - current_count} available slots")

        # DOCSTRING HELPFULNESS RATING: 8/10
        # TODO: Add examples of using this for subnet health monitoring and analytics
        # TODO: Show how to track subnet growth over time using historical queries
        # TODO: Explain relationship to neuron registration fees and difficulty scaling
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what Tempo represents (epoch length in blocks)
        # TODO: Add examples of calculating timing based on tempo and block time
        # TODO: Explain how tempo affects emissions, weight setting, and consensus
        # TODO: Show practical use cases for tempo-based scheduling
        # TODO: Add guidance on typical tempo values and their implications
        # TODO: Explain relationship to blocks_since_last_step and epoch timing
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what time frame the rate limit applies to
        # TODO: Add examples of how rate limiting affects transaction submission
        # TODO: Show how to handle rate limit errors and implement backoff strategies
        # TODO: Explain the relationship between rate limits and network congestion
        # TODO: Add guidance on optimal transaction timing and batching
        # TODO: Show examples of monitoring network load and adjusting behavior
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain typical use cases for waiting on blocks
        # TODO: Add examples of timing operations around epochs and emissions
        # TODO: Show how to handle timeout scenarios and error cases
        # TODO: Explain the relationship to block time (12 seconds) and timing calculations
        # TODO: Add guidance on when to use this vs other timing mechanisms
        # TODO: Show examples of coordinating actions with blockchain state

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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what weight values represent and how they're normalized
        # TODO: Add examples of interpreting weight distributions for consensus analysis
        # TODO: Show how to use weights for validator performance tracking
        # TODO: Explain the relationship between weights and trust/rank calculations
        # TODO: Add guidance on handling sparse weight data and missing entries
        # TODO: Show examples of weight evolution analysis over time
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what the rate limit represents (minimum blocks between weight updates)
        # TODO: Add examples of calculating when weight updates are allowed
        # TODO: Show how to handle rate limit errors and retry strategies
        # TODO: Explain the relationship to tempo and epoch timing
        # TODO: Add guidance on optimal weight update timing strategies
        # TODO: Show examples of coordinating weight updates with other validators
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain typical use cases for block timestamps (timing analysis, historical data)
        # TODO: Add examples of calculating time differences between blocks
        # TODO: Show how to use timestamps for epoch and tempo calculations
        # TODO: Explain timezone handling and UTC normalization
        # TODO: Add guidance on timestamp precision and accuracy considerations
        # TODO: Show examples of coordinating actions with specific block times
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Explain what subnet ownership means and the owner's responsibilities
        # TODO: Add examples of using owner information for subnet governance
        # TODO: Show how to handle cases where subnets don't have owners
        # TODO: Explain the relationship between subnet ownership and registration
        # TODO: Add guidance on when subnet ownership information is useful
        # TODO: Show examples of owner-based subnet filtering and analysis
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

        # DOCSTRING HELPFULNESS RATING: 5/10
        # TODO: Explain what validator permits represent and how they're earned
        # TODO: Add examples of using permit information for validator analysis
        # TODO: Show how permits relate to stake amounts and validator responsibilities
        # TODO: Explain the relationship between permits and weight setting privileges
        # TODO: Add guidance on interpreting permit distribution across subnet
        # TODO: Show examples of permit-based validator filtering and monitoring
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

        # DOCSTRING HELPFULNESS RATING: 4/10
        # TODO: Add comprehensive documentation for all arguments
        # TODO: Explain the difference between inclusion and finalization waiting
        # TODO: Add examples of using this method for different extrinsic types
        # TODO: Show error handling patterns and retry strategies
        # TODO: Explain nonce management and when to use custom nonces
        # TODO: Add guidance on transaction period and expiration handling
        # TODO: Show examples of signing with different key types (coldkey vs hotkey)
        # TODO: Explain return values and how to interpret success/failure

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
        amount: Balance,
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
            amount: Number of tokens to transfer.
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

        # DOCSTRING HELPFULNESS RATING: 8/10
        # TODO: Clarify the difference between this and unstake() with unstake_all=True
        # TODO: Explain when to use rate_tolerance vs None for price protection
        # TODO: Add guidance on checking current stake amounts before unstaking
        # TODO: Show how to handle partial failures in batch unstaking scenarios
        # TODO: Add examples of monitoring transaction status and handling timeouts
        # TODO: Explain fees and costs associated with unstaking operations
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

        # DOCSTRING HELPFULNESS RATING: 6/10
        # TODO: Add practical examples of batch unstaking scenarios
        # TODO: Explain how to handle list length mismatches between parameters
        # TODO: Add guidance on optimal batch sizes for performance
        # TODO: Show how to handle partial failures in batch operations
        # TODO: Explain the atomicity of batch operations (all or nothing vs partial success)
        # TODO: Add examples of stake rebalancing across multiple subnets
        # TODO: Explain fee calculations for batch operations
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

    # DOCSTRING HELPFULNESS RATING: 5/10
    # TODO: Add comprehensive documentation for all arguments
    # TODO: Explain when to use this vs manual initialization
    # TODO: Add examples of using this factory function
    # TODO: Show error handling patterns for initialization failures
    # TODO: Explain the difference between this and direct instantiation
    # TODO: Add guidance on when this convenience function is beneficial
    """
    sub = AsyncSubtensor(
        network=network, config=config, _mock=_mock, log_verbose=log_verbose
    )
    await sub.initialize()
    return sub
