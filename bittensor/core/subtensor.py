import copy
from datetime import datetime, timezone
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union, cast

import scalecodec
from async_substrate_interface.errors import SubstrateRequestException
from async_substrate_interface.substrate_addons import RetrySyncSubstrate
from async_substrate_interface.sync_substrate import SubstrateInterface
from async_substrate_interface.types import ScaleObj
from async_substrate_interface.utils.storage import StorageKey
from bittensor_drand import get_encrypted_commitment

from bittensor.core.async_subtensor import ProposalVoteData
from bittensor.core.axon import Axon
from bittensor.core.chain_data import (
    DelegatedInfo,
    DelegateInfo,
    DynamicInfo,
    MetagraphInfo,
    NeuronInfo,
    NeuronInfoLite,
    SelectiveMetagraphIndex,
    StakeInfo,
    SubnetInfo,
    SubnetIdentity,
    SubnetHyperparameters,
    WeightCommitInfo,
    decode_account_id,
)
from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.utils import (
    decode_block,
    decode_metadata,
    decode_revealed_commitment,
    decode_revealed_commitment_with_hotkey,
)
from bittensor.core.config import Config
from bittensor.core.errors import ChainError
from bittensor.core.extrinsics.children import (
    set_children_extrinsic,
    root_set_pending_childkey_cooldown_extrinsic,
)
from bittensor.core.extrinsics.liquidity import (
    add_liquidity_extrinsic,
    modify_liquidity_extrinsic,
    remove_liquidity_extrinsic,
    toggle_user_liquidity_extrinsic,
)
from bittensor.core.extrinsics.move_stake import (
    transfer_stake_extrinsic,
    swap_stake_extrinsic,
    move_stake_extrinsic,
)
from bittensor.core.extrinsics.registration import (
    burned_register_extrinsic,
    register_extrinsic,
    register_subnet_extrinsic,
    set_subnet_identity_extrinsic,
)
from bittensor.core.extrinsics.root import root_register_extrinsic
from bittensor.core.extrinsics.serving import (
    get_last_bonds_reset,
    publish_metadata_extrinsic,
    get_metadata,
    serve_axon_extrinsic,
)
from bittensor.core.extrinsics.staking import (
    add_stake_extrinsic,
    add_stake_multiple_extrinsic,
    set_auto_stake_extrinsic,
)
from bittensor.core.extrinsics.start_call import start_call_extrinsic
from bittensor.core.extrinsics.take import (
    decrease_take_extrinsic,
    increase_take_extrinsic,
)
from bittensor.core.extrinsics.transfer import transfer_extrinsic
from bittensor.core.extrinsics.unstaking import (
    unstake_all_extrinsic,
    unstake_extrinsic,
    unstake_multiple_extrinsic,
)
from bittensor.core.extrinsics.weights import (
    commit_timelocked_weights_extrinsic,
    commit_weights_extrinsic,
    reveal_weights_extrinsic,
    set_weights_extrinsic,
)
from bittensor.core.metagraph import Metagraph
from bittensor.core.settings import (
    version_as_int,
    SS58_FORMAT,
    TYPE_REGISTRY,
)
from bittensor.core.types import ExtrinsicResponse
from bittensor.core.types import (
    ParamWithTypes,
    Salt,
    SubtensorMixin,
    UIDs,
    Weights,
)
from bittensor.utils import (
    Certificate,
    decode_hex_identity_dict,
    format_error_message,
    is_valid_ss58_address,
    u16_normalized_float,
    u64_normalized_float,
    deprecated_message,
    get_transfer_fn_params,
    get_mechid_storage_index,
)
from bittensor.utils.balance import (
    Balance,
    fixed_to_float,
    FixedPoint,
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
    U16_MAX,
)

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from async_substrate_interface.sync_substrate import QueryMapResult
    from scalecodec.types import GenericCall


class Subtensor(SubtensorMixin):
    """Thin layer for interacting with Substrate Interface. Mostly a collection of frequently used calls."""

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional[Config] = None,
        log_verbose: bool = False,
        fallback_endpoints: Optional[list[str]] = None,
        retry_forever: bool = False,
        _mock: bool = False,
        archive_endpoints: Optional[list[str]] = None,
    ):
        """
        Initializes an instance of the Subtensor class.

        Parameters:
            network: The network name or type to connect to.
            config: Configuration object for the AsyncSubtensor instance.
            log_verbose: Enables or disables verbose logging.
            fallback_endpoints: List of fallback endpoints to use if default or provided network is not available.
            retry_forever: Whether to retry forever on connection errors.
            _mock: Whether this is a mock instance. Mainly just for use in testing.
            archive_endpoints: Similar to fallback_endpoints, but specifically only archive nodes. Will be used in cases
                where you are requesting a block that is too old for your current (presumably lite) node.

        Raises:
            Any exceptions raised during the setup, configuration, or connection process.
        """
        if config is None:
            config = self.config()
        self._config = copy.deepcopy(config)
        self.chain_endpoint, self.network = self.setup_config(network, self._config)

        self.log_verbose = log_verbose
        self._check_and_log_network_settings()

        logging.debug(
            f"Connecting to network: [blue]{self.network}[/blue], "
            f"chain_endpoint: [blue]{self.chain_endpoint}[/blue]> ..."
        )
        self.substrate = self._get_substrate(
            fallback_endpoints=fallback_endpoints,
            retry_forever=retry_forever,
            _mock=_mock,
            archive_endpoints=archive_endpoints,
        )
        if self.log_verbose:
            logging.info(
                f"Connected to {self.network} network and {self.chain_endpoint}."
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Closes the websocket connection."""
        self.substrate.close()

    def _get_substrate(
        self,
        fallback_endpoints: Optional[list[str]] = None,
        retry_forever: bool = False,
        _mock: bool = False,
        archive_endpoints: Optional[list[str]] = None,
    ) -> Union[SubstrateInterface, RetrySyncSubstrate]:
        """Creates the Substrate instance based on provided arguments.

        Parameters:
            fallback_endpoints: List of fallback chains endpoints to use if main network isn't available.
            retry_forever: Whether to retry forever on connection errors.
            _mock: Whether this is a mock instance. Mainly just for use in testing.
            archive_endpoints: Similar to fallback_endpoints, but specifically only archive nodes. Will be used in cases
                where you are requesting a block that is too old for your current (presumably lite) node.

        Returns:
            The instance of the SubstrateInterface or RetrySyncSubstrate class.
        """
        if fallback_endpoints or retry_forever or archive_endpoints:
            return RetrySyncSubstrate(
                url=self.chain_endpoint,
                ss58_format=SS58_FORMAT,
                type_registry=TYPE_REGISTRY,
                use_remote_preset=True,
                chain_name="Bittensor",
                fallback_chains=fallback_endpoints,
                retry_forever=retry_forever,
                _mock=_mock,
                archive_nodes=archive_endpoints,
            )
        return SubstrateInterface(
            url=self.chain_endpoint,
            ss58_format=SS58_FORMAT,
            type_registry=TYPE_REGISTRY,
            use_remote_preset=True,
            chain_name="Bittensor",
            _mock=_mock,
        )

    # Subtensor queries ===========================================================================================

    def query_constant(
        self, module_name: str, constant_name: str, block: Optional[int] = None
    ) -> Optional["ScaleObj"]:
        """
        Retrieves a constant from the specified module on the Bittensor blockchain. This function is used to access
        fixed parameters or values defined within the blockchain's modules, which are essential for understanding the
        network's configuration and rules.

        Parameters:
            module_name: The name of the module containing the constant.
            constant_name: The name of the constant to retrieve.
            block: The blockchain block number at which to query the constant.

        Returns:
            The value of the constant if found, `None` otherwise.

        Constants queried through this function can include critical network parameters such as inflation rates,
        consensus rules, or validation thresholds, providing a deeper understanding of the Bittensor network's
        operational parameters.
        """
        return self.substrate.get_constant(
            module_name=module_name,
            constant_name=constant_name,
            block_hash=self.determine_block_hash(block),
        )

    def query_map(
        self,
        module: str,
        name: str,
        params: Optional[list] = None,
        block: Optional[int] = None,
    ) -> "QueryMapResult":
        """
        Queries map storage from any module on the Bittensor blockchain. This function retrieves data structures that
        represent key-value mappings, essential for accessing complex and structured data within the blockchain modules.

        Parameters:
            module: The name of the module from which to query the map storage.
            name: The specific storage function within the module to query.
            params: Parameters to be passed to the query.
            block: The blockchain block number at which to perform the query.

        Returns:
            A data structure representing the map storage if found, `None` otherwise.

        This function is particularly useful for retrieving detailed and structured data from various blockchain
        modules, offering insights into the network's state and the relationships between its different components.
        """
        result = self.substrate.query_map(
            module=module,
            storage_function=name,
            params=params,
            block_hash=self.determine_block_hash(block=block),
        )
        return result

    def query_map_subtensor(
        self,
        name: str,
        params: Optional[list] = None,
        block: Optional[int] = None,
    ) -> "QueryMapResult":
        """
        Queries map storage from the Subtensor module on the Bittensor blockchain. This function is designed to retrieve
        a map-like data structure, which can include various neuron-specific details or network-wide attributes.

        Parameters:
            name: The name of the map storage function to query.
            params: A list of parameters to pass to the query function.
            block: The blockchain block number at which to perform the query.

        Returns:
            An object containing the map-like data structure, or `None` if not found.

        This function is particularly useful for analyzing and understanding complex network structures and
        relationships within the Bittensor ecosystem, such as interneuronal connections and stake distributions.
        """
        return self.substrate.query_map(
            module="SubtensorModule",
            storage_function=name,
            params=params,
            block_hash=self.determine_block_hash(block),
        )

    def query_module(
        self,
        module: str,
        name: str,
        params: Optional[list] = None,
        block: Optional[int] = None,
    ) -> Optional[Union["ScaleObj", Any, FixedPoint]]:
        """
        Queries any module storage on the Bittensor blockchain with the specified parameters and block number. This
        function is a generic query interface that allows for flexible and diverse data retrieval from various
        blockchain modules.

        Parameters:
            module: The name of the module from which to query data.
            name: The name of the storage function within the module.
            block: The blockchain block number at which to perform the query.
            params: A list of parameters to pass to the query function.

        Returns:
            An object containing the requested data if found, `None` otherwise.

        This versatile query function is key to accessing a wide range of data and insights from different parts of the
            Bittensor blockchain, enhancing the understanding and analysis of the network's state and dynamics.
        """
        return self.substrate.query(
            module=module,
            storage_function=name,
            params=params,
            block_hash=self.determine_block_hash(block),
        )

    def query_runtime_api(
        self,
        runtime_api: str,
        method: str,
        params: Optional[Union[list[Any], dict[str, Any]]] = None,
        block: Optional[int] = None,
    ) -> Any:
        """
        Queries the runtime API of the Bittensor blockchain, providing a way to interact with the underlying runtime and
        retrieve data encoded in Scale Bytes format. This function is essential for advanced users who need to interact
        with specific runtime methods and decode complex data types.

        Parameters:
            runtime_api: The name of the runtime API to query.
            method: The specific method within the runtime API to call.
            params: The parameters to pass to the method call.
            block: the block number for this query.

        Returns:
            The Scale Bytes encoded result from the runtime API call, or `None` if the call fails.

        This function enables access to the deeper layers of the Bittensor blockchain, allowing for detailed and
        specific interactions with the network's runtime environment.
        """
        block_hash = self.determine_block_hash(block)
        result = self.substrate.runtime_call(runtime_api, method, params, block_hash)

        return result.value

    def query_subtensor(
        self,
        name: str,
        params: Optional[list] = None,
        block: Optional[int] = None,
    ) -> Optional[Union["ScaleObj", Any]]:
        """
        Queries named storage from the Subtensor module on the Bittensor blockchain. This function is used to retrieve
        specific data or parameters from the blockchain, such as stake, rank, or other neuron-specific attributes.

        Parameters:
            name: The name of the storage function to query.
            params: A list of parameters to pass to the query function.
            block: The blockchain block number at which to perform the query.

        Returns:
            An object containing the requested data.

        This query function is essential for accessing detailed information about the network and its neurons, providing
        valuable insights into the state and dynamics of the Bittensor ecosystem.
        """
        return self.substrate.query(
            module="SubtensorModule",
            storage_function=name,
            params=params,
            block_hash=self.determine_block_hash(block),
        )

    def state_call(
        self, method: str, data: str, block: Optional[int] = None
    ) -> dict[Any, Any]:
        """
        Makes a state call to the Bittensor blockchain, allowing for direct queries of the blockchain's state. This
        function is typically used for advanced queries that require specific method calls and data inputs.

        Parameters:
            method: The method name for the state call.
            data: The data to be passed to the method.
            block: The blockchain block number at which to perform the state call.

        Returns:
            The result of the rpc call.

        The state call function provides a more direct and flexible way of querying blockchain data, useful for specific
        use cases where standard queries are insufficient.
        """
        block_hash = self.determine_block_hash(block)
        return self.substrate.rpc_request(
            method="state_call", params=[method, data], block_hash=block_hash
        )

    # Common subtensor calls ===========================================================================================

    @property
    def block(self) -> int:
        return self.get_current_block()

    def all_subnets(self, block: Optional[int] = None) -> Optional[list["DynamicInfo"]]:
        """
        Retrieves the subnet information for all subnets in the network.

        Parameters:
            block: The block number to query the subnet information from.

        Returns:
            A list of DynamicInfo objects, each containing detailed information about a subnet.

        """
        block_hash = self.determine_block_hash(block=block)
        query = self.substrate.runtime_call(
            api="SubnetInfoRuntimeApi",
            method="get_all_dynamic_info",
            block_hash=block_hash,
        )
        decoded = query.decode()
        try:
            subnet_prices = self.get_subnet_prices(block=block)
            for sn in decoded:
                sn.update(
                    {"price": subnet_prices.get(sn["netuid"], Balance.from_tao(0))}
                )
        except (SubstrateRequestException, ValueError) as e:
            logging.warning(f"Unable to fetch subnet prices for block {block}: {e}")

        return DynamicInfo.list_from_dicts(decoded)

    def blocks_since_last_step(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """Returns number of blocks since the last epoch of the subnet.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: the block number for this query.

        Returns:
            block number of the last step in the subnet.
        """
        query = self.query_subtensor(
            name="BlocksSinceLastStep", block=block, params=[netuid]
        )
        return query.value if query is not None and hasattr(query, "value") else query

    def blocks_since_last_update(self, netuid: int, uid: int) -> Optional[int]:
        """
        Returns the number of blocks since the last update for a specific UID in the subnetwork.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.

        Returns:
            The number of blocks since the last update, or ``None`` if the subnetwork or UID does not exist.
        """
        call = self.get_hyperparameter(param_name="LastUpdate", netuid=netuid)
        return None if not call else (self.get_current_block() - int(call[uid]))

    def bonds(
        self,
        netuid: int,
        block: Optional[int] = None,
        mechid: int = 0,
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the bond distribution set by neurons within a specific subnet of the Bittensor network.
            Bonds represent the investments or commitments made by neurons in one another, indicating a level of trust
            and perceived value. This bonding mechanism is integral to the network's market-based approach to
            measuring and rewarding machine intelligence.

        Parameters:
            netuid: Subnet identifier.
            block: the block number for this query.
            mechid: Subnet mechanism identifier.

        Returns:
            List of tuples mapping each neuron's UID to its bonds with other neurons.

        Understanding bond distributions is crucial for analyzing the trust dynamics and market behavior within the
            subnet. It reflects how neurons recognize and invest in each other's intelligence and contributions,
            supporting diverse and niche systems within the Bittensor ecosystem.
        """
        storage_index = get_mechid_storage_index(netuid, mechid)
        b_map_encoded = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Bonds",
            params=[storage_index],
            block_hash=self.determine_block_hash(block),
        )
        b_map = []
        for uid, b in b_map_encoded:
            if b.value is not None:
                b_map.append((uid, b.value))

        return b_map

    def commit_reveal_enabled(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[bool]:
        """
        Check if the commit-reveal mechanism is enabled for a given network at a specific block.

        Parameters:
            netuid: The network identifier for which to check the commit-reveal mechanism.
            block: The block number to query.

        Returns:
            Returns the integer value of the hyperparameter if available; otherwise, returns None.
        """
        call = self.get_hyperparameter(
            param_name="CommitRevealWeightsEnabled", block=block, netuid=netuid
        )
        return True if call is True else False

    def difficulty(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the 'Difficulty' hyperparameter for a specified subnet in the Bittensor network.

        This parameter is instrumental in determining the computational challenge required for neurons to participate in
        consensus and validation processes.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            The value of the 'Difficulty' hyperparameter if the subnet exists, ``None`` otherwise.

        The 'Difficulty' parameter directly impacts the network's security and integrity by setting the computational
        effort required for validating transactions and participating in the network's consensus mechanism.
        """
        call = self.get_hyperparameter(
            param_name="Difficulty", netuid=netuid, block=block
        )
        if call is None:
            return None
        return int(call)

    def does_hotkey_exist(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        """
        Returns true if the hotkey is known by the chain and there are accounts.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey.
            block: the block number for this query.

        Returns:
            `True` if the hotkey is known by the chain and there are accounts, `False` otherwise.
        """
        result = self.substrate.query(
            module="SubtensorModule",
            storage_function="Owner",
            params=[hotkey_ss58],
            block_hash=self.determine_block_hash(block),
        )
        return_val = (
            False
            if result is None
            # not the default key (0x0)
            else result != "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        )
        return return_val

    def get_admin_freeze_window(self, block: Optional[int] = None) -> int:
        """
        Returns the number of blocks when dependent transactions will be frozen for execution.

        Parameters:
            block: The block number for which the children are to be retrieved.

        Returns:
            AdminFreezeWindow as integer. The number of blocks are frozen.
        """

        return self.substrate.query(
            module="SubtensorModule",
            storage_function="AdminFreezeWindow",
            block_hash=self.determine_block_hash(block),
        ).value

    def get_all_subnets_info(self, block: Optional[int] = None) -> list["SubnetInfo"]:
        """
        Retrieves detailed information about all subnets within the Bittensor network. This function provides
        comprehensive data on each subnet, including its characteristics and operational parameters.

        Parameters:
            block: The blockchain block number for the query.

        Returns:
            A list of SubnetInfo objects, each containing detailed information about a subnet.

        Gaining insights into the subnets' details assists in understanding the network's composition, the roles of
        different subnets, and their unique features.
        """
        result = self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnets_info_v2",
            params=[],
            block=block,
        )
        if not result:
            return []
        try:
            subnets_prices = self.get_subnet_prices(block=block)

            for subnet in result:
                subnet.update({"price": subnets_prices.get(subnet["netuid"], 0)})
        except (SubstrateRequestException, ValueError) as e:
            logging.warning(f"Unable to fetch subnet prices for block {block}: {e}")

        return SubnetInfo.list_from_dicts(result)

    def get_auto_stakes(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
    ) -> dict[int, str]:
        """Fetches auto stake destinations for a given wallet across all subnets.

        Parameters:
            coldkey_ss58: Coldkey ss58 address.
            block: Subnet unique identifier.

        Returns:
            dict[int, str]:
                - netuid: The unique identifier of the subnet.
                - hotkey: The hotkey of the wallet.
        """
        block_hash = self.determine_block_hash(block=block)
        query = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="AutoStakeDestination",
            params=[coldkey_ss58],
            block_hash=block_hash,
        )

        pairs = {}
        for netuid, destination in query:
            hotkey_ss58 = decode_account_id(destination.value[0])
            if hotkey_ss58:
                pairs[int(netuid)] = hotkey_ss58

        return pairs

    def get_balance(self, address: str, block: Optional[int] = None) -> Balance:
        """
        Retrieves the balance for given coldkey. Always in TAO.

        Parameters:
            address: coldkey address.
            block: The blockchain block number for the query.

        Returns:
            Balance object in TAO.
        """
        balance = self.substrate.query(
            module="System",
            storage_function="Account",
            params=[address],
            block_hash=self.determine_block_hash(block),
        )
        return Balance(balance["data"]["free"])

    def get_balances(
        self,
        *addresses: str,
        block: Optional[int] = None,
    ) -> dict[str, Balance]:
        """
        Retrieves the balance for given coldkey(s)

        Parameters:
            addresses: coldkey addresses(s).
            block: The blockchain block number for the query.

        Returns:
            Dict of {address: Balance objects}.
        """
        if not (block_hash := self.determine_block_hash(block)):
            block_hash = self.substrate.get_chain_head()
        calls = [
            (
                self.substrate.create_storage_key(
                    "System", "Account", [address], block_hash=block_hash
                )
            )
            for address in addresses
        ]
        batch_call = self.substrate.query_multi(calls, block_hash=block_hash)
        results = {}
        for item in batch_call:
            value = item[1] or {"data": {"free": 0}}
            results.update({item[0].params[0]: Balance(value["data"]["free"])})
        return results

    def get_current_block(self) -> int:
        """
        Returns the current block number on the Bittensor blockchain. This function provides the latest block number,
            indicating the most recent state of the blockchain.

        Returns:
            int: The current chain block number.

        Knowing the current block number is essential for querying real-time data and performing time-sensitive
            operations on the blockchain. It serves as a reference point for network activities and data
            synchronization.
        """
        return self.substrate.get_block_number(None)

    @lru_cache(maxsize=128)
    def _get_block_hash(self, block_id: int):
        return self.substrate.get_block_hash(block_id)

    def get_block_hash(self, block: Optional[int] = None) -> str:
        """
        Retrieves the hash of a specific block on the Bittensor blockchain. The block hash is a unique identifier
        representing the cryptographic hash of the block's content, ensuring its integrity and immutability.

        Parameters:
            block: The block number for which the hash is to be retrieved.

        Returns:
            str: The cryptographic hash of the specified block.

        The block hash is a fundamental aspect of blockchain technology, providing a secure reference to each block's
        data. It is crucial for verifying transactions, ensuring data consistency, and maintaining the trustworthiness
        of the blockchain.
        """
        if block is not None:
            return self._get_block_hash(block)
        else:
            return self.substrate.get_chain_head()

    def determine_block_hash(self, block: Optional[int]) -> Optional[str]:
        if block is None:
            return None
        else:
            return self.get_block_hash(block=block)

    def encode_params(
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

    def get_hyperparameter(
        self, param_name: str, netuid: int, block: Optional[int] = None
    ) -> Optional[Any]:
        """
        Retrieves a specified hyperparameter for a specific subnet.

        Parameters:
            param_name: The name of the hyperparameter to retrieve.
            netuid: The unique identifier of the subnet.
            block: the block number at which to retrieve the hyperparameter.

        Returns:
            The value of the specified hyperparameter if the subnet exists, or None
        """
        block_hash = self.determine_block_hash(block)
        if not self.subnet_exists(netuid, block=block):
            logging.error(f"subnet {netuid} does not exist")
            return None

        result = self.substrate.query(
            module="SubtensorModule",
            storage_function=param_name,
            params=[netuid],
            block_hash=block_hash,
        )

        return getattr(result, "value", result)

    def get_parents(
        self, hotkey: str, netuid: int, block: Optional[int] = None
    ) -> list[tuple[float, str]]:
        """
        This method retrieves the parent of a given hotkey and netuid. It queries the SubtensorModule's ParentKeys
        storage function to get the children and formats them before returning as a tuple.

        Parameters:
            hotkey: The child hotkey SS58.
            netuid: The netuid.
            block: The block number for which the children are to be retrieved.

        Returns:
            A list of formatted parents [(proportion, parent)]
        """
        parents = self.substrate.query(
            module="SubtensorModule",
            storage_function="ParentKeys",
            params=[hotkey, netuid],
            block_hash=self.determine_block_hash(block),
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

    def get_children(
        self, hotkey: str, netuid: int, block: Optional[int] = None
    ) -> tuple[bool, list[tuple[float, str]], str]:
        """
        This method retrieves the children of a given hotkey and netuid. It queries the SubtensorModule's ChildKeys
        storage function to get the children and formats them before returning as a tuple.

        Parameters:
            hotkey: The hotkey value.
            netuid: The netuid value.
            block: The block number for which the children are to be retrieved.

        Returns:
            A tuple containing a boolean indicating success or failure, a list of formatted children, and an error
                message (if applicable)
        """
        try:
            children = self.substrate.query(
                module="SubtensorModule",
                storage_function="ChildKeys",
                params=[hotkey, netuid],
                block_hash=self.determine_block_hash(block),
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

    def get_children_pending(
        self,
        hotkey: str,
        netuid: int,
        block: Optional[int] = None,
    ) -> tuple[
        list[tuple[float, str]],
        int,
    ]:
        """
        This method retrieves the pending children of a given hotkey and netuid.
        It queries the SubtensorModule's PendingChildKeys storage function.

        Parameters:
            hotkey: The hotkey value.
            netuid: The netuid value.
            block: The block number for which the children are to be retrieved.

        Returns:
            - A list of children with their proportions.
            - The cool-down block number.
        """

        children, cooldown = self.substrate.query(
            module="SubtensorModule",
            storage_function="PendingChildKeys",
            params=[netuid, hotkey],
            block_hash=self.determine_block_hash(block),
        ).value

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

    def get_commitment(self, netuid: int, uid: int, block: Optional[int] = None) -> str:
        """
        Retrieves the on-chain commitment for a specific neuron in the Bittensor network.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.
            block: The block number to retrieve the commitment from. If None, the latest block is used.

        Returns:
            The commitment data as a string.
        """
        metagraph = self.metagraph(netuid)
        try:
            hotkey = metagraph.hotkeys[uid]  # type: ignore
        except IndexError:
            logging.error(
                "Your uid is not in the hotkeys. Please double-check your UID."
            )
            return ""

        metadata = cast(dict, get_metadata(self, netuid, hotkey, block))
        try:
            return decode_metadata(metadata)
        except Exception as error:
            logging.error(error)
            return ""

    def get_last_commitment_bonds_reset_block(
        self, netuid: int, uid: int
    ) -> Optional[int]:
        """
        Retrieves the last block number when the bonds reset were triggered by publish_metadata for a specific neuron.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.

        Returns:
            The block number when the bonds were last reset, or None if not found.
        """

        metagraph = self.metagraph(netuid)
        try:
            hotkey = metagraph.hotkeys[uid]
        except IndexError:
            logging.error(
                "Your uid is not in the hotkeys. Please double-check your UID."
            )
            return None
        block = get_last_bonds_reset(self, netuid, hotkey)
        if block is None:
            return None
        return decode_block(block)

    def get_all_commitments(
        self, netuid: int, block: Optional[int] = None
    ) -> dict[str, str]:
        query = self.query_map(
            module="Commitments",
            name="CommitmentOf",
            params=[netuid],
            block=block,
        )
        result = {}
        for id_, value in query:
            try:
                result[decode_account_id(id_[0])] = decode_metadata(value)
            except Exception as error:
                logging.error(
                    f"Error decoding [red]{id_}[/red] and [red]{value}[/red]: {error}"
                )
        return result

    def get_revealed_commitment_by_hotkey(
        self,
        netuid: int,
        hotkey_ss58_address: str,
        block: Optional[int] = None,
    ) -> Optional[tuple[tuple[int, str], ...]]:
        """Returns hotkey related revealed commitment for a given netuid.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            hotkey_ss58_address: The ss58 address of the committee member.
            block: The block number to retrieve the commitment from.

        Returns:
            A tuple of reveal block and commitment message.
        """
        if not is_valid_ss58_address(address=hotkey_ss58_address):
            raise ValueError(f"Invalid ss58 address {hotkey_ss58_address} provided.")

        query = self.query_module(
            module="Commitments",
            name="RevealedCommitments",
            params=[netuid, hotkey_ss58_address],
            block=block,
        )
        if query is None:
            return None
        return tuple(decode_revealed_commitment(pair) for pair in query)

    def get_revealed_commitment(
        self,
        netuid: int,
        uid: int,
        block: Optional[int] = None,
    ) -> Optional[tuple[tuple[int, str], ...]]:
        """Returns uid related revealed commitment for a given netuid.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            uid: The neuron uid to retrieve the commitment from.
            block: The block number to retrieve the commitment from.

        Returns:
            A tuple of reveal block and commitment message.

        Example of result:
            ( (12, "Alice message 1"), (152, "Alice message 2") )
            ( (12, "Bob message 1"), (147, "Bob message 2") )
        """
        try:
            meta_info = self.get_metagraph_info(netuid, block=block)
            if meta_info:
                hotkey_ss58_address = meta_info.hotkeys[uid]
            else:
                raise ValueError(f"Subnet with netuid {netuid} does not exist.")
        except IndexError:
            raise ValueError(f"Subnet {netuid} does not have a neuron with uid {uid}.")

        return self.get_revealed_commitment_by_hotkey(
            netuid=netuid, hotkey_ss58_address=hotkey_ss58_address, block=block
        )

    def get_all_revealed_commitments(
        self, netuid: int, block: Optional[int] = None
    ) -> dict[str, tuple[tuple[int, str], ...]]:
        """Returns all revealed commitments for a given netuid.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The block number to retrieve the commitment from.

        Returns:
            result: A dictionary of all revealed commitments in view {ss58_address: (reveal block, commitment message)}.

        Example of result:
        {
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY": ( (12, "Alice message 1"), (152, "Alice message 2") ),
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty": ( (12, "Bob message 1"), (147, "Bob message 2") ),
        }
        """
        query = self.query_map(
            module="Commitments",
            name="RevealedCommitments",
            params=[netuid],
            block=block,
        )

        result = {}
        for pair in query:
            hotkey_ss58_address, commitment_message = (
                decode_revealed_commitment_with_hotkey(pair)
            )
            result[hotkey_ss58_address] = commitment_message
        return result

    # TODO: remove in SDKv10
    def get_current_weight_commit_info(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[str, str, int]]:
        """
        Retrieves CRV3 weight commit information for a specific subnet.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

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
        result = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="CRV3WeightCommits",
            params=[netuid],
            block_hash=self.determine_block_hash(block),
        )

        commits = result.records[0][1] if result.records else []
        return [WeightCommitInfo.from_vec_u8(commit) for commit in commits]

    # TODO: remove in SDKv10
    def get_current_weight_commit_info_v2(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[str, int, str, int]]:
        """
        Retrieves CRV3 weight commit information for a specific subnet.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            A list of commit details, where each item contains:
                - ss58_address: The address of the committer.
                - commit_block: The block number when the commitment was made.
                - commit_message: The commit message.
                - reveal_round: The round when the commitment was revealed.

        The list may be empty if there are no commits found.
        """
        result = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="CRV3WeightCommitsV2",
            params=[netuid],
            block_hash=self.determine_block_hash(block),
        )

        commits = result.records[0][1] if result.records else []
        return [WeightCommitInfo.from_vec_u8_v2(commit) for commit in commits]

    def get_delegate_by_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional["DelegateInfo"]:
        """
        Retrieves detailed information about a delegate neuron based on its hotkey. This function provides a
        comprehensive view of the delegate's status, including its stakes, nominators, and reward distribution.

        Parameters:
            hotkey_ss58: The ``SS58`` address of the delegate's hotkey.
            block: The blockchain block number for the query.

        Returns:
            Detailed information about the delegate neuron, ``None`` if not found.

        This function is essential for understanding the roles and influence of delegate neurons within the Bittensor
        network's consensus and governance structures.
        """

        result = self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegate",
            params=[hotkey_ss58],
            block=block,
        )

        if not result:
            return None

        return DelegateInfo.from_dict(result)

    def get_delegate_identities(
        self, block: Optional[int] = None
    ) -> dict[str, ChainIdentity]:
        """
        Fetches delegates identities from the chain.

        Parameters:
            block: The blockchain block number for the query.

        Returns:
            Dict {ss58: ChainIdentity, ...}

        """
        identities = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="IdentitiesV2",
            block_hash=self.determine_block_hash(block),
        )

        return {
            decode_account_id(ss58_address[0]): ChainIdentity.from_dict(
                decode_hex_identity_dict(identity.value),
            )
            for ss58_address, identity in identities
        }

    def get_delegate_take(self, hotkey_ss58: str, block: Optional[int] = None) -> float:
        """
        Retrieves the delegate 'take' percentage for a neuron identified by its hotkey. The 'take' represents the
        percentage of rewards that the delegate claims from its nominators' stakes.

        Parameters:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            block: The blockchain block number for the query.

        Returns:
            float: The delegate take percentage.

        The delegate take is a critical parameter in the network's incentive structure, influencing the distribution of
        rewards among neurons and their nominators.
        """
        result = self.query_subtensor(
            name="Delegates",
            block=block,
            params=[hotkey_ss58],
        )

        return u16_normalized_float(result.value)  # type: ignore

    def get_delegated(
        self, coldkey_ss58: str, block: Optional[int] = None
    ) -> list[DelegatedInfo]:
        """
        Retrieves a list of delegates and their associated stakes for a given coldkey. This function identifies the
        delegates that a specific account has staked tokens on.

        Parameters:
            coldkey_ss58: The `SS58` address of the account's coldkey.
            block: The blockchain block number for the query.

        Returns:
            A list containing the delegated information for the specified coldkey.

        This function is important for account holders to understand their stake allocations and their involvement in
        the network's delegation and consensus mechanisms.
        """

        result = self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegated",
            params=[coldkey_ss58],
            block=block,
        )

        if not result:
            return []

        return DelegatedInfo.list_from_dicts(result)

    def get_delegates(self, block: Optional[int] = None) -> list["DelegateInfo"]:
        """
        Fetches all delegates on the chain

        Parameters:
            block: The blockchain block number for the query.

        Returns:
            List of DelegateInfo objects, or an empty list if there are no delegates.
        """
        result = self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegates",
            params=[],
            block=block,
        )
        if result:
            return DelegateInfo.list_from_dicts(result)
        else:
            return []

    def get_existential_deposit(self, block: Optional[int] = None) -> Optional[Balance]:
        """
        Retrieves the existential deposit amount for the Bittensor blockchain. Always in TAO.
        The existential deposit is the minimum amount of TAO required for an account to exist on the blockchain.
        Accounts with balances below this threshold can be reaped to conserve network resources.

        Parameters:
            block: The blockchain block number for the query.

        Returns:
            The existential deposit amount. Always in TAO.

        The existential deposit is a fundamental economic parameter in the Bittensor network, ensuring efficient use of
        storage and preventing the proliferation of dust accounts.
        """
        result = self.substrate.get_constant(
            module_name="Balances",
            constant_name="ExistentialDeposit",
            block_hash=self.determine_block_hash(block),
        )

        if result is None:
            raise Exception("Unable to retrieve existential deposit amount.")

        return Balance.from_rao(getattr(result, "value", 0))

    def get_hotkey_owner(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[str]:
        """
        Retrieves the owner of the given hotkey at a specific block hash.
        This function queries the blockchain for the owner of the provided hotkey. If the hotkey does not exist at the
        specified block hash, it returns None.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey.
            block: The blockchain block number for the query.

        Returns:
            The SS58 address of the owner if the hotkey exists, or None if it doesn't.
        """
        hk_owner_query = self.substrate.query(
            module="SubtensorModule",
            storage_function="Owner",
            params=[hotkey_ss58],
            block_hash=self.determine_block_hash(block),
        )
        exists = False
        if hk_owner_query:
            exists = self.does_hotkey_exist(hotkey_ss58, block=block)
        hotkey_owner = hk_owner_query if exists else None
        return hotkey_owner

    def get_minimum_required_stake(self) -> Balance:
        """
        Returns the minimum required stake for nominators in the Subtensor network.

        Returns:
            The minimum required stake as a Balance object in TAO.
        """
        result = self.substrate.query(
            module="SubtensorModule", storage_function="NominatorMinRequiredStake"
        )

        return Balance.from_rao(getattr(result, "value", 0))

    # TODO: update parameters order in SDKv10, rename `field_indices` to `selected_indices`
    def get_metagraph_info(
        self,
        netuid: int,
        field_indices: Optional[Union[list[SelectiveMetagraphIndex], list[int]]] = None,
        block: Optional[int] = None,
        mechid: int = 0,
    ) -> Optional[MetagraphInfo]:
        """
        Retrieves full or partial metagraph information for the specified subnet mechanism (netuid, mechid).

        Parameters:
            netuid: Subnet unique identifier.
            field_indices: Optional list of SelectiveMetagraphIndex or int values specifying which fields to retrieve.
                If not provided, all available fields will be returned.
            block: The block number at which to query the data.
            mechid: Subnet mechanism unique identifier.

        Returns:
            MetagraphInfo object with the requested subnet mechanism data, None if the subnet mechanism does not exist.

        Example:
            # Retrieve all fields from the metagraph from subnet 2 mechanism 0
            meta_info = subtensor.get_metagraph_info(netuid=2)

            # Retrieve all fields from the metagraph from subnet 2 mechanism 1
            meta_info = subtensor.get_metagraph_info(netuid=2, mechid=1)

            # Retrieve selective data from the metagraph from subnet 2 mechanism 0
            partial_meta_info = subtensor.get_metagraph_info(
                netuid=2,
                field_indices=[SelectiveMetagraphIndex.Name, SelectiveMetagraphIndex.OwnerHotkeys]
            )

            # Retrieve selective data from the metagraph from subnet 2 mechanism 1
            partial_meta_info = subtensor.get_metagraph_info(
                netuid=2,
                mechid=1,
                field_indices=[SelectiveMetagraphIndex.Name, SelectiveMetagraphIndex.OwnerHotkeys]
            )

        Notes:
            See also:
            - <https://docs.learnbittensor.org/glossary#metagraph>
            - <https://docs.learnbittensor.org/glossary#emission>
        """
        block_hash = self.determine_block_hash(block=block)

        indexes = (
            [
                f.value if isinstance(f, SelectiveMetagraphIndex) else f
                for f in field_indices
            ]
            if field_indices is not None
            else [f for f in range(len(SelectiveMetagraphIndex))]
        )

        query = self.substrate.runtime_call(
            api="SubnetInfoRuntimeApi",
            method="get_selective_mechagraph",
            params=[netuid, mechid, indexes if 0 in indexes else [0] + indexes],
            block_hash=block_hash,
        )
        if query is None or not hasattr(query, "value") or query.value is None:
            logging.error(
                f"Subnet mechanism {netuid}.{mechid if mechid else 0} does not exist."
            )
            return None

        return MetagraphInfo.from_dict(query.value)

    # TODO: update parameters order in SDKv10
    def get_all_metagraphs_info(
        self,
        block: Optional[int] = None,
        all_mechanisms: bool = False,
    ) -> Optional[list[MetagraphInfo]]:
        """
        Retrieves a list of MetagraphInfo objects for all subnets

        Parameters:
            block: The blockchain block number for the query.
            all_mechanisms: If True then returns all mechanisms, otherwise only those with index 0 for all subnets.

        Returns:
            List of MetagraphInfo objects for all existing subnets.

        Notes:
            See also: See <https://docs.learnbittensor.org/glossary#metagraph>
        """
        block_hash = self.determine_block_hash(block)
        method = "get_all_mechagraphs" if all_mechanisms else "get_all_metagraphs"
        query = self.substrate.runtime_call(
            api="SubnetInfoRuntimeApi",
            method=method,
            block_hash=block_hash,
        )
        if query is None or not hasattr(query, "value"):
            return None

        return MetagraphInfo.list_from_dicts(query.value)

    def get_netuids_for_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> list[int]:
        """
        Retrieves a list of subnet UIDs (netuids) for which a given hotkey is a member. This function identifies the
            specific subnets within the Bittensor network where the neuron associated with the hotkey is active.

        Parameters:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            block: The blockchain block number for the query.

        Returns:
            A list of netuids where the neuron is a member.
        """
        result = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="IsNetworkMember",
            params=[hotkey_ss58],
            block_hash=self.determine_block_hash(block),
        )
        netuids = []
        if result.records:
            for record in result:
                if record[1].value:
                    netuids.append(record[0])
        return netuids

    def get_neuron_certificate(
        self, hotkey: str, netuid: int, block: Optional[int] = None
    ) -> Optional[Certificate]:
        """
        Retrieves the TLS certificate for a specific neuron identified by its unique identifier (UID) within a specified
        subnet (netuid) of the Bittensor network.

        Parameters:
            hotkey: The hotkey to query.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            the certificate of the neuron if found, `None` otherwise.

        This function is used for certificate discovery for setting up mutual tls communication between neurons.
        """
        certificate_query = self.query_module(
            module="SubtensorModule",
            name="NeuronCertificates",
            block=block,
            params=[netuid, hotkey],
        )
        try:
            if certificate_query:
                certificate = cast(dict, certificate_query)
                return Certificate(certificate)
        except AttributeError:
            return None
        return None

    def get_all_neuron_certificates(
        self, netuid: int, block: Optional[int] = None
    ) -> dict[str, Certificate]:
        """
        Retrieves the TLS certificates for neurons within a specified subnet (netuid) of the Bittensor network.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            {ss58: Certificate} for the key/Certificate pairs on the subnet

        This function is used for certificate discovery for setting up mutual tls communication between neurons.
        """
        query_certificates = self.query_map(
            module="SubtensorModule",
            name="NeuronCertificates",
            params=[netuid],
            block=block,
        )
        output = {}
        for key, item in query_certificates:
            output[decode_account_id(key)] = Certificate(item.value)
        return output

    def get_liquidity_list(
        self,
        wallet: "Wallet",
        netuid: int,
        block: Optional[int] = None,
    ) -> Optional[list[LiquidityPosition]]:
        """
        Retrieves all liquidity positions for the given wallet on a specified subnet (netuid).
        Calculates associated fee rewards based on current global and tick-level fee data.

        Parameters:
            wallet: Wallet instance to fetch positions for.
            netuid: Subnet unique id.
            block: The blockchain block number for the query.

        Returns:
            List of liquidity positions, or None if subnet does not exist.
        """
        if not self.subnet_exists(netuid=netuid):
            logging.debug(f"Subnet {netuid} does not exist.")
            return None

        if not self.is_subnet_active(netuid=netuid):
            logging.debug(f"Subnet {netuid} is not active.")
            return None

        block_hash = self.determine_block_hash(block)

        # Fetch global fees and current price
        fee_global_tao_query_sk = self.substrate.create_storage_key(
            pallet="Swap",
            storage_function="FeeGlobalTao",
            params=[netuid],
            block_hash=block_hash,
        )
        fee_global_alpha_query_sk = self.substrate.create_storage_key(
            pallet="Swap",
            storage_function="FeeGlobalAlpha",
            params=[netuid],
            block_hash=block_hash,
        )
        sqrt_price_query_sk = self.substrate.create_storage_key(
            pallet="Swap",
            storage_function="AlphaSqrtPrice",
            params=[netuid],
            block_hash=block_hash,
        )
        fee_global_tao_query, fee_global_alpha_query, sqrt_price_query = (
            self.substrate.query_multi(
                [
                    fee_global_tao_query_sk,
                    fee_global_alpha_query_sk,
                    sqrt_price_query_sk,
                ],
                block_hash=block_hash,
            )
        )

        fee_global_tao = fixed_to_float(fee_global_tao_query[1])
        fee_global_alpha = fixed_to_float(fee_global_alpha_query[1])
        sqrt_price = fixed_to_float(sqrt_price_query[1])
        current_tick = price_to_tick(sqrt_price**2)

        # Fetch positions
        positions_response = self.query_map(
            module="Swap",
            name="Positions",
            block=block,
            params=[netuid, wallet.coldkeypub.ss58_address],
        )
        positions_values: list[tuple[dict, int, int]] = []
        positions_storage_keys: list[StorageKey] = []
        for _, p in positions_response:
            position = p.value

            tick_low_idx = position["tick_low"][0]
            tick_high_idx = position["tick_high"][0]

            tick_low_sk = self.substrate.create_storage_key(
                pallet="Swap",
                storage_function="Ticks",
                params=[netuid, tick_low_idx],
                block_hash=block_hash,
            )
            tick_high_sk = self.substrate.create_storage_key(
                pallet="Swap",
                storage_function="Ticks",
                params=[netuid, tick_high_idx],
                block_hash=block_hash,
            )
            positions_values.append((position, tick_low_idx, tick_high_idx))
            positions_storage_keys.extend([tick_low_sk, tick_high_sk])
        # query all our ticks at once
        ticks_query = self.substrate.query_multi(
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

    def get_neuron_for_pubkey_and_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional["NeuronInfo"]:
        """
        Retrieves information about a neuron based on its public key (hotkey SS58 address) and the specific subnet UID
        (netuid). This function provides detailed neuron information for a particular subnet within the Bittensor
        network.

        Parameters:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            Optional: Detailed information about the neuron if found, ``None`` otherwise.

        This function is crucial for accessing specific neuron data and understanding its status, stake, and other
        attributes within a particular subnet of the Bittensor ecosystem.
        """
        block_hash = self.determine_block_hash(block)
        uid = self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
        )
        if uid is None:
            return NeuronInfo.get_null_neuron()

        result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neuron",
            params=[netuid, uid.value],
            block=block,
        )

        if not result:
            return NeuronInfo.get_null_neuron()

        return NeuronInfo.from_dict(result)

    def get_next_epoch_start_block(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Calculates the first block number of the next epoch for the given subnet.

        If `block` is not provided, the current chain block will be used. Epochs are
        determined based on the subnet's tempo (i.e., blocks per epoch). The result
        is the block number at which the next epoch will begin.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The reference block to calculate from. If None, uses the current chain block height.

        Returns:
            int: The block number at which the next epoch will start.
        """
        block = block or self.block
        blocks_since_last_step = self.blocks_since_last_step(netuid=netuid, block=block)
        tempo = self.tempo(netuid=netuid, block=block)

        if block and blocks_since_last_step is not None and tempo:
            return block - blocks_since_last_step + tempo + 1
        return None

    def get_owned_hotkeys(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        reuse_block: bool = False,
    ) -> list[str]:
        """
        Retrieves all hotkeys owned by a specific coldkey address.

        Parameters:
            coldkey_ss58: The SS58 address of the coldkey to query.
            block: The blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            list[str]: A list of hotkey SS58 addresses owned by the coldkey.
        """
        block_hash = self.determine_block_hash(block)
        owned_hotkeys = self.substrate.query(
            module="SubtensorModule",
            storage_function="OwnedHotkeys",
            params=[coldkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return [decode_account_id(hotkey[0]) for hotkey in owned_hotkeys or []]

    def get_stake(
        self,
        coldkey_ss58: str,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
    ) -> Balance:
        """
        Returns the amount of Alpha staked by a specific coldkey to a specific hotkey within a given subnet.
        This function retrieves the delegated stake balance, referred to as the 'Alpha' value.

        Parameters:
            coldkey_ss58: The SS58 address of the coldkey that delegated the stake. This address owns the stake.
            hotkey_ss58: The ss58 address of the hotkey which the stake is on.
            netuid: The unique identifier of the subnet to query.
            block: The specific block number at which to retrieve the stake information.

        Returns:
            An object representing the amount of Alpha (TAO ONLY if the subnet's netuid is 0) currently staked from the
                specified coldkey to the specified hotkey within the given subnet.
        """
        alpha_shares_query = self.query_module(
            module="SubtensorModule",
            name="Alpha",
            block=block,
            params=[hotkey_ss58, coldkey_ss58, netuid],
        )
        alpha_shares = cast(FixedPoint, alpha_shares_query)

        hotkey_alpha_obj: ScaleObj = self.query_module(
            module="SubtensorModule",
            name="TotalHotkeyAlpha",
            block=block,
            params=[hotkey_ss58, netuid],
        )
        hotkey_alpha = hotkey_alpha_obj.value

        hotkey_shares_query = self.query_module(
            module="SubtensorModule",
            name="TotalHotkeyShares",
            block=block,
            params=[hotkey_ss58, netuid],
        )
        hotkey_shares = cast(FixedPoint, hotkey_shares_query)

        alpha_shares_as_float = fixed_to_float(alpha_shares)
        hotkey_shares_as_float = fixed_to_float(hotkey_shares)

        if hotkey_shares_as_float == 0:
            return Balance.from_rao(0).set_unit(netuid=netuid)

        stake = alpha_shares_as_float / hotkey_shares_as_float * hotkey_alpha

        return Balance.from_rao(int(stake)).set_unit(netuid=netuid)

    # TODO: remove unused parameters in SDK.v10
    def get_stake_add_fee(
        self,
        amount: Balance,
        netuid: int,
        coldkey_ss58: str,
        hotkey_ss58: str,
        block: Optional[int] = None,
    ) -> Balance:
        """
        Calculates the fee for adding new stake to a hotkey.

        Parameters:
            amount: Amount of stake to add in TAO
            netuid: Netuid of subnet
            coldkey_ss58: SS58 address of coldkey
            hotkey_ss58: SS58 address of hotkey
            block: Block number at which to perform the calculation

        Returns:
            The calculated stake fee as a Balance object
        """
        return self.get_stake_operations_fee(amount=amount, netuid=netuid, block=block)

    def get_mechanism_emission_split(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[list[int]]:
        """Returns the emission percentages allocated to each subnet mechanism.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            A list of integers representing the percentage of emission allocated to each subnet mechanism (rounded to
            whole numbers). Returns None if emission is evenly split or if the data is unavailable.
        """
        block_hash = self.determine_block_hash(block)
        result = self.substrate.query(
            module="SubtensorModule",
            storage_function="MechanismEmissionSplit",
            params=[netuid],
            block_hash=block_hash,
        )
        if result is None or not hasattr(result, "value"):
            return None

        return [round(i / sum(result.value) * 100) for i in result.value]

    def get_mechanism_count(
        self,
        netuid: int,
        block: Optional[int] = None,
    ) -> int:
        """Retrieves the number of mechanisms for the given subnet.

        Parameters:
            netuid: Subnet identifier.
            block: The blockchain block number for the query.

        Returns:
            The number of mechanisms for the given subnet.
        """
        block_hash = self.determine_block_hash(block)
        query = self.substrate.query(
            module="SubtensorModule",
            storage_function="MechanismCountCurrent",
            params=[netuid],
            block_hash=block_hash,
        )
        return query.value if query is not None and hasattr(query, "value") else 1

    def get_subnet_info(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional["SubnetInfo"]:
        """
        Retrieves detailed information about subnet within the Bittensor network.
        This function provides comprehensive data on subnet, including its characteristics and operational parameters.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            SubnetInfo: A SubnetInfo objects, each containing detailed information about a subnet.

        Gaining insights into the subnet's details assists in understanding the network's composition, the roles of
        different subnets, and their unique features.
        """
        result = self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_info_v2",
            params=[netuid],
            block=block,
        )
        if not result:
            return None
        return SubnetInfo.from_dict(result)

    def get_subnet_price(
        self,
        netuid: int,
        block: Optional[int] = None,
    ) -> Balance:
        """Gets the current Alpha price in TAO for all subnets.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            The current Alpha price in TAO units for the specified subnet.
        """
        # SN0 price is always 1 TAO
        if netuid == 0:
            return Balance.from_tao(1)

        block_hash = self.determine_block_hash(block=block)
        price_rao = self.substrate.runtime_call(
            api="SwapRuntimeApi",
            method="current_alpha_price",
            params=[netuid],
            block_hash=block_hash,
        ).value
        return Balance.from_rao(price_rao)

    def get_subnet_prices(
        self,
        block: Optional[int] = None,
    ) -> dict[int, Balance]:
        """Gets the current Alpha price in TAO for a specified subnet.

        Parameters:
            block: The blockchain block number for the query.

        Returns:
            dict:
                - subnet unique ID
                - The current Alpha price in TAO units for the specified subnet.
        """
        block_hash = self.determine_block_hash(block=block)

        current_sqrt_prices = self.substrate.query_map(
            module="Swap",
            storage_function="AlphaSqrtPrice",
            block_hash=block_hash,
            page_size=129,  # total number of subnets
        )

        prices = {}
        for id_, current_sqrt_price in current_sqrt_prices:
            current_sqrt_price = fixed_to_float(current_sqrt_price)
            current_price = current_sqrt_price * current_sqrt_price
            current_price_in_tao = Balance.from_rao(int(current_price * 1e9))
            prices.update({id_: current_price_in_tao})

        # SN0 price is always 1 TAO
        prices.update({0: Balance.from_tao(1)})
        return prices

    # TODO: update order in SDKv10
    def get_timelocked_weight_commits(
        self,
        netuid: int,
        block: Optional[int] = None,
        mechid: int = 0,
    ) -> list[tuple[str, int, str, int]]:
        """
        Retrieves CRv4 weight commit information for a specific subnet.

        Parameters:
            netuid: Subnet identifier.
            block: The blockchain block number for the query.
            mechid: Subnet mechanism identifier.

        Returns:
            A list of commit details, where each item contains:
                - ss58_address: The address of the committer.
                - commit_block: The block number when the commitment was made.
                - commit_message: The commit message.
                - reveal_round: The round when the commitment was revealed.

            The list may be empty if there are no commits found.
        """
        storage_index = get_mechid_storage_index(netuid, mechid)
        result = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="TimelockedWeightCommits",
            params=[storage_index],
            block_hash=self.determine_block_hash(block=block),
        )

        commits = result.records[0][1] if result.records else []
        return [WeightCommitInfo.from_vec_u8_v2(commit) for commit in commits]

    # TODO: remove unused parameters in SDK.v10
    def get_unstake_fee(
        self,
        amount: Balance,
        netuid: int,
        coldkey_ss58: str,
        hotkey_ss58: str,
        block: Optional[int] = None,
    ) -> Balance:
        """
        Calculates the fee for unstaking from a hotkey.

        Parameters:
            amount: Amount of stake to unstake in TAO
            netuid: Netuid of subnet
            coldkey_ss58: SS58 address of coldkey
            hotkey_ss58: SS58 address of hotkey
            block: Block number at which to perform the calculation

        Returns:
            The calculated stake fee as a Balance object
        """
        return self.get_stake_operations_fee(amount=amount, netuid=netuid, block=block)

    # TODO: remove unused parameters in SDK.v10
    def get_stake_movement_fee(
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

        Parameters:
            amount: Amount of stake to move in TAO
            origin_netuid: Netuid of origin subnet
            origin_hotkey_ss58: SS58 address of origin hotkey
            origin_coldkey_ss58: SS58 address of origin coldkey
            destination_netuid: Netuid of destination subnet
            destination_hotkey_ss58: SS58 address of destination hotkey
            destination_coldkey_ss58: SS58 address of destination coldkey
            block: Block number at which to perform the calculation

        Returns:
            The calculated stake fee as a Balance object
        """
        return self.get_stake_operations_fee(
            amount=amount, netuid=origin_netuid, block=block
        )

    def get_stake_for_coldkey_and_hotkey(
        self,
        coldkey_ss58: str,
        hotkey_ss58: str,
        netuids: Optional[UIDs] = None,
        block: Optional[int] = None,
    ) -> dict[int, StakeInfo]:
        """
        Retrieves all coldkey-hotkey pairing stake across specified (or all) subnets

        Parameters:
            coldkey_ss58: The SS58 address of the coldkey.
            hotkey_ss58: The SS58 address of the hotkey.
            netuids: The subnet IDs to query for. Set to `None` for all subnets.
            block: The block number at which to query the stake information.

        Returns:
            A {netuid: StakeInfo} pairing of all stakes across all subnets.
        """
        if netuids is None:
            all_netuids = self.get_subnets(block=block)
        else:
            all_netuids = netuids
        results = [
            self.query_runtime_api(
                "StakeInfoRuntimeApi",
                "get_stake_info_for_hotkey_coldkey_netuid",
                params=[hotkey_ss58, coldkey_ss58, netuid],
                block=block,
            )
            for netuid in all_netuids
        ]
        return {
            netuid: StakeInfo.from_dict(result)
            for (netuid, result) in zip(all_netuids, results)
        }

    def get_stake_for_coldkey(
        self, coldkey_ss58: str, block: Optional[int] = None
    ) -> list["StakeInfo"]:
        """
        Retrieves the stake information for a given coldkey.

        Parameters:
            coldkey_ss58 The SS58 address of the coldkey.
            block: The block number at which to query the stake information.

        Returns:
            OA list of StakeInfo objects, or ``None`` if no stake information is found.
        """
        result = self.query_runtime_api(
            runtime_api="StakeInfoRuntimeApi",
            method="get_stake_info_for_coldkey",
            params=[coldkey_ss58],
            block=block,
        )

        if result is None:
            return []
        stakes: list[StakeInfo] = StakeInfo.list_from_dicts(result)
        return [stake for stake in stakes if stake.stake > 0]

    get_stake_info_for_coldkey = get_stake_for_coldkey

    def get_stake_for_hotkey(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Balance:
        """
        Retrieves the stake information for a given hotkey.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey.
            netuid: The subnet ID to query for.
            block: The block number at which to query the stake information.
        """
        hotkey_alpha_query = self.query_subtensor(
            name="TotalHotkeyAlpha", params=[hotkey_ss58, netuid], block=block
        )
        hotkey_alpha = cast(ScaleObj, hotkey_alpha_query)
        balance = Balance.from_rao(hotkey_alpha.value)
        balance.set_unit(netuid=netuid)
        return balance

    get_hotkey_stake = get_stake_for_hotkey

    def get_stake_operations_fee(
        self,
        amount: Balance,
        netuid: int,
        block: Optional[int] = None,
    ):
        """Returns fee for any stake operation in specified subnet.

        Parameters:
            amount: Amount of stake to add in Alpha/TAO.
            netuid: Netuid of subnet.
            block: Block number at which to perform the calculation.

        Returns:
            The calculated stake fee as a Balance object.
        """
        block_hash = self.determine_block_hash(block=block)
        result = self.substrate.query(
            module="Swap",
            storage_function="FeeRate",
            params=[netuid],
            block_hash=block_hash,
        )
        return amount * (result.value / U16_MAX)

    def get_stake_weight(self, netuid: int, block: Optional[int] = None) -> list[float]:
        """
        Retrieves the stake weight for all hotkeys in a given subnet.

        Parameters:
            netuid: Netuid of subnet.
            block: Block number at which to perform the calculation.

        Returns:
            A list of stake weights for all hotkeys in the specified subnet.
        """
        block_hash = self.determine_block_hash(block=block)
        result = self.substrate.query(
            module="SubtensorModule",
            storage_function="StakeWeight",
            params=[netuid],
            block_hash=block_hash,
        )
        return [u16_normalized_float(w) for w in result]

    def get_subnet_burn_cost(self, block: Optional[int] = None) -> Optional[Balance]:
        """
        Retrieves the burn cost for registering a new subnet within the Bittensor network. This cost represents the
        amount of Tao that needs to be locked or burned to establish a new subnet.

        Parameters:
            block: The blockchain block number for the query.

        Returns:
            int: The burn cost for subnet registration.

        The subnet burn cost is an important economic parameter, reflecting the network's mechanisms for controlling the
        proliferation of subnets and ensuring their commitment to the network's long-term viability.
        """
        lock_cost = self.query_runtime_api(
            runtime_api="SubnetRegistrationRuntimeApi",
            method="get_network_registration_cost",
            params=[],
            block=block,
        )

        if lock_cost is not None:
            return Balance.from_rao(lock_cost)
        else:
            return lock_cost

    def get_subnet_hyperparameters(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[Union[list, "SubnetHyperparameters"]]:
        """
        Retrieves the hyperparameters for a specific subnet within the Bittensor network. These hyperparameters define
        the operational settings and rules governing the subnet's behavior.

        Parameters:
            netuid: The network UID of the subnet to query.
            block: The blockchain block number for the query.

        Returns:
            The subnet's hyperparameters, or `None` if not available.

        Understanding the hyperparameters is crucial for comprehending how subnets are configured and managed, and how
        they interact with the network's consensus and incentive mechanisms.
        """
        result = self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_hyperparams_v2",
            params=[netuid],
            block=block,
        )

        if not result:
            return None

        return SubnetHyperparameters.from_dict(result)

    def get_subnet_reveal_period_epochs(
        self, netuid: int, block: Optional[int] = None
    ) -> int:
        """Retrieve the SubnetRevealPeriodEpochs hyperparameter."""
        return cast(
            int,
            self.get_hyperparameter(
                param_name="RevealPeriodEpochs", block=block, netuid=netuid
            ),
        )

    def get_subnets(self, block: Optional[int] = None) -> UIDs:
        """
        Retrieves the list of all subnet unique identifiers (netuids) currently present in the Bittensor network.

        Parameters:
            block: The blockchain block number for the query.

        Returns:
            A list of subnet netuids.

        This function provides a comprehensive view of the subnets within the Bittensor network,
        offering insights into its diversity and scale.
        """
        result = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            block_hash=self.determine_block_hash(block),
        )
        subnets = []
        if result.records:
            for netuid, exists in result:
                if exists:
                    subnets.append(netuid)
        return subnets

    def get_total_subnets(self, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the total number of subnets within the Bittensor network as of a specific blockchain block.

        Parameters:
            block: The blockchain block number for the query.

        Returns:
            The total number of subnets in the network.

        Understanding the total number of subnets is essential for assessing the network's growth and the extent of its
        decentralized infrastructure.
        """
        result = self.substrate.query(
            module="SubtensorModule",
            storage_function="TotalNetworks",
            params=[],
            block_hash=self.determine_block_hash(block),
        )
        return getattr(result, "value", None)

    def get_transfer_fee(
        self,
        wallet: "Wallet",
        dest: str,
        value: Optional[Balance],
        keep_alive: bool = True,
    ) -> Balance:
        """
        Calculates the transaction fee for transferring tokens from a wallet to a specified destination address. This
        function simulates the transfer to estimate the associated cost, taking into account the current network
        conditions and transaction complexity.

        Parameters:
            wallet: The wallet from which the transfer is initiated.
            dest: The ``SS58`` address of the destination account.
            value: The amount of tokens to be transferred, specified as a Balance object, or in Tao or Rao units.
            keep_alive: Whether the transfer fee should be calculated based on keeping the wallet alive (existential
                deposit) or not.

        Returns:
            The estimated transaction fee for the transfer, represented as a Balance object.

        Estimating the transfer fee is essential for planning and executing token transactions, ensuring that the wallet
        has sufficient funds to cover both the transfer amount and the associated costs. This function provides a
        crucial tool for managing financial operations within the Bittensor network.
        """
        if value is not None:
            value = check_and_convert_to_balance(value)
        call_params: dict[str, Union[int, str, bool]]
        call_function, call_params = get_transfer_fn_params(value, dest, keep_alive)

        call = self.substrate.compose_call(
            call_module="Balances",
            call_function=call_function,
            call_params=call_params,
        )

        try:
            payment_info = self.substrate.get_payment_info(
                call=call, keypair=wallet.coldkeypub
            )
        except Exception as e:
            logging.error(f":cross_mark: [red]Failed to get payment info: [/red]{e}")
            payment_info = {"partial_fee": int(2e7)}  # assume  0.02 Tao

        return Balance.from_rao(payment_info["partial_fee"])

    def get_vote_data(
        self, proposal_hash: str, block: Optional[int] = None
    ) -> Optional["ProposalVoteData"]:
        """
        Retrieves the voting data for a specific proposal on the Bittensor blockchain. This data includes information
        about how senate members have voted on the proposal.

        Parameters:
            proposal_hash: The hash of the proposal for which voting data is requested.
            block: The blockchain block number for the query.

        Returns:
            An object containing the proposal's voting data, or `None` if not found.

        This function is important for tracking and understanding the decision-making processes within the Bittensor
        network, particularly how proposals are received and acted upon by the governing body.
        """
        vote_data: dict[str, Any] = self.substrate.query(
            module="Triumvirate",
            storage_function="Voting",
            params=[proposal_hash],
            block_hash=self.determine_block_hash(block),
        )

        if vote_data is None:
            return None

        return ProposalVoteData.from_dict(vote_data)

    def get_uid_for_hotkey_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Retrieves the unique identifier (UID) for a neuron's hotkey on a specific subnet.

        Parameters:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            The UID of the neuron if it is registered on the subnet, ``None`` otherwise.

        The UID is a critical identifier within the network, linking the neuron's hotkey to its operational and
        governance activities on a particular subnet.
        """
        result = self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=self.determine_block_hash(block),
        )
        return getattr(result, "value", result)

    def filter_netuids_by_registered_hotkeys(
        self,
        all_netuids: Iterable[int],
        filter_for_netuids: Iterable[int],
        all_hotkeys: Iterable["Wallet"],
        block: Optional[int],
    ) -> list[int]:
        """
        Filters a given list of all netuids for certain specified netuids and hotkeys

        Parameters:
            all_netuids: A list of netuids to filter.
            filter_for_netuids: A subset of all_netuids to filter from the main list.
            all_hotkeys: Hotkeys to filter from the main list.
            block: The blockchain block number for the query.

        Returns:
            The filtered list of netuids.
        """
        self._get_block_hash(block)  # just used to cache the block hash
        netuids_with_registered_hotkeys = [
            item
            for sublist in [
                self.get_netuids_for_hotkey(
                    wallet.hotkey.ss58_address,
                    block=block,
                )
                for wallet in all_hotkeys
            ]
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

    def immunity_period(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Retrieves the 'ImmunityPeriod' hyperparameter for a specific subnet. This parameter defines the duration during
        which new neurons are protected from certain network penalties or restrictions.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            The value of the 'ImmunityPeriod' hyperparameter if the subnet exists, ``None`` otherwise.

        The 'ImmunityPeriod' is a critical aspect of the network's governance system, ensuring that new participants
        have a grace period to establish themselves and contribute to the network without facing immediate punitive
        actions.
        """
        call = self.get_hyperparameter(
            param_name="ImmunityPeriod", netuid=netuid, block=block
        )
        return None if call is None else int(call)

    def is_in_admin_freeze_window(
        self,
        netuid: int,
        block: Optional[int] = None,
    ) -> bool:
        """
        Returns True if the current block is within the terminal freeze window of the tempo
        for the given subnet. During this window, admin ops are prohibited to avoid interference
        with validator weight submissions.

        Parameters:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            bool: True if in freeze window, else False.
        """
        # SN0 doesn't have admin_freeze_window
        if netuid == 0:
            return False

        next_epoch_start_block = self.get_next_epoch_start_block(
            netuid=netuid, block=block
        )

        if next_epoch_start_block is not None:
            remaining = next_epoch_start_block - self.block
            window = self.get_admin_freeze_window(block=block)
            return remaining < window
        return False

    def is_fast_blocks(self):
        """Returns True if the node is running with fast blocks. False if not."""
        return self.query_constant("SubtensorModule", "DurationOfStartCall").value == 10

    def is_hotkey_delegate(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        """
        Determines whether a given hotkey (public key) is a delegate on the Bittensor network. This function checks if
        the neuron associated with the hotkey is part of the network's delegation system.

        Parameters:
            hotkey_ss58: The SS58 address of the neuron's hotkey.
            block: The blockchain block number for the query.

        Returns:
            `True` if the hotkey is a delegate, `False` otherwise.

        Being a delegate is a significant status within the Bittensor network, indicating a neuron's involvement in
        consensus and governance processes.
        """
        delegates = self.get_delegates(block)
        return hotkey_ss58 in [info.hotkey_ss58 for info in delegates]

    def is_hotkey_registered(
        self,
        hotkey_ss58: str,
        netuid: Optional[int] = None,
        block: Optional[int] = None,
    ) -> bool:
        """
        Determines whether a given hotkey (public key) is registered in the Bittensor network, either globally across
        any subnet or specifically on a specified subnet. This function checks the registration status of a neuron
        identified by its hotkey, which is crucial for validating its participation and activities within the network.

        Parameters:
            hotkey_ss58: The SS58 address of the neuron's hotkey.
            netuid: The unique identifier of the subnet to check the registration. If `None`, the registration is
                checked across all subnets.
            block: The blockchain block number at which to perform the query.

        Returns:
            `True` if the hotkey is registered in the specified context (either any subnet or a specific subnet),
                `False` otherwise.

        This function is important for verifying the active status of neurons in the Bittensor network. It aids in
        understanding whether a neuron is eligible to participate in network processes such as consensus, validation,
        and incentive distribution based on its registration status.
        """
        if netuid is None:
            return self.is_hotkey_registered_any(hotkey_ss58, block)
        else:
            return self.is_hotkey_registered_on_subnet(hotkey_ss58, netuid, block)

    def is_hotkey_registered_any(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
    ) -> bool:
        """
        Checks if a neuron's hotkey is registered on any subnet within the Bittensor network.

        Parameters:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            block: The blockchain block number for the query.

        Returns:
            ``True`` if the hotkey is registered on any subnet, False otherwise.

        This function is essential for determining the network-wide presence and participation of a neuron.
        """
        hotkeys = self.get_netuids_for_hotkey(hotkey_ss58, block)
        return len(hotkeys) > 0

    def is_hotkey_registered_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> bool:
        """Checks if the hotkey is registered on a given netuid."""
        return (
            self.get_uid_for_hotkey_on_subnet(hotkey_ss58, netuid, block=block)
            is not None
        )

    def is_subnet_active(self, netuid: int, block: Optional[int] = None) -> bool:
        """Verify if subnet with provided netuid is active.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            True if subnet is active, False otherwise.

        This means whether the `start_call` was initiated or not.
        """
        query = self.query_subtensor(
            name="FirstEmissionBlockNumber",
            block=block,
            params=[netuid],
        )
        return True if query and query.value > 0 else False

    def last_drand_round(self) -> Optional[int]:
        """
        Retrieves the last drand round emitted in bittensor. This corresponds when committed weights will be revealed.

        Returns:
            int: The latest Drand round emitted in bittensor.
        """
        result = self.substrate.query(
            module="Drand", storage_function="LastStoredRound"
        )
        return getattr(result, "value", None)

    def max_weight_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        """
        Returns network MaxWeightsLimit hyperparameter.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.

        Returns:
            The value of the MaxWeightsLimit hyperparameter, or ``None`` if the subnetwork does not exist or the
            parameter is not found.
        """
        call = self.get_hyperparameter(
            param_name="MaxWeightsLimit", netuid=netuid, block=block
        )
        return None if call is None else u16_normalized_float(int(call))

    # TODO: update parameters order in SDKv10
    def metagraph(
        self,
        netuid: int,
        lite: bool = True,
        block: Optional[int] = None,
        mechid: int = 0,
    ) -> "Metagraph":
        """
        Returns a synced metagraph for a specified subnet within the Bittensor network.
        The metagraph represents the network's structure, including neuron connections and interactions.

        Parameters:
            netuid: The network UID of the subnet to query.
            lite: If true, returns a metagraph using a lightweight sync (no weights, no bonds).
            block: Block number for synchronization, or `None` for the latest block.
            mechid: Subnet mechanism identifier.

        Returns:
            The metagraph representing the subnet's structure and neuron relationships.

        The metagraph is an essential tool for understanding the topology and dynamics of the Bittensor network's
        decentralized architecture, particularly in relation to neuron interconnectivity and consensus processes.
        """
        metagraph = Metagraph(
            network=self.chain_endpoint,
            netuid=netuid,
            lite=lite,
            sync=False,
            subtensor=self,
            mechid=mechid,
        )
        metagraph.sync(block=block, lite=lite, subtensor=self)

        return metagraph

    def min_allowed_weights(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Returns network MinAllowedWeights hyperparameter.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.

        Returns:
            The value of the MinAllowedWeights hyperparameter, or ``None`` if the subnetwork does not exist or the
            parameter is not found.
        """
        call = self.get_hyperparameter(
            param_name="MinAllowedWeights", netuid=netuid, block=block
        )
        return None if call is None else int(call)

    def neuron_for_uid(
        self, uid: int, netuid: int, block: Optional[int] = None
    ) -> "NeuronInfo":
        """
        Retrieves detailed information about a specific neuron identified by its unique identifier (UID) within a
        specified subnet (netuid) of the Bittensor network. This function provides a comprehensive view of a neuron's
        attributes, including its stake, rank, and operational status.

        Parameters:
            uid: The unique identifier of the neuron.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            Detailed information about the neuron if found, a null neuron otherwise

        This function is crucial for analyzing individual neurons' contributions and status within a specific subnet,
        offering insights into their roles in the network's consensus and validation mechanisms.
        """
        if uid is None:
            return NeuronInfo.get_null_neuron()

        result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neuron",
            params=[netuid, uid],
            block=block,
        )

        if not result:
            return NeuronInfo.get_null_neuron()

        return NeuronInfo.from_dict(result)

    def neurons(self, netuid: int, block: Optional[int] = None) -> list["NeuronInfo"]:
        """
        Retrieves a list of all neurons within a specified subnet of the Bittensor network.
        This function provides a snapshot of the subnet's neuron population, including each neuron's attributes and
        network interactions.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            A list of NeuronInfo objects detailing each neuron's characteristics in the subnet.

        Understanding the distribution and status of neurons within a subnet is key to comprehending the network's
        decentralized structure and the dynamics of its consensus and governance processes.
        """
        result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons",
            params=[netuid],
            block=block,
        )

        if not result:
            return []

        return NeuronInfo.list_from_dicts(result)

    def neurons_lite(
        self, netuid: int, block: Optional[int] = None
    ) -> list["NeuronInfoLite"]:
        """
        Retrieves a list of neurons in a 'lite' format from a specific subnet of the Bittensor network.
        This function provides a streamlined view of the neurons, focusing on key attributes such as stake and network
        participation.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            A list of simplified neuron information for the subnet.

        This function offers a quick overview of the neuron population within a subnet, facilitating efficient analysis
        of the network's decentralized structure and neuron dynamics.
        """
        result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons_lite",
            params=[netuid],
            block=block,
        )

        if not result:
            return []

        return NeuronInfoLite.list_from_dicts(result)

    def query_identity(
        self, coldkey_ss58: str, block: Optional[int] = None
    ) -> Optional[ChainIdentity]:
        """
        Queries the identity of a neuron on the Bittensor blockchain using the given key. This function retrieves
        detailed identity information about a specific neuron, which is a crucial aspect of the network's decentralized
        identity and governance system.

        Parameters:
            coldkey_ss58: Coldkey used to query the neuron's identity (technically the neuron's coldkey SS58 address).
            block: The blockchain block number for the query.

        Returns:
            An object containing the identity information of the neuron if found, ``None`` otherwise.

        The identity information can include various attributes such as the neuron's stake, rank, and other
        network-specific details, providing insights into the neuron's role and status within the Bittensor network.

        Note:
            See the `Bittensor CLI documentation <https://docs.bittensor.com/reference/btcli>`_ for supported identity
            parameters.
        """
        identity_info = cast(
            dict,
            self.substrate.query(
                module="SubtensorModule",
                storage_function="IdentitiesV2",
                params=[coldkey_ss58],
                block_hash=self.determine_block_hash(block),
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

    def recycle(self, netuid: int, block: Optional[int] = None) -> Optional[Balance]:
        """
        Retrieves the 'Burn' hyperparameter for a specified subnet. The 'Burn' parameter represents the amount of Tao
        that is effectively recycled within the Bittensor network.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            The value of the 'Burn' hyperparameter if the subnet exists, None otherwise.

        Understanding the 'Burn' rate is essential for analyzing the network registration usage, particularly how it is
            correlated with user activity and the overall cost of participation in a given subnet.
        """
        call = self.get_hyperparameter(param_name="Burn", netuid=netuid, block=block)
        return None if call is None else Balance.from_rao(int(call))

    def subnet(self, netuid: int, block: Optional[int] = None) -> Optional[DynamicInfo]:
        """
        Retrieves the subnet information for a single subnet in the network.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The block number to query the subnet information from.

        Returns:
            A DynamicInfo object, containing detailed information about a subnet.
        """
        block_hash = self.determine_block_hash(block=block)

        query = self.substrate.runtime_call(
            api="SubnetInfoRuntimeApi",
            method="get_dynamic_info",
            params=[netuid],
            block_hash=block_hash,
        )

        if isinstance(decoded := query.decode(), dict):
            try:
                price = self.get_subnet_price(netuid=netuid, block=block)
            except (SubstrateRequestException, ValueError):
                price = None
            return DynamicInfo.from_dict({**decoded, "price": price})
        return None

    def subnet_exists(self, netuid: int, block: Optional[int] = None) -> bool:
        """
        Checks if a subnet with the specified unique identifier (netuid) exists within the Bittensor network.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            `True` if the subnet exists, `False` otherwise.

        This function is critical for verifying the presence of specific subnets in the network, enabling a deeper
        understanding of the network's structure and composition.
        """
        result = self.substrate.query(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            params=[netuid],
            block_hash=self.determine_block_hash(block),
        )
        return getattr(result, "value", False)

    def subnetwork_n(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Returns network SubnetworkN hyperparameter.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.

        Returns:
            The value of the SubnetworkN hyperparameter, or ``None`` if the subnetwork does not exist or the parameter
                is not found.
        """
        call = self.get_hyperparameter(
            param_name="SubnetworkN", netuid=netuid, block=block
        )
        return None if call is None else int(call)

    def tempo(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Returns network Tempo hyperparameter.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.

        Returns:
            The value of the Tempo hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not
                found.
        """
        call = self.get_hyperparameter(param_name="Tempo", netuid=netuid, block=block)
        return None if call is None else int(call)

    def tx_rate_limit(self, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the transaction rate limit for the Bittensor network as of a specific blockchain block.
        This rate limit sets the maximum number of transactions that can be processed within a given time frame.

        Parameters:
            block: The blockchain block number for the query.

        Returns:
            The transaction rate limit of the network, None if not available.

        The transaction rate limit is an essential parameter for ensuring the stability and scalability of the Bittensor
            network. It helps in managing network load and preventing congestion, thereby maintaining efficient and
            timely transaction processing.
        """
        result = self.query_subtensor("TxRateLimit", block=block)
        return getattr(result, "value", None)

    def wait_for_block(self, block: Optional[int] = None):
        """
        Waits until a specific block is reached on the chain. If no block is specified,
        waits for the next block.

        Parameters:
            block: The block number to wait for. If None, waits for the next block.

        Returns:
            True if the target block was reached, False if timeout occurred.

        Example:
            import bittensor as bt
            subtensor = bt.Subtensor()

            subtensor.wait_for_block() # Waits for the next block
            subtensor.wait_for_block(block=1234) # Waits for a specific block
        """

        def handler(block_data: dict):
            logging.debug(
                f"reached block {block_data['header']['number']}. Waiting for block {target_block}"
            )
            if block_data["header"]["number"] >= target_block:
                return True
            return None

        current_block = self.substrate.get_block()
        current_block_hash = current_block.get("header", {}).get("hash")
        if block is not None:
            target_block = block
        else:
            target_block = current_block["header"]["number"] + 1

        self.substrate.get_block_handler(
            current_block_hash, header_only=True, subscription_handler=handler
        )
        return True

    # TODO: update order in SDKv10
    def weights(
        self,
        netuid: int,
        block: Optional[int] = None,
        mechid: int = 0,
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the weight distribution set by neurons within a specific subnet of the Bittensor network.
        This function maps each neuron's UID to the weights it assigns to other neurons, reflecting the network's trust
        and value assignment mechanisms.

        Parameters:
            netuid: The network UID of the subnet to query.
            block: Block number for synchronization, or ``None`` for the latest block.
            mechid: Subnet mechanism identifier.

        Returns:
            A list of tuples mapping each neuron's UID to its assigned weights.

        The weight distribution is a key factor in the network's consensus algorithm and the ranking of neurons,
        influencing their influence and reward allocation within the subnet.
        """
        storage_index = get_mechid_storage_index(netuid, mechid)
        w_map_encoded = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Weights",
            params=[storage_index],
            block_hash=self.determine_block_hash(block),
        )
        w_map = [(uid, w.value or []) for uid, w in w_map_encoded]

        return w_map

    def weights_rate_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Returns network WeightsSetRateLimit hyperparameter.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.

        Returns:
            The value of the WeightsSetRateLimit hyperparameter, or ``None`` if the subnetwork does not exist or the
            parameter is not found.
        """
        call = self.get_hyperparameter(
            param_name="WeightsSetRateLimit", netuid=netuid, block=block
        )
        return None if call is None else int(call)

    def get_timestamp(self, block: Optional[int] = None) -> datetime:
        """
        Retrieves the datetime timestamp for a given block

        Parameters:
            block: The blockchain block number for the query.

        Returns:
            datetime object for the timestamp of the block
        """
        unix = cast(ScaleObj, self.query_module("Timestamp", "Now", block=block)).value
        return datetime.fromtimestamp(unix / 1000, tz=timezone.utc)

    def get_subnet_owner_hotkey(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[str]:
        """
        Retrieves the hotkey of the subnet owner for a given network UID.

        This function queries the subtensor network to fetch the hotkey of the owner of a subnet specified by its
        netuid. If no data is found or the query fails, the function returns None.

        Parameters:
            netuid: The network UID of the subnet to fetch the owner's hotkey for.
            block: The specific block number to query the data from.

        Returns:
            The hotkey of the subnet owner if available; None otherwise.
        """
        return self.query_subtensor(
            name="SubnetOwnerHotkey", params=[netuid], block=block
        )

    def get_subnet_validator_permits(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[list[bool]]:
        """
        Retrieves the list of validator permits for a given subnet as boolean values.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.

        Returns:
            A list of boolean values representing validator permits, or None if not available.
        """
        query = self.query_subtensor(
            name="ValidatorPermit",
            params=[netuid],
            block=block,
        )
        return query.value if query is not None and hasattr(query, "value") else query

    # Extrinsics helper ================================================================================================

    def sign_and_send_extrinsic(
        self,
        call: "GenericCall",
        wallet: "Wallet",
        sign_with: str = "coldkey",
        use_nonce: bool = False,
        nonce_key: str = "hotkey",
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        calling_function: Optional[str] = None,
    ) -> ExtrinsicResponse:
        """
        Helper method to sign and submit an extrinsic call to chain.

        Parameters:
            call: a prepared Call object
            wallet: the wallet whose coldkey will be used to sign the extrinsic
            sign_with: the wallet's keypair to use for the signing. Options are "coldkey", "hotkey", "coldkeypub"
            use_nonce: unique identifier for the transaction related with hot/coldkey.
            nonce_key: the type on nonce to use. Options are "hotkey" or "coldkey".
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: raises the relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: whether to wait until the extrinsic call is included on the chain
            wait_for_finalization: whether to wait until the extrinsic call is finalized on the chain
            calling_function: the name of the calling function.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Raises:
            SubstrateRequestException: Substrate request exception.
        """
        extrinsic_response = ExtrinsicResponse(extrinsic_function=calling_function)
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
            next_nonce = self.substrate.get_account_next_index(
                getattr(wallet, nonce_key).ss58_address
            )
            extrinsic_data["nonce"] = next_nonce
        if period is not None:
            extrinsic_data["era"] = {"period": period}

        extrinsic_response.extrinsic = self.substrate.create_signed_extrinsic(
            **extrinsic_data
        )
        try:
            response = self.substrate.submit_extrinsic(
                extrinsic=extrinsic_response.extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                extrinsic_response.message = (
                    "Not waiting for finalization or inclusion."
                )
                return extrinsic_response

            if response.is_success:
                extrinsic_response.message = "Success"
                return extrinsic_response

            response_error_message = response.error_message

            if raise_error:
                raise ChainError.from_error(response_error_message)

            extrinsic_response.success = False
            extrinsic_response.message = format_error_message(response_error_message)
            extrinsic_response.error = response_error_message
            return extrinsic_response

        except SubstrateRequestException as error:
            if raise_error:
                raise

            extrinsic_response.success = False
            extrinsic_response.message = format_error_message(error)
            extrinsic_response.error = error
            return extrinsic_response

    # Extrinsics =======================================================================================================

    def add_stake(
        self,
        wallet: "Wallet",
        netuid: int,
        hotkey_ss58: str,
        amount: Balance,
        safe_staking: bool = False,
        allow_partial_stake: bool = False,
        rate_tolerance: float = 0.005,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Adds a stake from the specified wallet to the neuron identified by the SS58 address of its hotkey in specified
        subnet. Staking is a fundamental process in the Bittensor network that enables neurons to participate actively
        and earn incentives.

        Parameters:
            wallet: The wallet to be used for staking.
            netuid: The unique identifier of the subnet to which the neuron belongs.
            hotkey_ss58: The `ss58` address of the hotkey account to stake to default to the wallet's hotkey.
            amount: The amount of TAO to stake.
            safe_staking: If true, enables price safety checks to protect against fluctuating prices. The stake will
                only execute if the price change doesn't exceed the rate tolerance.
            allow_partial_stake: If true and safe_staking is enabled, allows partial staking when the full amount would
                exceed the price tolerance. If false, the entire stake fails if it would exceed the tolerance.
            rate_tolerance: The maximum allowed price change ratio when staking. For example, 0.005 = 0.5% maximum price
                increase. Only used when safe_staking is True.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function enables neurons to increase their stake in the network, enhancing their influence and potential.
        When safe_staking is enabled, it provides protection against price fluctuations during the time stake is
        executed and the time it is actually processed by the chain.
        """
        amount = check_and_convert_to_balance(amount)
        return add_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            amount=amount,
            safe_staking=safe_staking,
            allow_partial_stake=allow_partial_stake,
            rate_tolerance=rate_tolerance,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def add_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        liquidity: Balance,
        price_low: Balance,
        price_high: Balance,
        hotkey: Optional[str] = None,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Adds liquidity to the specified price range.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            liquidity: The amount of liquidity to be added.
            price_low: The lower bound of the price tick range. In TAO.
            price_high: The upper bound of the price tick range. In TAO.
            hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note: Adding is allowed even when user liquidity is enabled in specified subnet. Call `toggle_user_liquidity`
        method to enable/disable user liquidity.
        """
        return add_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            liquidity=liquidity,
            price_low=price_low,
            price_high=price_high,
            hotkey=hotkey,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def add_stake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        netuids: UIDs,
        amounts: list[Balance],
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Adds stakes to multiple neurons identified by their hotkey SS58 addresses.
        This bulk operation allows for efficient staking across different neurons from a single wallet.

        Parameters:
            wallet: The wallet used for staking.
            hotkey_ss58s: List of ``SS58`` addresses of hotkeys to stake to.
            netuids: List of subnet UIDs.
            amounts: List of corresponding TAO amounts to bet for each netuid and hotkey.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function is essential for managing stakes across multiple neurons, reflecting the dynamic and collaborative
        nature of the Bittensor network.
        """
        return add_stake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def burned_register(
        self,
        wallet: "Wallet",
        netuid: int,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Registers a neuron on the Bittensor network by recycling TAO. This method of registration involves recycling
        TAO tokens, allowing them to be re-mined by performing work on the network.

        Parameters:
            wallet: The wallet associated with the neuron to be registered.
            netuid: The unique identifier of the subnet.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """

        if netuid == 0:
            return root_register_extrinsic(
                subtensor=self,
                wallet=wallet,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        return burned_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def commit_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        salt: Salt,
        uids: UIDs,
        weights: Weights,
        version_key: int = version_as_int,
        max_retries: int = 5,
        period: Optional[int] = 16,
        raise_error: bool = True,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        mechid: int = 0,
    ) -> ExtrinsicResponse:
        """
        Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
        This action serves as a commitment or snapshot of the neuron's current weight distribution.

        Parameters:
            wallet: The wallet associated with the neuron committing the weights.
            netuid: The unique identifier of the subnet.
            salt: list of randomly generated integers as salt to generated weighted hash.
            uids: NumPy array of neuron UIDs for which weights are being committed.
            weights: NumPy array of weight values corresponding to each UID.
            version_key: Version key for compatibility with the network.
            max_retries: The number of maximum attempts to commit weights.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            mechid: Subnet mechanism unique identifier.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function allows neurons to create a tamper-proof record of their weight distribution at a specific point in
        time, enhancing transparency and accountability within the Bittensor network.
        """
        retries = 0
        response = ExtrinsicResponse(
            False, "No attempt made. Perhaps it is too soon to commit weights!"
        )

        logging.info(
            f"Committing weights with params: "
            f"netuid=[blue]{netuid}[/blue], uids=[blue]{uids}[/blue], weights=[blue]{weights}[/blue], "
            f"version_key=[blue]{version_key}[/blue]"
        )

        while retries < max_retries and response.success is False:
            try:
                response = commit_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    mechid=mechid,
                    uids=uids,
                    weights=weights,
                    salt=salt,
                    period=period,
                    raise_error=raise_error,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                if response.success:
                    break
            except Exception as error:
                response.error = error if not response.error else response.error
                logging.error(f"Error committing weights: {error}")
            retries += 1

        return response

    def modify_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        position_id: int,
        liquidity_delta: Balance,
        hotkey: Optional[str] = None,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Modifies liquidity in liquidity position by adding or removing liquidity from it.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            position_id: The id of the position record in the pool.
            liquidity_delta: The amount of liquidity to be added or removed (add if positive or remove if negative).
            hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Example:
            import bittensor as bt

            subtensor = bt.subtensor(network="local")
            my_wallet = bt.Wallet()

            # if `liquidity_delta` is negative
            my_liquidity_delta = Balance.from_tao(100) * -1

            subtensor.modify_liquidity(
                wallet=my_wallet,
                netuid=123,
                position_id=2,
                liquidity_delta=my_liquidity_delta
            )

            # if `liquidity_delta` is positive
            my_liquidity_delta = Balance.from_tao(120)

            subtensor.modify_liquidity(
                wallet=my_wallet,
                netuid=123,
                position_id=2,
                liquidity_delta=my_liquidity_delta
            )

        Note: Modifying is allowed even when user liquidity is enabled in specified subnet. Call `toggle_user_liquidity`
        to enable/disable user liquidity.
        """
        return modify_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            position_id=position_id,
            liquidity_delta=liquidity_delta,
            hotkey=hotkey,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def move_stake(
        self,
        wallet: "Wallet",
        origin_hotkey_ss58: str,
        origin_netuid: int,
        destination_hotkey_ss58: str,
        destination_netuid: int,
        amount: Optional[Balance] = None,
        move_all_stake: bool = False,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Moves stake to a different hotkey and/or subnet.

        Parameters:
            wallet: The wallet to move stake from.
            origin_hotkey_ss58: The SS58 address of the source hotkey.
            origin_netuid: The netuid of the source subnet.
            destination_hotkey_ss58: The SS58 address of the destination hotkey.
            destination_netuid: The netuid of the destination subnet.
            amount: Amount of stake to move.
            move_all_stake: If true, moves all stake from the source hotkey to the destination hotkey.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        amount = check_and_convert_to_balance(amount)
        return move_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            origin_hotkey_ss58=origin_hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_hotkey_ss58=destination_hotkey_ss58,
            destination_netuid=destination_netuid,
            amount=amount,
            move_all_stake=move_all_stake,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def register(
        self,
        wallet: "Wallet",
        netuid: int,
        max_allowed_attempts: int = 3,
        output_in_place: bool = True,
        cuda: bool = False,
        dev_id: Union[list[int], int] = 0,
        tpb: int = 256,
        num_processes: Optional[int] = None,
        update_interval: Optional[int] = None,
        log_verbose: bool = False,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Registers a neuron on the Bittensor subnet with provided netuid using the provided wallet.

        Registration is a critical step for a neuron to become an active participant in the network, enabling it to
        stake, set weights, and receive incentives.

        Parameters:
            wallet: The wallet associated with the neuron to be registered.
            netuid: The unique identifier of the subnet.
            max_allowed_attempts: Maximum number of attempts to register the wallet.
            output_in_place: If true, prints the progress of the proof of work to the console in-place. Meaning the
                progress is printed on the same lines.
            cuda: If ``true``, the wallet should be registered using CUDA device(s).
            dev_id: The CUDA device id to use, or a list of device ids.
            tpb: The number of threads per block (CUDA).
            num_processes: The number of processes to use to register.
            update_interval: The number of nonces to solve between updates.
            log_verbose: If ``true``, the registration process will log more information.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function facilitates the entry of new neurons into the network, supporting the decentralized growth and
        scalability of the Bittensor ecosystem.
        """
        return register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            max_allowed_attempts=max_allowed_attempts,
            tpb=tpb,
            update_interval=update_interval,
            num_processes=num_processes,
            cuda=cuda,
            dev_id=dev_id,
            output_in_place=output_in_place,
            log_verbose=log_verbose,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def register_subnet(
        self,
        wallet: "Wallet",
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Registers a new subnetwork on the Bittensor network.

        Parameters:
            wallet: The wallet to be used for subnet registration.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return register_subnet_extrinsic(
            subtensor=self,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def remove_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        position_id: int,
        hotkey: Optional[str] = None,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Remove liquidity and credit balances back to wallet's hotkey stake.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            position_id: The id of the position record in the pool.
            hotkey: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            - Adding is allowed even when user liquidity is enabled in specified subnet. Call `toggle_user_liquidity`
        extrinsic to enable/disable user liquidity.
            - To get the `position_id` use `get_liquidity_list` method.
        """
        return remove_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            position_id=position_id,
            hotkey=hotkey,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def reveal_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: UIDs,
        weights: Weights,
        salt: Salt,
        max_retries: int = 5,
        version_key: int = version_as_int,
        period: Optional[int] = 16,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        mechid: int = 0,
    ) -> ExtrinsicResponse:
        """
        Reveals the weights for a specific subnet on the Bittensor blockchain using the provided wallet.
        This action serves as a revelation of the neuron's previously committed weight distribution.

        Parameters:
            wallet: Bittensor Wallet instance.
            netuid: The unique identifier of the subnet.
            uids: NumPy array of neuron UIDs for which weights are being revealed.
            weights: NumPy array of weight values corresponding to each UID.
            salt: NumPy array of salt values corresponding to the hash function.
            max_retries: The number of maximum attempts to reveal weights.
            version_key: Version key for compatibility with the network.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            mechid: The subnet mechanism unique identifier.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function allows neurons to reveal their previously committed weight distribution, ensuring transparency and
        accountability within the Bittensor network.

        See also: <https://docs.learnbittensor.org/glossary#commit-reveal>,
        """
        retries = 0
        response = ExtrinsicResponse(
            False, "No attempt made. Perhaps it is too soon to reveal weights!"
        )

        while retries < max_retries and response.success is False:
            try:
                response = reveal_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    mechid=mechid,
                    uids=uids,
                    weights=weights,
                    salt=salt,
                    version_key=version_key,
                    period=period,
                    raise_error=raise_error,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
            except Exception as error:
                response.error = error if not response.error else response.error
                logging.error(f"Error revealing weights: {error}")
            retries += 1

        return response

    def root_register(
        self,
        wallet: "Wallet",
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Register neuron by recycling some TAO.

        Parameters:
            wallet (bittensor_wallet.Wallet): Bittensor wallet instance.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """

        return root_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def root_set_pending_childkey_cooldown(
        self,
        wallet: "Wallet",
        cooldown: int,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Sets the pending childkey cooldown.

        Parameters:
            wallet: bittensor wallet instance.
            cooldown: the number of blocks to setting pending childkey cooldown.
            period (Optional[int]): The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note: This operation can only be successfully performed if your wallet has root privileges.
        """
        return root_set_pending_childkey_cooldown_extrinsic(
            subtensor=self,
            wallet=wallet,
            cooldown=cooldown,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def set_auto_stake(
        self,
        wallet: "Wallet",
        netuid: int,
        hotkey_ss58: str,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> tuple[bool, str]:
        """Sets the coldkey to automatically stake to the hotkey within specific subnet mechanism.

        Parameters:
            wallet: Bittensor Wallet instance.
            netuid: The subnet unique identifier.
            hotkey_ss58: The SS58 address of the validator's hotkey to which the miner automatically stakes all rewards
                received from the specified subnet immediately upon receipt.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            tuple[bool, str]:
                `True` if the extrinsic executed successfully, `False` otherwise.
                `message` is a string value describing the success or potential error.
        """
        return set_auto_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            hotkey_ss58=hotkey_ss58,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def set_children(
        self,
        wallet: "Wallet",
        hotkey: str,
        netuid: int,
        children: list[tuple[float, str]],
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Allows a coldkey to set children-keys.

        Parameters:
            wallet: bittensor wallet instance.
            hotkey: The ``SS58`` address of the neuron's hotkey.
            netuid: The netuid value.
            children: A list of children with their proportions.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return set_children_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey=hotkey,
            netuid=netuid,
            children=children,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def set_delegate_take(
        self,
        wallet: "Wallet",
        hotkey_ss58: str,
        take: float,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        raise_error: bool = False,
        period: Optional[int] = None,
    ) -> ExtrinsicResponse:
        """
        Sets the delegate 'take' percentage for a neuron identified by its hotkey.
        The 'take' represents the percentage of rewards that the delegate claims from its nominators' stakes.

        Parameters:
            wallet: bittensor wallet instance.
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            take: Percentage reward for the delegate.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

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

        current_take = self.get_delegate_take(hotkey_ss58)
        current_take_u16 = int(current_take * 0xFFFF)

        if current_take_u16 == take_u16:
            message = f"The take for {hotkey_ss58} is already set to {take}"
            logging.info(f":white_heavy_check_mark: [green]{message}[/green].")
            return ExtrinsicResponse(True, message)

        logging.info(f"Updating {hotkey_ss58} take: current={current_take} new={take}")

        extrinsic_call = (
            increase_take_extrinsic
            if current_take_u16 < take_u16
            else decrease_take_extrinsic
        )

        response = extrinsic_call(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            take=take_u16,
            period=period,
            raise_error=raise_error,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
        )

        if response.success:
            logging.info(":white_heavy_check_mark: [green]Take Updated[/green]")

        return response

    def set_subnet_identity(
        self,
        wallet: "Wallet",
        netuid: int,
        subnet_identity: SubnetIdentity,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Sets the identity of a subnet for a specific wallet and network.

        Parameters:
            wallet: The wallet instance that will authorize the transaction.
            netuid: The unique ID of the network on which the operation takes place.
            subnet_identity: The identity data of the subnet including attributes like name, GitHub repository, contact,
                URL, discord, description, and any additional metadata.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return set_subnet_identity_extrinsic(
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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def set_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: UIDs,
        weights: Weights,
        block_time: float = 12.0,
        commit_reveal_version: int = 4,
        max_retries: int = 5,
        version_key: int = version_as_int,
        period: Optional[int] = 8,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        mechid: int = 0,
    ) -> ExtrinsicResponse:
        """
        Sets the interneuronal weights for the specified neuron. This process involves specifying the influence or trust
        a neuron places on other neurons in the network, which is a fundamental aspect of Bittensor's decentralized
        learning architecture.

        Parameters:
            wallet: The wallet associated with the subnet validator setting the weights.
            netuid: The unique identifier of the subnet.
            uids: The list of subnet miner neuron UIDs that the weights are being set for.
            weights: The corresponding weights to be set for each UID, representing the validator's evaluation of each
                miner's performance.
            block_time: The number of seconds for block duration.
            commit_reveal_version: The version of the chain commit-reveal protocol to use.
            max_retries: The number of maximum attempts to set weights.
            version_key: Version key for compatibility with the network.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            mechid: The subnet mechanism unique identifier.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function is crucial in shaping the network's collective intelligence, where each neuron's learning and
        contribution are influenced by the weights it sets towards others.

        Notes:
            See <https://docs.learnbittensor.org/glossary#yuma-consensus>
        """

        def _blocks_weight_limit() -> bool:
            bslu = cast(int, self.blocks_since_last_update(netuid, cast(int, uid)))
            wrl = cast(int, self.weights_rate_limit(netuid))
            return bslu > wrl

        retries = 0
        response = ExtrinsicResponse(
            False, "No attempt made. Perhaps it is too soon to set weights!"
        )
        if (
            uid := self.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, netuid)
        ) is None:
            return ExtrinsicResponse(
                False,
                f"Hotkey {wallet.hotkey.ss58_address} not registered in subnet {netuid}",
            )

        if self.commit_reveal_enabled(netuid=netuid):
            # go with `commit_reveal_weights_extrinsic` extrinsic

            while (
                retries < max_retries
                and response.success is False
                and _blocks_weight_limit()
            ):
                logging.info(
                    f"Committing weights for subnet [blue]{netuid}[/blue]. "
                    f"Attempt [blue]{retries + 1}[blue] of [green]{max_retries}[/green]."
                )
                try:
                    response = commit_timelocked_weights_extrinsic(
                        subtensor=self,
                        wallet=wallet,
                        netuid=netuid,
                        mechid=mechid,
                        uids=uids,
                        weights=weights,
                        block_time=block_time,
                        commit_reveal_version=commit_reveal_version,
                        version_key=version_key,
                        period=period,
                        raise_error=raise_error,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )

                except Exception as error:
                    response.error = error if not response.error else response.error
                    logging.error(f"Error setting weights: {error}")
                retries += 1

            return response
        else:
            # go with `set_mechanism_weights_extrinsic`

            while (
                retries < max_retries
                and response.success is False
                and _blocks_weight_limit()
            ):
                try:
                    logging.info(
                        f"Setting weights for subnet [blue]{netuid}[/blue]. "
                        f"Attempt [blue]{retries + 1}[/blue] of [green]{max_retries}[/green]."
                    )
                    response = set_weights_extrinsic(
                        subtensor=self,
                        wallet=wallet,
                        netuid=netuid,
                        mechid=mechid,
                        uids=uids,
                        weights=weights,
                        version_key=version_key,
                        period=period,
                        raise_error=raise_error,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )
                except Exception as error:
                    response.error = error if not response.error else response.error
                    logging.error(f"Error setting weights: {error}")
                retries += 1

            return response

    def serve_axon(
        self,
        netuid: int,
        axon: "Axon",
        certificate: Optional[Certificate] = None,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Registers an ``Axon`` serving endpoint on the Bittensor network for a specific neuron.

        This function is used to set up the Axon, a key component of a neuron that handles incoming queries and data
        processing tasks.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            axon: The Axon instance to be registered for serving.
            certificate: Certificate to use for TLS. If ``None``, no TLS will be used.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        By registering an Axon, the neuron becomes an active part of the network's distributed computing infrastructure,
        contributing to the collective intelligence of Bittensor.
        """
        return serve_axon_extrinsic(
            subtensor=self,
            netuid=netuid,
            axon=axon,
            certificate=certificate,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def set_commitment(
        self,
        wallet: "Wallet",
        netuid: int,
        data: str,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Commits arbitrary data to the Bittensor network by publishing metadata.

        This method allows neurons to publish arbitrary data to the blockchain, which can be used for various purposes
        such as sharing model updates, configuration data, or other network-relevant information.


        Parameters:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the data.
            netuid (int): The unique identifier of the subnetwork.
            data (str): The data to be committed to the network.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Example:
            # Commit some data to subnet 1
            success = await subtensor.commit(wallet=my_wallet, netuid=1, data="Hello Bittensor!")

            # Commit with custom period
            success = await subtensor.commit(wallet=my_wallet, netuid=1, data="Model update v2.0", period=100)

        Note: See <https://docs.learnbittensor.org/glossary#commit-reveal>
        """
        return publish_metadata_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            data_type=f"Raw{len(data)}",
            data=data.encode(),
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def set_reveal_commitment(
        self,
        wallet,
        netuid: int,
        data: str,
        blocks_until_reveal: int = 360,
        block_time: Union[int, float] = 12,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Commits arbitrary data to the Bittensor network by publishing metadata.

        Parameters:
            wallet: The wallet associated with the neuron committing the data.
            netuid: The unique identifier of the subnetwork.
            data: The data to be committed to the network.
            blocks_until_reveal: The number of blocks from now after which the data will be revealed. Then number of
                blocks in one epoch.
            block_time: The number of seconds between each block.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note: A commitment can be set once per subnet epoch and is reset at the next epoch in the chain automatically.
        """

        encrypted, reveal_round = get_encrypted_commitment(
            data, blocks_until_reveal, block_time
        )

        # increase reveal_round in return + 1 because we want to fetch data from the chain after that round was revealed
        # and stored.
        data_ = {"encrypted": encrypted, "reveal_round": reveal_round}
        response = publish_metadata_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            data_type="TimelockEncrypted",
            data=data_,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        response.data = data_
        return response

    def start_call(
        self,
        wallet: "Wallet",
        netuid: int,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicResponse:
        """
        Submits a start_call extrinsic to the blockchain, to trigger the start call process for a subnet (used to start
            a new subnet's emission mechanism).

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return start_call_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def swap_stake(
        self,
        wallet: "Wallet",
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        safe_swapping: bool = False,
        allow_partial_stake: bool = False,
        rate_tolerance: float = 0.005,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Moves stake between subnets while keeping the same coldkey-hotkey pair ownership.
        Like subnet hopping - same owner, same hotkey, just changing which subnet the stake is in.

        Parameters:
            wallet: The wallet to swap stake from.
            hotkey_ss58: The SS58 address of the hotkey whose stake is being swapped.
            origin_netuid: The netuid from which stake is removed.
            destination_netuid: The netuid to which stake is added.
            amount: The amount to swap.
            safe_swapping: If true, enables price safety checks to protect against fluctuating prices. The swap
                will only execute if the price ratio between subnets doesn't exceed the rate tolerance.
            allow_partial_stake: If true and safe_staking is enabled, allows partial stake swaps when the full amount
                would exceed the price tolerance. If false, the entire swap fails if it would exceed the tolerance.
            rate_tolerance: The maximum allowed increase in the price ratio between subnets
                (origin_price/destination_price). For example, 0.005 = 0.5% maximum increase. Only used when
                safe_staking is True.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        The price ratio for swap_stake in safe mode is calculated as: origin_subnet_price / destination_subnet_price
        When safe_staking is enabled, the swap will only execute if:
            - With allow_partial_stake=False: The entire swap amount can be executed without the price ratio
            increasing more than rate_tolerance
            - With allow_partial_stake=True: A partial amount will be swapped up to the point where the
            price ratio would increase by rate_tolerance
        """
        amount = check_and_convert_to_balance(amount)
        return swap_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            amount=amount,
            safe_swapping=safe_swapping,
            allow_partial_stake=allow_partial_stake,
            rate_tolerance=rate_tolerance,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def toggle_user_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        enable: bool,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Allow to toggle user liquidity for specified subnet.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            enable: Boolean indicating whether to enable user liquidity.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note: The call can be executed successfully by the subnet owner only.
        """
        return toggle_user_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            enable=enable,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def transfer(
        self,
        wallet: "Wallet",
        destination: str,
        amount: Optional[Balance],
        transfer_all: bool = False,
        keep_alive: bool = True,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicResponse:
        """
        Transfer token of amount to destination.

        Parameters:
            wallet: Source wallet for the transfer.
            destination: Destination address for the transfer.
            amount: Number of tokens to transfer. `None` is transferring all.
            transfer_all: Flag to transfer all tokens.
            keep_alive: Flag to keep the connection alive.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        if amount is not None:
            amount = check_and_convert_to_balance(amount)
        return transfer_extrinsic(
            subtensor=self,
            wallet=wallet,
            destination=destination,
            amount=amount,
            transfer_all=transfer_all,
            keep_alive=keep_alive,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def transfer_stake(
        self,
        wallet: "Wallet",
        destination_coldkey_ss58: str,
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Transfers stake from one subnet to another while changing the coldkey owner.

        Parameters:
            wallet: The wallet to transfer stake from.
            destination_coldkey_ss58: The destination coldkey SS58 address.
            hotkey_ss58: The hotkey SS58 address associated with the stake.
            origin_netuid: The source subnet UID.
            destination_netuid: The destination subnet UID.
            amount: Amount to transfer.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        amount = check_and_convert_to_balance(amount)
        return transfer_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            destination_coldkey_ss58=destination_coldkey_ss58,
            hotkey_ss58=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            amount=amount,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def unstake(
        self,
        wallet: "Wallet",
        netuid: int,
        hotkey_ss58: str,
        amount: Balance,
        allow_partial_stake: bool = False,
        rate_tolerance: float = 0.005,
        safe_unstaking: bool = False,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting
        individual neuron stakes within the Bittensor network.

        Parameters:
            wallet: The wallet associated with the neuron from which the stake is being removed.
            netuid: The unique identifier of the subnet.
            hotkey_ss58: The ``SS58`` address of the hotkey account to unstake from.
            amount: The amount of alpha to unstake. If not specified, unstakes all. Alpha amount.
            allow_partial_stake: If true and safe_staking is enabled, allows partial unstaking when
                the full amount would exceed the price tolerance. If false, the entire unstake fails if it would
                exceed the tolerance.
            rate_tolerance: The maximum allowed price change ratio when unstaking. For example,
                0.005 = 0.5% maximum price decrease. Only used when safe_staking is True.
            safe_unstaking: If true, enables price safety checks to protect against fluctuating prices. The unstake
                will only execute if the price change doesn't exceed the rate tolerance.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function supports flexible stake management, allowing neurons to adjust their network participation and
        potential reward accruals. When safe_staking is enabled, it provides protection against price fluctuations
        during the time unstake is executed and the time it is actually processed by the chain.
        """
        amount = check_and_convert_to_balance(amount)
        return unstake_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            hotkey_ss58=hotkey_ss58,
            amount=amount,
            allow_partial_stake=allow_partial_stake,
            rate_tolerance=rate_tolerance,
            safe_unstaking=safe_unstaking,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def unstake_all(
        self,
        wallet: "Wallet",
        hotkey: str,
        netuid: int,
        rate_tolerance: Optional[float] = 0.005,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Unstakes all TAO/Alpha associated with a hotkey from the specified subnets on the Bittensor network.

        Parameters:
            wallet: The wallet of the stake owner.
            hotkey: The SS58 address of the hotkey to unstake from.
            netuid: The unique identifier of the subnet.
            rate_tolerance: The maximum allowed price change ratio when unstaking. For example, 0.005 = 0.5% maximum
                price decrease. If not passed (None), then unstaking goes without price limit.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Example:
            # If you would like to unstake all stakes in all subnets safely:
            import bittensor as bt

            subtensor = bt.Subtensor()
            wallet = bt.Wallet("my_wallet")
            netuid = 14
            hotkey = "5%SOME_HOTKEY%"

            wallet_stakes = subtensor.get_stake_info_for_coldkey(coldkey_ss58=wallet.coldkey.ss58_address)

            for stake in wallet_stakes:
                result = subtensor.unstake_all(
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
        return unstake_all_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey=hotkey,
            netuid=netuid,
            rate_tolerance=rate_tolerance,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def unstake_multiple(
        self,
        wallet: "Wallet",
        netuids: UIDs,
        hotkey_ss58s: list[str],
        amounts: Optional[list[Balance]] = None,
        unstake_all: bool = False,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Performs batch unstaking from multiple hotkey accounts, allowing a neuron to reduce its staked amounts
            efficiently. This function is useful for managing the distribution of stakes across multiple neurons.

        Parameters:
            wallet: The wallet linked to the coldkey from which the stakes are being withdrawn.
            netuids: Subnets unique IDs.
            hotkey_ss58s: A list of hotkey `SS58` addresses to unstake from.
            amounts: The amounts of TAO to unstake from each hotkey. If not provided, unstakes all.
            unstake_all: If true, unstakes all tokens. If `True` amounts are ignored.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function allows for strategic reallocation or withdrawal of stakes, aligning with the dynamic stake
        management aspect of the Bittensor network.
        """
        return unstake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            unstake_all=unstake_all,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
