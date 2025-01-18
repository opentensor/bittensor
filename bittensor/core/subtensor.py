import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import numpy as np
from async_substrate_interface.sync_substrate import SubstrateInterface
from numpy.typing import NDArray
import scalecodec
from scalecodec.base import RuntimeConfiguration
from scalecodec.type_registry import load_type_registry_preset

from bittensor.core import SubtensorMixin
from bittensor.core.chain_data import custom_rpc_type_registry
from bittensor.core.metagraph import Metagraph
from bittensor.core.settings import version_as_int, SS58_FORMAT, TYPE_REGISTRY
from bittensor.core.types import ParamWithTypes
from bittensor.utils import torch
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import ProposalVoteData
    from bittensor.core.axon import Axon
    from bittensor.core.config import Config
    from bittensor.core.chain_data.delegate_info import DelegateInfo
    from bittensor.core.chain_data.neuron_info import NeuronInfo
    from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite
    from bittensor.core.chain_data.stake_info import StakeInfo
    from bittensor.core.chain_data.subnet_hyperparameters import SubnetHyperparameters
    from bittensor.core.chain_data.subnet_info import SubnetInfo
    from bittensor.utils.balance import Balance
    from bittensor.utils import Certificate
    from async_substrate_interface.sync_substrate import QueryMapResult
    from bittensor.utils.delegates_details import DelegatesDetails
    from scalecodec.types import ScaleType


class Subtensor(SubtensorMixin):
    """
    TODO docstring
    """

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional["Config"] = None,
        _mock: bool = False,
        log_verbose: bool = False,
    ):
        """
        Initializes an instance of the AsyncSubtensor class.

        Arguments:
            network (str): The network name or type to connect to.
            config (Optional[Config]): Configuration object for the AsyncSubtensor instance.
            _mock: Whether this is a mock instance. Mainly just for use in testing.
            log_verbose (bool): Enables or disables verbose logging.

        Raises:
            Any exceptions raised during the setup, configuration, or connection process.
        """
        if config is None:
            config = self.config()
        self._config = copy.deepcopy(config)
        self.chain_endpoint, self.network = self.setup_config(network, self._config)
        self._mock = _mock

        self.log_verbose = log_verbose
        self._check_and_log_network_settings()

        logging.debug(
            f"Connecting to <network: [blue]{self.network}[/blue], "
            f"chain_endpoint: [blue]{self.chain_endpoint}[/blue]> ..."
        )
        self.substrate = SubstrateInterface(
            url=self.chain_endpoint,
            ss58_format=SS58_FORMAT,
            type_registry=TYPE_REGISTRY,
            use_remote_preset=True,
            chain_name="Bittensor",
            _mock=_mock,
        )
        if self.log_verbose:
            logging.info(
                f"Connected to {self.network} network and {self.chain_endpoint}."
            )

    def close(self):
        """
        Does nothing. Exists for backwards compatibility purposes.
        """
        pass

    # Subtensor queries ===========================================================================================

    def query_constant(
        self, module_name: str, constant_name: str, block: Optional[int] = None
    ) -> Optional["ScaleType"]:
        """
        Retrieves a constant from the specified module on the Bittensor blockchain. This function is used to access
            fixed parameters or values defined within the blockchain's modules, which are essential for understanding
            the network's configuration and rules.

        Args:
            module_name: The name of the module containing the constant.
            constant_name: The name of the constant to retrieve.
            block: The blockchain block number at which to query the constant.

        Returns:
            Optional[scalecodec.ScaleType]: The value of the constant if found, `None` otherwise.

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
        block: Optional[int] = None,
        params: Optional[list] = None,
    ) -> "QueryMapResult":
        """
        Queries map storage from any module on the Bittensor blockchain. This function retrieves data structures that
            represent key-value mappings, essential for accessing complex and structured data within the blockchain
            modules.

        Args:
            module: The name of the module from which to query the map storage.
            name: The specific storage function within the module to query.
            block: The blockchain block number at which to perform the query.
            params: Parameters to be passed to the query.

        Returns:
            result: A data structure representing the map storage if found, `None` otherwise.

        This function is particularly useful for retrieving detailed and structured data from various blockchain
            modules, offering insights into the network's state and the relationships between its different components.
        """
        result = self.substrate.query_map(
            module=module,
            storage_function=name,
            params=params,
            block_hash=self.determine_block_hash(block=block),
        )
        return getattr(result, "value", None)

    def query_map_subtensor(
        self, name: str, block: Optional[int] = None, params: Optional[list] = None
    ) -> "QueryMapResult":
        """
        Queries map storage from the Subtensor module on the Bittensor blockchain. This function is designed to retrieve
            a map-like data structure, which can include various neuron-specific details or network-wide attributes.

        Args:
            name: The name of the map storage function to query.
            block: The blockchain block number at which to perform the query.
            params: A list of parameters to pass to the query function.

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
        block: Optional[int] = None,
        params: Optional[list] = None,
    ) -> "ScaleType":
        """
        Queries any module storage on the Bittensor blockchain with the specified parameters and block number. This
            function is a generic query interface that allows for flexible and diverse data retrieval from various
            blockchain modules.

        Args:
            module (str): The name of the module from which to query data.
            name (str): The name of the storage function within the module.
            block (Optional[int]): The blockchain block number at which to perform the query.
            params (Optional[list[object]]): A list of parameters to pass to the query function.

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
        params: Optional[Union[list[int], dict[str, int]]] = None,
        block: Optional[int] = None,
    ) -> Optional[str]:
        """
        Queries the runtime API of the Bittensor blockchain, providing a way to interact with the underlying runtime and
            retrieve data encoded in Scale Bytes format. This function is essential for advanced users who need to
            interact with specific runtime methods and decode complex data types.

        Args:
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

        call_definition = TYPE_REGISTRY["runtime_api"][runtime_api]["methods"][method]

        data = (
            "0x"
            if params is None
            else self.encode_params(call_definition=call_definition, params=params)
        )
        api_method = f"{runtime_api}_{method}"

        json_result = self.substrate.rpc_request(
            method="state_call",
            params=[api_method, data, block_hash] if block_hash else [api_method, data],
        )

        if json_result is None:
            return None

        return_type = call_definition["type"]

        as_scale_bytes = scalecodec.ScaleBytes(json_result["result"])  # type: ignore

        rpc_runtime_config = RuntimeConfiguration()
        rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
        rpc_runtime_config.update_type_registry(custom_rpc_type_registry)

        obj = rpc_runtime_config.create_scale_object(return_type, as_scale_bytes)
        if obj.data.to_hex() == "0x0400":  # RPC returned None result
            return None

        return obj.decode()

    def query_subtensor(
        self, name: str, block: Optional[int] = None, params: Optional[list] = None
    ) -> "ScaleType":
        """
        Queries named storage from the Subtensor module on the Bittensor blockchain. This function is used to retrieve
            specific data or parameters from the blockchain, such as stake, rank, or other neuron-specific attributes.

        Args:
            name: The name of the storage function to query.
            block: The blockchain block number at which to perform the query.
            params: A list of parameters to pass to the query function.

        Returns:
            query_response (scalecodec.ScaleType): An object containing the requested data.

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

        Args:
            method: The method name for the state call.
            data: The data to be passed to the method.
            block: The blockchain block number at which to perform the state call.

        Returns:
            result (dict[Any, Any]): The result of the rpc call.

        The state call function provides a more direct and flexible way of querying blockchain data, useful for specific
            use cases where standard queries are insufficient.
        """
        block_hash = self.determine_block_hash(block)
        return self.substrate.rpc_request(
            method="state_call",
            params=[method, data, block_hash] if block_hash else [method, data],
        )

    # Common subtensor calls ===========================================================================================

    @property
    def block(self) -> int:
        return self.get_current_block()

    def blocks_since_last_update(self, netuid: int, uid: int) -> Optional[int]:
        """
        Returns the number of blocks since the last update for a specific UID in the subnetwork.

        Arguments:
            netuid (int): The unique identifier of the subnetwork.
            uid (int): The unique identifier of the neuron.

        Returns:
            Optional[int]: The number of blocks since the last update, or ``None`` if the subnetwork or UID does not
                exist.
        """
        call = self.get_hyperparameter(param_name="LastUpdate", netuid=netuid)
        return None if call is None else (self.get_current_block() - int(call[uid]))

    def bonds(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the bond distribution set by neurons within a specific subnet of the Bittensor network.
            Bonds represent the investments or commitments made by neurons in one another, indicating a level of trust
            and perceived value. This bonding mechanism is integral to the network's market-based approach to
            measuring and rewarding machine intelligence.

        Args:
            netuid: The network UID of the subnet to query.
            block: the block number for this query.

        Returns:
            List of tuples mapping each neuron's UID to its bonds with other neurons.

        Understanding bond distributions is crucial for analyzing the trust dynamics and market behavior within the
            subnet. It reflects how neurons recognize and invest in each other's intelligence and contributions,
            supporting diverse and niche systems within the Bittensor ecosystem.
        """
        b_map_encoded = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Bonds",
            params=[netuid],
            block_hash=self.determine_block_hash(block),
        )
        b_map = []
        for uid, b in b_map_encoded:
            b_map.append((uid, b.value))

        return b_map

    def commit(self, wallet, netuid: int, data: str) -> bool:
        """
        Commits arbitrary data to the Bittensor network by publishing metadata.

        Arguments:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the data.
            netuid (int): The unique identifier of the subnetwork.
            data (str): The data to be committed to the network.
        """
        # TODO add
        return publish_metadata(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            data_type=f"Raw{len(data)}",
            data=data.encode(),
        )

    def commit_reveal_enabled(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[bool]:
        return self.execute_coroutine(
            self.async_subtensor.commit_reveal_enabled(netuid=netuid, block=block)
        )

    def difficulty(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.difficulty(netuid=netuid, block=block),
        )

    def does_hotkey_exist(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.does_hotkey_exist(hotkey_ss58=hotkey_ss58, block=block)
        )

    def get_all_subnets_info(self, block: Optional[int] = None) -> list["SubnetInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.get_all_subnets_info(block=block),
        )

    def get_balance(self, address: str, block: Optional[int] = None) -> "Balance":
        return self.execute_coroutine(
            self.async_subtensor.get_balance(address, block=block),
        )

    def get_balances(
        self,
        *addresses: str,
        block: Optional[int] = None,
    ) -> dict[str, "Balance"]:
        return self.execute_coroutine(
            self.async_subtensor.get_balances(*addresses, block=block),
        )

    def get_current_block(self) -> int:
        return self.execute_coroutine(
            coroutine=self.async_subtensor.get_current_block(),
        )

    @lru_cache(maxsize=128)
    def get_block_hash(self, block: Optional[int] = None) -> str:
        return self.execute_coroutine(
            coroutine=self.async_subtensor.get_block_hash(block=block),
        )

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

        Arguments:
            param_name (str): The name of the hyperparameter to retrieve.
            netuid (int): The unique identifier of the subnet.
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

    def get_children(
        self, hotkey: str, netuid: int, block: Optional[int] = None
    ) -> tuple[bool, list, str]:
        return self.execute_coroutine(
            self.async_subtensor.get_children(
                hotkey=hotkey, netuid=netuid, block=block
            ),
        )

    def get_commitment(self, netuid: int, uid: int, block: Optional[int] = None) -> str:
        return self.execute_coroutine(
            self.async_subtensor.get_commitment(netuid=netuid, uid=uid, block=block),
        )

    def get_current_weight_commit_info(
        self, netuid: int, block: Optional[int] = None
    ) -> list:
        return self.execute_coroutine(
            self.async_subtensor.get_current_weight_commit_info(
                netuid=netuid, block=block
            ),
        )

    def get_delegate_by_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional["DelegateInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.get_delegate_by_hotkey(
                hotkey_ss58=hotkey_ss58, block=block
            ),
        )

    def get_delegate_identities(
        self, block: Optional[int] = None
    ) -> dict[str, "DelegatesDetails"]:
        return self.execute_coroutine(
            self.async_subtensor.get_delegate_identities(block=block),
        )

    def get_delegate_take(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[float]:
        return self.execute_coroutine(
            self.async_subtensor.get_delegate_take(
                hotkey_ss58=hotkey_ss58, block=block
            ),
        )

    def get_delegated(
        self, coldkey_ss58: str, block: Optional[int] = None
    ) -> list[tuple["DelegateInfo", "Balance"]]:
        return self.execute_coroutine(
            self.async_subtensor.get_delegated(coldkey_ss58=coldkey_ss58, block=block),
        )

    def get_delegates(self, block: Optional[int] = None) -> list["DelegateInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.get_delegates(block=block),
        )

    def get_existential_deposit(
        self, block: Optional[int] = None
    ) -> Optional["Balance"]:
        return self.execute_coroutine(
            self.async_subtensor.get_existential_deposit(block=block),
        )

    def get_hotkey_owner(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[str]:
        return self.execute_coroutine(
            self.async_subtensor.get_hotkey_owner(hotkey_ss58=hotkey_ss58, block=block),
        )

    def get_minimum_required_stake(self) -> "Balance":
        return self.execute_coroutine(
            self.async_subtensor.get_minimum_required_stake(),
        )

    def get_netuids_for_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None, reuse_block: bool = False
    ) -> list[int]:
        return self.execute_coroutine(
            self.async_subtensor.get_netuids_for_hotkey(
                hotkey_ss58=hotkey_ss58, block=block, reuse_block=reuse_block
            ),
        )

    def get_neuron_certificate(
        self, hotkey: str, netuid: int, block: Optional[int] = None
    ) -> Optional["Certificate"]:
        return self.execute_coroutine(
            self.async_subtensor.get_neuron_certificate(hotkey, netuid, block=block),
        )

    def get_neuron_for_pubkey_and_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional["NeuronInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.get_neuron_for_pubkey_and_subnet(
                hotkey_ss58, netuid, block=block
            ),
        )

    def get_stake_for_coldkey_and_hotkey(
        self, hotkey_ss58: str, coldkey_ss58: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        return self.execute_coroutine(
            self.async_subtensor.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=hotkey_ss58, coldkey_ss58=coldkey_ss58, block=block
            ),
        )

    def get_stake_info_for_coldkey(
        self, coldkey_ss58: str, block: Optional[int] = None
    ) -> list["StakeInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.get_stake_info_for_coldkey(
                coldkey_ss58=coldkey_ss58, block=block
            ),
        )

    def get_subnet_burn_cost(self, block: Optional[int] = None) -> Optional[str]:
        return self.execute_coroutine(
            self.async_subtensor.get_subnet_burn_cost(block=block),
        )

    def get_subnet_hyperparameters(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[Union[list, "SubnetHyperparameters"]]:
        return self.execute_coroutine(
            self.async_subtensor.get_subnet_hyperparameters(netuid=netuid, block=block),
        )

    def get_subnet_reveal_period_epochs(
        self, netuid: int, block: Optional[int] = None
    ) -> int:
        return self.execute_coroutine(
            self.async_subtensor.get_subnet_reveal_period_epochs(
                netuid=netuid, block=block
            ),
        )

    def get_subnets(self, block: Optional[int] = None) -> list[int]:
        return self.execute_coroutine(
            self.async_subtensor.get_subnets(block=block),
        )

    def get_total_stake_for_coldkey(
        self, ss58_address: str, block: Optional[int] = None
    ) -> "Balance":
        result = self.execute_coroutine(
            self.async_subtensor.get_total_stake_for_coldkey(ss58_address, block=block),
        )
        return result

    def get_total_stake_for_coldkeys(
        self, *ss58_addresses: str, block: Optional[int] = None
    ) -> dict[str, "Balance"]:
        return self.execute_coroutine(
            self.async_subtensor.get_total_stake_for_coldkeys(
                *ss58_addresses, block=block
            ),
        )

    def get_total_stake_for_hotkey(
        self, ss58_address: str, block: Optional[int] = None
    ) -> "Balance":
        result = self.execute_coroutine(
            self.async_subtensor.get_total_stake_for_hotkey(ss58_address, block=block),
        )
        return result

    def get_total_stake_for_hotkeys(
        self, *ss58_addresses: str, block: Optional[int] = None
    ) -> dict[str, "Balance"]:
        return self.execute_coroutine(
            self.async_subtensor.get_total_stake_for_hotkeys(
                *ss58_addresses, block=block
            ),
        )

    def get_total_subnets(self, block: Optional[int] = None) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.get_total_subnets(block=block),
        )

    def get_transfer_fee(
        self, wallet: "Wallet", dest: str, value: Union["Balance", float, int]
    ) -> "Balance":
        return self.execute_coroutine(
            self.async_subtensor.get_transfer_fee(
                wallet=wallet, dest=dest, value=value
            ),
        )

    def get_vote_data(
        self, proposal_hash: str, block: Optional[int] = None
    ) -> Optional["ProposalVoteData"]:
        return self.execute_coroutine(
            self.async_subtensor.get_vote_data(
                proposal_hash=proposal_hash, block=block
            ),
        )

    def get_uid_for_hotkey_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.get_uid_for_hotkey_on_subnet(
                hotkey_ss58=hotkey_ss58, netuid=netuid, block=block
            ),
        )

    def filter_netuids_by_registered_hotkeys(
        self,
        all_netuids: Iterable[int],
        filter_for_netuids: Iterable[int],
        all_hotkeys: Iterable["Wallet"],
        block: Optional[int],
    ) -> list[int]:
        return self.execute_coroutine(
            self.async_subtensor.filter_netuids_by_registered_hotkeys(
                all_netuids=all_netuids,
                filter_for_netuids=filter_for_netuids,
                all_hotkeys=all_hotkeys,
                block=block,
            ),
        )

    def immunity_period(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.immunity_period(netuid=netuid, block=block),
        )

    def is_hotkey_delegate(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.is_hotkey_delegate(
                hotkey_ss58=hotkey_ss58, block=block
            ),
        )

    def is_hotkey_registered(
        self,
        hotkey_ss58: str,
        netuid: Optional[int] = None,
        block: Optional[int] = None,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.is_hotkey_registered(
                hotkey_ss58=hotkey_ss58, netuid=netuid, block=block
            ),
        )

    def is_hotkey_registered_any(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.is_hotkey_registered_any(
                hotkey_ss58=hotkey_ss58,
                block=block,
            ),
        )

    def is_hotkey_registered_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> bool:
        return self.get_uid_for_hotkey_on_subnet(hotkey_ss58, netuid, block) is not None

    def last_drand_round(self) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.last_drand_round(),
        )

    def max_weight_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        return self.execute_coroutine(
            self.async_subtensor.max_weight_limit(netuid=netuid, block=block),
        )

    def metagraph(
        self, netuid: int, lite: bool = True, block: Optional[int] = None
    ) -> "Metagraph":
        metagraph = Metagraph(
            network=self.chain_endpoint,
            netuid=netuid,
            lite=lite,
            sync=False,
            subtensor=self,
        )
        metagraph.sync(block=block, lite=lite, subtensor=self)

        return metagraph

    def min_allowed_weights(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.min_allowed_weights(netuid=netuid, block=block),
        )

    def neuron_for_uid(
        self, uid: int, netuid: int, block: Optional[int] = None
    ) -> "NeuronInfo":
        return self.execute_coroutine(
            self.async_subtensor.neuron_for_uid(uid=uid, netuid=netuid, block=block),
        )

    def neurons(self, netuid: int, block: Optional[int] = None) -> list["NeuronInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.neurons(netuid=netuid, block=block),
        )

    def neurons_lite(
        self, netuid: int, block: Optional[int] = None
    ) -> list["NeuronInfoLite"]:
        return self.execute_coroutine(
            self.async_subtensor.neurons_lite(netuid=netuid, block=block),
        )

    def query_identity(self, key: str, block: Optional[int] = None) -> Optional[str]:
        return self.execute_coroutine(
            self.async_subtensor.query_identity(key=key, block=block),
        )

    def recycle(self, netuid: int, block: Optional[int] = None) -> Optional["Balance"]:
        return self.execute_coroutine(
            self.async_subtensor.recycle(netuid=netuid, block=block),
        )

    def subnet_exists(self, netuid: int, block: Optional[int] = None) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.subnet_exists(netuid=netuid, block=block),
        )

    def subnetwork_n(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.subnetwork_n(netuid=netuid, block=block),
        )

    def tempo(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.tempo(netuid=netuid, block=block),
        )

    def tx_rate_limit(self, block: Optional[int] = None) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.tx_rate_limit(block=block),
        )

    def weights(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        return self.execute_coroutine(
            self.async_subtensor.weights(netuid=netuid, block=block),
        )

    def weights_rate_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.weights_rate_limit(netuid=netuid, block=block),
        )

    # Extrinsics =======================================================================================================

    def add_stake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Optional[Union["Balance", float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.add_stake(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                amount=amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

    def add_stake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        amounts: Optional[list[Union["Balance", float]]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.add_stake_multiple(
                wallet=wallet,
                hotkey_ss58s=hotkey_ss58s,
                amounts=amounts,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

    def burned_register(
        self,
        wallet: "Wallet",
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.burned_register(
                wallet=wallet,
                netuid=netuid,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

    def commit_weights(
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
    ) -> tuple[bool, str]:
        return self.execute_coroutine(
            self.async_subtensor.commit_weights(
                wallet=wallet,
                netuid=netuid,
                salt=salt,
                uids=uids,
                weights=weights,
                version_key=version_key,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                max_retries=max_retries,
            ),
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
        return self.execute_coroutine(
            self.async_subtensor.register(
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
            ),
        )

    def reveal_weights(
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
    ) -> tuple[bool, str]:
        return self.execute_coroutine(
            self.async_subtensor.reveal_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                salt=salt,
                version_key=version_key,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                max_retries=max_retries,
            ),
        )

    def root_register(
        self,
        wallet: "Wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.root_register(
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        )

    def root_set_weights(
        self,
        wallet: "Wallet",
        netuids: list[int],
        weights: list[float],
        version_key: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.root_set_weights(
                wallet=wallet,
                netuids=netuids,
                weights=weights,
                version_key=version_key,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

    def set_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: Union[NDArray[np.int64], "torch.LongTensor", list],
        weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
        version_key: int = version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
    ) -> tuple[bool, str]:
        return self.execute_coroutine(
            self.async_subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                version_key=version_key,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                max_retries=max_retries,
            )
        )

    def serve_axon(
        self,
        netuid: int,
        axon: "Axon",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        certificate: Optional["Certificate"] = None,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.serve_axon(
                netuid=netuid,
                axon=axon,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                certificate=certificate,
            ),
        )

    def transfer(
        self,
        wallet: "Wallet",
        dest: str,
        amount: Union["Balance", float],
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        transfer_all: bool = False,
        keep_alive: bool = True,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.transfer(
                wallet=wallet,
                destination=dest,
                amount=amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                transfer_all=transfer_all,
                keep_alive=keep_alive,
            ),
        )

    def unstake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Optional[Union["Balance", float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.unstake(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                amount=amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

    def unstake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        amounts: Optional[list[Union["Balance", float]]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.unstake_multiple(
                wallet=wallet,
                hotkey_ss58s=hotkey_ss58s,
                amounts=amounts,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )
