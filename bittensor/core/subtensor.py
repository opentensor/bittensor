import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union, cast

import numpy as np
import requests
import scalecodec
from async_substrate_interface.errors import SubstrateRequestException
from async_substrate_interface.types import ScaleObj
from async_substrate_interface.sync_substrate import SubstrateInterface
from async_substrate_interface.utils import json
from numpy.typing import NDArray

from bittensor.core.async_subtensor import ProposalVoteData
from bittensor.core.axon import Axon
from bittensor.core.chain_data import (
    DelegateInfo,
    DynamicInfo,
    MetagraphInfo,
    NeuronInfo,
    NeuronInfoLite,
    StakeInfo,
    SubnetHyperparameters,
    WeightCommitInfo,
    SubnetIdentity,
    SubnetInfo,
    DelegatedInfo,
    decode_account_id,
)
from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.config import Config
from bittensor.core.extrinsics.commit_reveal import commit_reveal_v3_extrinsic
from bittensor.core.extrinsics.commit_weights import (
    commit_weights_extrinsic,
    reveal_weights_extrinsic,
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
from bittensor.core.extrinsics.root import (
    root_register_extrinsic,
    set_root_weights_extrinsic,
)
from bittensor.core.extrinsics.serving import (
    publish_metadata,
    get_metadata,
    serve_axon_extrinsic,
)
from bittensor.core.extrinsics.set_weights import set_weights_extrinsic
from bittensor.core.extrinsics.staking import (
    add_stake_extrinsic,
    add_stake_multiple_extrinsic,
)
from bittensor.core.extrinsics.transfer import transfer_extrinsic
from bittensor.core.extrinsics.unstaking import (
    unstake_extrinsic,
    unstake_multiple_extrinsic,
)
from bittensor.core.metagraph import Metagraph
from bittensor.core.settings import (
    version_as_int,
    SS58_FORMAT,
    TYPE_REGISTRY,
    DELEGATES_DETAILS_URL,
)
from bittensor.core.types import ParamWithTypes, SubtensorMixin
from bittensor.utils import (
    torch,
    format_error_message,
    decode_hex_identity_dict,
    u16_normalized_float,
    _decode_hex_identity_dict,
    Certificate,
    u64_normalized_float,
)
from bittensor.utils.balance import (
    Balance,
    fixed_to_float,
    FixedPoint,
    check_and_convert_to_balance,
)
from bittensor.utils.btlogging import logging
from bittensor.utils.delegates_details import DelegatesDetails
from bittensor.utils.weight_utils import generate_weight_hash

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from async_substrate_interface.sync_substrate import QueryMapResult
    from scalecodec.types import GenericCall


class Subtensor(SubtensorMixin):
    """Thin layer for interacting with Substrate Interface. Mostly a collection of frequently-used calls."""

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional["Config"] = None,
        _mock: bool = False,
        log_verbose: bool = False,
    ):
        """
        Initializes an instance of the Subtensor class.

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
            f"Connecting to network: [blue]{self.network}[/blue], "
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Closes the websocket connection
        """
        self.substrate.close()

    # Subtensor queries ===========================================================================================

    def query_constant(
        self, module_name: str, constant_name: str, block: Optional[int] = None
    ) -> Optional["ScaleObj"]:
        """
        Retrieves a constant from the specified module on the Bittensor blockchain. This function is used to access
            fixed parameters or values defined within the blockchain's modules, which are essential for understanding
            the network's configuration and rules.

        Args:
            module_name: The name of the module containing the constant.
            constant_name: The name of the constant to retrieve.
            block: The blockchain block number at which to query the constant.

        Returns:
            Optional[async_substrate_interface.types.ScaleObj]: The value of the constant if found, `None` otherwise.

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
        return result

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
    ) -> Optional[Union["ScaleObj", Any, FixedPoint]]:
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
        params: Optional[Union[list[Any], dict[str, Any]]] = None,
        block: Optional[int] = None,
    ) -> Any:
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
        result = self.substrate.runtime_call(runtime_api, method, params, block_hash)

        return result.value

    def query_subtensor(
        self, name: str, block: Optional[int] = None, params: Optional[list] = None
    ) -> Optional[Union["ScaleObj", Any]]:
        """
        Queries named storage from the Subtensor module on the Bittensor blockchain. This function is used to retrieve
            specific data or parameters from the blockchain, such as stake, rank, or other neuron-specific attributes.

        Args:
            name: The name of the storage function to query.
            block: The blockchain block number at which to perform the query.
            params: A list of parameters to pass to the query function.

        Returns:
            query_response: An object containing the requested data.

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

    def all_subnets(self, block: Optional[int] = None) -> Optional[list["DynamicInfo"]]:
        """
        Retrieves the subnet information for all subnets in the network.

        Args:
            block (Optional[int]): The block number to query the subnet information from.

        Returns:
            Optional[DynamicInfo]: A list of DynamicInfo objects, each containing detailed information about a subnet.

        """
        block_hash = self.determine_block_hash(block)
        query = self.substrate.runtime_call(
            "SubnetInfoRuntimeApi",
            "get_all_dynamic_info",
            block_hash=block_hash,
        )
        return DynamicInfo.list_from_dicts(query.decode())

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
            if b.value is not None:
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
        return publish_metadata(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            data_type=f"Raw{len(data)}",
            data=data.encode(),
        )

    # add explicit alias
    set_commitment = commit

    def commit_reveal_enabled(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[bool]:
        """
        Check if commit-reveal mechanism is enabled for a given network at a specific block.

        Arguments:
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

        Arguments:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.

        Returns:
            Optional[int]: The value of the 'Difficulty' hyperparameter if the subnet exists, ``None`` otherwise.

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

        Args:
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
            else result != "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        )
        return return_val

    def get_all_subnets_info(self, block: Optional[int] = None) -> list["SubnetInfo"]:
        """
        Retrieves detailed information about all subnets within the Bittensor network. This function provides
            comprehensive data on each subnet, including its characteristics and operational parameters.

        Arguments:
            block: The blockchain block number for the query.

        Returns:
            list[SubnetInfo]: A list of SubnetInfo objects, each containing detailed information about a subnet.

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
        else:
            return SubnetInfo.list_from_dicts(result)

    def get_balance(self, address: str, block: Optional[int] = None) -> Balance:
        """
        Retrieves the balance for given coldkey.

        Arguments:
            address (str): coldkey address.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Balance object.
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

        Arguments:
            addresses (str): coldkey addresses(s).
            block (Optional[int]): The blockchain block number for the query.

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

        Arguments:
            block (int): The block number for which the hash is to be retrieved.

        Returns:
            str: The cryptographic hash of the specified block.

        The block hash is a fundamental aspect of blockchain technology, providing a secure reference to each block's
            data. It is crucial for verifying transactions, ensuring data consistency, and maintaining the
            trustworthiness of the blockchain.
        """
        if block:
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
    ) -> tuple[bool, list[tuple[float, str]], str]:
        """
        This method retrieves the children of a given hotkey and netuid. It queries the SubtensorModule's ChildKeys
            storage function to get the children and formats them before returning as a tuple.

        Arguments:
            hotkey (str): The hotkey value.
            netuid (int): The netuid value.
            block (Optional[int]): The block number for which the children are to be retrieved.

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

    def get_commitment(self, netuid: int, uid: int, block: Optional[int] = None) -> str:
        """
        Retrieves the on-chain commitment for a specific neuron in the Bittensor network.

        Arguments:
            netuid (int): The unique identifier of the subnetwork.
            uid (int): The unique identifier of the neuron.
            block (Optional[int]): The block number to retrieve the commitment from. If None, the latest block is used.
                Default is ``None``.

        Returns:
            str: The commitment data as a string.
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

        except TypeError:
            return ""

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
            result[decode_account_id(id_[0])] = decode_account_id(value)
        return result

    def get_current_weight_commit_info(
        self, netuid: int, block: Optional[int] = None
    ) -> list:
        """
        Retrieves CRV3 weight commit information for a specific subnet.

        Arguments:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query. Default is ``None``.

        Returns:
            list: A list of commit details, where each entry is a dictionary with keys 'who', 'serialized_commit', and
            'reveal_round', or an empty list if no data is found.
        """
        result = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="CRV3WeightCommits",
            params=[netuid],
            block_hash=self.determine_block_hash(block),
        )

        commits = result.records[0][1] if result.records else []
        return [WeightCommitInfo.from_vec_u8(commit) for commit in commits]

    def get_delegate_by_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional["DelegateInfo"]:
        """
        Retrieves detailed information about a delegate neuron based on its hotkey. This function provides a
            comprehensive view of the delegate's status, including its stakes, nominators, and reward distribution.

        Arguments:
            hotkey_ss58 (str): The ``SS58`` address of the delegate's hotkey.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[DelegateInfo]: Detailed information about the delegate neuron, ``None`` if not found.

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
    ) -> dict[str, "DelegatesDetails"]:
        """
        Fetches delegates identities from the chain and GitHub. Preference is given to chain data, and missing info is
            filled-in by the info from GitHub. At some point, we want to totally move away from fetching this info from
            GitHub, but chain data is still limited in that regard.

        Arguments:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Dict {ss58: DelegatesDetails, ...}

        """
        block_hash = self.determine_block_hash(block)
        response = requests.get(DELEGATES_DETAILS_URL)
        identities_info = self.substrate.query_map(
            module="Registry", storage_function="IdentityOf", block_hash=block_hash
        )

        all_delegates_details = {}
        for ss58_address, identity in identities_info:
            all_delegates_details.update(
                {
                    decode_account_id(
                        ss58_address[0]
                    ): DelegatesDetails.from_chain_data(
                        decode_hex_identity_dict(identity.value["info"])
                    )
                }
            )
        if response.ok:
            all_delegates: dict[str, Any] = json.loads(response.content)

            for delegate_hotkey, delegate_details in all_delegates.items():
                delegate_info = all_delegates_details.setdefault(
                    delegate_hotkey,
                    DelegatesDetails(
                        display=delegate_details.get("name", ""),
                        web=delegate_details.get("url", ""),
                        additional=delegate_details.get("description", ""),
                        pgp_fingerprint=delegate_details.get("fingerprint", ""),
                    ),
                )
                delegate_info.display = delegate_info.display or delegate_details.get(
                    "name", ""
                )
                delegate_info.web = delegate_info.web or delegate_details.get("url", "")
                delegate_info.additional = (
                    delegate_info.additional or delegate_details.get("description", "")
                )
                delegate_info.pgp_fingerprint = (
                    delegate_info.pgp_fingerprint
                    or delegate_details.get("fingerprint", "")
                )

        return all_delegates_details

    def get_delegate_take(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[float]:
        """
        Retrieves the delegate 'take' percentage for a neuron identified by its hotkey. The 'take' represents the
            percentage of rewards that the delegate claims from its nominators' stakes.

        Arguments:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[float]: The delegate take percentage, None if not available.

        The delegate take is a critical parameter in the network's incentive structure, influencing the distribution of
            rewards among neurons and their nominators.
        """
        result = self.query_subtensor(
            name="Delegates",
            block=block,
            params=[hotkey_ss58],
        )
        return (
            None
            if result is None
            else u16_normalized_float(getattr(result, "value", 0))
        )

    def get_delegated(
        self, coldkey_ss58: str, block: Optional[int] = None
    ) -> list[tuple["DelegateInfo", Balance]]:
        """
        Retrieves a list of delegates and their associated stakes for a given coldkey. This function identifies the
        delegates that a specific account has staked tokens on.

        Arguments:
            coldkey_ss58 (str): The `SS58` address of the account's coldkey.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            A list of tuples, each containing a delegate's information and staked amount.

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

        Arguments:
            block (Optional[int]): The blockchain block number for the query.

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
        Retrieves the existential deposit amount for the Bittensor blockchain.
        The existential deposit is the minimum amount of TAO required for an account to exist on the blockchain.
        Accounts with balances below this threshold can be reaped to conserve network resources.

        Arguments:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            The existential deposit amount.

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

        Arguments:
            hotkey_ss58 (str): The SS58 address of the hotkey.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[str]: The SS58 address of the owner if the hotkey exists, or None if it doesn't.
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
        This method retries the substrate call up to three times with exponential backoff in case of failures.

        Returns:
            Balance: The minimum required stake as a Balance object.

        Raises:
            Exception: If the substrate call fails after the maximum number of retries.
        """
        result = self.substrate.query(
            module="SubtensorModule", storage_function="NominatorMinRequiredStake"
        )

        return Balance.from_rao(getattr(result, "value", 0))

    def get_metagraph_info(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[MetagraphInfo]:
        """
        Retrieves the MetagraphInfo dataclass from the node for a single subnet (netuid)

        Arguments:
            netuid: The NetUID of the subnet.
            block: the block number at which to retrieve the hyperparameter. Do not specify if using block_hash or
                reuse_block

        Returns:
            MetagraphInfo dataclass
        """
        block_hash = self.determine_block_hash(block)
        query = self.substrate.runtime_call(
            "SubnetInfoRuntimeApi",
            "get_metagraph",
            params=[netuid],
            block_hash=block_hash,
        )
        if query.value is None:
            logging.error(f"Subnet {netuid} does not exist.")
            return None
        return MetagraphInfo.from_dict(query.value)

    def get_all_metagraphs_info(
        self, block: Optional[int] = None
    ) -> list[MetagraphInfo]:
        """
        Retrieves a list of MetagraphInfo objects for all subnets

        Arguments:
            block: the block number at which to retrieve the hyperparameter. Do not specify if using block_hash or
                reuse_block

        Returns:
            MetagraphInfo dataclass
        """
        block_hash = self.determine_block_hash(block)
        query = self.substrate.runtime_call(
            "SubnetInfoRuntimeApi",
            "get_all_metagraphs",
            block_hash=block_hash,
        )
        return MetagraphInfo.list_from_dicts(query.value)

    def get_netuids_for_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> list[int]:
        """
        Retrieves a list of subnet UIDs (netuids) for which a given hotkey is a member. This function identifies the
            specific subnets within the Bittensor network where the neuron associated with the hotkey is active.

        Arguments:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            block (Optional[int]): The blockchain block number for the query.

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
        Retrieves the TLS certificate for a specific neuron identified by its unique identifier (UID) within a
            specified subnet (netuid) of the Bittensor network.

        Arguments:
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

        Arguments:
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

    def get_neuron_for_pubkey_and_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional["NeuronInfo"]:
        """
        Retrieves information about a neuron based on its public key (hotkey SS58 address) and the specific subnet UID
            (netuid). This function provides detailed neuron information for a particular subnet within the Bittensor
            network.

        Arguments:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[bittensor.core.chain_data.neuron_info.NeuronInfo]: Detailed information about the neuron if found,
                ``None`` otherwise.

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

    def get_stake(
        self,
        coldkey_ss58: str,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
    ) -> Balance:
        """
        Returns the stake under a coldkey - hotkey pairing.

        Args:
            hotkey_ss58 (str): The SS58 address of the hotkey.
            coldkey_ss58 (str): The SS58 address of the coldkey.
            netuid (int): The subnet ID
            block (Optional[int]): The block number at which to query the stake information.

        Returns:
            Balance: The stake under the coldkey - hotkey pairing.
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

    def get_stake_for_coldkey_and_hotkey(
        self,
        coldkey_ss58: str,
        hotkey_ss58: str,
        netuids: Optional[list[int]] = None,
        block: Optional[int] = None,
    ) -> dict[int, StakeInfo]:
        """
        Retrieves all coldkey-hotkey pairing stake across specified (or all) subnets

        Arguments:
            coldkey_ss58 (str): The SS58 address of the coldkey.
            hotkey_ss58 (str): The SS58 address of the hotkey.
            netuids (Optional[list[int]]): The subnet IDs to query for. Set to `None` for all subnets.
            block (Optional[int]): The block number at which to query the stake information.

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

        Args:
            coldkey_ss58 (str): The SS58 address of the coldkey.
            block (Optional[int]): The block number at which to query the stake information.

        Returns:
            Optional[list[StakeInfo]]: A list of StakeInfo objects, or ``None`` if no stake information is found.
        """
        result = self.query_runtime_api(
            runtime_api="StakeInfoRuntimeApi",
            method="get_stake_info_for_coldkey",
            params=[coldkey_ss58],  # type: ignore
            block=block,
        )

        if result is None:
            return []
        stakes = StakeInfo.list_from_dicts(result)  # type: ignore
        return [stake for stake in stakes if stake.stake > 0]

    get_stake_info_for_coldkey = get_stake_for_coldkey

    def get_stake_for_hotkey(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Balance:
        """
        Retrieves the stake information for a given hotkey.

        Args:
            hotkey_ss58: The SS58 address of the hotkey.
            netuid: The subnet ID to query for.
            block: The block number at which to query the stake information. Do not specify if also specifying
                block_hash or reuse_block
        """
        hotkey_alpha_query = self.query_subtensor(
            name="TotalHotkeyAlpha", params=[hotkey_ss58, netuid], block=block
        )
        hotkey_alpha = cast(ScaleObj, hotkey_alpha_query)
        balance = Balance.from_rao(hotkey_alpha.value)
        balance.set_unit(netuid=netuid)
        return balance

    get_hotkey_stake = get_stake_for_hotkey

    def get_subnet_burn_cost(self, block: Optional[int] = None) -> Optional[Balance]:
        """
        Retrieves the burn cost for registering a new subnet within the Bittensor network. This cost represents the
            amount of Tao that needs to be locked or burned to establish a new subnet.

        Arguments:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            int: The burn cost for subnet registration.

        The subnet burn cost is an important economic parameter, reflecting the network's mechanisms for controlling
            the proliferation of subnets and ensuring their commitment to the network's long-term viability.
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

        Arguments:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            The subnet's hyperparameters, or `None` if not available.

        Understanding the hyperparameters is crucial for comprehending how subnets are configured and managed, and how
            they interact with the network's consensus and incentive mechanisms.
        """
        result = self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_hyperparams",
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

    def get_subnets(self, block: Optional[int] = None) -> list[int]:
        """
        Retrieves the list of all subnet unique identifiers (netuids) currently present in the Bittensor network.

        Arguments:
            block (Optional[int]): The blockchain block number for the query.

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

        Arguments:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[str]: The total number of subnets in the network.

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

    def get_transfer_fee(self, wallet: "Wallet", dest: str, value: Balance) -> Balance:
        """
        Calculates the transaction fee for transferring tokens from a wallet to a specified destination address. This
            function simulates the transfer to estimate the associated cost, taking into account the current network
            conditions and transaction complexity.

        Arguments:
            wallet (bittensor_wallet.Wallet): The wallet from which the transfer is initiated.
            dest (str): The ``SS58`` address of the destination account.
            value (Union[bittensor.utils.balance.Balance, float, int]): The amount of tokens to be transferred,
                specified as a Balance object, or in Tao (float) or Rao (int) units.

        Returns:
            bittensor.utils.balance.Balance: The estimated transaction fee for the transfer, represented as a Balance
                object.

        Estimating the transfer fee is essential for planning and executing token transactions, ensuring that the wallet
            has sufficient funds to cover both the transfer amount and the associated costs. This function provides a
            crucial tool for managing financial operations within the Bittensor network.
        """
        value = check_and_convert_to_balance(value)
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
            logging.error(f":cross_mark: [red]Failed to get payment info: [/red]{e}")
            payment_info = {"partial_fee": int(2e7)}  # assume  0.02 Tao

        return Balance.from_rao(payment_info["partial_fee"])

    def get_vote_data(
        self, proposal_hash: str, block: Optional[int] = None
    ) -> Optional["ProposalVoteData"]:
        """
        Retrieves the voting data for a specific proposal on the Bittensor blockchain. This data includes information
            about how senate members have voted on the proposal.

        Arguments:
            proposal_hash (str): The hash of the proposal for which voting data is requested.
            block (Optional[int]): The blockchain block number for the query.

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
        else:
            return ProposalVoteData(vote_data)

    def get_uid_for_hotkey_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Retrieves the unique identifier (UID) for a neuron's hotkey on a specific subnet.

        Arguments:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The UID of the neuron if it is registered on the subnet, ``None`` otherwise.

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

        Arguments:
            all_netuids (Iterable[int]): A list of netuids to filter.
            filter_for_netuids (Iterable[int]): A subset of all_netuids to filter from the main list.
            all_hotkeys (Iterable[Wallet]): Hotkeys to filter from the main list.
            block (Optional[int]): The blockchain block number for the query.

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

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The value of the 'ImmunityPeriod' hyperparameter if the subnet exists, ``None`` otherwise.

        The 'ImmunityPeriod' is a critical aspect of the network's governance system, ensuring that new participants
            have a grace period to establish themselves and contribute to the network without facing immediate
            punitive actions.
        """
        call = self.get_hyperparameter(
            param_name="ImmunityPeriod", netuid=netuid, block=block
        )
        return None if call is None else int(call)

    def is_hotkey_delegate(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        """
        Determines whether a given hotkey (public key) is a delegate on the Bittensor network. This function checks if
            the neuron associated with the hotkey is part of the network's delegation system.

        Arguments:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block (Optional[int]): The blockchain block number for the query.

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
            identified by its hotkey, which is crucial for validating its participation and activities within the
            network.

        Args:
            hotkey_ss58: The SS58 address of the neuron's hotkey.
            netuid: The unique identifier of the subnet to check the registration. If `None`, the
                registration is checked across all subnets.
            block: The blockchain block number at which to perform the query.

        Returns:
            bool: `True` if the hotkey is registered in the specified context (either any subnet or a specific subnet),
                `False` otherwise.

        This function is important for verifying the active status of neurons in the Bittensor network. It aids in
            understanding whether a neuron is eligible to participate in network processes such as consensus,
            validation, and incentive distribution based on its registration status.
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

        Arguments:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            bool: ``True`` if the hotkey is registered on any subnet, False otherwise.

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

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[float]: The value of the MaxWeightsLimit hyperparameter, or ``None`` if the subnetwork does not
                exist or the parameter is not found.
        """
        call = self.get_hyperparameter(
            param_name="MaxWeightsLimit", netuid=netuid, block=block
        )
        return None if call is None else u16_normalized_float(int(call))

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
        """
        Returns network MinAllowedWeights hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The value of the MinAllowedWeights hyperparameter, or ``None`` if the subnetwork does not
                exist or the parameter is not found.
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
            specified subnet (netuid) of the Bittensor network. This function provides a comprehensive view of a
            neuron's attributes, including its stake, rank, and operational status.

        Arguments:
            uid (int): The unique identifier of the neuron.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

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

        Arguments:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

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

        Arguments:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

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

    def query_identity(self, coldkey_ss58: str, block: Optional[int] = None) -> dict:
        """
        Queries the identity of a neuron on the Bittensor blockchain using the given key. This function retrieves
            detailed identity information about a specific neuron, which is a crucial aspect of the network's
            decentralized identity and governance system.

        Arguments:
            coldkey_ss58 (str): The coldkey used to query the neuron's identity (technically the neuron's coldkey SS58
                address).
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            An object containing the identity information of the neuron if found, ``None`` otherwise.

        The identity information can include various attributes such as the neuron's stake, rank, and other
            network-specific details, providing insights into the neuron's role and status within the Bittensor network.

        Note:
            See the `Bittensor CLI documentation <https://docs.bittensor.com/reference/btcli>`_ for supported identity
                parameters.
        """
        identity_info = self.substrate.query(
            module="SubtensorModule",
            storage_function="IdentitiesV2",
            params=[coldkey_ss58],
            block_hash=self.determine_block_hash(block),
        )
        if not identity_info:
            return {}
        try:
            return _decode_hex_identity_dict(identity_info)
        except TypeError:
            return {}

    def recycle(self, netuid: int, block: Optional[int] = None) -> Optional[Balance]:
        """
        Retrieves the 'Burn' hyperparameter for a specified subnet. The 'Burn' parameter represents the amount of Tao
            that is effectively recycled within the Bittensor network.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[Balance]: The value of the 'Burn' hyperparameter if the subnet exists, None otherwise.

        Understanding the 'Burn' rate is essential for analyzing the network registration usage, particularly how it is
            correlated with user activity and the overall cost of participation in a given subnet.
        """
        call = self.get_hyperparameter(param_name="Burn", netuid=netuid, block=block)
        return None if call is None else Balance.from_rao(int(call))

    def subnet(self, netuid: int, block: Optional[int] = None) -> Optional[DynamicInfo]:
        """
        Retrieves the subnet information for a single subnet in the network.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The block number to query the subnet information from.

        Returns:
            Optional[DynamicInfo]: A DynamicInfo object, containing detailed information about a subnet.

        """
        block_hash = self.determine_block_hash(block)

        query = self.substrate.runtime_call(
            "SubnetInfoRuntimeApi",
            "get_dynamic_info",
            params=[netuid],
            block_hash=block_hash,
        )
        subnet = DynamicInfo.from_dict(query.decode())  # type: ignore
        return subnet

    def subnet_exists(self, netuid: int, block: Optional[int] = None) -> bool:
        """
        Checks if a subnet with the specified unique identifier (netuid) exists within the Bittensor network.

        Arguments:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            `True` if the subnet exists, `False` otherwise.

        This function is critical for verifying the presence of specific subnets in the network,
        enabling a deeper understanding of the network's structure and composition.
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

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The value of the SubnetworkN hyperparameter, or ``None`` if the subnetwork does not exist or
                the parameter is not found.
        """
        call = self.get_hyperparameter(
            param_name="SubnetworkN", netuid=netuid, block=block
        )
        return None if call is None else int(call)

    def tempo(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        """
        Returns network Tempo hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The value of the Tempo hyperparameter, or ``None`` if the subnetwork does not exist or the
                parameter is not found.
        """
        call = self.get_hyperparameter(param_name="Tempo", netuid=netuid, block=block)
        return None if call is None else int(call)

    def tx_rate_limit(self, block: Optional[int] = None) -> Optional[int]:
        """
        Retrieves the transaction rate limit for the Bittensor network as of a specific blockchain block.
        This rate limit sets the maximum number of transactions that can be processed within a given time frame.

        Args:
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The transaction rate limit of the network, None if not available.

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

        Args:
            block (Optional[int]): The block number to wait for. If None, waits for next block.

        Returns:
            bool: True if the target block was reached, False if timeout occurred.

        Example:
            >>> subtensor.wait_for_block() # Waits for next block
            >>> subtensor.wait_for_block(block=1234) # Waits for specific block
        """

        def handler(block_data: dict):
            logging.debug(
                f'reached block {block_data["header"]["number"]}. Waiting for block {target_block}'
            )
            if block_data["header"]["number"] >= target_block:
                return True

        current_block = self.substrate.get_block()
        current_block_hash = current_block.get("header", {}).get("hash")
        if block is not None:
            target_block = block
        else:
            target_block = current_block["header"]["number"] + 1

        self.substrate._get_block_handler(
            current_block_hash, header_only=True, subscription_handler=handler
        )
        return True

    def weights(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the weight distribution set by neurons within a specific subnet of the Bittensor network.
        This function maps each neuron's UID to the weights it assigns to other neurons, reflecting the network's trust
            and value assignment mechanisms.

        Arguments:
            netuid (int): The network UID of the subnet to query.
            block (Optional[int]): Block number for synchronization, or ``None`` for the latest block.

        Returns:
            A list of tuples mapping each neuron's UID to its assigned weights.

        The weight distribution is a key factor in the network's consensus algorithm and the ranking of neurons,
            influencing their influence and reward allocation within the subnet.
        """
        w_map_encoded = self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Weights",
            params=[netuid],
            block_hash=self.determine_block_hash(block),
        )
        w_map = [(uid, w.value or []) for uid, w in w_map_encoded]

        return w_map

    def weights_rate_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        """
        Returns network WeightsSetRateLimit hyperparameter.

        Arguments:
            netuid (int): The unique identifier of the subnetwork.
            block (Optional[int]): The blockchain block number for the query.

        Returns:
            Optional[int]: The value of the WeightsSetRateLimit hyperparameter, or ``None`` if the subnetwork does not
                exist or the parameter is not found.
        """
        call = self.get_hyperparameter(
            param_name="WeightsSetRateLimit", netuid=netuid, block=block
        )
        return None if call is None else int(call)

    # Extrinsics helper ================================================================================================

    def sign_and_send_extrinsic(
        self,
        call: "GenericCall",
        wallet: "Wallet",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        sign_with: str = "coldkey",
        use_nonce: bool = False,
        period: Optional[int] = None,
        nonce_key: str = "hotkey",
    ) -> tuple[bool, str]:
        """
        Helper method to sign and submit an extrinsic call to chain.

        Arguments:
            call (scalecodec.types.GenericCall): a prepared Call object
            wallet (bittensor_wallet.Wallet): the wallet whose coldkey will be used to sign the extrinsic
            wait_for_inclusion (bool): whether to wait until the extrinsic call is included on the chain
            wait_for_finalization (bool): whether to wait until the extrinsic call is finalized on the chain
            sign_with: the wallet's keypair to use for the signing. Options are "coldkey", "hotkey", "coldkeypub"

        Returns:
            (success, error message)
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
            next_nonce = self.substrate.get_account_next_index(
                getattr(wallet, nonce_key).ss58_address
            )
            extrinsic_data["nonce"] = next_nonce
        if period is not None:
            extrinsic_data["era"] = {"period": period}

        extrinsic = self.substrate.create_signed_extrinsic(**extrinsic_data)
        try:
            response = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True, ""

            if response.is_success:
                return True, ""

            return False, format_error_message(response.error_message)

        except SubstrateRequestException as e:
            return False, format_error_message(e)

    # Extrinsics =======================================================================================================

    def add_stake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        netuid: Optional[int] = None,
        amount: Optional[Balance] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Adds the specified amount of stake to a neuron identified by the hotkey ``SS58`` address.
        Staking is a fundamental process in the Bittensor network that enables neurons to participate actively and earn
            incentives.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet to be used for staking.
            hotkey_ss58 (Optional[str]): The ``SS58`` address of the hotkey associated with the neuron.
            netuid (Optional[int]): The unique identifier of the subnet to which the neuron belongs.
            amount (Balance): The amount of TAO to stake.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            bool: ``True`` if the staking is successful, False otherwise.

        This function enables neurons to increase their stake in the network, enhancing their influence and potential
            rewards in line with Bittensor's consensus and reward mechanisms.
        """
        amount = check_and_convert_to_balance(amount)
        return add_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def add_stake_multiple(
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

        Args:
            wallet (bittensor_wallet.Wallet): The wallet used for staking.
            hotkey_ss58s (list[str]): List of ``SS58`` addresses of hotkeys to stake to.
            netuids (list[int]): List of network UIDs to stake to.
            amounts (list[Balance]): Corresponding amounts of TAO to stake for each hotkey.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            bool: ``True`` if the staking is successful for all specified neurons, False otherwise.

        This function is essential for managing stakes across multiple neurons, reflecting the dynamic and collaborative
            nature of the Bittensor network.
        """
        return add_stake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            netuids=netuids,
            amounts=amounts,
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
        Registers a neuron on the Bittensor network by recycling TAO. This method of registration involves recycling
            TAO tokens, allowing them to be re-mined by performing work on the network.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron to be registered.
            netuid (int): The unique identifier of the subnet.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block. Defaults to
                `False`.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
                Defaults to `True`.

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
        """
        Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
        This action serves as a commitment or snapshot of the neuron's current weight distribution.

        Arguments:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the weights.
            netuid (int): The unique identifier of the subnet.
            salt (list[int]): list of randomly generated integers as salt to generated weighted hash.
            uids (np.ndarray): NumPy array of neuron UIDs for which weights are being committed.
            weights (np.ndarray): NumPy array of weight values corresponding to each UID.
            version_key (int): Version key for compatibility with the network. Default is ``int representation of
                Bittensor version.``.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is
                ``False``.
            max_retries (int): The number of maximum attempts to commit weights. Default is ``5``.

        Returns:
            tuple[bool, str]: ``True`` if the weight commitment is successful, False otherwise. And `msg`, a string
                value describing the success or potential error.

        This function allows neurons to create a tamper-proof record of their weight distribution at a specific point
            in time, enhancing transparency and accountability within the Bittensor network.
        """
        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to commit weights!"

        logging.info(
            f"Committing weights with params: netuid={netuid}, uids={uids}, weights={weights}, "
            f"version_key={version_key}"
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

    def move_stake(
        self,
        wallet: "Wallet",
        origin_hotkey: str,
        origin_netuid: int,
        destination_hotkey: str,
        destination_netuid: int,
        amount: Balance,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Moves stake to a different hotkey and/or subnet.

        Args:
            wallet (bittensor.wallet): The wallet to move stake from.
            origin_hotkey (str): The SS58 address of the source hotkey.
            origin_netuid (int): The netuid of the source subnet.
            destination_hotkey (str): The SS58 address of the destination hotkey.
            destination_netuid (int): The netuid of the destination subnet.
            amount (Balance): Amount of stake to move.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            success (bool): True if the stake movement was successful.
        """
        amount = check_and_convert_to_balance(amount)
        return move_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            origin_hotkey=origin_hotkey,
            origin_netuid=origin_netuid,
            destination_hotkey=destination_hotkey,
            destination_netuid=destination_netuid,
            amount=amount,
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

        Registration is a critical step for a neuron to become an active participant in the network, enabling it to
            stake, set weights, and receive incentives.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron to be registered.
            netuid (int): The unique identifier of the subnet.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Defaults to `False`.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Defaults to
                `True`.
            max_allowed_attempts (int): Maximum number of attempts to register the wallet.
            output_in_place (bool): If true, prints the progress of the proof of work to the console in-place. Meaning
                the progress is printed on the same lines. Defaults to `True`.
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
            tpb=tpb,
            update_interval=update_interval,
            num_processes=num_processes,
            cuda=cuda,
            dev_id=dev_id,
            output_in_place=output_in_place,
            log_verbose=log_verbose,
        )

    def register_subnet(
        self,
        wallet: "Wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        """
        Registers a new subnetwork on the Bittensor network.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet to be used for subnet registration.
            wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning true, or returns
                false if the extrinsic fails to enter the block within the timeout. Default is False.
            wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
                true, or returns false if the extrinsic fails to be finalized within the timeout. Default is True.

        Returns:
            bool: True if the subnet registration was successful, False otherwise.
        """
        return register_subnet_extrinsic(
            subtensor=self,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        """
        Reveals the weights for a specific subnet on the Bittensor blockchain using the provided wallet.
        This action serves as a revelation of the neuron's previously committed weight distribution.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron revealing the weights.
            netuid (int): The unique identifier of the subnet.
            uids (np.ndarray): NumPy array of neuron UIDs for which weights are being revealed.
            weights (np.ndarray): NumPy array of weight values corresponding to each UID.
            salt (np.ndarray): NumPy array of salt values corresponding to the hash function.
            version_key (int): Version key for compatibility with the network. Default is ``int representation of
                Bittensor version``.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is
                ``False``.
            max_retries (int): The number of maximum attempts to reveal weights. Default is ``5``.

        Returns:
            tuple[bool, str]: ``True`` if the weight revelation is successful, False otherwise. And `msg`, a string
                value describing the success or potential error.

        This function allows neurons to reveal their previously committed weight distribution, ensuring transparency
            and accountability within the Bittensor network.
        """
        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to reveal weights!"

        while retries < max_retries and success is False:
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

    def root_register(
        self,
        wallet: "Wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        """
        Register neuron by recycling some TAO.

        Arguments:
            wallet (bittensor_wallet.Wallet): Bittensor wallet instance.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is
                ``False``.

        Returns:
            `True` if registration was successful, otherwise `False`.
        """
        logging.info(
            f"Registering on netuid [blue]0[/blue] on network: [blue]{self.network}[/blue]"
        )

        # Check current recycle amount
        logging.info("Fetching recycle amount & balance.")
        block = self.get_current_block()

        try:
            recycle_call = cast(
                int, self.get_hyperparameter(param_name="Burn", netuid=0, block=block)
            )
            balance = self.get_balance(wallet.coldkeypub.ss58_address, block=block)
        except TypeError as e:
            logging.error(f"Unable to retrieve current recycle. {e}")
            return False

        current_recycle = Balance.from_rao(int(recycle_call))

        # Check balance is sufficient
        if balance < current_recycle:
            logging.error(
                f"[red]Insufficient balance {balance} to register neuron. "
                f"Current recycle is {current_recycle} TAO[/red]."
            )
            return False

        return root_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        """
        Set weights for root network.

        Arguments:
            wallet (bittensor_wallet.Wallet): bittensor wallet instance.
            netuids (list[int]): The list of subnet uids.
            weights (list[float]): The list of weights to be set.
            version_key (int, optional): Version key for compatibility with the network. Default is ``0``.
            wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block. Defaults to
                ``False``.
            wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
                Defaults to ``False``.

        Returns:
            `True` if the setting of weights is successful, `False` otherwise.
        """
        netuids_ = np.array(netuids, dtype=np.int64)
        weights_ = np.array(weights, dtype=np.float32)
        logging.info(f"Setting weights in network: [blue]{self.network}[/blue]")
        return set_root_weights_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids_,
            weights=weights_,
            version_key=version_key,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
        )

    def set_subnet_identity(
        self,
        wallet: "Wallet",
        netuid: int,
        subnet_identity: SubnetIdentity,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> tuple[bool, str]:
        """
        Sets the identity of a subnet for a specific wallet and network.

        Arguments:
            wallet (Wallet): The wallet instance that will authorize the transaction.
            netuid (int): The unique ID of the network on which the operation takes place.
            subnet_identity (SubnetIdentity): The identity data of the subnet including attributes like name, GitHub
                repository, contact, URL, discord, description, and any additional metadata.
            wait_for_inclusion (bool): Indicates if the function should wait for the transaction to be included in the block.
            wait_for_finalization (bool): Indicates if the function should wait for the transaction to reach finalization.

        Returns:
            tuple[bool, str]: A tuple where the first element is a boolean indicating success or failure of the
             operation, and the second element is a message providing additional information.
        """
        return set_subnet_identity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            subnet_name=subnet_identity.subnet_name,
            github_repo=subnet_identity.github_repo,
            subnet_contact=subnet_identity.subnet_contact,
            subnet_url=subnet_identity.subnet_url,
            discord=subnet_identity.discord,
            description=subnet_identity.description,
            additional=subnet_identity.additional,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        """
        Sets the inter-neuronal weights for the specified neuron. This process involves specifying the influence or
            trust a neuron places on other neurons in the network, which is a fundamental aspect of Bittensor's
            decentralized learning architecture.

        Arguments:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron setting the weights.
            netuid (int): The unique identifier of the subnet.
            uids (Union[NDArray[np.int64], torch.LongTensor, list]): The list of neuron UIDs that the weights are being
                set for.
            weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The corresponding weights to be set for each
                UID.
            version_key (int): Version key for compatibility with the network.  Default is int representation of
                Bittensor version.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is
                ``False``.
            max_retries (int): The number of maximum attempts to set weights. Default is ``5``.

        Returns:
            tuple[bool, str]: ``True`` if the setting of weights is successful, False otherwise. And `msg`, a string
                value describing the success or potential error.

        This function is crucial in shaping the network's collective intelligence, where each neuron's learning and
            contribution are influenced by the weights it sets towards others【81†source】.
        """

        def _blocks_weight_limit() -> bool:
            bslu = cast(int, self.blocks_since_last_update(netuid, cast(int, uid)))
            wrl = cast(int, self.weights_rate_limit(netuid))
            return bslu > wrl

        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to commit weights!"
        if (
            uid := self.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, netuid)
        ) is None:
            return (
                False,
                f"Hotkey {wallet.hotkey.ss58_address} not registered in subnet {netuid}",
            )

        if self.commit_reveal_enabled(netuid=netuid) is True:
            # go with `commit reveal v3` extrinsic

            while retries < max_retries and success is False and _blocks_weight_limit():
                logging.info(
                    f"Committing weights for subnet #{netuid}. Attempt {retries + 1} of {max_retries}."
                )
                success, message = commit_reveal_v3_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    uids=uids,
                    weights=weights,
                    version_key=version_key,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                retries += 1
            return success, message
        else:
            # go with classic `set weights extrinsic`

            while retries < max_retries and success is False and _blocks_weight_limit():
                try:
                    logging.info(
                        f"Setting weights for subnet #[blue]{netuid}[/blue]. "
                        f"Attempt [blue]{retries + 1} of {max_retries}[/blue]."
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

    def serve_axon(
        self,
        netuid: int,
        axon: "Axon",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        certificate: Optional[Certificate] = None,
    ) -> bool:
        """
        Registers an ``Axon`` serving endpoint on the Bittensor network for a specific neuron. This function is used to
            set up the Axon, a key component of a neuron that handles incoming queries and data processing tasks.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            axon (bittensor.core.axon.Axon): The Axon instance to be registered for serving.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is
                ``True``.
            certificate (bittensor.utils.Certificate): Certificate to use for TLS. If ``None``, no TLS will be used.
                Defaults to ``None``.

        Returns:
            bool: ``True`` if the Axon serve registration is successful, False otherwise.

        By registering an Axon, the neuron becomes an active part of the network's distributed computing infrastructure,
            contributing to the collective intelligence of Bittensor.
        """
        return serve_axon_extrinsic(
            subtensor=self,
            netuid=netuid,
            axon=axon,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            certificate=certificate,
        )

    def swap_stake(
        self,
        wallet: "Wallet",
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Moves stake between subnets while keeping the same coldkey-hotkey pair ownership.
        Like subnet hopping - same owner, same hotkey, just changing which subnet the stake is in.

        Args:
            wallet (bittensor.wallet): The wallet to swap stake from.
            hotkey_ss58 (str): The SS58 address of the hotkey whose stake is being swapped.
            origin_netuid (int): The netuid from which stake is removed.
            destination_netuid (int): The netuid to which stake is added.
            amount (Union[Balance, float]): The amount to swap.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            success (bool): True if the extrinsic was successful.
        """
        amount = check_and_convert_to_balance(amount)
        return swap_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def transfer(
        self,
        wallet: "Wallet",
        dest: str,
        amount: Balance,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        transfer_all: bool = False,
        keep_alive: bool = True,
    ) -> bool:
        """
        Transfer token of amount to destination.

        Arguments:
            wallet (bittensor_wallet.Wallet): Source wallet for the transfer.
            dest (str): Destination address for the transfer.
            amount (float): Amount of tokens to transfer.
            transfer_all (bool): Flag to transfer all tokens. Default is ``False``.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.  Default is ``True``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.  Default is
                ``False``.
            keep_alive (bool): Flag to keep the connection alive. Default is ``True``.

        Returns:
            `True` if the transferring was successful, otherwise `False`.
        """
        amount = check_and_convert_to_balance(amount)
        return transfer_extrinsic(
            subtensor=self,
            wallet=wallet,
            dest=dest,
            amount=amount,
            transfer_all=transfer_all,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            keep_alive=keep_alive,
        )

    def transfer_stake(
        self,
        wallet: "Wallet",
        destination_coldkey_ss58: str,
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Transfers stake from one subnet to another while changing the coldkey owner.

        Args:
            wallet (bittensor.wallet): The wallet to transfer stake from.
            destination_coldkey_ss58 (str): The destination coldkey SS58 address.
            hotkey_ss58 (str): The hotkey SS58 address associated with the stake.
            origin_netuid (int): The source subnet UID.
            destination_netuid (int): The destination subnet UID.
            amount (Union[Balance, float, int]): Amount to transfer.
            wait_for_inclusion (bool): If true, waits for inclusion before returning.
            wait_for_finalization (bool): If true, waits for finalization before returning.

        Returns:
            success (bool): True if the transfer was successful.
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
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def unstake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        netuid: Optional[int] = None,
        amount: Optional[Balance] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting
            individual neuron stakes within the Bittensor network.

        Args:
            wallet (bittensor_wallet.wallet): The wallet associated with the neuron from which the stake is being
                removed.
            hotkey_ss58 (Optional[str]): The ``SS58`` address of the hotkey account to unstake from.
            netuid (Optional[int]): The unique identifier of the subnet.
            amount (Balance): The amount of TAO to unstake. If not specified, unstakes all.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            bool: ``True`` if the unstaking process is successful, False otherwise.

        This function supports flexible stake management, allowing neurons to adjust their network participation and
            potential reward accruals.
        """
        amount = check_and_convert_to_balance(amount)
        return unstake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    def unstake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        netuids: list[int],
        amounts: Optional[list[Balance]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Performs batch unstaking from multiple hotkey accounts, allowing a neuron to reduce its staked amounts
            efficiently. This function is useful for managing the distribution of stakes across multiple neurons.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet linked to the coldkey from which the stakes are being
                withdrawn.
            hotkey_ss58s (List[str]): A list of hotkey ``SS58`` addresses to unstake from.
            netuids (List[int]): The list of subnet uids.
            amounts (List[Balance]): The amounts of TAO to unstake from each hotkey. If not provided,
                unstakes all available stakes.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

        Returns:
            bool: ``True`` if the batch unstaking is successful, False otherwise.

        This function allows for strategic reallocation or withdrawal of stakes, aligning with the dynamic stake
            management aspect of the Bittensor network.
        """
        return unstake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            netuids=netuids,
            amounts=amounts,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
