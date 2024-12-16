import asyncio
from itertools import chain
import ssl
from typing import Optional, Any, Union, TypedDict, Iterable, TYPE_CHECKING

import aiohttp
from async_lru import alru_cache
import numpy as np
import scalecodec
from bittensor_wallet import Wallet
from bittensor_wallet.utils import SS58_FORMAT
from numpy.typing import NDArray
from scalecodec import GenericCall
from scalecodec.base import RuntimeConfiguration
from scalecodec.type_registry import load_type_registry_preset
from substrateinterface.exceptions import SubstrateRequestException

from bittensor.core.chain_data import (
    DelegateInfo,
    custom_rpc_type_registry,
    StakeInfo,
    NeuronInfoLite,
    NeuronInfo,
    SubnetHyperparameters,
    decode_account_id,
    SubnetInfo,
    PrometheusInfo,
)
from bittensor.core.extrinsics.async_registration import register_extrinsic
from bittensor.core.extrinsics.async_root import (
    set_root_weights_extrinsic,
    root_register_extrinsic,
)
from bittensor.core.extrinsics.async_transfer import transfer_extrinsic
from bittensor.core.extrinsics.async_weights import (
    commit_weights_extrinsic,
    set_weights_extrinsic,
)
from bittensor.core.extrinsics.async_serving import serve_axon_extrinsic
from bittensor.core.extrinsics.async_unstaking import unstake_extrinsic
from bittensor.core.settings import (
    TYPE_REGISTRY,
    DEFAULTS,
    NETWORK_MAP,
    DELEGATES_DETAILS_URL,
    DEFAULT_NETWORK,
)
from bittensor.core.settings import version_as_int
from bittensor.utils import (
    torch,
    ss58_to_vec_u8,
    format_error_message,
    decode_hex_identity_dict,
    validate_chain_endpoint,
    hex_to_bytes,
    Certificate,
    u16_normalized_float,
    networking,
)
from bittensor.utils.substrate_interface import AsyncSubstrateInterface
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.delegates_details import DelegatesDetails
from bittensor.utils.weight_utils import generate_weight_hash
from bittensor.core.metagraph import AsyncMetagraph

if TYPE_CHECKING:
    from scalecodec import ScaleType
    from bittensor.utils.substrate_interface import QueryMapResult
    from bittensor.core.axon import Axon


class ParamWithTypes(TypedDict):
    name: str  # Name of the parameter.
    type: str  # ScaleType string of the parameter.


class ProposalVoteData:
    index: int
    threshold: int
    ayes: list[str]
    nays: list[str]
    end: int

    def __init__(self, proposal_dict: dict) -> None:
        self.index = proposal_dict["index"]
        self.threshold = proposal_dict["threshold"]
        self.ayes = self.decode_ss58_tuples(proposal_dict["ayes"])
        self.nays = self.decode_ss58_tuples(proposal_dict["nays"])
        self.end = proposal_dict["end"]

    @staticmethod
    def decode_ss58_tuples(line: tuple):
        """Decodes a tuple of ss58 addresses formatted as bytes tuples."""
        return [decode_account_id(line[x][0]) for x in range(len(line))]


def _decode_hex_identity_dict(info_dictionary: dict[str, Any]) -> dict[str, Any]:
    """Decodes a dictionary of hexadecimal identities."""
    for k, v in info_dictionary.items():
        if isinstance(v, dict):
            item = next(iter(v.values()))
        else:
            item = v
        if isinstance(item, tuple) and item:
            if len(item) > 1:
                try:
                    info_dictionary[k] = (
                        bytes(item).hex(sep=" ", bytes_per_sep=2).upper()
                    )
                except UnicodeDecodeError:
                    logging.error(f"Could not decode: {k}: {item}.")
            else:
                try:
                    info_dictionary[k] = bytes(item[0]).decode("utf-8")
                except UnicodeDecodeError:
                    logging.error(f"Could not decode: {k}: {item}.")
        else:
            info_dictionary[k] = item

    return info_dictionary


class AsyncSubtensor:
    """Thin layer for interacting with Substrate Interface. Mostly a collection of frequently-used calls."""

    def __init__(self, network: str = DEFAULT_NETWORK) -> None:
        if network in NETWORK_MAP:
            self.chain_endpoint = NETWORK_MAP[network]
            self.network = network
            if network == "local":
                logging.warning(
                    "Warning: Verify your local subtensor is running on port 9944."
                )
        else:
            is_valid, _ = validate_chain_endpoint(network)
            if is_valid:
                self.chain_endpoint = network
                if network in NETWORK_MAP.values():
                    self.network = next(
                        key for key, value in NETWORK_MAP.items() if value == network
                    )
                else:
                    self.network = "custom"
            else:
                logging.info(
                    f"Network not specified or not valid. Using default chain endpoint: "
                    f"[blue]{NETWORK_MAP[DEFAULTS.subtensor.network]}[/blue]."
                )
                logging.info(
                    "You can set this for commands with the [blue]--network[/blue] flag, or by setting this in the "
                    "config."
                )
                self.chain_endpoint = NETWORK_MAP[DEFAULTS.subtensor.network]
                self.network = DEFAULTS.subtensor.network

        self.substrate = AsyncSubstrateInterface(
            chain_endpoint=self.chain_endpoint,
            ss58_format=SS58_FORMAT,
            type_registry=TYPE_REGISTRY,
            use_remote_preset=True,
            chain_name="Bittensor",
        )

    def __str__(self):
        return f"Network: {self.network}, Chain: {self.chain_endpoint}"

    async def __aenter__(self):
        logging.info(
            f"[magenta]Connecting to Substrate:[/magenta] [blue]{self}[/blue][magenta]...[/magenta]"
        )
        try:
            async with self.substrate:
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

    async def close(self):
        await self.substrate.close()

    async def encode_params(
        self,
        call_definition: dict[str, list["ParamWithTypes"]],
        params: Union[list[Any], dict[str, Any]],
    ) -> str:
        """Returns a hex encoded string of the params using their types."""
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

    async def query_constant(
        self,
        module_name: str,
        constant_name: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["ScaleType"]:
        """
        Retrieves a constant from the specified module on the Bittensor blockchain. This function is used to access
            fixed parameters or values defined within the blockchain's modules, which are essential for understanding
            the network's configuration and rules.

        Args:
            module_name: The name of the module containing the constant.
            constant_name: The name of the constant to retrieve.
            block: The blockchain block number at which to query the constant. Do not specify if using block_hash or
                reuse_block
            block_hash: the hash of th blockchain block at which to query the constant. Do not specify if using block
                or reuse_block
            reuse_block: Whether to reuse the blockchain block at which to query the constant.

        Returns:
            Optional[scalecodec.ScaleType]: The value of the constant if found, `None` otherwise.

        Constants queried through this function can include critical network parameters such as inflation rates,
            consensus rules, or validation thresholds, providing a deeper understanding of the Bittensor network's
            operational parameters.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.get_constant(
            module_name=module_name,
            constant_name=constant_name,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def get_hyperparameter(
        self,
        param_name: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Any]:
        """
        Retrieves a specified hyperparameter for a specific subnet.

        Args:
            param_name (str): The name of the hyperparameter to retrieve.
            netuid (int): The unique identifier of the subnet.
            block: the block number at which to retrieve the hyperparameter. Do not specify if using block_hash or
                reuse_block
            block_hash (Optional[str]): The hash of blockchain block number for the query. Do not specify if using
                block or reuse_block
            reuse_block (bool): Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The value of the specified hyperparameter if the subnet exists, or None
        """

        if not await self.subnet_exists(netuid, block_hash):
            logging.error(f"subnet {netuid} does not exist")
            return None

        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function=param_name,
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        return result

    # Common subtensor methods =========================================================================================

    async def get_current_block(self) -> int:
        """
        Returns the current block number on the Bittensor blockchain. This function provides the latest block number,
            indicating the most recent state of the blockchain.

        Returns:
            int: The current chain block number.

        Knowing the current block number is essential for querying real-time data and performing time-sensitive
            operations on the blockchain. It serves as a reference point for network activities and data
            synchronization.
        """
        return await self.substrate.get_block_number(None)

    @alru_cache(maxsize=128)
    async def _get_block_hash(self, block_id: int):
        return await self.substrate.get_block_hash(block_id)

    async def get_block_hash(self, block_id: Optional[int] = None):
        """
        Retrieves the hash of a specific block on the Bittensor blockchain. The block hash is a unique identifier
            representing the cryptographic hash of the block's content, ensuring its integrity and immutability.

        Args:
            block_id (int): The block number for which the hash is to be retrieved.

        Returns:
            str: The cryptographic hash of the specified block.

        The block hash is a fundamental aspect of blockchain technology, providing a secure reference to each block's
            data. It is crucial for verifying transactions, ensuring data consistency, and maintaining the
            trustworthiness of the blockchain.
        """
        if block_id:
            return await self._get_block_hash(block_id)
        else:
            return await self.substrate.get_chain_head()

    async def _determine_block_hash(
        self, block: Optional[int], block_hash: Optional[str], reuse_block: bool = False
    ) -> Optional[str]:
        if len([x for x in [block, block_hash, reuse_block] if x]) > 1:
            raise ValueError(
                "Of `block`, `block_hash`, or `reuse_block` only one can be specified."
            )
        else:
            if block_hash:
                return block_hash
            elif block:
                return await self.get_block_hash(block)
            else:
                return None

    async def is_hotkey_registered_any(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Checks if a neuron's hotkey is registered on any subnet within the Bittensor network.

        Args:
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            block: the blockchain block number on which to check. Do not specify if using block_hash or reuse_block
            block_hash: The blockchain block_hash representation of block id. Do not specify if using block
                or reuse_block
            reuse_block (bool): Whether to reuse the last-used block hash. Do not specify if using block_hash or block

        Returns:
            bool: `True` if the hotkey is registered on any subnet, `False` otherwise.

        This function is essential for determining the network-wide presence and participation of a neuron.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        return (
            len(await self.get_netuids_for_hotkey(hotkey_ss58, block_hash, reuse_block))
            > 0
        )

    async def get_subnet_burn_cost(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the burn cost for registering a new subnet within the Bittensor network. This cost represents the
            amount of Tao that needs to be locked or burned to establish a new subnet.

        Args:
            block: The blockchain block number on which to check. Do not specify if using block_hash or reuse_block
            block_hash: The blockchain block_hash of the block id. Do not specify if using block or
                reuse_block
            reuse_block: Whether to reuse the last-used block hash. Do not specify if using block_hash or block

        Returns:
            The burn cost for subnet registration.

        The subnet burn cost is an important economic parameter, reflecting the network's mechanisms for controlling
            the proliferation of subnets and ensuring their commitment to the network's long-term viability.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        lock_cost = await self.query_runtime_api(
            runtime_api="SubnetRegistrationRuntimeApi",
            method="get_network_registration_cost",
            params=[],
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        return lock_cost

    async def get_total_subnets(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the total number of subnets within the Bittensor network as of a specific blockchain block.

        Args:
            block: The blockchain block number on which to check. Do not specify if using block_hash or reuse_block
            block_hash (Optional[str]): The blockchain block_hash representation of block id. Do not specify if using
                block or reuse_block
            reuse_block: Whether to reuse the last-used block hash. Do not specify if using block_hash or block

        Returns:
            The total number of subnets in the network.

        Understanding the total number of subnets is essential for assessing the network's growth and the extent of its
            decentralized infrastructure.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="TotalNetworks",
            params=[],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return result

    async def get_subnets(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[int]:
        """
        Retrieves the list of all subnet unique identifiers (netuids) currently present in the Bittensor network.

        Args:
            block: The blockchain block number on which to check. Do not specify if using block_hash or reuse_block
            block_hash (Optional[str]): The hash of the block to retrieve the subnet unique identifiers from. Do not
                specify if using block or reuse_block
            reuse_block (bool): Whether to reuse the last-used block hash. Do not set if using block_hash or block

        Returns:
            A list of subnet netuids.

        This function provides a comprehensive view of the subnets within the Bittensor network,
        offering insights into its diversity and scale.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return (
            []
            if result is None or not hasattr(result, "records")
            else [netuid async for netuid, exists in result if exists]
        )

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

        Args:
            hotkey_ss58: The SS58 address of the neuron's hotkey.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block number for the query. Do not specify if using
                block or reuse_block
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or
                block

        Returns:
            `True` if the hotkey is a delegate, `False` otherwise.

        Being a delegate is a significant status within the Bittensor network, indicating a neuron's involvement in
            consensus and governance processes.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        delegates = await self.get_delegates(
            block_hash=block_hash, reuse_block=reuse_block
        )
        return hotkey_ss58 in [info.hotkey_ss58 for info in delegates]

    async def get_delegates(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[DelegateInfo]:
        """
        Fetches all delegates on the chain

        Args:
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: hash of the blockchain block number for the query. Do not specify if using block or reuse_block
            reuse_block: whether to reuse the last-used block hash. Do not set if using block_hash or block

        Returns:
            List of DelegateInfo objects, or an empty list if there are no delegates.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        hex_bytes_result = await self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegates",
            params=[],
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if hex_bytes_result is not None:
            return DelegateInfo.list_from_vec_u8(hex_to_bytes(hex_bytes_result))
        else:
            return []

    async def get_stake_info_for_coldkey(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[StakeInfo]:
        """
        Retrieves stake information associated with a specific coldkey. This function provides details about the stakes
            held by an account, including the staked amounts and associated delegates.

        Args:
            coldkey_ss58 (str): The ``SS58`` address of the account's coldkey.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block number for the query. Do not specify if using block or
                reuse_block
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block

        Returns:
            A list of StakeInfo objects detailing the stake allocations for the account.

        Stake information is vital for account holders to assess their investment and participation in the network's
            delegation and consensus processes.
        """
        encoded_coldkey = ss58_to_vec_u8(coldkey_ss58)
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        hex_bytes_result = await self.query_runtime_api(
            runtime_api="StakeInfoRuntimeApi",
            method="get_stake_info_for_coldkey",
            params=[encoded_coldkey],
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if hex_bytes_result is None:
            return []

        return StakeInfo.list_from_vec_u8(hex_to_bytes(hex_bytes_result))

    async def get_stake_for_coldkey_and_hotkey(
        self,
        hotkey_ss58: str,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """
        Retrieves stake information associated with a specific coldkey and hotkey.

        Args:
            hotkey_ss58: the hotkey SS58 address to query
            coldkey_ss58: the coldkey SS58 address to query
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: the hash of the blockchain block number for the query. Do not specify if using block or
                reuse_block
            reuse_block: whether to reuse the last-used block hash. Do not set if using block_hash or block

        Returns:
            Stake Balance for the given coldkey and hotkey
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        _result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Stake",
            params=[hotkey_ss58, coldkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return Balance.from_rao(_result or 0)

    async def query_runtime_api(
        self,
        runtime_api: str,
        method: str,
        params: Optional[Union[list[list[int]], dict[str, int], list[int]]],
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[str]:
        """
        Queries the runtime API of the Bittensor blockchain, providing a way to interact with the underlying runtime and
            retrieve data encoded in Scale Bytes format. This function is essential for advanced users who need to
            interact with specific runtime methods and decode complex data types.

        Args:
            runtime_api: The name of the runtime API to query.
            method: The specific method within the runtime API to call.
            params: The parameters to pass to the method call.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block number at which to perform the query. Do not specify if
                using block or reuse_block
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block

        Returns:
            The Scale Bytes encoded result from the runtime API call, or `None` if the call fails.

        This function enables access to the deeper layers of the Bittensor blockchain, allowing for detailed and
            specific interactions with the network's runtime environment.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)

        call_definition = TYPE_REGISTRY["runtime_api"][runtime_api]["methods"][method]

        data = (
            "0x"
            if params is None
            else await self.encode_params(
                call_definition=call_definition, params=params
            )
        )
        api_method = f"{runtime_api}_{method}"

        json_result = await self.substrate.rpc_request(
            method="state_call",
            params=[api_method, data, block_hash] if block_hash else [api_method, data],
            reuse_block_hash=reuse_block,
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

    async def get_balance(
        self,
        *addresses: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, Balance]:
        """
        Retrieves the balance for given coldkey(s)

        Args:
            addresses: coldkey addresses(s).
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: the block hash. Do not specify if using block or block
            reuse_block: whether to reuse the last-used block hash. Do not set if using block_hash or block

        Returns:
            Dict of {address: Balance objects}.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        if reuse_block:
            block_hash = self.substrate.last_block_hash
        else:
            block_hash = await self.substrate.get_chain_head()
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

    async def get_transfer_fee(
        self, wallet: "Wallet", dest: str, value: Union["Balance", float, int]
    ) -> "Balance":
        """
        Calculates the transaction fee for transferring tokens from a wallet to a specified destination address. This
            function simulates the transfer to estimate the associated cost, taking into account the current network
            conditions and transaction complexity.

        Args:
            wallet: The wallet from which the transfer is initiated.
            dest: The `SS58` address of the destination account.
            value: The amount of tokens to be transferred, specified as a Balance object, or in Tao (float) or
                Rao (int) units.

        Returns:
            The estimated transaction fee for the transfer, represented as a Balance object.

        Estimating the transfer fee is essential for planning and executing token transactions, ensuring that the
            wallet has sufficient funds to cover both the transfer amount and the associated costs.
            This function provides a crucial tool for managing financial operations within the Bittensor network.
        """
        if isinstance(value, float):
            value = Balance.from_tao(value)
        elif isinstance(value, int):
            value = Balance.from_rao(value)

        if isinstance(value, Balance):
            call = await self.substrate.compose_call(
                call_module="Balances",
                call_function="transfer_allow_death",
                call_params={"dest": dest, "value": value.rao},
            )

            try:
                payment_info = await self.substrate.get_payment_info(
                    call=call, keypair=wallet.coldkeypub
                )
            except Exception as e:
                logging.error(
                    f":cross_mark: [red]Failed to get payment info: [/red]{e}"
                )
                payment_info = {"partialFee": int(2e7)}  # assume  0.02 Tao

            return Balance.from_rao(payment_info["partialFee"])
        else:
            fee = Balance.from_rao(int(2e7))
            logging.error(
                "To calculate the transaction fee, the value must be Balance, float, or int. Received type: %s. Fee "
                "is %s",
                type(value),
                2e7,
            )
            return fee

    async def get_total_stake_for_coldkey(
        self,
        *ss58_addresses,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, Balance]:
        """
        Returns the total stake held on a coldkey.

        Args:
            ss58_addresses: The SS58 address(es) of the coldkey(s)
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the block number to retrieve the stake from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block

        Returns:
            Dict in view {address: Balance objects}.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        if reuse_block:
            block_hash = self.substrate.last_block_hash
        elif not block_hash:
            block_hash = await self.substrate.get_chain_head()
        calls = [
            (
                await self.substrate.create_storage_key(
                    "SubtensorModule",
                    "TotalColdkeyStake",
                    [address],
                    block_hash=block_hash,
                )
            )
            for address in ss58_addresses
        ]
        batch_call = await self.substrate.query_multi(calls, block_hash=block_hash)
        results = {}
        for item in batch_call:
            results.update({item[0].params[0]: Balance.from_rao(item[1] or 0)})
        return results

    async def get_total_stake_for_hotkey(
        self,
        *ss58_addresses,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, Balance]:
        """
        Returns the total stake held on a hotkey.

        Args:
            ss58_addresses: The SS58 address(es) of the hotkey(s)
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the block number to retrieve the stake from. Do not specify if using block
                or reuse_block
            reuse_block: Whether to reuse the last-used block hash when retrieving info. Do not set if using block_hash
                or block

        Returns:
            Dict {address: Balance objects}.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        results = await self.substrate.query_multiple(
            params=[s for s in ss58_addresses],
            module="SubtensorModule",
            storage_function="TotalHotkeyStake",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return {k: Balance.from_rao(r or 0) for (k, r) in results.items()}

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

        Args:
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block number at which to perform the query. Do not specify if using
                block or reuse_block
            reuse_block: Whether to reuse the last-used block hash when retrieving info. Do not set if using block_hash
                or block.

        Returns:
            A list of netuids where the neuron is a member.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="IsNetworkMember",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return (
            [record[0] async for record in result if record[1]]
            if result and hasattr(result, "records")
            else []
        )

    async def subnet_exists(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Checks if a subnet with the specified unique identifier (netuid) exists within the Bittensor network.

        Args:
            netuid: The unique identifier of the subnet.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block number at which to check the subnet existence. Do not specify
                if using block or reuse_block
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            `True` if the subnet exists, `False` otherwise.

        This function is critical for verifying the presence of specific subnets in the network,
        enabling a deeper understanding of the network's structure and composition.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return result

    async def filter_netuids_by_registered_hotkeys(
        self,
        all_netuids: Iterable[int],
        filter_for_netuids: Iterable[int],
        all_hotkeys: Iterable[Wallet],
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[int]:
        """
        Filters a given list of all netuids for certain specified netuids and hotkeys

        Args:
            all_netuids: A list of netuids to filter.
            filter_for_netuids: A subset of all_netuids to filter from the main list
            all_hotkeys: Hotkeys to filter from the main list
            block: the block number for this query. Do not specify if using block_hash or reuse_block.
            block_hash: hash of the blockchain block number at which to perform the query. Do not specify if using block
                or reuse_block
            reuse_block: whether to reuse the last-used blockchain hash when retrieving info. Do not set if using
                block_hash or block.

        Returns:
            The filtered list of netuids.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
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

        Args:
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: Block hash at which to query the deposit amount. Do not specify if using block or reuse_block
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            The existential deposit amount.

        The existential deposit is a fundamental economic parameter in the Bittensor network, ensuring efficient use of
            storage and preventing the proliferation of dust accounts.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.get_constant(
            module_name="Balances",
            constant_name="ExistentialDeposit",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        if result is None:
            raise Exception("Unable to retrieve existential deposit amount.")

        return Balance.from_rao(result)

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

        Args:
            netuid: The unique identifier of the subnet.
            block: the block number for this query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the blockchain block number for the query. Do not specify if using block
                or reuse_block
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            A list of NeuronInfo objects detailing each neuron's characteristics in the subnet.

        Understanding the distribution and status of neurons within a subnet is key to comprehending the network's
            decentralized structure and the dynamics of its consensus and governance processes.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        hex_bytes_result = await self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons",
            params=[netuid],
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if hex_bytes_result is None:
            return []

        return NeuronInfo.list_from_vec_u8(hex_to_bytes(hex_bytes_result))

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

        Args:
            netuid: The unique identifier of the subnet.
            block: the block number for this query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the blockchain block number for the query. Do not specify if using block
                or reuse_block
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            A list of simplified neuron information for the subnet.

        This function offers a quick overview of the neuron population within a subnet, facilitating efficient analysis
            of the network's decentralized structure and neuron dynamics.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        hex_bytes_result = await self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons_lite",
            params=[netuid],
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if hex_bytes_result is None:
            return []

        return NeuronInfoLite.list_from_vec_u8(hex_to_bytes(hex_bytes_result))

    async def get_neuron_for_pubkey_and_subnet(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> NeuronInfo:
        """
        Retrieves information about a neuron based on its public key (hotkey SS58 address) and the specific subnet
            UID (netuid). This function provides detailed neuron information for a particular subnet within the
            Bittensor network.

        Args:
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The blockchain block number at which to perform the query. Do not specify if using block
                or reuse_block
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            Detailed information about the neuron if found, null neuron if not.

        This function is crucial for accessing specific neuron data and understanding its status, stake, and other
            attributes within a particular subnet of the Bittensor ecosystem.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        uid = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        if uid is None:
            return NeuronInfo.get_null_neuron()

        params = [netuid, uid]
        json_body = await self.substrate.rpc_request(
            method="neuronInfo_getNeuron",
            params=params,
        )

        if not (result := json_body.get("result", None)):
            return NeuronInfo.get_null_neuron()

        return NeuronInfo.from_vec_u8(bytes(result))

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
            specified subnet (netuid) of the Bittensor network. This function provides a comprehensive view of a
            neuron's attributes, including its stake, rank, and operational status.

        Args:
            uid: The unique identifier of the neuron.
            netuid: The unique identifier of the subnet.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block number for the query. Do not specify if using block or
                reuse_block
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            Detailed information about the neuron if found, a null neuron otherwise

        This function is crucial for analyzing individual neurons' contributions and status within a specific subnet,
            offering insights into their roles in the network's consensus and validation mechanisms.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        if uid is None:
            return NeuronInfo.get_null_neuron()

        if reuse_block:
            block_hash = self.substrate.last_block_hash

        params = [netuid, uid, block_hash] if block_hash else [netuid, uid]
        json_body = await self.substrate.rpc_request(
            method="neuronInfo_getNeuron",
            params=params,  # custom rpc method
        )
        if not (result := json_body.get("result", None)):
            return NeuronInfo.get_null_neuron()

        bytes_result = bytes(result)
        return NeuronInfo.from_vec_u8(bytes_result)

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

        Args:
            coldkey_ss58: The `SS58` address of the account's coldkey.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block number for the query. Do not specify if using block or
                reuse_block
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            A list of tuples, each containing a delegate's information and staked amount.

        This function is important for account holders to understand their stake allocations and their involvement in
            the network's delegation and consensus mechanisms.
        """
        block_hash = (
            bh
            if (bh := await self._determine_block_hash(block, block_hash, reuse_block))
            else (self.substrate.last_block_hash if reuse_block else None)
        )
        encoded_coldkey = ss58_to_vec_u8(coldkey_ss58)
        json_body = await self.substrate.rpc_request(
            method="delegateInfo_getDelegated",
            params=([block_hash, encoded_coldkey] if block_hash else [encoded_coldkey]),
        )

        if not (result := json_body.get("result")):
            return []

        return DelegateInfo.delegated_list_from_vec_u8(bytes(result))

    async def query_identity(
        self,
        key: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict:
        """
        Queries the identity of a neuron on the Bittensor blockchain using the given key. This function retrieves
            detailed identity information about a specific neuron, which is a crucial aspect of the network's
            decentralized identity and governance system.

        Args:
            key: The key used to query the neuron's identity, typically the neuron's SS58 address.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block number at which to perform the query. Do not specify if using
                block_hash or block.
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            An object containing the identity information of the neuron if found, ``None`` otherwise.

        The identity information can include various attributes such as the neuron's stake, rank, and other
            network-specific details, providing insights into the neuron's role and status within the Bittensor network.

        Note:
            See the `Bittensor CLI documentation <https://docs.bittensor.com/reference/btcli>`_ for supported
            identity parameters.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        identity_info = await self.substrate.query(
            module="Registry",
            storage_function="IdentityOf",
            params=[key],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        try:
            return _decode_hex_identity_dict(identity_info["info"])
        except TypeError:
            return {}

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

        Args:
            netuid: The network UID of the subnet to query.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block for the query. Do not specify if using reuse_block or block.
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            A list of tuples mapping each neuron's UID to its assigned weights.

        The weight distribution is a key factor in the network's consensus algorithm and the ranking of neurons,
            influencing their influence and reward allocation within the subnet.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        # TODO look into seeing if we can speed this up with storage query
        w_map_encoded = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Weights",
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        w_map = [(uid, w or []) async for uid, w in w_map_encoded]

        return w_map

    async def bonds(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the bond distribution set by neurons within a specific subnet of the Bittensor network.
            Bonds represent the investments or commitments made by neurons in one another, indicating a level of trust
            and perceived value. This bonding mechanism is integral to the network's market-based approach to
            measuring and rewarding machine intelligence.

        Args:
            netuid: The network UID of the subnet to query.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block number for the query. Do not specify if using reuse_block or
                block.
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            List of tuples mapping each neuron's UID to its bonds with other neurons.

        Understanding bond distributions is crucial for analyzing the trust dynamics and market behavior within the
            subnet. It reflects how neurons recognize and invest in each other's intelligence and contributions,
            supporting diverse and niche systems within the Bittensor ecosystem.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        b_map_encoded = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Bonds",
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        b_map = [(uid, b) async for uid, b in b_map_encoded]

        return b_map

    async def does_hotkey_exist(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Returns true if the hotkey is known by the chain and there are accounts.

        Args:
            hotkey_ss58: The SS58 address of the hotkey.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the block number to check the hotkey against. Do not specify if using reuse_block
                or block.
            reuse_block: Whether to reuse the last-used blockchain hash. Do not set if using block_hash or block.

        Returns:
            `True` if the hotkey is known by the chain and there are accounts, `False` otherwise.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        _result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Owner",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        result = decode_account_id(_result[0])
        return_val = (
            False
            if result is None
            else result != "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        )
        return return_val

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

        Args:
            hotkey_ss58: The SS58 address of the hotkey.
            block: the block number for this query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the block at which to check the hotkey ownership. Do not specify if using
                reuse_block or block.
            reuse_block: Whether to reuse the last-used blockchain hash. Do not set if using block_hash or block.

        Returns:
            Optional[str]: The SS58 address of the owner if the hotkey exists, or None if it doesn't.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        hk_owner_query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Owner",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        val = decode_account_id(hk_owner_query[0])
        if val:
            exists = await self.does_hotkey_exist(hotkey_ss58, block_hash=block_hash)
        else:
            exists = False
        hotkey_owner = val if exists else None
        return hotkey_owner

    async def sign_and_send_extrinsic(
        self,
        call: "GenericCall",
        wallet: "Wallet",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> tuple[bool, str]:
        """
        Helper method to sign and submit an extrinsic call to chain.

        Args:
            call (scalecodec.types.GenericCall): a prepared Call object
            wallet (bittensor_wallet.Wallet): the wallet whose coldkey will be used to sign the extrinsic
            wait_for_inclusion (bool): whether to wait until the extrinsic call is included on the chain
            wait_for_finalization (bool): whether to wait until the extrinsic call is finalized on the chain

        Returns:
            (success, error message)
        """
        extrinsic = await self.substrate.create_signed_extrinsic(
            call=call, keypair=wallet.coldkey
        )  # sign with coldkey
        try:
            response = await self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True, ""
            if await response.is_success:
                return True, ""
            else:
                return False, format_error_message(await response.error_message)
        except SubstrateRequestException as e:
            return False, format_error_message(e)

    async def get_children(self, hotkey: str, netuid: int) -> tuple[bool, list, str]:
        """
        This method retrieves the children of a given hotkey and netuid. It queries the SubtensorModule's ChildKeys
            storage function to get the children and formats them before returning as a tuple.

        Args:
            hotkey: The hotkey value.
            netuid: The netuid value.

        Returns:
            A tuple containing a boolean indicating success or failure, a list of formatted children, and an error
                message (if applicable)
        """
        try:
            children = await self.substrate.query(
                module="SubtensorModule",
                storage_function="ChildKeys",
                params=[hotkey, netuid],
            )
            if children:
                formatted_children = []
                for proportion, child in children:
                    # Convert U64 to int
                    formatted_child = decode_account_id(child[0])
                    int_proportion = int(proportion)
                    formatted_children.append((int_proportion, formatted_child))
                return True, formatted_children, ""
            else:
                return True, [], ""
        except SubstrateRequestException as e:
            return False, [], format_error_message(e)

    async def get_subnet_hyperparameters(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Union[list, SubnetHyperparameters]]:
        """
        Retrieves the hyperparameters for a specific subnet within the Bittensor network. These hyperparameters define
            the operational settings and rules governing the subnet's behavior.

        Args:
            netuid: The network UID of the subnet to query.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the blockchain block number for the query. Do not specify if using bloc or
                reuse_block.
            reuse_block: Whether to reuse the last-used blockchain hash. Do not set if using block_hash or reuse_block.

        Returns:
            The subnet's hyperparameters, or `None` if not available.

        Understanding the hyperparameters is crucial for comprehending how subnets are configured and managed, and how
            they interact with the network's consensus and incentive mechanisms.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        hex_bytes_result = await self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_hyperparams",
            params=[netuid],
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if hex_bytes_result is None:
            return []

        return SubnetHyperparameters.from_vec_u8(hex_to_bytes(hex_bytes_result))

    async def get_vote_data(
        self,
        proposal_hash: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[ProposalVoteData]:
        """
        Retrieves the voting data for a specific proposal on the Bittensor blockchain. This data includes information
            about how senate members have voted on the proposal.

        Args:
            proposal_hash: The hash of the proposal for which voting data is requested.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the blockchain block number to query the voting data. Do not specify if using block
                or reuse_block.
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or
                block.

        Returns:
            An object containing the proposal's voting data, or `None` if not found.

        This function is important for tracking and understanding the decision-making processes within the Bittensor
            network, particularly how proposals are received and acted upon by the governing body.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        vote_data = await self.substrate.query(
            module="Triumvirate",
            storage_function="Voting",
            params=[proposal_hash],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        if vote_data is None:
            return None
        else:
            return ProposalVoteData(vote_data)

    async def get_delegate_identities(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, DelegatesDetails]:
        """
        Fetches delegates identities from the chain and GitHub. Preference is given to chain data, and missing info is
            filled-in by the info from GitHub. At some point, we want to totally move away from fetching this info from
            GitHub, but chain data is still limited in that regard.

        Args:
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: the hash of the blockchain block for the query Do not specify if using block or reuse_block.
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block.

        Returns:
            Dict {ss58: DelegatesDetails, ...}

        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        timeout = aiohttp.ClientTimeout(10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            identities_info, response = await asyncio.gather(
                self.substrate.query_map(
                    module="Registry",
                    storage_function="IdentityOf",
                    block_hash=block_hash,
                    reuse_block_hash=reuse_block,
                ),
                session.get(DELEGATES_DETAILS_URL),
            )

            all_delegates_details = {
                decode_account_id(ss58_address[0]): DelegatesDetails.from_chain_data(
                    decode_hex_identity_dict(identity["info"])
                )
                for ss58_address, identity in identities_info
            }

            if response.ok:
                all_delegates: dict[str, Any] = await response.json(content_type=None)

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
                    delegate_info.display = (
                        delegate_info.display or delegate_details.get("name", "")
                    )
                    delegate_info.web = delegate_info.web or delegate_details.get(
                        "url", ""
                    )
                    delegate_info.additional = (
                        delegate_info.additional
                        or delegate_details.get("description", "")
                    )
                    delegate_info.pgp_fingerprint = (
                        delegate_info.pgp_fingerprint
                        or delegate_details.get("fingerprint", "")
                    )

        return all_delegates_details

    async def is_hotkey_registered(
        self,
        netuid: int,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """Checks to see if the hotkey is registered on a given netuid"""
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        if result is not None:
            return True
        else:
            return False

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

        Args:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The blockchain block_hash representation of the block id. Do not specify if using block or
                reuse_block
            reuse_block (bool): Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or
                reuse_block.

        Returns:
            The UID of the neuron if it is registered on the subnet, `None` otherwise.

        The UID is a critical identifier within the network, linking the neuron's hotkey to its operational and
            governance activities on a particular subnet.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return result

    async def weights_rate_limit(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Returns network WeightsSetRateLimit hyperparameter.

        Args:
            netuid: The unique identifier of the subnetwork.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The blockchain block_hash representation of the block id. Do not specify if using block or
                reuse_block
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or block

        Returns:
            The value of the WeightsSetRateLimit hyperparameter, or `None` if the subnetwork does not exist or the
            parameter is not found.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="WeightsSetRateLimit",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def blocks_since_last_update(self, netuid: int, uid: int) -> Optional[int]:
        """
        Returns the number of blocks since the last update for a specific UID in the subnetwork.

        Args:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.

        Returns:
            The number of blocks since the last update, or `None` if the subnetwork or UID does not exist.
        """
        call = await self.get_hyperparameter(param_name="LastUpdate", netuid=netuid)
        return None if call is None else await self.get_current_block() - int(call[uid])

    async def commit_reveal_enabled(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Check if commit-reveal mechanism is enabled for a given network at a specific block.

        Arguments:
            netuid: The network identifier for which to check the commit-reveal mechanism.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash of block at which to check the parameter. Do not set if using block or
                reuse_block.
            reuse_block: Whether to reuse the last-used blockchain block hash. Do not set if using block_hash or
                block.

        Returns:
            Returns the integer value of the hyperparameter if available; otherwise, returns None.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="CommitRevealWeightsEnabled",
            block_hash=block_hash,
            netuid=netuid,
            reuse_block=reuse_block,
        )
        return True if call is True else False

    async def get_subnet_reveal_period_epochs(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> int:
        """Retrieve the SubnetRevealPeriodEpochs hyperparameter."""
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        return await self.get_hyperparameter(
            param_name="RevealPeriodEpochs",
            block_hash=block_hash,
            netuid=netuid,
            reuse_block=reuse_block,
        )

    # Extrinsics =======================================================================================================

    async def transfer(
        self,
        wallet: "Wallet",
        destination: str,
        amount: float,
        transfer_all: bool,
    ) -> bool:
        """
        Transfer token of amount to destination.

        Args:
            wallet (bittensor_wallet.Wallet): Source wallet for the transfer.
            destination (str): Destination address for the transfer.
            amount (float): Amount of tokens to transfer.
            transfer_all (bool): Flag to transfer all tokens.

        Returns:
            `True` if the transferring was successful, otherwise `False`.
        """
        return await transfer_extrinsic(
            self,
            wallet,
            destination,
            Balance.from_tao(amount),
            transfer_all,
        )

    async def root_register(
        self,
        wallet: "Wallet",
        netuid: int,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> bool:
        """
        Register neuron by recycling some TAO.

        Args:
            wallet: Bittensor wallet instance.
            netuid: Subnet uniq id.
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is `False`.

        Returns:
            `True` if registration was successful, otherwise `False`.
        """
        logging.info(
            f"Registering on netuid [blue]0[/blue] on network: [blue]{self.network}[/blue]"
        )

        # Check current recycle amount
        logging.info("Fetching recycle amount & balance.")
        block_hash = await self.substrate.get_chain_head()
        recycle_call, balance_ = await asyncio.gather(
            self.get_hyperparameter(
                param_name="Burn", netuid=netuid, block_hash=block_hash
            ),
            self.get_balance(wallet.coldkeypub.ss58_address, block_hash=block_hash),
        )
        current_recycle = Balance.from_rao(int(recycle_call))
        try:
            balance: Balance = balance_[wallet.coldkeypub.ss58_address]
        except TypeError as e:
            logging.error(f"Unable to retrieve current recycle. {e}")
            return False
        except KeyError:
            logging.error("Unable to retrieve current balance.")
            return False

        # Check balance is sufficient
        if balance < current_recycle:
            logging.error(
                f"[red]Insufficient balance {balance} to register neuron. "
                f"Current recycle is {current_recycle} TAO[/red]."
            )
            return False

        return await root_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def register(
        self: "AsyncSubtensor",
        wallet: "Wallet",
        netuid: int,
        processors: int,
        update_interval: int,
        output_in_place: bool,
        verbose: bool,
        use_cuda: bool,
        dev_id: Union[list[int], int],
        threads_per_block: int,
    ):
        """Register neuron."""
        return await register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            tpb=threads_per_block,
            update_interval=update_interval,
            num_processes=processors,
            cuda=use_cuda,
            dev_id=dev_id,
            output_in_place=output_in_place,
            log_verbose=verbose,
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
    ) -> tuple[bool, str]:
        """
        Sets the interneuronal weights for the specified neuron. This process involves specifying the influence or
            trust a neuron places on other neurons in the network, which is a fundamental aspect of Bittensor's
            decentralized learning architecture.

        Args:
            wallet: The wallet associated with the neuron setting the weights.
            netuid: The unique identifier of the subnet.
            uids: The list of neuron UIDs that the weights are being set for.
            weights: The corresponding weights to be set for each UID.
            version_key: Version key for compatibility with the network.  Default is int representation of Bittensor
                version.
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is `False`.
            max_retries: The number of maximum attempts to set weights. Default is `5`.

        Returns:
            `True` if the setting of weights is successful, `False` otherwise. And `msg`, a string value describing the
                success or potential error.

        This function is crucial in shaping the network's collective intelligence, where each neuron's learning and
            contribution are influenced by the weights it sets towards others【81†source】.
        """
        if (await self.commit_reveal_enabled(netuid=netuid)) is True:
            # go with `commit reveal v3` extrinsic
            raise NotImplementedError(
                "Not implemented yet for AsyncSubtensor. Coming soon."
            )
        else:
            # go with classic `set weights extrinsic`
            uid = await self.get_uid_for_hotkey_on_subnet(
                wallet.hotkey.ss58_address, netuid
            )
            retries = 0
            success = False
            message = "No attempt made. Perhaps it is too soon to set weights!"
            while (
                retries < max_retries
                and await self.blocks_since_last_update(netuid, uid)
                > await self.weights_rate_limit(netuid)
                and success is False
            ):
                try:
                    logging.info(
                        f"Setting weights for subnet #[blue]{netuid}[/blue]. "
                        f"Attempt [blue]{retries + 1} of {max_retries}[/blue]."
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
                    )
                except Exception as e:
                    logging.error(f"Error setting weights: {e}")
                finally:
                    retries += 1

            return success, message

    async def root_set_weights(
        self,
        wallet: "Wallet",
        netuids: list[int],
        weights: list[float],
    ) -> bool:
        """
        Set weights for root network.

        Args:
            wallet (bittensor_wallet.Wallet): bittensor wallet instance.
            netuids (list[int]): The list of subnet uids.
            weights (list[float]): The list of weights to be set.

        Returns:
            `True` if the setting of weights is successful, `False` otherwise.
        """
        netuids_ = np.array(netuids, dtype=np.int64)
        weights_ = np.array(weights, dtype=np.float32)
        logging.info(f"Setting weights in network: [blue]{self.network}[/blue]")
        # Run the set weights operation.
        return await set_root_weights_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids_,
            weights=weights_,
            version_key=0,
            wait_for_finalization=True,
            wait_for_inclusion=True,
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
    ) -> tuple[bool, str]:
        """
        Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
        This action serves as a commitment or snapshot of the neuron's current weight distribution.

        Args:
            wallet: The wallet associated with the neuron committing the weights.
            netuid: The unique identifier of the subnet.
            salt: list of randomly generated integers as salt to generated weighted hash.
            uids: NumPy array of neuron UIDs for which weights are being committed.
            weights: NumPy array of weight values corresponding to each UID.
            version_key: Version key for compatibility with the network. Default is int representation of Bittensor
                version.
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is `False`.
            max_retries: The number of maximum attempts to commit weights. Default is `5`.

        Returns:
            (success, message) where message is a string value describing the success or potential error

        This function allows neurons to create a tamper-proof record of their weight distribution at a specific point
            in time, enhancing transparency and accountability within the Bittensor network.
        """
        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to commit weights!"

        logging.info(
            f"Committing weights with params: "
            f"netuid={netuid}, uids={uids}, weights={weights}, version_key={version_key}"
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

        while retries < max_retries:
            try:
                success, message = await commit_weights_extrinsic(
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

    async def get_all_subnets_info(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ):
        """
        Retrieves detailed information about all subnets within the Bittensor network. This function provides
            comprehensive data on each subnet, including its characteristics and operational parameters.

        Args:
            block: The blockchain block number for the query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the blockchain block. Do not specify if using block or reuse_block

        Returns:
            list[SubnetInfo]: A list of SubnetInfo objects, each containing detailed information about a subnet.

        Gaining insights into the subnets' details assists in understanding the network's composition, the roles of
            different subnets, and their unique features.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        hex_bytes_result = await self.query_runtime_api(
            "SubnetInfoRuntimeApi",
            "get_subnets_info",
            params=[],
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if not hex_bytes_result:
            return []
        else:
            return SubnetInfo.list_from_vec_u8(hex_to_bytes(hex_bytes_result))

    async def get_minimum_required_stake(self):
        """
        Returns the minimum required stake for nominators in the Subtensor network.

        This method retries the substrate call up to three times with exponential backoff in case of failures.

        Returns:
            Balance: The minimum required stake as a Balance object.

        Raises:
            Exception: If the substrate call fails after the maximum number of retries.
        """
        result = await self.substrate.query(
            module="SubtensorModule", storage_function="NominatorMinRequiredStake"
        )
        return Balance.from_rao(result)

    async def tempo(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Returns network Tempo hyperparameter.

        Args:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The value of the Tempo hyperparameter, or `None` if the subnetwork does not exist or the parameter
            is not found.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="Tempo",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def difficulty(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the 'Difficulty' hyperparameter for a specified subnet in the Bittensor network.

        This parameter is instrumental in determining the computational challenge required for neurons to participate
            in consensus and validation processes.

        Args:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query. Do not specify if using block_hash or reuse_block
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The value of the 'Difficulty' hyperparameter if the subnet exists, `None` otherwise.

        The 'Difficulty' parameter directly impacts the network's security and integrity by setting the computational
            effort required for validating transactions and participating in the network's consensus mechanism.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="Difficulty",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if call is None:
            return None
        return int(call)

    async def query_subtensor(
        self,
        name: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
        params: Optional[list] = None,
    ) -> "ScaleType":
        """
        Queries named storage from the Subtensor module on the Bittensor blockchain. This function is used to retrieve
            specific data or parameters from the blockchain, such as stake, rank, or other neuron-specific attributes.

        Args:
            name: The name of the storage function to query.
            block: The blockchain block number at which to perform the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.
            params: A list of parameters to pass to the query function.

        Returns:
            query_response (scalecodec.ScaleType): An object containing the requested data.

        This query function is essential for accessing detailed information about the network and its neurons, providing
            valuable insights into the state and dynamics of the Bittensor ecosystem.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.query(
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
    ) -> "ScaleType":
        """
        Queries any module storage on the Bittensor blockchain with the specified parameters and block number. This
            function is a generic query interface that allows for flexible and diverse data retrieval from various
            blockchain modules.

        Args:
            module (str): The name of the module from which to query data.
            name (str): The name of the storage function within the module.
            block (Optional[int]): The blockchain block number at which to perform the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.
            params (Optional[list[object]]): A list of parameters to pass to the query function.

        Returns:
            An object containing the requested data if found, `None` otherwise.

        This versatile query function is key to accessing a wide range of data and insights from different parts of the
            Bittensor blockchain, enhancing the understanding and analysis of the network's state and dynamics.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.query(
            module=module,
            storage_function=name,
            params=params,
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
    ) -> "QueryMapResult":
        """
        Queries map storage from any module on the Bittensor blockchain. This function retrieves data structures that
            represent key-value mappings, essential for accessing complex and structured data within the blockchain
            modules.

        Args:
            module: The name of the module from which to query the map storage.
            name: The specific storage function within the module to query.
            block: The blockchain block number at which to perform the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.
            params: Parameters to be passed to the query.

        Returns:
            result: A data structure representing the map storage if found, `None` otherwise.

        This function is particularly useful for retrieving detailed and structured data from various blockchain
            modules, offering insights into the network's state and the relationships between its different components.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.query_map(
            module=module,
            storage_function=name,
            params=params,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def query_map_subtensor(
        self,
        name: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
        params: Optional[list] = None,
    ) -> "QueryMapResult":
        """
        Queries map storage from the Subtensor module on the Bittensor blockchain. This function is designed to retrieve
            a map-like data structure, which can include various neuron-specific details or network-wide attributes.

        Args:
            name: The name of the map storage function to query.
            block: The blockchain block number at which to perform the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.
            params: A list of parameters to pass to the query function.

        Returns:
            An object containing the map-like data structure, or `None` if not found.

        This function is particularly useful for analyzing and understanding complex network structures and
            relationships within the Bittensor ecosystem, such as interneuronal connections and stake distributions.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.query_map(
            module="SubtensorModule",
            storage_function=name,
            params=params,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def get_neuron_certificate(
        self,
        hotkey: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Certificate]:
        """
        Retrieves the TLS certificate for a specific neuron identified by its unique identifier (UID)
        within a specified subnet (netuid) of the Bittensor network.

        Args:
            hotkey: The hotkey to query.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            the certificate of the neuron if found, `None` otherwise.

        This function is used for certificate discovery for setting up mutual tls communication between neurons
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        certificate = await self.query_module(
            module="SubtensorModule",
            name="NeuronCertificates",
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[netuid, hotkey],
        )
        try:
            if certificate:
                return "".join(
                    chr(i)
                    for i in chain(
                        [certificate["algorithm"]],
                        certificate["public_key"][0],
                    )
                )

        except AttributeError:
            return None
        return None

    async def serve_axon(
        self,
        netuid: int,
        axon: "Axon",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        certificate: Optional[Certificate] = None,
    ) -> bool:
        """
        Registers an `Axon` serving endpoint on the Bittensor network for a specific neuron. This function is used to
            set up the Axon, a key component of a neuron that handles incoming queries and data processing tasks.

        Args:
            netuid: The unique identifier of the subnetwork.
            axon: The Axon instance to be registered for serving.
            wait_for_inclusion: Waits for the transaction to be included in a block. Default is `False`.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain. Default is `True`.
            certificate: the certificate of the neuron

        Returns:
            `True` if the Axon serve registration is successful, `False` otherwise.

        By registering an Axon, the neuron becomes an active part of the network's distributed computing infrastructure,
            contributing to the collective intelligence of Bittensor.
        """
        return await serve_axon_extrinsic(
            self, netuid, axon, wait_for_inclusion, wait_for_finalization, certificate
        )

    async def is_hotkey_registered_on_subnet(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Checks if a neuron's hotkey is registered on a specific subnet within the Bittensor network.

        Args:
            hotkey_ss58 (str): The `SS58` address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number at which to perform the check.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            bool: `True` if the hotkey is registered on the specified subnet, False otherwise.

        This function helps in assessing the participation of a neuron in a particular subnet, indicating its specific
            area of operation or influence within the network.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        return (
            await self.get_uid_for_hotkey_on_subnet(
                hotkey_ss58, netuid, block_hash=block_hash, reuse_block=reuse_block
            )
        ) is not None

    @property
    async def block(self) -> int:
        """
        Returns current chain block.

        Returns:
            block: Current chain block.
        """
        return await self.get_current_block()

    async def metagraph(
        self, netuid: int, lite: bool = True, block: Optional[int] = None
    ) -> AsyncMetagraph:
        """
        Returns a synced metagraph for a specified subnet within the Bittensor network. The metagraph represents the
            network's structure, including neuron connections and interactions.

        Args:
            netuid: The network UID of the subnet to query.
            lite: If true, returns a metagraph using a lightweight sync (no weights, no bonds). Default is `True`.
            block: Block number for synchronization, or `None` for the latest block.

        Returns:
            The metagraph representing the subnet's structure and neuron relationships.

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

    async def get_delegate_take(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[float]:
        """
        Retrieves the delegate 'take' percentage for a neuron identified by its hotkey. The 'take' represents the
            percentage of rewards that the delegate claims from its nominators' stakes.

        Args:
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The delegate take percentage, `None` if not available.

        The delegate take is a critical parameter in the network's incentive structure, influencing the distribution of
            rewards among neurons and their nominators.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        _result = await self.query_subtensor(
            "Delegates",
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[hotkey_ss58],
        )
        return None if _result is None else u16_normalized_float(_result)

    async def get_prometheus_info(
        self,
        netuid: int,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[PrometheusInfo]:
        """
        Returns the prometheus information for this hotkey account.

        Args:
            netuid: The unique identifier of the subnetwork.
            hotkey_ss58: The SS58 address of the hotkey.
            block: The block number to retrieve the prometheus information from. If `None`, the latest block is used.
                Default is `None`.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            A PrometheusInfo object containing the prometheus information, or `None` if the prometheus information is
                not found.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        result = await self.query_subtensor(
            "Prometheus",
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[netuid, hotkey_ss58],
        )
        if result is not None:
            return PrometheusInfo(
                ip=networking.int_to_ip(result["ip"]),
                ip_type=result["ip_type"],
                port=result["port"],
                version=result["version"],
                block=result["block"],
            )
        return None

    async def tx_rate_limit(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the transaction rate limit for the Bittensor network as of a specific blockchain block.
        This rate limit sets the maximum number of transactions that can be processed within a given time frame.

        Args:
            block: The blockchain block number at which to perform the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The transaction rate limit of the network, None if not available.

        The transaction rate limit is an essential parameter for ensuring the stability and scalability of the Bittensor
            network. It helps in managing network load and preventing congestion, thereby maintaining efficient and
            timely transaction processing.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        result = await self.query_subtensor(
            "TxRateLimit", block_hash=block_hash, reuse_block=reuse_block
        )
        return result

    async def max_weight_limit(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[float]:
        """
        Returns network MaxWeightsLimit hyperparameter.

        Args:
            netuid: The unique identifier of the subnetwork.
            block: The block number to retrieve the parameter from. If `None`, the latest block is used.
                Default is `None`.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The value of the MaxWeightsLimit hyperparameter, or `None` if the subnetwork does not exist or the parameter
                is not found.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="MaxWeightsLimit",
            block_hash=block_hash,
            netuid=netuid,
            reuse_block=reuse_block,
        )
        return None if call is None else u16_normalized_float(int(call))

    async def subnetwork_n(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Returns network SubnetworkN hyperparameter.

        Args:
            netuid: The unique identifier of the subnetwork.
            block: The block number to retrieve the parameter from. If `None`, the latest block is used.
                Default is `None`.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The value of the SubnetworkN hyperparameter, or `None` if the subnetwork does not exist or the parameter is
                not found.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="SubnetworkN",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def recycle(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["Balance"]:
        """
        Retrieves the 'Burn' hyperparameter for a specified subnet. The 'Burn' parameter represents the amount of Tao
            that is effectively recycled within the Bittensor network.

        Args:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            Optional[Balance]: The value of the 'Burn' hyperparameter if the subnet exists, None otherwise.

        Understanding the 'Burn' rate is essential for analyzing the network registration usage, particularly how it is
            correlated with user activity and the overall cost of participation in a given subnet.
        """
        block_hash = await self._determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="Burn",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else Balance.from_rao(int(call))

    async def unstake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Optional[Union["Balance", float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting
            individual neuron stakes within the Bittensor network.

        Args:
            wallet: The wallet associated with the neuron from which the stake is being removed.
            hotkey_ss58: The `SS58` address of the hotkey account to unstake from.
            amount: The amount of TAO to unstake. If not specified, unstakes all.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            `True` if the unstaking process is successful, `False` otherwise.

        This function supports flexible stake management, allowing neurons to adjust their network participation and
            potential reward accruals.
        """
        return unstake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
